from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import vtk
from scipy.ndimage import (
    gaussian_filter,
    label,
    binary_opening,
    binary_closing,
    binary_erosion,
    binary_dilation,
)
from scipy.optimize import minimize
from vtk.util.numpy_support import numpy_to_vtk


@dataclass
class BoreholeConfig:
    chosen_axis: int = 0
    frac_start: float = 0.01
    frac_end: float = 0.69
    n_slices: int = 35
    thickness_frac: float = 0.02
    grid_n: int = 250
    subsample: int = 25000
    smooth_sigma: float = 4.0
    low_density_q: float = 0.10
    pad_frac_a: float = 0.10
    pad_frac_b: float = 0.10
    do_morph: bool = True
    open_iters: int = 1
    close_iters: int = 1
    use_occupancy_gate: bool = True
    occ_dilate_iters: int = 6
    min_comp_area: int = 150
    max_comp_frac: float = 0.20
    do_mirror: bool = True
    mirror_axis: int = 1
    mirror_mode: str = "augment"
    ellipse_q: float = 90.0
    use_outer_band_refinement: bool = True
    outer_band_sectors: int = 24
    outer_band_keep_per_sector: int = 6
    outer_band_min_points: int = 48
    use_ellipse_refinement: bool = True
    ellipse_refine_maxiter: int = 200
    inlier_tau_frac: float = 0.10
    fit_score_w_resid_med: float = 1.0
    fit_score_w_resid_p90: float = 0.35
    fit_score_w_inlier: float = 0.003
    fit_score_w_center: float = 0.75
    fit_score_w_axis: float = 0.004
    fit_score_w_theta: float = 0.00004
    fit_score_w_comp_center: float = 0.35
    fit_score_w_anchor: float = 0.2
    use_component_centroid_prior: bool = True
    use_confident_anchor: bool = True
    anchor_warmup_slices: int = 6
    anchor_min_inlier: float = 0.97
    anchor_max_resid_med: float = 0.0005
    enable_continuity: bool = True
    max_ar: float = 4.0
    max_axis_jump: float = 2.0
    max_center_jump: Optional[float] = None
    enable_short_gap_fill: bool = True
    max_interp_gap: int = 2
    n_ring: int = 240
    cap_ends: bool = True
    max_axis_gap: Optional[float] = None
    max_gap_factor: float = 2.5
    fill_holes: bool = True
    fill_hole_size: float = 1e6
    smooth_surface: bool = False
    smooth_n_iters: int = 30
    smooth_relaxation: float = 0.08
    rng_seed: int = 0


def get_bounds(points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins, maxs, maxs - mins


def mirror_points(points: np.ndarray, axis: int, mode: str) -> np.ndarray:
    mirrored = points.copy()
    mirrored[:, axis] *= -1.0
    if mode == "augment":
        return np.vstack([points, mirrored])
    if mode == "replace":
        return mirrored
    raise ValueError("mode must be 'augment' or 'replace'")


def extract_slice(points: np.ndarray, axis: int, frac: float, thickness_frac: float):
    mins, _, ranges = get_bounds(points)
    coord = points[:, axis]
    center = mins[axis] + frac * ranges[axis]
    thickness = thickness_frac * ranges[axis]
    mask = (coord >= center - thickness / 2) & (coord <= center + thickness / 2)
    return points[mask], center, thickness


def project_to_2d(slice_points: np.ndarray, axis: int):
    dims = [0, 1, 2]
    dims.remove(axis)
    a, b = dims
    return slice_points[:, a], slice_points[:, b], tuple(dims)


def subsample_xy(A: np.ndarray, B: np.ndarray, max_n: int, seed: int):
    n = len(A)
    if n <= max_n:
        return A, B
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return A[idx], B[idx]


def bbox_aspect(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h = (y1 - y0 + 1)
    w = (x1 - x0 + 1)
    return max(h / w, w / h), (x0, x1, y0, y1)


def compactness_from_mask(mask: np.ndarray):
    area = int(mask.sum())
    er = binary_erosion(mask)
    boundary = mask & (~er)
    perimeter = int(boundary.sum())
    if area <= 0 or perimeter <= 0:
        return 0.0, area, perimeter
    compact = (4.0 * np.pi * area) / (perimeter * perimeter)
    return float(compact), area, perimeter


def fit_ellipse_pca(points2d: np.ndarray, q: float):
    center = points2d.mean(axis=0)
    X = points2d - center
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    Y = X @ vecs
    a = np.percentile(np.abs(Y[:, 0]), q)
    b = np.percentile(np.abs(Y[:, 1]), q)
    if b > a:
        a, b = b, a
        vecs = vecs[:, ::-1]
    return center, (float(a), float(b)), vecs


def ellipse_frame_metrics(points2d: np.ndarray, center: np.ndarray, axes: tuple, R: np.ndarray):
    X = points2d - center
    U = X @ R
    a, b = axes
    rho = np.sqrt((U[:, 0] / (a + 1e-12)) ** 2 + (U[:, 1] / (b + 1e-12)) ** 2)
    ang = np.mod(np.arctan2(U[:, 1] / (b + 1e-12), U[:, 0] / (a + 1e-12)), 2.0 * np.pi)
    return rho, ang


def refine_candidates_to_outer_band(points2d: np.ndarray, cfg: BoreholeConfig):
    if (not cfg.use_outer_band_refinement) or (points2d.shape[0] < cfg.outer_band_min_points):
        return points2d
    center0, axes0, R0 = fit_ellipse_pca(points2d, q=cfg.ellipse_q)
    rho, ang = ellipse_frame_metrics(points2d, center0, axes0, R0)
    sector_ids = np.floor(ang / (2.0 * np.pi) * cfg.outer_band_sectors).astype(int)
    sector_ids = np.clip(sector_ids, 0, cfg.outer_band_sectors - 1)
    keep_idx = []
    for sid in range(cfg.outer_band_sectors):
        idx = np.flatnonzero(sector_ids == sid)
        if idx.size == 0:
            continue
        order = idx[np.argsort(rho[idx])[::-1]]
        keep_idx.extend(order[: cfg.outer_band_keep_per_sector].tolist())
    if len(keep_idx) < max(24, points2d.shape[0] // 4):
        return points2d
    keep_idx = np.array(sorted(set(keep_idx)), dtype=int)
    refined = points2d[keep_idx]
    return refined if refined.shape[0] >= 20 else points2d


def ellipse_points(center, axes, R, n):
    a, b = axes
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = np.column_stack([a * np.cos(t), b * np.sin(t)])
    return circ @ R.T + center


def ellipse_residuals(points2d: np.ndarray, center: np.ndarray, axes: tuple, R: np.ndarray):
    a, b = axes
    X = points2d - center
    U = X @ R
    rho = np.sqrt((U[:, 0] / (a + 1e-12)) ** 2 + (U[:, 1] / (b + 1e-12)) ** 2)
    mean_axis = 0.5 * (a + b)
    return np.abs(rho - 1.0) * mean_axis


def ellipse_inlier_ratio(points2d: np.ndarray, center: np.ndarray, axes: tuple, R: np.ndarray, tau_frac: float):
    res = ellipse_residuals(points2d, center, axes, R)
    tau = tau_frac * (0.5 * (axes[0] + axes[1]))
    return float(np.mean(res <= tau)), res


def rotation_angle_deg(R: np.ndarray):
    theta = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return theta + 180.0 if theta < 0.0 else theta


def rotation_matrix(theta_rad: float):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotation_matrix_from_angle_deg(theta_deg: float):
    return rotation_matrix(np.radians(theta_deg))


def ellipse_param_vector(center: np.ndarray, axes: tuple, R: np.ndarray):
    theta = np.arctan2(R[1, 0], R[0, 0])
    return np.array([center[0], center[1], np.log(axes[0]), np.log(axes[1]), theta], dtype=float)


def ellipse_from_param_vector(params: np.ndarray):
    cx, cy, loga, logb, theta = params
    a = float(np.exp(loga))
    b = float(np.exp(logb))
    if b > a:
        a, b = b, a
        theta += np.pi / 2.0
    R = rotation_matrix(theta)
    return np.array([cx, cy], dtype=float), (a, b), R


def ellipse_residual_objective(params: np.ndarray, points2d: np.ndarray):
    center, axes, R = ellipse_from_param_vector(params)
    res = ellipse_residuals(points2d, center, axes, R)
    scale = 0.5 * (axes[0] + axes[1]) + 1e-12
    u = res / scale
    return float(np.mean(np.where(u < 0.25, u * u, 0.5 * u)))


def refine_ellipse(points2d: np.ndarray, center: np.ndarray, axes: tuple, R: np.ndarray, cfg: BoreholeConfig):
    if (not cfg.use_ellipse_refinement) or (points2d.shape[0] < 20):
        return center, axes, R
    x0 = ellipse_param_vector(center, axes, R)
    result = minimize(
        ellipse_residual_objective,
        x0,
        args=(points2d,),
        method="Powell",
        options={"maxiter": cfg.ellipse_refine_maxiter, "disp": False},
    )
    if not result.success:
        return center, axes, R
    center1, axes1, R1 = ellipse_from_param_vector(result.x)
    if (axes1[0] <= 0.0) or (axes1[1] <= 0.0):
        return center, axes, R
    return center1, axes1, R1


def fit_variant(points2d: np.ndarray, cfg: BoreholeConfig):
    C0, axes0, R0 = fit_ellipse_pca(points2d, q=cfg.ellipse_q)
    C1, axes1, R1 = refine_ellipse(points2d, C0, axes0, R0, cfg)
    return C1, axes1, R1, C0, axes0, R0


def angle_diff_deg(a0: float, a1: float):
    d = abs(a1 - a0) % 180.0
    return float(min(d, 180.0 - d))


def component_centroid(boundary_pts: np.ndarray):
    return None if boundary_pts.shape[0] == 0 else np.mean(boundary_pts, axis=0)


def score_fit_candidate(
    points2d: np.ndarray,
    center: np.ndarray,
    axes: tuple,
    R: np.ndarray,
    last_center2d,
    last_axes,
    last_R,
    cfg: BoreholeConfig,
    comp_center2d=None,
    anchor_center2d=None,
    use_anchor_penalty: bool = False,
):
    inlier_ratio, res = ellipse_inlier_ratio(points2d, center, axes, R, cfg.inlier_tau_frac)
    resid_med = float(np.median(res))
    resid_p90 = float(np.percentile(res, 90))
    center_pen = 0.0
    axis_pen = 0.0
    theta_pen = 0.0
    comp_center_pen = 0.0
    anchor_pen = 0.0
    if (last_center2d is not None) and (last_axes is not None):
        center_pen = float(np.linalg.norm(center - last_center2d))
        axis_pen = float(np.mean(np.abs(np.asarray(axes) - np.asarray(last_axes))))
    if last_R is not None:
        theta_pen = angle_diff_deg(rotation_angle_deg(last_R), rotation_angle_deg(R))
    if cfg.use_component_centroid_prior and (comp_center2d is not None):
        comp_center_pen = float(np.linalg.norm(center - comp_center2d))
    if use_anchor_penalty and (anchor_center2d is not None):
        anchor_pen = float(np.linalg.norm(center - anchor_center2d))
    score = (
        cfg.fit_score_w_resid_med * resid_med +
        cfg.fit_score_w_resid_p90 * resid_p90 -
        cfg.fit_score_w_inlier * inlier_ratio +
        cfg.fit_score_w_center * center_pen +
        cfg.fit_score_w_axis * axis_pen +
        cfg.fit_score_w_theta * theta_pen +
        cfg.fit_score_w_comp_center * comp_center_pen +
        cfg.fit_score_w_anchor * anchor_pen
    )
    return {
        "score": float(score),
        "resid_med": resid_med,
        "resid_p90": resid_p90,
        "inlier": inlier_ratio,
    }


def select_best_fit(cand, last_center2d, last_axes, last_R, cfg: BoreholeConfig, comp_center2d=None, anchor_center2d=None, use_anchor_penalty: bool = False):
    variants = []
    C_hat, axes_hat, R_hat, C0, axes0, R0 = fit_variant(cand, cfg)
    variants.append({
        "name": "raw_refined",
        "points": cand,
        "center": C_hat,
        "axes": axes_hat,
        "R": R_hat,
        "pre_center": C0,
        "pre_axes": axes0,
        "pre_R": R0,
    })
    cand_outer = refine_candidates_to_outer_band(cand, cfg)
    if cand_outer.shape[0] >= 20 and cand_outer.shape[0] != cand.shape[0]:
        C_hat, axes_hat, R_hat, C0, axes0, R0 = fit_variant(cand_outer, cfg)
        variants.append({
            "name": "outer_refined",
            "points": cand_outer,
            "center": C_hat,
            "axes": axes_hat,
            "R": R_hat,
            "pre_center": C0,
            "pre_axes": axes0,
            "pre_R": R0,
        })
    best = None
    for variant in variants:
        metrics = score_fit_candidate(
            variant["points"],
            variant["center"],
            variant["axes"],
            variant["R"],
            last_center2d,
            last_axes,
            last_R,
            cfg,
            comp_center2d=comp_center2d,
            anchor_center2d=anchor_center2d,
            use_anchor_penalty=use_anchor_penalty,
        )
        variant["metrics"] = metrics
        if (best is None) or (metrics["score"] < best["metrics"]["score"]):
            best = variant
    return best, variants


def boundary_band_raw_points(A: np.ndarray, B: np.ndarray, boundary_pts: np.ndarray, band: float):
    P = np.column_stack([A, B])
    Q = boundary_pts
    N = P.shape[0]
    keep = np.zeros(N, dtype=bool)
    band2 = band * band
    chunk = 4000
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        d2 = ((P[i0:i1, None, :] - Q[None, :, :]) ** 2).sum(axis=2)
        keep[i0:i1] = (d2.min(axis=1) <= band2)
    return P[keep]


def remove_border_touching_components(low_mask: np.ndarray):
    lbl, n = label(low_mask)
    if n == 0:
        return low_mask * False, lbl, []
    border = np.zeros_like(low_mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_ids = np.unique(lbl[border & (lbl > 0)])
    keep_ids = [i for i in range(1, n + 1) if i not in set(border_ids)]
    return np.isin(lbl, keep_ids), lbl, keep_ids


def interp_angle_deg(a0: float, a1: float, t: float):
    delta = ((a1 - a0 + 90.0) % 180.0) - 90.0
    return float((a0 + t * delta) % 180.0)


def sample_ellipse_mask(center, axes, R, aa, bb):
    XY = np.stack([aa.ravel(), bb.ravel()], axis=1)
    local = (XY - center) @ R
    mask = ((local[:, 0] / (axes[0] + 1e-12)) ** 2 + (local[:, 1] / (axes[1] + 1e-12)) ** 2) <= 1.0
    return mask.reshape(aa.shape)


def ellipse_iou(center0, axes0, R0, center1, axes1, R1, aa, bb):
    m0 = sample_ellipse_mask(center0, axes0, R0, aa, bb)
    m1 = sample_ellipse_mask(center1, axes1, R1, aa, bb)
    inter = np.count_nonzero(m0 & m1)
    union = np.count_nonzero(m0 | m1)
    return 0.0 if union == 0 else float(inter / union)


def continuity_report(kept, grid_n=256):
    if len(kept) < 2:
        return [], []
    centers = np.array([k[2] for k in kept], dtype=float)
    all_a = centers[:, 0]
    all_b = centers[:, 1]
    pad_a = 0.2 * max(all_a.max() - all_a.min(), 1e-6)
    pad_b = 0.2 * max(all_b.max() - all_b.min(), 1e-6)
    aa_lin = np.linspace(all_a.min() - pad_a, all_a.max() + pad_a, grid_n)
    bb_lin = np.linspace(all_b.min() - pad_b, all_b.max() + pad_b, grid_n)
    AA, BB = np.meshgrid(aa_lin, bb_lin)
    rows = []
    pairs = []
    for idx in range(1, len(kept)):
        prev = kept[idx - 1]
        cur = kept[idx]
        delta_center = float(np.linalg.norm(cur[2] - prev[2]))
        delta_major_pct = float((cur[3][0] - prev[3][0]) / (prev[3][0] + 1e-12))
        delta_minor_pct = float((cur[3][1] - prev[3][1]) / (prev[3][1] + 1e-12))
        delta_theta = angle_diff_deg(rotation_angle_deg(prev[4]), rotation_angle_deg(cur[4]))
        iou = ellipse_iou(prev[2], prev[3], prev[4], cur[2], cur[3], cur[4], AA, BB)
        row = {
            "delta_center": delta_center,
            "delta_major_pct": delta_major_pct,
            "delta_minor_pct": delta_minor_pct,
            "delta_theta_deg": delta_theta,
            "iou_prev": iou,
        }
        rows.append(row)
        pairs.append(row)
    return rows, pairs


def build_section_from_params(axis_center, center2d, axes, R, dims, cfg: BoreholeConfig):
    ell2 = ellipse_points(center2d, axes, R, n=cfg.n_ring)
    pts3 = np.zeros((ell2.shape[0], 3), dtype=float)
    pts3[:, cfg.chosen_axis] = axis_center
    pts3[:, dims[0]] = ell2[:, 0]
    pts3[:, dims[1]] = ell2[:, 1]
    return pts3


def fill_short_gaps(sections, kept, fracs, dims, cfg: BoreholeConfig):
    if (not cfg.enable_short_gap_fill) or (len(kept) < 2):
        return sections, kept
    by_idx = {item[5]: (sections[pos], item) for pos, item in enumerate(kept)}
    accepted_idx = sorted(by_idx)
    for left_idx, right_idx in zip(accepted_idx[:-1], accepted_idx[1:]):
        gap = right_idx - left_idx - 1
        if gap <= 0 or gap > cfg.max_interp_gap:
            continue
        left = by_idx[left_idx][1]
        right = by_idx[right_idx][1]
        theta0 = rotation_angle_deg(left[4])
        theta1 = rotation_angle_deg(right[4])
        for miss_idx in range(left_idx + 1, right_idx):
            t = (miss_idx - left_idx) / float(right_idx - left_idx)
            frac = float(fracs[miss_idx])
            axis_center = (1.0 - t) * left[1] + t * right[1]
            center2d = (1.0 - t) * left[2] + t * right[2]
            axes = tuple((1.0 - t) * np.asarray(left[3]) + t * np.asarray(right[3]))
            theta = interp_angle_deg(theta0, theta1, t)
            R = rotation_matrix_from_angle_deg(theta)
            sec = build_section_from_params(axis_center, center2d, axes, R, dims, cfg)
            item = (frac, axis_center, center2d, axes, R, miss_idx, "interp", np.nan, np.nan, np.nan)
            by_idx[miss_idx] = (sec, item)
    full_idx = sorted(by_idx)
    return [by_idx[i][0] for i in full_idx], [by_idx[i][1] for i in full_idx]


def resample_closed_polyline(points: np.ndarray, n: int):
    P = np.asarray(points, dtype=float)
    P_closed = np.vstack([P, P[0]])
    seg = np.linalg.norm(np.diff(P_closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    t = np.linspace(0.0, total, n + 1)[:-1]
    out = np.empty((n, 3), dtype=float)
    j = 0
    for i, ti in enumerate(t):
        while not (s[j] <= ti <= s[j + 1]):
            j += 1
            if j >= len(seg):
                j = len(seg) - 1
                break
        denom = s[j + 1] - s[j]
        a = 0.0 if denom == 0 else (ti - s[j]) / denom
        out[i] = (1 - a) * P_closed[j] + a * P_closed[j + 1]
    return out


def best_cyclic_shift(prev: np.ndarray, cur: np.ndarray):
    n = prev.shape[0]
    best_k, best_val = 0, np.inf
    for k in range(n):
        D = prev - np.roll(cur, -k, axis=0)
        val = float(np.sum(D * D))
        if val < best_val:
            best_val = val
            best_k = k
    return best_k


def align_loop_to_prev(cur: np.ndarray, prev: np.ndarray):
    k1 = best_cyclic_shift(prev, cur)
    cur1 = np.roll(cur, -k1, axis=0)
    err1 = float(np.sum((prev - cur1) ** 2))
    cur_rev = cur[::-1].copy()
    k2 = best_cyclic_shift(prev, cur_rev)
    cur2 = np.roll(cur_rev, -k2, axis=0)
    err2 = float(np.sum((prev - cur2) ** 2))
    return cur1 if err1 <= err2 else cur2


def polydata_from_points_faces(points: np.ndarray, faces: list[list[int]]):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(points, deep=True))
    polys = vtk.vtkCellArray()
    for face in faces:
        polys.InsertNextCell(len(face))
        for idx in face:
            polys.InsertCellPoint(int(idx))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(polys)
    return poly


def loft_between_loops(loop0: np.ndarray, loop1: np.ndarray):
    A = np.asarray(loop0, dtype=float)
    B = np.asarray(loop1, dtype=float)
    n = A.shape[0]
    pts = np.vstack([A, B])
    faces = []
    for i in range(n):
        i0 = i
        i1 = (i + 1) % n
        a0, a1 = i0, i1
        b0, b1 = n + i0, n + i1
        faces.append([a0, a1, b1])
        faces.append([a0, b1, b0])
    return polydata_from_points_faces(pts, faces)


def cap_from_loop(loop: np.ndarray):
    P = np.asarray(loop, dtype=float)
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(P, deep=True))
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(P.shape[0])
    for i in range(P.shape[0]):
        polygon.GetPointIds().SetId(i, i)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(polygon)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(polys)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    return tri.GetOutput()


def append_polydata(meshes):
    append = vtk.vtkAppendPolyData()
    for mesh in meshes:
        if mesh is not None:
            append.AddInputData(mesh)
    append.Update()
    return append.GetOutput()


def infer_axis_gap_limit(axis_centers, cfg: BoreholeConfig):
    if cfg.max_axis_gap is not None:
        return cfg.max_axis_gap
    if len(axis_centers) < 2:
        return None
    diffs = np.diff(np.sort(np.asarray(axis_centers, dtype=float)))
    nominal = float(np.median(diffs))
    return None if nominal <= 0 else cfg.max_gap_factor * nominal


def contiguous_blocks(axis_centers, gap_limit):
    if not axis_centers:
        return []
    blocks = [[0]]
    for i in range(1, len(axis_centers)):
        gap = abs(axis_centers[i] - axis_centers[i - 1])
        if (gap_limit is not None) and (gap > gap_limit):
            blocks.append([i])
        else:
            blocks[-1].append(i)
    return blocks


def postprocess_surface(mesh: vtk.vtkPolyData, cfg: BoreholeConfig):
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(mesh)
    clean.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(clean.GetOutput())
    tri.Update()
    out = tri.GetOutput()
    if cfg.fill_holes:
        fill = vtk.vtkFillHolesFilter()
        fill.SetInputData(out)
        fill.SetHoleSize(cfg.fill_hole_size)
        fill.Update()
        out = fill.GetOutput()
    if cfg.smooth_surface:
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputData(out)
        smooth.SetNumberOfIterations(cfg.smooth_n_iters)
        smooth.SetRelaxationFactor(cfg.smooth_relaxation)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOff()
        smooth.Update()
        out = smooth.GetOutput()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(out)
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def boundary_nonmanifold_edge_count(mesh: vtk.vtkPolyData):
    feat = vtk.vtkFeatureEdges()
    feat.SetInputData(mesh)
    feat.BoundaryEdgesOn()
    feat.NonManifoldEdgesOn()
    feat.FeatureEdgesOff()
    feat.ManifoldEdgesOff()
    feat.Update()
    return feat.GetOutput().GetNumberOfCells()


def add_field_array(poly: vtk.vtkPolyData, name: str, values):
    arr = numpy_to_vtk(np.asarray(values), deep=True)
    arr.SetName(name)
    poly.GetFieldData().AddArray(arr)


def reconstruct_borehole(points: np.ndarray, cfg: Optional[BoreholeConfig] = None):
    cfg = cfg or BoreholeConfig()
    work_points = np.asarray(points, dtype=float)
    if work_points.ndim != 2 or work_points.shape[1] != 3:
        raise ValueError("Expected Nx3 point array.")
    if cfg.do_mirror:
        work_points = mirror_points(work_points, cfg.mirror_axis, cfg.mirror_mode)

    fracs = np.linspace(cfg.frac_start, cfg.frac_end, cfg.n_slices)
    sections = []
    kept = []
    last_axes = None
    last_center2d = None
    anchor_center2d = None
    loft_dims = [0, 1, 2]
    loft_dims.remove(cfg.chosen_axis)
    loft_dims = tuple(loft_dims)

    for i, frac in enumerate(fracs, start=1):
        slice_pts, axis_center, _ = extract_slice(work_points, cfg.chosen_axis, frac=frac, thickness_frac=cfg.thickness_frac)
        if slice_pts.shape[0] < 500:
            continue
        A, B, dims = project_to_2d(slice_pts, cfg.chosen_axis)
        A_s, B_s = subsample_xy(A, B, cfg.subsample, cfg.rng_seed)
        a_min, a_max = float(np.min(A)), float(np.max(A))
        b_min, b_max = float(np.min(B)), float(np.max(B))
        pad_a = cfg.pad_frac_a * (a_max - a_min + 1e-12)
        pad_b = cfg.pad_frac_b * (b_max - b_min + 1e-12)
        a_min_p, a_max_p = a_min - pad_a, a_max + pad_a
        b_min_p, b_max_p = b_min - pad_b, b_max + pad_b
        H, a_edges, b_edges = np.histogram2d(
            A_s,
            B_s,
            bins=cfg.grid_n,
            range=[[a_min_p, a_max_p], [b_min_p, b_max_p]],
        )
        D = gaussian_filter(H.T, sigma=cfg.smooth_sigma)
        aa = 0.5 * (a_edges[:-1] + a_edges[1:])
        bb = 0.5 * (b_edges[:-1] + b_edges[1:])
        if cfg.use_occupancy_gate:
            occ = binary_dilation((H.T > 0), iterations=cfg.occ_dilate_iters)
        else:
            occ = np.ones_like(D, dtype=bool)
        thr = float(np.quantile(D[occ], cfg.low_density_q))
        low = (D <= thr) & occ
        if cfg.do_morph:
            low = binary_opening(low, iterations=cfg.open_iters)
            low = binary_closing(low, iterations=cfg.close_iters)
        low_interior, _, keep_ids = remove_border_touching_components(low)
        if len(keep_ids) == 0:
            continue
        lbl_cav, ncomp = label(low_interior)
        best = None
        best_score = -np.inf
        best_boundary_pts = None
        for cid in range(1, ncomp + 1):
            comp = (lbl_cav == cid)
            area = int(comp.sum())
            if area < cfg.min_comp_area:
                continue
            frac_area = area / float(cfg.grid_n * cfg.grid_n)
            if frac_area > cfg.max_comp_frac:
                continue
            compact, _, _ = compactness_from_mask(comp)
            if compact <= 1e-8:
                continue
            asp, _ = bbox_aspect(comp)
            boundary_pix = comp & (~binary_erosion(comp))
            bi, ai = np.nonzero(boundary_pix)
            if len(ai) < 30:
                continue
            boundary_pts = np.column_stack([aa[ai], bb[bi]])
            score = (6.0 * compact) - (2.0 * frac_area) - (0.15 * (asp - 1.0))
            if score > best_score:
                best_score = score
                best = (cid, area, frac_area, compact, asp)
                best_boundary_pts = boundary_pts
        if best is None:
            continue
        comp_center2d = component_centroid(best_boundary_pts)
        cell_a = (a_max_p - a_min_p) / cfg.grid_n
        cell_b = (b_max_p - b_min_p) / cfg.grid_n
        band = 2.0 * max(cell_a, cell_b)
        cand = boundary_band_raw_points(A, B, best_boundary_pts, band=band)
        if cand.shape[0] < 50:
            continue
        last_R = kept[-1][4] if kept else None
        use_anchor_penalty = (
            cfg.use_confident_anchor and
            (anchor_center2d is not None) and
            (len(kept) < cfg.anchor_warmup_slices)
        )
        best_fit, _ = select_best_fit(
            cand,
            last_center2d,
            last_axes,
            last_R,
            cfg,
            comp_center2d=comp_center2d,
            anchor_center2d=anchor_center2d,
            use_anchor_penalty=use_anchor_penalty,
        )
        C_hat, axes_hat, R_hat = best_fit["center"], best_fit["axes"], best_fit["R"]
        AR = float(axes_hat[0] / (axes_hat[1] + 1e-12))
        ok = AR <= cfg.max_ar
        if cfg.enable_continuity and (last_axes is not None):
            a_prev, b_prev = last_axes
            a_hat, b_hat = axes_hat
            if (
                (a_hat > cfg.max_axis_jump * a_prev) or
                (b_hat > cfg.max_axis_jump * b_prev) or
                (a_hat < a_prev / cfg.max_axis_jump) or
                (b_hat < b_prev / cfg.max_axis_jump)
            ):
                ok = False
            if (cfg.max_center_jump is not None) and (last_center2d is not None):
                if np.linalg.norm(C_hat - last_center2d) > cfg.max_center_jump:
                    ok = False
        if not ok:
            continue
        resid_med = best_fit["metrics"]["resid_med"]
        resid_p90 = best_fit["metrics"]["resid_p90"]
        inlier_ratio = best_fit["metrics"]["inlier"]
        last_axes = axes_hat
        last_center2d = C_hat.copy()
        pts3 = build_section_from_params(axis_center, C_hat, axes_hat, R_hat, dims, cfg)
        sections.append(pts3)
        kept.append((frac, axis_center, C_hat, axes_hat, R_hat, i - 1, "observed", resid_med, resid_p90, inlier_ratio))
        if (
            cfg.use_confident_anchor and
            (anchor_center2d is None) and
            (inlier_ratio >= cfg.anchor_min_inlier) and
            (resid_med <= cfg.anchor_max_resid_med)
        ):
            anchor_center2d = C_hat.copy()

    if len(sections) < 2:
        raise RuntimeError(f"Not enough accepted slices to loft. Accepted={len(sections)}")

    sections, kept = fill_short_gaps(sections, kept, fracs, loft_dims, cfg)
    loops = [resample_closed_polyline(L, cfg.n_ring) for L in sections]
    aligned = [loops[0]]
    for i in range(1, len(loops)):
        aligned.append(align_loop_to_prev(loops[i], aligned[-1]))
    axis_centers = [k[1] for k in kept]
    gap_limit = infer_axis_gap_limit(axis_centers, cfg)
    blocks = contiguous_blocks(axis_centers, gap_limit)
    parts = []
    for block in blocks:
        if len(block) < 2:
            continue
        for j0, j1 in zip(block[:-1], block[1:]):
            parts.append(loft_between_loops(aligned[j0], aligned[j1]))
        if cfg.cap_ends:
            parts.append(cap_from_loop(aligned[block[0]]))
            parts.append(cap_from_loop(aligned[block[-1]]))
    if not parts:
        raise RuntimeError("No loft segments were created.")
    surface = postprocess_surface(append_polydata(parts), cfg)
    rows, pair_rows = continuity_report(kept)
    field = {
        "accepted_slice_count": len(kept),
        "observed_slice_count": int(sum(1 for item in kept if item[6] == "observed")),
        "interpolated_slice_count": int(sum(1 for item in kept if item[6] == "interp")),
        "boundary_nonmanifold_edge_segments": boundary_nonmanifold_edge_count(surface),
    }
    obs = [item for item in kept if item[6] == "observed"]
    if obs:
        field["resid_med_mean"] = float(np.mean([item[7] for item in obs]))
        field["resid_p90_mean"] = float(np.mean([item[8] for item in obs]))
        field["inlier_mean"] = float(np.mean([item[9] for item in obs]))
    if pair_rows:
        field["center_jump_median"] = float(np.median([r["delta_center"] for r in pair_rows]))
        field["major_pct_abs_median"] = float(np.median(np.abs([r["delta_major_pct"] for r in pair_rows])))
        field["minor_pct_abs_median"] = float(np.median(np.abs([r["delta_minor_pct"] for r in pair_rows])))
        field["theta_median"] = float(np.median([r["delta_theta_deg"] for r in pair_rows]))
        field["iou_median"] = float(np.median([r["iou_prev"] for r in pair_rows]))
    for key, value in field.items():
        add_field_array(surface, key, [value])
    if obs:
        add_field_array(surface, "slice_axis_center", [item[1] for item in obs])
        add_field_array(surface, "slice_resid_med", [item[7] for item in obs])
        add_field_array(surface, "slice_resid_p90", [item[8] for item in obs])
        add_field_array(surface, "slice_inlier", [item[9] for item in obs])
    return {
        "surface": surface,
        "kept": kept,
        "continuity_rows": rows,
        "continuity_pairs": pair_rows,
        "summary": field,
    }
