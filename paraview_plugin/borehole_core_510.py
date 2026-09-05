import math

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


class BoreholeConfig510:
    def __init__(self):
        self.chosen_axis = 0
        self.frac_start = 0.01
        self.frac_end = 0.69
        self.n_slices = 35
        self.thickness_frac = 0.02
        self.grid_n = 220
        self.subsample = 20000
        self.smooth_sigma = 3.0
        self.low_density_q = 0.10
        self.pad_frac_a = 0.10
        self.pad_frac_b = 0.10
        self.open_iters = 1
        self.close_iters = 1
        self.occ_dilate_iters = 5
        self.min_comp_area = 120
        self.max_comp_frac = 0.20
        self.do_mirror = True
        self.mirror_axis = 1
        self.ellipse_q = 90.0
        self.use_outer_band_refinement = True
        self.outer_band_sectors = 24
        self.outer_band_keep_per_sector = 6
        self.max_ar = 4.0
        self.max_axis_jump = 2.0
        self.enable_short_gap_fill = True
        self.max_interp_gap = 2
        self.n_ring = 240
        self.cap_ends = True
        self.max_gap_factor = 2.5
        self.fill_holes = True
        self.fill_hole_size = 1e6
        self.rng_seed = 0


def get_bounds(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins, maxs, maxs - mins


def mirror_points(points, axis):
    mirrored = points.copy()
    mirrored[:, axis] *= -1.0
    return np.vstack([points, mirrored])


def extract_slice(points, axis, frac, thickness_frac):
    mins, _, ranges = get_bounds(points)
    center = mins[axis] + frac * ranges[axis]
    thickness = thickness_frac * ranges[axis]
    coord = points[:, axis]
    mask = (coord >= center - thickness / 2.0) & (coord <= center + thickness / 2.0)
    return points[mask], center


def project_to_2d(slice_points, axis):
    dims = [0, 1, 2]
    dims.remove(axis)
    return slice_points[:, dims[0]], slice_points[:, dims[1]], tuple(dims)


def subsample_xy(A, B, max_n, seed):
    if len(A) <= max_n:
        return A, B
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), size=max_n, replace=False)
    return A[idx], B[idx]


def gaussian_kernel1d(sigma):
    if sigma <= 0:
        return np.array([1.0])
    radius = max(1, int(3.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def convolve_axis_reflect(arr, kernel, axis):
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(arr, dtype=float)
        for i, w in enumerate(kernel):
            out += w * padded[i:i + arr.shape[0], :]
        return out
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="reflect")
    out = np.zeros_like(arr, dtype=float)
    for i, w in enumerate(kernel):
        out += w * padded[:, i:i + arr.shape[1]]
    return out


def gaussian_filter_numpy(arr, sigma):
    kernel = gaussian_kernel1d(sigma)
    return convolve_axis_reflect(convolve_axis_reflect(arr, kernel, 0), kernel, 1)


def dilate(mask, iterations=1):
    out = mask.astype(bool)
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] |
            p[1:-1, :-2] | p[1:-1, 2:] | p[:-2, :-2] |
            p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        )
    return out


def erode(mask, iterations=1):
    out = mask.astype(bool)
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] &
            p[1:-1, :-2] & p[1:-1, 2:] & p[:-2, :-2] &
            p[:-2, 2:] & p[2:, :-2] & p[2:, 2:]
        )
    return out


def connected_components(mask):
    mask = mask.astype(bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    H, W = mask.shape
    for y in range(H):
        for x in range(W):
            if (not mask[y, x]) or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def remove_border_touching_components(low_mask):
    labels, n = connected_components(low_mask)
    if n == 0:
        return low_mask * False, labels, []
    border_ids = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ])).tolist())
    keep_ids = [i for i in range(1, n + 1) if i not in border_ids]
    return np.isin(labels, keep_ids), labels, keep_ids


def bbox_aspect(mask):
    ys, xs = np.nonzero(mask)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    return max(float(h) / max(w, 1), float(w) / max(h, 1))


def compactness_from_mask(mask):
    area = int(mask.sum())
    boundary = mask & (~erode(mask))
    perimeter = int(boundary.sum())
    if area <= 0 or perimeter <= 0:
        return 0.0, area
    return float((4.0 * math.pi * area) / (perimeter * perimeter)), area


def boundary_band_raw_points(A, B, boundary_pts, band):
    P = np.column_stack([A, B])
    keep = np.zeros(P.shape[0], dtype=bool)
    band2 = band * band
    for i0 in range(0, P.shape[0], 4000):
        i1 = min(i0 + 4000, P.shape[0])
        d2 = ((P[i0:i1, None, :] - boundary_pts[None, :, :]) ** 2).sum(axis=2)
        keep[i0:i1] = d2.min(axis=1) <= band2
    return P[keep]


def fit_ellipse_pca(points2d, q):
    center = points2d.mean(axis=0)
    X = points2d - center
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(vals)[::-1]]
    Y = X @ vecs
    a = float(np.percentile(np.abs(Y[:, 0]), q))
    b = float(np.percentile(np.abs(Y[:, 1]), q))
    if b > a:
        a, b = b, a
        vecs = vecs[:, ::-1]
    return center, (a, b), vecs


def ellipse_points(center, axes, R, n):
    a, b = axes
    t = np.linspace(0, 2.0 * math.pi, n, endpoint=False)
    circ = np.column_stack([a * np.cos(t), b * np.sin(t)])
    return circ @ R.T + center


def ellipse_residuals(points2d, center, axes, R):
    a, b = axes
    U = (points2d - center) @ R
    rho = np.sqrt((U[:, 0] / (a + 1e-12)) ** 2 + (U[:, 1] / (b + 1e-12)) ** 2)
    return np.abs(rho - 1.0) * (0.5 * (a + b))


def inlier_and_residuals(points2d, center, axes, R):
    res = ellipse_residuals(points2d, center, axes, R)
    tau = 0.10 * (0.5 * (axes[0] + axes[1]))
    return float(np.mean(res <= tau)), res


def rotation_angle_deg(R):
    theta = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return theta + 180.0 if theta < 0.0 else theta


def rotation_matrix_from_angle_deg(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def interp_angle_deg(a0, a1, t):
    delta = ((a1 - a0 + 90.0) % 180.0) - 90.0
    return float((a0 + t * delta) % 180.0)


def refine_candidates_to_outer_band(points2d, cfg):
    if (not cfg.use_outer_band_refinement) or points2d.shape[0] < 48:
        return points2d
    center, axes, R = fit_ellipse_pca(points2d, cfg.ellipse_q)
    U = (points2d - center) @ R
    rho = np.sqrt((U[:, 0] / (axes[0] + 1e-12)) ** 2 + (U[:, 1] / (axes[1] + 1e-12)) ** 2)
    ang = np.mod(np.arctan2(U[:, 1] / (axes[1] + 1e-12), U[:, 0] / (axes[0] + 1e-12)), 2.0 * math.pi)
    sectors = np.floor(ang / (2.0 * math.pi) * cfg.outer_band_sectors).astype(int)
    keep = []
    for sid in range(cfg.outer_band_sectors):
        idx = np.flatnonzero(sectors == sid)
        if idx.size:
            order = idx[np.argsort(rho[idx])[::-1]]
            keep.extend(order[:cfg.outer_band_keep_per_sector].tolist())
    if len(keep) < max(24, points2d.shape[0] // 4):
        return points2d
    return points2d[np.array(sorted(set(keep)), dtype=int)]


def choose_fit(cand, last_center, last_axes, last_R, cfg):
    variants = [("raw", cand)]
    outer = refine_candidates_to_outer_band(cand, cfg)
    if outer.shape[0] >= 20 and outer.shape[0] != cand.shape[0]:
        variants.append(("outer", outer))
    best = None
    for name, pts in variants:
        center, axes, R = fit_ellipse_pca(pts, cfg.ellipse_q)
        inlier, res = inlier_and_residuals(pts, center, axes, R)
        score = float(np.median(res) + 0.35 * np.percentile(res, 90) - 0.003 * inlier)
        if last_center is not None:
            score += 0.75 * float(np.linalg.norm(center - last_center))
        if last_axes is not None:
            score += 0.004 * float(np.mean(np.abs(np.asarray(axes) - np.asarray(last_axes))))
        if best is None or score < best["score"]:
            best = {
                "name": name,
                "points": pts,
                "center": center,
                "axes": axes,
                "R": R,
                "score": score,
                "resid_med": float(np.median(res)),
                "resid_p90": float(np.percentile(res, 90)),
                "inlier": inlier,
            }
    return best


def build_section(axis_center, center2d, axes, R, dims, cfg):
    ell2 = ellipse_points(center2d, axes, R, cfg.n_ring)
    pts3 = np.zeros((ell2.shape[0], 3), dtype=float)
    pts3[:, cfg.chosen_axis] = axis_center
    pts3[:, dims[0]] = ell2[:, 0]
    pts3[:, dims[1]] = ell2[:, 1]
    return pts3


def fill_short_gaps(sections, kept, fracs, dims, cfg):
    if (not cfg.enable_short_gap_fill) or len(kept) < 2:
        return sections, kept
    by_idx = {item[5]: (sections[pos], item) for pos, item in enumerate(kept)}
    for left_idx, right_idx in zip(sorted(by_idx)[:-1], sorted(by_idx)[1:]):
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
            R = rotation_matrix_from_angle_deg(interp_angle_deg(theta0, theta1, t))
            sec = build_section(axis_center, center2d, axes, R, dims, cfg)
            by_idx[miss_idx] = (sec, (frac, axis_center, center2d, axes, R, miss_idx, "interp", np.nan, np.nan, np.nan))
    full_idx = sorted(by_idx)
    return [by_idx[i][0] for i in full_idx], [by_idx[i][1] for i in full_idx]


def resample_closed_polyline(points, n):
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
            j = min(j + 1, len(seg) - 1)
            if j == len(seg) - 1:
                break
        denom = s[j + 1] - s[j]
        alpha = 0.0 if denom == 0 else (ti - s[j]) / denom
        out[i] = (1.0 - alpha) * P_closed[j] + alpha * P_closed[j + 1]
    return out


def best_cyclic_shift(prev, cur):
    best_k, best_val = 0, np.inf
    for k in range(prev.shape[0]):
        D = prev - np.roll(cur, -k, axis=0)
        val = float(np.sum(D * D))
        if val < best_val:
            best_k, best_val = k, val
    return best_k


def align_loop_to_prev(cur, prev):
    k1 = best_cyclic_shift(prev, cur)
    cur1 = np.roll(cur, -k1, axis=0)
    err1 = float(np.sum((prev - cur1) ** 2))
    cur_rev = cur[::-1].copy()
    k2 = best_cyclic_shift(prev, cur_rev)
    cur2 = np.roll(cur_rev, -k2, axis=0)
    err2 = float(np.sum((prev - cur2) ** 2))
    return cur1 if err1 <= err2 else cur2


def polydata_from_points_faces(points, faces):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(points, dtype=float), deep=True))
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for idx in face:
            cells.InsertCellPoint(int(idx))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)
    return poly


def loft_between_loops(loop0, loop1):
    A = np.asarray(loop0, dtype=float)
    B = np.asarray(loop1, dtype=float)
    n = A.shape[0]
    faces = []
    for i in range(n):
        i0, i1 = i, (i + 1) % n
        b0, b1 = n + i0, n + i1
        faces.append([i0, i1, b1])
        faces.append([i0, b1, b0])
    return polydata_from_points_faces(np.vstack([A, B]), faces)


def cap_from_loop(loop):
    center_idx = loop.shape[0]
    pts = np.vstack([loop, loop.mean(axis=0)])
    faces = []
    for i in range(loop.shape[0]):
        faces.append([center_idx, i, (i + 1) % loop.shape[0]])
    return polydata_from_points_faces(pts, faces)


def append_polydata(meshes):
    app = vtk.vtkAppendPolyData()
    for mesh in meshes:
        app.AddInputData(mesh)
    app.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(app.GetOutput())
    clean.Update()
    return clean.GetOutput()


def boundary_nonmanifold_edge_count(mesh):
    feat = vtk.vtkFeatureEdges()
    feat.SetInputData(mesh)
    feat.BoundaryEdgesOn()
    feat.NonManifoldEdgesOn()
    feat.FeatureEdgesOff()
    feat.ManifoldEdgesOff()
    feat.Update()
    return feat.GetOutput().GetNumberOfCells()


def add_field_array(poly, name, values):
    arr = numpy_to_vtk(np.asarray(values), deep=True)
    arr.SetName(name)
    poly.GetFieldData().AddArray(arr)


def reconstruct_borehole(points, cfg=None):
    cfg = cfg or BoreholeConfig510()
    work = np.asarray(points, dtype=float)
    if cfg.do_mirror:
        work = mirror_points(work, cfg.mirror_axis)
    fracs = np.linspace(cfg.frac_start, cfg.frac_end, cfg.n_slices)
    sections = []
    kept = []
    last_axes = None
    last_center = None
    dims_for_loft = None

    for i, frac in enumerate(fracs):
        slice_pts, axis_center = extract_slice(work, cfg.chosen_axis, float(frac), cfg.thickness_frac)
        if slice_pts.shape[0] < 500:
            continue
        A, B, dims = project_to_2d(slice_pts, cfg.chosen_axis)
        dims_for_loft = dims
        A_s, B_s = subsample_xy(A, B, cfg.subsample, cfg.rng_seed)
        a_min, a_max = float(np.min(A)), float(np.max(A))
        b_min, b_max = float(np.min(B)), float(np.max(B))
        pad_a = cfg.pad_frac_a * (a_max - a_min + 1e-12)
        pad_b = cfg.pad_frac_b * (b_max - b_min + 1e-12)
        a_min_p, a_max_p = a_min - pad_a, a_max + pad_a
        b_min_p, b_max_p = b_min - pad_b, b_max + pad_b
        H, a_edges, b_edges = np.histogram2d(
            A_s, B_s,
            bins=cfg.grid_n,
            range=[[a_min_p, a_max_p], [b_min_p, b_max_p]],
        )
        D = gaussian_filter_numpy(H.T, cfg.smooth_sigma)
        aa = 0.5 * (a_edges[:-1] + a_edges[1:])
        bb = 0.5 * (b_edges[:-1] + b_edges[1:])
        occ = dilate(H.T > 0, cfg.occ_dilate_iters)
        if not np.any(occ):
            continue
        low = (D <= float(np.quantile(D[occ], cfg.low_density_q))) & occ
        low = dilate(erode(low, cfg.open_iters), cfg.close_iters)
        low_interior, _, keep_ids = remove_border_touching_components(low)
        if not keep_ids:
            continue
        labels, ncomp = connected_components(low_interior)
        best_boundary = None
        best_score = -np.inf
        for cid in range(1, ncomp + 1):
            comp = labels == cid
            compact, area = compactness_from_mask(comp)
            if area < cfg.min_comp_area:
                continue
            frac_area = area / float(cfg.grid_n * cfg.grid_n)
            if frac_area > cfg.max_comp_frac or compact <= 1e-8:
                continue
            asp = bbox_aspect(comp)
            boundary = comp & (~erode(comp))
            bi, ai = np.nonzero(boundary)
            if len(ai) < 30:
                continue
            score = (6.0 * compact) - (2.0 * frac_area) - (0.15 * (asp - 1.0))
            if score > best_score:
                best_score = score
                best_boundary = np.column_stack([aa[ai], bb[bi]])
        if best_boundary is None:
            continue
        band = 2.0 * max((a_max_p - a_min_p) / cfg.grid_n, (b_max_p - b_min_p) / cfg.grid_n)
        cand = boundary_band_raw_points(A, B, best_boundary, band)
        if cand.shape[0] < 50:
            continue
        fit = choose_fit(cand, last_center, last_axes, None, cfg)
        center, axes, R = fit["center"], fit["axes"], fit["R"]
        ar = axes[0] / (axes[1] + 1e-12)
        ok = ar <= cfg.max_ar
        if last_axes is not None:
            if axes[0] > cfg.max_axis_jump * last_axes[0] or axes[1] > cfg.max_axis_jump * last_axes[1]:
                ok = False
            if axes[0] < last_axes[0] / cfg.max_axis_jump or axes[1] < last_axes[1] / cfg.max_axis_jump:
                ok = False
        if not ok:
            continue
        section = build_section(axis_center, center, axes, R, dims, cfg)
        sections.append(section)
        kept.append((float(frac), axis_center, center, axes, R, i, "observed", fit["resid_med"], fit["resid_p90"], fit["inlier"]))
        last_center = center.copy()
        last_axes = axes

    if len(sections) < 2:
        raise RuntimeError("Not enough accepted slices to loft.")
    sections, kept = fill_short_gaps(sections, kept, fracs, dims_for_loft, cfg)
    loops = [resample_closed_polyline(sec, cfg.n_ring) for sec in sections]
    aligned = [loops[0]]
    for i in range(1, len(loops)):
        aligned.append(align_loop_to_prev(loops[i], aligned[-1]))
    parts = []
    for j0, j1 in zip(range(len(aligned) - 1), range(1, len(aligned))):
        parts.append(loft_between_loops(aligned[j0], aligned[j1]))
    if cfg.cap_ends:
        parts.append(cap_from_loop(aligned[0]))
        parts.append(cap_from_loop(aligned[-1]))
    surface = append_polydata(parts)
    if cfg.fill_holes:
        fill = vtk.vtkFillHolesFilter()
        fill.SetInputData(surface)
        fill.SetHoleSize(cfg.fill_hole_size)
        fill.Update()
        surface = fill.GetOutput()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    surface = normals.GetOutput()

    observed = [k for k in kept if k[6] == "observed"]
    add_field_array(surface, "accepted_slice_count", [len(kept)])
    add_field_array(surface, "observed_slice_count", [len(observed)])
    add_field_array(surface, "interpolated_slice_count", [len(kept) - len(observed)])
    add_field_array(surface, "boundary_nonmanifold_edge_segments", [boundary_nonmanifold_edge_count(surface)])
    if observed:
        add_field_array(surface, "inlier_mean", [float(np.mean([k[9] for k in observed]))])
        add_field_array(surface, "resid_med_mean", [float(np.mean([k[7] for k in observed]))])
        add_field_array(surface, "resid_p90_mean", [float(np.mean([k[8] for k in observed]))])
    return surface
