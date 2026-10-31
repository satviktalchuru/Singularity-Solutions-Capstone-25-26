import math

import numpy as np


class BoreholeConfig510:
    def __init__(self):
        self.chosen_axis = 0
        self.frac_start = 0.01
        self.frac_end = 0.69
        self.n_slices = 35
        self.thickness_frac = 0.02
        self.target_slice_points = 1500
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
        self.max_component_candidates = 12
        self.enable_dog_candidates = True
        self.dog_scales = (3.0, 5.0, 8.0, 12.0, 16.0)
        self.dog_max_candidates = 6
        self.dog_density_quantile = 0.30
        self.enable_global_tracking_fallback = True
        self.global_tracking_trigger_frac = 0.80
        self.global_tracking_max_gap = 3
        self.global_tracking_max_center_step = 1.0
        self.global_tracking_max_axis_jump = 1.5
        self.global_tracking_mode_switch_penalty = 0.12
        self.global_tracking_medium_spacing_ratio = 0.55
        self.enable_reference_corridor = True
        self.reference_bins = 48
        self.reference_min_points_per_bin = 20
        self.reference_proposal_max_error = 2.5
        self.reference_max_center_error = 1.5
        self.reference_max_axis_scale = 2.5
        self.reference_score_weight = 0.30
        self.reference_scale_score_weight = 0.20
        self.point_signal_score_weight = 0.25
        self.enable_reference_radial_candidates = True
        self.reference_radial_sectors = 72
        self.reference_radial_keep_per_sector = 5
        self.reference_radial_min_layers = 1
        self.reference_radial_max_scale = 2.5
        self.reference_radial_min_coverage = 0.35
        self.reference_radial_ellipse_q = 80.0
        self.reference_radial_fix_center_below_coverage = 0.90
        self.enable_adaptive_radial_routing = True
        self.reference_radial_stable_layers = 5
        self.reference_radial_stable_ellipse_q = 90.0
        self.reference_radial_dense_spacing_ratio = 0.25
        self.adaptive_route_replacement_margin = 0.03
        self.adaptive_route_off_mode_margin = 0.03
        self.adaptive_stable_max_residual_ratio = 1.10
        self.min_boundary_candidates = 50
        self.boundary_band_cells = 2.0
        self.adaptive_boundary_band = True
        self.max_boundary_band_cells = 8.0
        self.boundary_band_growth = 1.5
        self.do_mirror = True
        self.mirror_axis = 1
        self.ellipse_q = 90.0
        self.use_outer_band_refinement = True
        self.outer_band_sectors = 24
        self.outer_band_keep_per_sector = 6
        self.max_ar = 4.0
        self.max_axis_jump = 2.0
        self.enable_trajectory_gate = True
        self.trajectory_min_history = 3
        self.trajectory_window = 5
        self.max_center_prediction_frac = 0.75
        self.max_axis_prediction_frac = 0.45
        self.enable_confidence_gate = True
        self.min_slice_confidence = 0.35
        self.confidence_sectors = 24
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


def characterize_point_cloud(points, chosen_axis):
    _, _, spans = get_bounds(points)
    positive_spans = spans[spans > 1e-12]
    occupied_volume = float(np.prod(positive_spans)) if positive_spans.size else 0.0
    nominal_spacing = (
        (occupied_volume / float(len(points))) ** (1.0 / 3.0)
        if occupied_volume > 0.0 and len(points) > 0 else 0.0
    )
    transverse = [idx for idx in range(3) if idx != chosen_axis]
    return {
        "input_point_count": int(len(points)),
        "axis_span": float(spans[chosen_axis]),
        "transverse_span_0": float(spans[transverse[0]]),
        "transverse_span_1": float(spans[transverse[1]]),
        "nominal_point_spacing": float(nominal_spacing),
    }


def configure_from_reference_geometry(points, reference_points, cfg=None):
    """Infer dataset-dependent sampling parameters from soil and reference geometry."""
    cfg = cfg or BoreholeConfig510()
    points = np.asarray(points, dtype=float)
    reference_points = np.asarray(reference_points, dtype=float)
    if points.ndim != 2 or reference_points.ndim != 2:
        raise ValueError("points and reference_points must be two-dimensional arrays")

    reference_spans = np.ptp(reference_points, axis=0)
    cfg.chosen_axis = int(np.argmax(reference_spans))
    axis = cfg.chosen_axis
    soil_min = np.min(points, axis=0)
    soil_max = np.max(points, axis=0)
    soil_spans = soil_max - soil_min
    axis_span = float(soil_spans[axis])
    if axis_span <= 1e-12:
        raise ValueError("The inferred drilling axis has zero span in the soil point cloud")

    reference_min = float(np.min(reference_points[:, axis]))
    reference_max = float(np.max(reference_points[:, axis]))
    cfg.frac_start = float(np.clip((reference_min - soil_min[axis]) / axis_span, 0.0, 1.0))
    cfg.frac_end = float(np.clip((reference_max - soil_min[axis]) / axis_span, 0.0, 1.0))

    transverse = [dim for dim in range(3) if dim != axis]
    mirror_candidates = []
    for dim in transverse:
        span = max(float(soil_spans[dim]), 1e-12)
        near_zero_min = abs(float(soil_min[dim])) <= 0.05 * span
        near_zero_max = abs(float(soil_max[dim])) <= 0.05 * span
        one_sided = soil_min[dim] >= -0.05 * span or soil_max[dim] <= 0.05 * span
        if one_sided and (near_zero_min or near_zero_max):
            mirror_candidates.append(dim)
    cfg.do_mirror = bool(mirror_candidates)
    if mirror_candidates:
        cfg.mirror_axis = min(mirror_candidates, key=lambda dim: soil_spans[dim])

    characterization = characterize_point_cloud(points, axis)
    physical_thickness = max(
        4.0 * characterization["nominal_point_spacing"],
        axis_span / 500.0,
        axis_span * cfg.target_slice_points / float(max(len(points), 1)),
    )
    cfg.thickness_frac = float(np.clip(physical_thickness / axis_span, 0.001, 0.05))
    interval_length = max(reference_max - reference_min, physical_thickness)
    cfg.n_slices = int(np.clip(
        np.ceil(interval_length / physical_thickness) + 1,
        35,
        200,
    ))
    return cfg


def mirror_points(points, axis):
    mirrored = points.copy()
    mirrored[:, axis] *= -1.0
    return np.vstack([points, mirrored])


def build_reference_trajectory(reference_points, cfg):
    """Build a robust transverse centerline and local scale from reference geometry."""
    points = np.asarray(reference_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return None
    if cfg.do_mirror:
        points = mirror_points(points, cfg.mirror_axis)
    axis = cfg.chosen_axis
    dims = [idx for idx in range(3) if idx != axis]
    coordinates = points[:, axis]
    lo, hi = float(np.min(coordinates)), float(np.max(coordinates))
    if hi - lo <= 1e-12:
        return None
    edges = np.linspace(lo, hi, max(3, int(cfg.reference_bins)) + 1)
    rows = []
    for index in range(len(edges) - 1):
        if index == len(edges) - 2:
            mask = (coordinates >= edges[index]) & (coordinates <= edges[index + 1])
        else:
            mask = (coordinates >= edges[index]) & (coordinates < edges[index + 1])
        section = points[mask][:, dims]
        if section.shape[0] < cfg.reference_min_points_per_bin:
            continue
        center = np.median(section, axis=0)
        low = np.percentile(section, 5.0, axis=0)
        high = np.percentile(section, 95.0, axis=0)
        half_spans = 0.5 * (high - low)
        scale = max(float(np.max(half_spans)), 1e-12)
        rows.append((float(np.median(coordinates[mask])), center, scale))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda item: item[0])
    return {
        "axis": np.asarray([item[0] for item in rows], dtype=float),
        "centers": np.asarray([item[1] for item in rows], dtype=float),
        "scales": np.asarray([item[2] for item in rows], dtype=float),
        "dims": tuple(dims),
    }


def reference_at_axis(reference, axis_center):
    if reference is None:
        return None
    axis = reference["axis"]
    if axis_center < axis[0] or axis_center > axis[-1]:
        return None
    center = np.array([
        np.interp(axis_center, axis, reference["centers"][:, dim])
        for dim in range(2)
    ])
    scale = float(np.interp(axis_center, axis, reference["scales"]))
    return center, max(scale, 1e-12)


def sample_image_at_points(image, points, aa, bb):
    if points.shape[0] == 0:
        return 0.0
    cols = np.searchsorted(aa, points[:, 0], side="left")
    rows = np.searchsorted(bb, points[:, 1], side="left")
    cols = np.clip(cols, 0, len(aa) - 1)
    rows = np.clip(rows, 0, len(bb) - 1)
    return float(np.mean(image[rows, cols]))


def classify_radial_mode(coverage, _edge_proximity, spacing_ratio, cfg):
    """Select a boundary estimator from observable per-slice properties."""
    if coverage < cfg.reference_radial_fix_center_below_coverage:
        return "partial"
    if spacing_ratio <= cfg.reference_radial_dense_spacing_ratio:
        return "stable"
    return "inner"


def apply_adaptive_route_margin(candidates, preferred_mode, cfg):
    """Keep the inner fit unless an adaptive radial fit is measurably better."""
    baseline = [
        item for item in candidates
        if item.get("radial_mode") == "inner" and
        not item.get("geometry_failed", False) and
        not item.get("confidence_failed", False) and
        not item.get("reference_failed", False)
    ]
    if not baseline:
        return
    baseline_item = max(baseline, key=lambda item: item["selection_score"])
    baseline_score = baseline_item["selection_score"]
    baseline_residual = (
        baseline_item["fit"]["resid_p90"] /
        max(float(np.mean(baseline_item["axes"])), 1e-12)
    )
    for item in candidates:
        mode = item.get("radial_mode")
        if mode not in ("stable", "partial"):
            continue
        margin = cfg.adaptive_route_replacement_margin
        if mode != preferred_mode:
            margin += cfg.adaptive_route_off_mode_margin
        score_failed = item["selection_score"] < baseline_score + margin
        residual_failed = False
        if mode == "stable":
            normalized_residual = (
                item["fit"]["resid_p90"] /
                max(float(np.mean(item["axes"])), 1e-12)
            )
            residual_failed = (
                normalized_residual >
                cfg.adaptive_stable_max_residual_ratio * baseline_residual
            )
        item["route_failed"] = score_failed or residual_failed


def radial_inner_envelope(A, B, center, scale, cfg, min_layers=None, max_layers=None):
    points = np.column_stack([A, B])
    delta = points - center
    radius = np.linalg.norm(delta, axis=1)
    valid = (radius > 1e-12) & (radius <= cfg.reference_radial_max_scale * scale)
    if not np.any(valid):
        return np.empty((0, 2)), 0.0
    points = points[valid]
    radius = radius[valid]
    angles = np.mod(np.arctan2(delta[valid, 1], delta[valid, 0]), 2.0 * math.pi)
    sectors = np.floor(
        angles * cfg.reference_radial_sectors / (2.0 * math.pi)
    ).astype(int)
    min_layers = (
        cfg.reference_radial_min_layers if min_layers is None else int(min_layers)
    )
    max_layers = (
        cfg.reference_radial_keep_per_sector if max_layers is None else int(max_layers)
    )
    min_layers = max(1, min_layers)
    max_layers = max(min_layers, max_layers)
    ordered_by_sector = []
    occupied = 0
    for sector in range(cfg.reference_radial_sectors):
        indices = np.flatnonzero(sectors == sector)
        if indices.size == 0:
            continue
        occupied += 1
        order = indices[np.argsort(radius[indices])]
        ordered_by_sector.append(order[:max_layers])
    coverage = occupied / float(cfg.reference_radial_sectors)
    if coverage < cfg.reference_radial_min_coverage:
        return np.empty((0, 2)), coverage
    selected = []
    for rank in range(max_layers):
        selected.extend([
            int(order[rank]) for order in ordered_by_sector if rank < order.size
        ])
        if (
            rank + 1 >= min_layers and
            len(selected) >= cfg.min_boundary_candidates
        ):
            break
    return points[np.asarray(selected, dtype=int)], coverage


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


def adaptive_boundary_candidates(A, B, boundary_pts, cell_size, cfg):
    band_cells = float(cfg.boundary_band_cells)
    candidates = boundary_band_raw_points(A, B, boundary_pts, band_cells * cell_size)
    if not cfg.adaptive_boundary_band:
        return candidates, band_cells
    while candidates.shape[0] < cfg.min_boundary_candidates and band_cells < cfg.max_boundary_band_cells:
        band_cells = min(
            cfg.max_boundary_band_cells,
            max(band_cells + 1.0, band_cells * cfg.boundary_band_growth),
        )
        candidates = boundary_band_raw_points(A, B, boundary_pts, band_cells * cell_size)
    return candidates, band_cells


def dog_blob_proposals(D, occ, aa, bb, cfg):
    if not cfg.enable_dog_candidates or not np.any(occ):
        return []
    density_limit = float(np.quantile(D[occ], cfg.dog_density_quantile))
    noise_scale = float(np.std(D[occ])) + 1e-12
    peaks = []
    for sigma in cfg.dog_scales:
        small = gaussian_filter_numpy(D, sigma)
        large = gaussian_filter_numpy(D, 1.6 * sigma)
        response = (large - small) / noise_scale
        response = np.where(occ & (D <= density_limit), response, -np.inf)
        order = np.argsort(response.ravel())[::-1]
        accepted_for_scale = 0
        for flat_idx in order:
            value = float(response.ravel()[flat_idx])
            if not np.isfinite(value) or value <= 0.0:
                break
            row, col = np.unravel_index(flat_idx, response.shape)
            if any((row - item[0]) ** 2 + (col - item[1]) ** 2 < (0.8 * sigma) ** 2 for item in peaks):
                continue
            peaks.append((row, col, float(sigma), value))
            accepted_for_scale += 1
            if accepted_for_scale >= 2:
                break
    peaks.sort(key=lambda item: item[3], reverse=True)
    selected = []
    for row, col, sigma, value in peaks:
        if any(
            (row - item[0]) ** 2 + (col - item[1]) ** 2 < (0.75 * max(sigma, item[2])) ** 2
            for item in selected
        ):
            continue
        selected.append((row, col, sigma, value))
        if len(selected) >= cfg.dog_max_candidates:
            break
    cell_a = float(np.median(np.diff(aa))) if len(aa) > 1 else 1.0
    cell_b = float(np.median(np.diff(bb))) if len(bb) > 1 else 1.0
    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    proposals = []
    for row, col, sigma, value in selected:
        radius_pixels = math.sqrt(2.0) * sigma
        boundary = np.column_stack([
            aa[col] + radius_pixels * cell_a * np.cos(angles),
            bb[row] + radius_pixels * cell_b * np.sin(angles),
        ])
        proposals.append({
            "boundary": boundary,
            "proposal_quality": float(value / (1.0 + value)),
            "source": "dog",
        })
    return proposals


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


def fit_ellipse_fixed_center(points2d, center):
    """Fit a centered quadratic ellipse, including from an incomplete boundary arc."""
    center = np.asarray(center, dtype=float)
    local = np.asarray(points2d, dtype=float) - center
    design = np.column_stack([
        local[:, 0] ** 2,
        2.0 * local[:, 0] * local[:, 1],
        local[:, 1] ** 2,
    ])
    keep = np.ones(local.shape[0], dtype=bool)
    coefficients = None
    for _ in range(2):
        if np.count_nonzero(keep) < 6:
            return None
        coefficients = np.linalg.lstsq(
            design[keep], np.ones(np.count_nonzero(keep)), rcond=None
        )[0]
        algebraic_residual = np.abs(design @ coefficients - 1.0)
        cutoff = np.percentile(algebraic_residual, 90.0)
        keep = algebraic_residual <= cutoff
    quadratic = np.array([
        [coefficients[0], coefficients[1]],
        [coefficients[1], coefficients[2]],
    ])
    values, vectors = np.linalg.eigh(quadratic)
    if np.any(values <= 1e-12):
        return None
    axes = 1.0 / np.sqrt(values)
    order = np.argsort(axes)[::-1]
    axes = axes[order]
    vectors = vectors[:, order]
    return center, (float(axes[0]), float(axes[1])), vectors


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


def choose_fit(
    cand, last_center, last_axes, last_R, cfg, allow_outer=True, fit_q=None,
    fixed_center=None,
):
    q = cfg.ellipse_q if fit_q is None else float(fit_q)
    if fixed_center is not None:
        variants = [("fixed_center", cand, np.asarray(fixed_center, dtype=float))]
    else:
        variants = [("raw", cand, None)]
        if allow_outer:
            outer = refine_candidates_to_outer_band(cand, cfg)
            if outer.shape[0] >= 20 and outer.shape[0] != cand.shape[0]:
                variants.append(("outer", outer, None))
    best = None
    for name, pts, constrained_center in variants:
        if constrained_center is None:
            fit_geometry = fit_ellipse_pca(pts, q)
        else:
            fit_geometry = fit_ellipse_fixed_center(pts, constrained_center)
            if fit_geometry is None:
                continue
        center, axes, R = fit_geometry
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


def predict_trajectory(kept, axis_center, cfg):
    observed = [item for item in kept if item[6] == "observed"]
    if len(observed) < cfg.trajectory_min_history:
        return None
    history = observed[-max(cfg.trajectory_min_history, cfg.trajectory_window):]
    x = np.asarray([item[1] for item in history], dtype=float)
    if np.ptp(x) <= 1e-12:
        return None
    centers = np.asarray([item[2] for item in history], dtype=float)
    axes = np.asarray([item[3] for item in history], dtype=float)
    predicted_center = np.array([
        np.polyval(np.polyfit(x, centers[:, dim], 1), axis_center)
        for dim in range(2)
    ])
    predicted_axes = np.array([
        np.polyval(np.polyfit(x, axes[:, dim], 1), axis_center)
        for dim in range(2)
    ])
    if np.any(predicted_axes <= 0.0):
        return None
    return predicted_center, predicted_axes


def trajectory_errors(center, axes, prediction):
    if prediction is None:
        return 0.0, 0.0
    predicted_center, predicted_axes = prediction
    radius_scale = max(float(np.mean(predicted_axes)), 1e-12)
    center_error = float(np.linalg.norm(center - predicted_center) / radius_scale)
    axis_error = float(np.max(np.abs(np.asarray(axes) - predicted_axes) / (predicted_axes + 1e-12)))
    return center_error, axis_error


def angular_coverage(points, center, axes, R, sectors):
    if points.shape[0] == 0:
        return 0.0
    local = (points - center) @ R
    angles = np.mod(
        np.arctan2(local[:, 1] / (axes[1] + 1e-12), local[:, 0] / (axes[0] + 1e-12)),
        2.0 * math.pi,
    )
    occupied = np.unique(np.floor(angles * sectors / (2.0 * math.pi)).astype(int))
    return float(len(occupied) / float(sectors))


def slice_confidence(fit, prediction_errors, cfg):
    center_error, axis_error = prediction_errors
    radius_scale = max(float(np.mean(fit["axes"])), 1e-12)
    normalized_p90 = fit["resid_p90"] / radius_scale
    residual_score = math.exp(-normalized_p90 / 0.15)
    inlier_score = float(np.clip((fit["inlier"] - 0.50) / 0.50, 0.0, 1.0))
    coverage = angular_coverage(
        fit["points"], fit["center"], fit["axes"], fit["R"], cfg.confidence_sectors
    )
    if prediction_errors == (0.0, 0.0):
        trajectory_score = 1.0
    else:
        trajectory_score = math.exp(
            -0.5 * center_error / max(cfg.max_center_prediction_frac, 1e-12)
            -0.5 * axis_error / max(cfg.max_axis_prediction_frac, 1e-12)
        )
    confidence = (
        0.35 * inlier_score +
        0.30 * residual_score +
        0.20 * coverage +
        0.15 * trajectory_score
    )
    return float(confidence), float(coverage), float(normalized_p90)


def select_global_candidate_path(candidate_slices, cfg):
    states = {}
    for slice_pos, entry in enumerate(candidate_slices):
        for cand_pos, candidate in enumerate(entry["candidates"]):
            if (
                candidate.get("reference_failed", False) or
                candidate.get("route_failed", False)
            ):
                continue
            fit = candidate["fit"]
            axes = np.asarray(candidate["axes"], dtype=float)
            if axes[1] <= 0.0 or axes[0] / axes[1] > cfg.max_ar:
                continue
            base_confidence, _, _ = slice_confidence(fit, (0.0, 0.0), cfg)
            if cfg.enable_confidence_gate and base_confidence < cfg.min_slice_confidence:
                continue
            reference_penalty = cfg.reference_score_weight * candidate.get("reference_error", 0.0)
            best_state = {
                "count": 1,
                "score": base_confidence - reference_penalty,
                "previous": None,
                "confidence": base_confidence,
            }
            for prev_slice_pos in range(max(0, slice_pos - cfg.global_tracking_max_gap), slice_pos):
                previous_entry = candidate_slices[prev_slice_pos]
                gap = entry["slice_index"] - previous_entry["slice_index"] - 1
                if gap < 0 or gap >= cfg.global_tracking_max_gap:
                    continue
                for prev_cand_pos, previous in enumerate(previous_entry["candidates"]):
                    prev_key = (prev_slice_pos, prev_cand_pos)
                    if prev_key not in states:
                        continue
                    prev_axes = np.asarray(previous["axes"], dtype=float)
                    axis_ratio = np.maximum(axes / (prev_axes + 1e-12), prev_axes / (axes + 1e-12))
                    if float(np.max(axis_ratio)) > cfg.global_tracking_max_axis_jump:
                        continue
                    radius_scale = max(float(np.mean(prev_axes)), 1e-12)
                    center_step = float(
                        np.linalg.norm(candidate["center"] - previous["center"]) / radius_scale
                    )
                    if center_step > cfg.global_tracking_max_center_step:
                        continue
                    axis_change = float(np.mean(np.abs(np.log((axes + 1e-12) / (prev_axes + 1e-12)))))
                    mode_switch = (
                        candidate.get("radial_mode") or candidate.get("source")
                    ) != (
                        previous.get("radial_mode") or previous.get("source")
                    )
                    previous_state = states[prev_key]
                    score = (
                        previous_state["score"] + base_confidence -
                        0.35 * center_step - 0.25 * axis_change - 0.10 * gap -
                        reference_penalty -
                        cfg.global_tracking_mode_switch_penalty * float(mode_switch)
                    )
                    proposal = {
                        "count": previous_state["count"] + 1,
                        "score": score,
                        "previous": prev_key,
                        "confidence": base_confidence,
                    }
                    if (proposal["score"], proposal["count"]) > (
                        best_state["score"], best_state["count"]
                    ):
                        best_state = proposal
            states[(slice_pos, cand_pos)] = best_state
    if not states:
        return []
    end_key = max(states, key=lambda key: (states[key]["score"], states[key]["count"]))
    path = []
    key = end_key
    while key is not None:
        slice_pos, cand_pos = key
        entry = candidate_slices[slice_pos]
        candidate = dict(entry["candidates"][cand_pos])
        candidate["slice_index"] = entry["slice_index"]
        candidate["frac"] = entry["frac"]
        candidate["axis_center"] = entry["axis_center"]
        candidate["dims"] = entry["dims"]
        candidate["confidence"] = states[key]["confidence"]
        path.append(candidate)
        key = states[key]["previous"]
    return list(reversed(path))


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


def mesh_from_loops(loops, cap_ends):
    if not loops:
        raise RuntimeError("No section loops were available for mesh construction.")
    n_ring = loops[0].shape[0]
    if any(loop.shape != (n_ring, 3) for loop in loops):
        raise RuntimeError("All section loops must contain the same number of 3D points.")

    point_blocks = [np.asarray(loop, dtype=float) for loop in loops]
    faces = []
    for section_index in range(len(loops) - 1):
        base0 = section_index * n_ring
        base1 = (section_index + 1) * n_ring
        for i in range(n_ring):
            i0 = i
            i1 = (i + 1) % n_ring
            faces.append([base0 + i0, base0 + i1, base1 + i1])
            faces.append([base0 + i0, base1 + i1, base1 + i0])

    if cap_ends:
        start_center = len(point_blocks) * n_ring
        point_blocks.append(np.mean(loops[0], axis=0, keepdims=True))
        end_center = start_center + 1
        point_blocks.append(np.mean(loops[-1], axis=0, keepdims=True))
        end_base = (len(loops) - 1) * n_ring
        for i in range(n_ring):
            i1 = (i + 1) % n_ring
            faces.append([start_center, i, i1])
            faces.append([end_center, end_base + i, end_base + i1])

    return np.vstack(point_blocks), np.asarray(faces, dtype=np.int64)


def reconstruct_borehole(points, cfg=None, reference_points=None, point_signal=None):
    cfg = cfg or BoreholeConfig510()
    work = np.asarray(points, dtype=float)
    characterization = characterize_point_cloud(work, cfg.chosen_axis)
    reference_trajectory = (
        build_reference_trajectory(reference_points, cfg)
        if reference_points is not None and cfg.enable_reference_corridor else None
    )
    reference_spacing_ratio_median = float("inf")
    if reference_trajectory is not None:
        reference_spacing_ratio_median = float(np.median(
            characterization["nominal_point_spacing"] /
            np.maximum(reference_trajectory["scales"], 1e-12)
        ))
    use_layered_baseline = (
        reference_trajectory is not None and (
            reference_spacing_ratio_median > cfg.global_tracking_medium_spacing_ratio or
            (
                point_signal is not None and
                cfg.reference_radial_dense_spacing_ratio < reference_spacing_ratio_median <=
                cfg.global_tracking_medium_spacing_ratio
            )
        )
    )
    baseline_radial_layers = (
        cfg.reference_radial_stable_layers
        if use_layered_baseline else cfg.reference_radial_min_layers
    )
    signal_work = None
    if point_signal is not None:
        signal_work = np.asarray(point_signal, dtype=float).reshape(-1)
        if signal_work.shape[0] != work.shape[0]:
            raise ValueError("point_signal must contain one value per input point")
    if cfg.do_mirror:
        work = mirror_points(work, cfg.mirror_axis)
        if signal_work is not None:
            signal_work = np.concatenate([signal_work, signal_work])
    fracs = np.linspace(cfg.frac_start, cfg.frac_end, cfg.n_slices)
    sections = []
    kept = []
    last_axes = None
    last_center = None
    dims_for_loft = None
    trajectory_rejected = 0
    confidence_rejected = 0
    center_prediction_errors = []
    axis_prediction_errors = []
    accepted_confidences = []
    boundary_band_cells_used = []
    boundary_band_expanded_count = 0
    component_candidates_considered = []
    selected_source_counts = {"component": 0, "dog": 0, "radial": 0}
    selected_fit_counts = {"raw": 0, "outer": 0, "fixed_center": 0}
    selected_radial_mode_counts = {"inner": 0, "stable": 0, "partial": 0}
    classified_radial_mode_counts = {"inner": 0, "stable": 0, "partial": 0}
    all_candidate_slices = []
    global_tracking_used = False
    reference_rejected = 0
    accepted_reference_errors = []
    accepted_reference_axis_scales = []
    accepted_signal_scores = []
    diagnostics = {
        "slice_fail_insufficient_points": 0,
        "slice_fail_empty_occupancy": 0,
        "slice_fail_no_interior_component": 0,
        "slice_fail_component_filters": 0,
        "slice_fail_boundary_candidates": 0,
        "slice_fail_geometry_continuity": 0,
        "slice_fail_trajectory": 0,
        "slice_fail_confidence": 0,
        "slice_fail_reference_corridor": 0,
        "slice_accepted_observed": 0,
    }

    for i, frac in enumerate(fracs):
        mins, _, ranges = get_bounds(work)
        axis_center = mins[cfg.chosen_axis] + float(frac) * ranges[cfg.chosen_axis]
        thickness = cfg.thickness_frac * ranges[cfg.chosen_axis]
        slice_mask = np.abs(work[:, cfg.chosen_axis] - axis_center) <= thickness / 2.0
        slice_pts = work[slice_mask]
        slice_signal = signal_work[slice_mask] if signal_work is not None else None
        if slice_pts.shape[0] < 500:
            diagnostics["slice_fail_insufficient_points"] += 1
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
            diagnostics["slice_fail_empty_occupancy"] += 1
            continue
        preferred_radial_mode = "inner"
        reference_prior = reference_at_axis(reference_trajectory, axis_center)
        low = (D <= float(np.quantile(D[occ], cfg.low_density_q))) & occ
        low = dilate(erode(low, cfg.open_iters), cfg.close_iters)
        low_interior, _, keep_ids = remove_border_touching_components(low)
        component_proposals = []
        if keep_ids:
            labels, ncomp = connected_components(low_interior)
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
                component_proposals.append({
                    "boundary": np.column_stack([aa[ai], bb[bi]]),
                    "component_score": float(score),
                    "proposal_quality": float(1.0 / (1.0 + math.exp(-score))),
                    "source": "component",
                })
        component_proposals.extend(dog_blob_proposals(D, occ, aa, bb, cfg))
        if reference_prior is not None and cfg.enable_reference_radial_candidates:
            baseline_points, radial_coverage = radial_inner_envelope(
                A, B, reference_prior[0], reference_prior[1], cfg,
                min_layers=baseline_radial_layers,
                max_layers=max(baseline_radial_layers, cfg.reference_radial_keep_per_sector),
            )
            inner_points, _ = radial_inner_envelope(
                A, B, reference_prior[0], reference_prior[1], cfg,
                min_layers=1, max_layers=1,
            )
            radial_edge_proximity = float(min(
                reference_prior[0][0] - a_min,
                a_max - reference_prior[0][0],
                reference_prior[0][1] - b_min,
                b_max - reference_prior[0][1],
            ) / reference_prior[1])
            radial_spacing_ratio = float(
                characterization["nominal_point_spacing"] / reference_prior[1]
            )
            preferred_radial_mode = classify_radial_mode(
                radial_coverage, radial_edge_proximity, radial_spacing_ratio, cfg
            )
            classified_radial_mode_counts[preferred_radial_mode] += 1
            incomplete = preferred_radial_mode == "partial"
            if baseline_points.shape[0] >= cfg.min_boundary_candidates:
                component_proposals.append({
                    "boundary": baseline_points,
                    "raw_candidates": baseline_points,
                    "proposal_quality": float(radial_coverage),
                    "angular_coverage": float(radial_coverage),
                    "source": "radial",
                    "radial_mode": "inner",
                    "fit_q": cfg.reference_radial_ellipse_q,
                    "fixed_center": None,
                    "radial_spacing_ratio": radial_spacing_ratio,
                    "radial_edge_proximity": radial_edge_proximity,
                    "allow_outer": False,
                })
                if incomplete:
                    component_proposals.append({
                        "boundary": inner_points,
                        "raw_candidates": inner_points,
                        "proposal_quality": float(radial_coverage),
                        "angular_coverage": float(radial_coverage),
                        "source": "radial",
                        "radial_mode": "partial",
                        "fit_q": cfg.reference_radial_ellipse_q,
                        "fixed_center": reference_prior[0],
                        "radial_spacing_ratio": radial_spacing_ratio,
                        "radial_edge_proximity": radial_edge_proximity,
                        "allow_outer": False,
                    })
            if cfg.enable_adaptive_radial_routing and not incomplete:
                stable_points, _ = radial_inner_envelope(
                    A, B, reference_prior[0], reference_prior[1], cfg,
                    min_layers=cfg.reference_radial_stable_layers,
                    max_layers=cfg.reference_radial_stable_layers,
                )
                if stable_points.shape[0] >= cfg.min_boundary_candidates:
                    component_proposals.append({
                        "boundary": stable_points,
                        "raw_candidates": stable_points,
                        "proposal_quality": float(radial_coverage),
                        "angular_coverage": float(radial_coverage),
                        "source": "radial",
                        "radial_mode": "stable",
                        "fit_q": cfg.reference_radial_stable_ellipse_q,
                        "fixed_center": None,
                        "radial_spacing_ratio": radial_spacing_ratio,
                        "radial_edge_proximity": radial_edge_proximity,
                        "allow_outer": True,
                    })
        if not component_proposals:
            if keep_ids:
                diagnostics["slice_fail_component_filters"] += 1
            else:
                diagnostics["slice_fail_no_interior_component"] += 1
            continue
        signal_image = None
        if slice_signal is not None:
            signal_sum = np.histogram2d(
                A, B, bins=cfg.grid_n,
                range=[[a_min_p, a_max_p], [b_min_p, b_max_p]],
                weights=slice_signal,
            )[0].T
            signal_count = np.histogram2d(
                A, B, bins=cfg.grid_n,
                range=[[a_min_p, a_max_p], [b_min_p, b_max_p]],
            )[0].T
            signal_image = signal_sum / np.maximum(signal_count, 1.0)
            occupied_signal = signal_image[signal_count > 0]
            if occupied_signal.size:
                low_signal, high_signal = np.percentile(occupied_signal, [50.0, 95.0])
                signal_image = np.clip(
                    (signal_image - low_signal) / (high_signal - low_signal + 1e-12), 0.0, 1.0
                )
            for proposal in component_proposals:
                proposal["signal_score"] = sample_image_at_points(
                    signal_image, proposal["boundary"], aa, bb
                )
        if reference_prior is not None:
            prior_center, prior_scale = reference_prior
            for proposal in component_proposals:
                proposal_center = np.mean(proposal["boundary"], axis=0)
                proposal["reference_error"] = float(
                    np.linalg.norm(proposal_center - prior_center) / prior_scale
                )
            component_proposals = [
                proposal for proposal in component_proposals
                if proposal["reference_error"] <= cfg.reference_proposal_max_error
            ]
        if not component_proposals:
            diagnostics["slice_fail_reference_corridor"] += 1
            reference_rejected += 1
            continue
        component_proposals.sort(
            key=lambda item: (
                item["proposal_quality"] -
                cfg.reference_score_weight * item.get("reference_error", 0.0) +
                cfg.point_signal_score_weight * item.get("signal_score", 0.0)
            ),
            reverse=True,
        )
        component_proposals = component_proposals[:cfg.max_component_candidates]
        component_candidates_considered.append(len(component_proposals))
        cell_size = max((a_max_p - a_min_p) / cfg.grid_n, (b_max_p - b_min_p) / cfg.grid_n)
        prediction = predict_trajectory(kept, axis_center, cfg) if cfg.enable_trajectory_gate else None
        reference = prediction
        if reference is None and last_center is not None and last_axes is not None:
            reference = (last_center, np.asarray(last_axes, dtype=float))
        fit_center = reference[0] if reference is not None else (
            reference_prior[0] if reference_prior is not None else None
        )
        fit_axes = tuple(reference[1]) if reference is not None else None
        evaluated = []
        slice_band_cells = []
        for proposal in component_proposals:
            if "raw_candidates" in proposal:
                cand = proposal["raw_candidates"]
                band_cells_used = 0.0
            else:
                cand, band_cells_used = adaptive_boundary_candidates(
                    A, B, proposal["boundary"], cell_size, cfg
                )
            slice_band_cells.append(band_cells_used)
            if cand.shape[0] < cfg.min_boundary_candidates:
                continue
            fit = choose_fit(
                cand,
                fit_center,
                fit_axes,
                None,
                cfg,
                allow_outer=proposal.get(
                    "allow_outer", proposal["source"] != "radial"
                ),
                fit_q=proposal.get("fit_q", cfg.ellipse_q),
                fixed_center=proposal.get("fixed_center"),
            )
            center, axes, R = fit["center"], fit["axes"], fit["R"]
            ar = axes[0] / (axes[1] + 1e-12)
            geometry_failed = ar > cfg.max_ar
            if last_axes is not None:
                if axes[0] > cfg.max_axis_jump * last_axes[0] or axes[1] > cfg.max_axis_jump * last_axes[1]:
                    geometry_failed = True
                if axes[0] < last_axes[0] / cfg.max_axis_jump or axes[1] < last_axes[1] / cfg.max_axis_jump:
                    geometry_failed = True
            center_error, axis_error = trajectory_errors(center, axes, reference)
            trajectory_failed = (
                reference is not None and cfg.enable_trajectory_gate and
                (center_error > cfg.max_center_prediction_frac or axis_error > cfg.max_axis_prediction_frac)
            )
            confidence, _, _ = slice_confidence(fit, (center_error, axis_error), cfg)
            confidence_failed = cfg.enable_confidence_gate and confidence < cfg.min_slice_confidence
            reference_error = 0.0
            reference_failed = False
            reference_axis_scale = 0.0
            reference_scale_penalty = 0.0
            if reference_prior is not None:
                reference_error = float(
                    np.linalg.norm(center - reference_prior[0]) / reference_prior[1]
                )
                reference_axis_scale = float(np.max(np.asarray(axes)) / reference_prior[1])
                reference_scale_penalty = abs(math.log(max(reference_axis_scale, 1e-12)))
                reference_failed = (
                    reference_error > cfg.reference_max_center_error or
                    reference_axis_scale > cfg.reference_max_axis_scale
                )
            selection_score = (
                0.75 * confidence + 0.25 * proposal["proposal_quality"] -
                0.10 * center_error - 0.10 * axis_error -
                cfg.reference_score_weight * reference_error -
                cfg.reference_scale_score_weight * reference_scale_penalty +
                cfg.point_signal_score_weight * proposal.get("signal_score", 0.0)
            )
            evaluated.append({
                "fit": fit,
                "center": center,
                "axes": axes,
                "R": R,
                "confidence": confidence,
                "center_error": center_error,
                "axis_error": axis_error,
                "geometry_failed": geometry_failed,
                "trajectory_failed": trajectory_failed,
                "confidence_failed": confidence_failed,
                "reference_error": reference_error,
                "reference_axis_scale": reference_axis_scale,
                "reference_failed": reference_failed,
                "selection_score": selection_score,
                "band_cells": band_cells_used,
                "source": proposal["source"],
                "radial_mode": proposal.get("radial_mode"),
                "radial_spacing_ratio": proposal.get("radial_spacing_ratio"),
                "adaptive_route": proposal.get("radial_mode") in ("stable", "partial"),
                "route_failed": False,
                "signal_score": proposal.get("signal_score", 0.0),
            })
        if not evaluated:
            if slice_band_cells:
                boundary_band_cells_used.append(max(slice_band_cells))
                if max(slice_band_cells) > cfg.boundary_band_cells:
                    boundary_band_expanded_count += 1
            diagnostics["slice_fail_boundary_candidates"] += 1
            continue
        apply_adaptive_route_margin(evaluated, preferred_radial_mode, cfg)
        dense_reference_sampling = (
            reference_spacing_ratio_median <= cfg.reference_radial_dense_spacing_ratio
        )
        unsupported_medium_stable = (
            point_signal is None and
            cfg.reference_radial_dense_spacing_ratio < reference_spacing_ratio_median <=
            cfg.global_tracking_medium_spacing_ratio
        )
        if unsupported_medium_stable:
            for item in evaluated:
                if item.get("radial_mode") == "stable":
                    item["route_failed"] = True
        elif dense_reference_sampling:
            for item in evaluated:
                if item.get("radial_mode") == "stable":
                    item["route_failed"] = False
            stable_available = any(
                item.get("radial_mode") == "stable"
                for item in evaluated
            )
            for item in evaluated:
                if stable_available and item.get("radial_mode") == "inner":
                    item["route_failed"] = True
        all_candidate_slices.append({
            "slice_index": i,
            "frac": float(frac),
            "axis_center": axis_center,
            "dims": dims,
            "candidates": evaluated,
        })
        valid = [
            item for item in evaluated
            if not item["geometry_failed"] and
            not item["trajectory_failed"] and
            not item["confidence_failed"] and
            not item["reference_failed"] and
            not item["route_failed"] and
            (
                not item["adaptive_route"] or
                (
                    item.get("radial_mode") == "stable" and
                    reference_spacing_ratio_median <=
                    cfg.reference_radial_dense_spacing_ratio
                ) or (
                    item.get("radial_mode") == "partial" and
                    preferred_radial_mode == "partial"
                )
            )
        ]
        if not valid:
            if all(item["geometry_failed"] for item in evaluated):
                diagnostics["slice_fail_geometry_continuity"] += 1
            elif all(item["geometry_failed"] or item["trajectory_failed"] for item in evaluated):
                diagnostics["slice_fail_trajectory"] += 1
                trajectory_rejected += 1
            elif all(item["reference_failed"] for item in evaluated):
                diagnostics["slice_fail_reference_corridor"] += 1
                reference_rejected += 1
            else:
                diagnostics["slice_fail_confidence"] += 1
                confidence_rejected += 1
            continue
        selected = max(valid, key=lambda item: item["selection_score"])
        selected_source_counts[selected["source"]] += 1
        if selected.get("radial_mode") is not None:
            selected_radial_mode_counts[selected["radial_mode"]] += 1
        fit = selected["fit"]
        selected_fit_counts[fit["name"]] += 1
        center, axes, R = selected["center"], selected["axes"], selected["R"]
        confidence = selected["confidence"]
        center_prediction_errors.append(selected["center_error"])
        axis_prediction_errors.append(selected["axis_error"])
        boundary_band_cells_used.append(selected["band_cells"])
        if selected["band_cells"] > cfg.boundary_band_cells:
            boundary_band_expanded_count += 1
        section = build_section(axis_center, center, axes, R, dims, cfg)
        sections.append(section)
        kept.append((float(frac), axis_center, center, axes, R, i, "observed", fit["resid_med"], fit["resid_p90"], fit["inlier"]))
        last_center = center.copy()
        last_axes = axes
        accepted_confidences.append(confidence)
        if reference_prior is not None:
            accepted_reference_errors.append(selected["reference_error"])
            accepted_reference_axis_scales.append(selected["reference_axis_scale"])
        if signal_image is not None:
            accepted_signal_scores.append(selected["signal_score"])
        diagnostics["slice_accepted_observed"] += 1

    fallback_trigger = max(2, int(math.ceil(cfg.global_tracking_trigger_frac * cfg.n_slices)))
    medium_density_global = (
        cfg.reference_radial_dense_spacing_ratio < reference_spacing_ratio_median <=
        cfg.global_tracking_medium_spacing_ratio and
        point_signal is not None
    )
    if (
        cfg.enable_global_tracking_fallback and
        (len(sections) < fallback_trigger or medium_density_global)
    ):
        global_path = select_global_candidate_path(all_candidate_slices, cfg)
        if len(global_path) > len(sections):
            global_tracking_used = True
            sections = []
            kept = []
            accepted_confidences = []
            accepted_reference_errors = []
            accepted_reference_axis_scales = []
            accepted_signal_scores = []
            selected_source_counts = {"component": 0, "dog": 0, "radial": 0}
            selected_fit_counts = {"raw": 0, "outer": 0, "fixed_center": 0}
            selected_radial_mode_counts = {"inner": 0, "stable": 0, "partial": 0}
            for item in global_path:
                fit = item["fit"]
                section = build_section(
                    item["axis_center"], item["center"], item["axes"], item["R"], item["dims"], cfg
                )
                sections.append(section)
                kept.append((
                    item["frac"], item["axis_center"], item["center"], item["axes"], item["R"],
                    item["slice_index"], "observed", fit["resid_med"], fit["resid_p90"], fit["inlier"],
                ))
                accepted_confidences.append(item["confidence"])
                if reference_trajectory is not None:
                    accepted_reference_errors.append(item.get("reference_error", 0.0))
                    accepted_reference_axis_scales.append(item.get("reference_axis_scale", 0.0))
                if point_signal is not None:
                    accepted_signal_scores.append(item.get("signal_score", 0.0))
                selected_source_counts[item["source"]] += 1
                if item.get("radial_mode") is not None:
                    selected_radial_mode_counts[item["radial_mode"]] += 1
                selected_fit_counts[fit["name"]] += 1
            diagnostics["slice_accepted_observed"] = len(global_path)

    if len(sections) < 2:
        raise RuntimeError("Not enough accepted slices to loft.")
    sections, kept = fill_short_gaps(sections, kept, fracs, dims_for_loft, cfg)
    loops = [resample_closed_polyline(sec, cfg.n_ring) for sec in sections]
    aligned = [loops[0]]
    for i in range(1, len(loops)):
        aligned.append(align_loop_to_prev(loops[i], aligned[-1]))
    mesh_points, mesh_faces = mesh_from_loops(aligned, cfg.cap_ends)
    observed = [k for k in kept if k[6] == "observed"]
    metrics = {
        "accepted_slice_count": len(kept),
        "observed_slice_count": len(observed),
        "interpolated_slice_count": len(kept) - len(observed),
        "observed_slice_fraction": len(observed) / float(cfg.n_slices),
        "interpolated_slice_fraction": (len(kept) - len(observed)) / float(max(len(kept), 1)),
        "trajectory_rejected_slice_count": trajectory_rejected,
        "confidence_rejected_slice_count": confidence_rejected,
        "global_tracking_used": int(global_tracking_used),
        "reference_corridor_used": int(reference_trajectory is not None),
        "reference_rejected_slice_count": reference_rejected,
        "point_signal_used": int(point_signal is not None),
        "reference_spacing_ratio_median": (
            reference_spacing_ratio_median
            if np.isfinite(reference_spacing_ratio_median) else 0.0
        ),
        "baseline_radial_layers": int(baseline_radial_layers),
    }
    metrics.update(diagnostics)
    metrics.update(characterization)
    metrics["boundary_band_expanded_slice_count"] = boundary_band_expanded_count
    if boundary_band_cells_used:
        metrics["boundary_band_cells_mean"] = float(np.mean(boundary_band_cells_used))
        metrics["boundary_band_cells_max"] = float(np.max(boundary_band_cells_used))
    if component_candidates_considered:
        metrics["component_candidates_mean"] = float(np.mean(component_candidates_considered))
        metrics["component_candidates_max"] = int(np.max(component_candidates_considered))
    metrics["selected_component_slice_count"] = selected_source_counts["component"]
    metrics["selected_dog_slice_count"] = selected_source_counts["dog"]
    metrics["selected_radial_slice_count"] = selected_source_counts["radial"]
    metrics["selected_radial_inner_slice_count"] = selected_radial_mode_counts["inner"]
    metrics["selected_radial_stable_slice_count"] = selected_radial_mode_counts["stable"]
    metrics["selected_radial_partial_slice_count"] = selected_radial_mode_counts["partial"]
    metrics["classified_radial_inner_slice_count"] = classified_radial_mode_counts["inner"]
    metrics["classified_radial_stable_slice_count"] = classified_radial_mode_counts["stable"]
    metrics["classified_radial_partial_slice_count"] = classified_radial_mode_counts["partial"]
    metrics["selected_raw_fit_slice_count"] = selected_fit_counts["raw"]
    metrics["selected_outer_fit_slice_count"] = selected_fit_counts["outer"]
    metrics["selected_fixed_center_slice_count"] = selected_fit_counts["fixed_center"]
    if observed:
        observed_axes = np.asarray([k[3] for k in observed], dtype=float)
        metrics["inlier_mean"] = float(np.mean([k[9] for k in observed]))
        metrics["resid_med_mean"] = float(np.mean([k[7] for k in observed]))
        metrics["resid_p90_mean"] = float(np.mean([k[8] for k in observed]))
        metrics["major_axis_mean"] = float(np.mean(observed_axes[:, 0]))
        metrics["minor_axis_mean"] = float(np.mean(observed_axes[:, 1]))
        mean_radius = float(np.mean(observed_axes))
        metrics["resid_p90_normalized"] = metrics["resid_p90_mean"] / max(mean_radius, 1e-12)
        if len(observed) > 1:
            observed_centers = np.asarray([k[2] for k in observed], dtype=float)
            center_steps = np.linalg.norm(np.diff(observed_centers, axis=0), axis=1)
            radius_scale = np.maximum(np.mean(observed_axes[:-1], axis=1), 1e-12)
            normalized_steps = center_steps / radius_scale
            axis_changes = np.abs(np.diff(observed_axes, axis=0)) / np.maximum(
                observed_axes[:-1], 1e-12
            )
            metrics["center_step_normalized_median"] = float(np.median(normalized_steps))
            metrics["center_step_normalized_max"] = float(np.max(normalized_steps))
            metrics["major_axis_change_median"] = float(np.median(axis_changes[:, 0]))
            metrics["major_axis_change_max"] = float(np.max(axis_changes[:, 0]))
            metrics["minor_axis_change_median"] = float(np.median(axis_changes[:, 1]))
            metrics["minor_axis_change_max"] = float(np.max(axis_changes[:, 1]))
    if center_prediction_errors:
        metrics["center_prediction_error_max"] = float(np.max(center_prediction_errors))
        metrics["center_prediction_error_median"] = float(np.median(center_prediction_errors))
        metrics["axis_prediction_error_max"] = float(np.max(axis_prediction_errors))
        metrics["axis_prediction_error_median"] = float(np.median(axis_prediction_errors))
    if accepted_confidences:
        metrics["slice_confidence_mean"] = float(np.mean(accepted_confidences))
        metrics["slice_confidence_min"] = float(np.min(accepted_confidences))
    if accepted_reference_errors:
        metrics["reference_center_error_mean"] = float(np.mean(accepted_reference_errors))
        metrics["reference_center_error_max"] = float(np.max(accepted_reference_errors))
    if accepted_reference_axis_scales:
        metrics["reference_axis_scale_mean"] = float(np.mean(accepted_reference_axis_scales))
        metrics["reference_axis_scale_max"] = float(np.max(accepted_reference_axis_scales))
    if accepted_signal_scores:
        metrics["selected_signal_score_mean"] = float(np.mean(accepted_signal_scores))
        metrics["selected_signal_score_min"] = float(np.min(accepted_signal_scores))
    return mesh_points, mesh_faces, metrics
