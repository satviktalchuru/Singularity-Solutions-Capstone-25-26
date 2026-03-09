import os
import time
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from scipy.ndimage import (
    gaussian_filter,
    label,
    binary_opening,
    binary_closing,
    binary_erosion,
    binary_dilation,
)

# ============================================================
# Config
# ============================================================
vtk_path = "/Users/akhilgorla/Downloads/prestress2800000.vtk"

# Slice axis: 0=x slice => work in (y,z)
CHOSEN_AXIS = 0

# Slices to process (fractions along chosen axis)
FRAC_START = 0.20
FRAC_END = 0.80
N_SLICES = 35
THICKNESS_FRAC = 0.02

# Density field
GRID_N = 250
SUBSAMPLE = 25000
SMOOTH_SIGMA = 4.0
LOW_DENSITY_Q = 0.10

# Padding
PAD_FRAC_A = 0.10
PAD_FRAC_B = 0.10

# Morphology
DO_MORPH = True
OPEN_ITERS = 1
CLOSE_ITERS = 1

# Occupancy gating (keeps us near points)
USE_OCCUPANCY_GATE = True
OCC_DILATE_ITERS = 6

# Candidate filtering/scoring
MIN_COMP_AREA = 150
MAX_COMP_FRAC = 0.20  # reject huge “cavities” (often outside/border artifacts)

# Mirroring
DO_MIRROR = True
MIRROR_AXIS = 1
MIRROR_MODE = "augment"  # keep original + mirrored

# Ellipse fit (your v2 PCA ellipse)
ELLIPSE_Q = 90  # percentile for PCA-frame radii

# Continuity guards (prevents one slice from blowing up the loft)
ENABLE_CONTINUITY = True
MAX_AR = 4.0                # if AR is ridiculous, skip slice
MAX_AXIS_JUMP = 2.0         # allow a,b to jump at most this factor vs last kept
MAX_CENTER_JUMP = None      # set to a value (in coords) if you want

# Loft controls
N_RING = 240                # target number of vertices per ring
CAP_ENDS = True             # for watertight STL
MAX_AXIS_GAP = None         # set (in axis units) to prevent bridging large gaps (None disables)

# Debug output
OUT_DIR = "loft_debug_v4"
SAVE_DEBUG = True
SAVE_EVERY = 1              # save debug image for every slice
SHOW_PLOT_LAST = True       # show last plot interactively
RNG_SEED = 0

# Exports
RUN_TAG = "mirrored" if DO_MIRROR else "original"
OUT_VTK = f"tunnel_loft_{RUN_TAG}_axis{CHOSEN_AXIS}_v4.vtk"
OUT_STL = f"tunnel_loft_{RUN_TAG}_axis{CHOSEN_AXIS}_v4.stl"

# ============================================================
# Helpers
# ============================================================
def timed(msg: str):
    print(msg, flush=True)
    return time.perf_counter()

def done(t0: float, msg: str):
    print(f"{msg}: {time.perf_counter() - t0:.3f}s", flush=True)

def get_bounds(points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins, maxs, maxs - mins

def mirror_points(points: np.ndarray, axis: int = 1, mode: str = "augment") -> np.ndarray:
    mirrored = points.copy()
    mirrored[:, axis] *= -1.0
    if mode == "augment":
        return np.vstack([points, mirrored])
    elif mode == "replace":
        return mirrored
    else:
        raise ValueError("mode must be 'augment' or 'replace'")

def extract_slice(points: np.ndarray, axis: int, frac: float, thickness_frac: float):
    mins, maxs, ranges = get_bounds(points)
    coord = points[:, axis]
    center = mins[axis] + frac * ranges[axis]
    thickness = thickness_frac * ranges[axis]
    mask = (coord >= center - thickness / 2) & (coord <= center + thickness / 2)
    return points[mask], center, thickness

def project_to_2d(slice_points: np.ndarray, axis: int):
    dims = [0, 1, 2]
    dims.remove(axis)
    a, b = dims
    return slice_points[:, a], slice_points[:, b], (a, b)

def subsample_xy(A: np.ndarray, B: np.ndarray, max_n: int, seed: int = 0):
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
    A = int(mask.sum())
    er = binary_erosion(mask)
    boundary = mask & (~er)
    P = int(boundary.sum())
    if A <= 0 or P <= 0:
        return 0.0, A, P
    compact = (4.0 * np.pi * A) / (P * P)
    return float(compact), A, P

def fit_ellipse_pca(points2d: np.ndarray, q: float = 90.0):
    """
    v2 ellipse approximation:
      center = mean
      orientation = PCA eigenvectors
      radii = percentile of |coords| in PCA frame
    """
    C = points2d.mean(axis=0)
    X = points2d - C

    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]  # columns are principal axes

    Y = X @ vecs
    a = np.percentile(np.abs(Y[:, 0]), q)
    b = np.percentile(np.abs(Y[:, 1]), q)

    if b > a:
        a, b = b, a
        vecs = vecs[:, ::-1]

    return C, (a, b), vecs

def ellipse_points(center, axes, R, n=240):
    a, b = axes
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    circ = np.column_stack([a*np.cos(t), b*np.sin(t)])
    return circ @ R.T + center

def boundary_band_raw_points(A: np.ndarray, B: np.ndarray, boundary_pts: np.ndarray, band: float):
    """
    Select raw slice points within distance 'band' of boundary_pts.
    """
    P = np.column_stack([A, B])     # (N,2)
    Q = boundary_pts                # (M,2)

    N = P.shape[0]
    chunk = 4000
    keep = np.zeros(N, dtype=bool)
    band2 = band * band

    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        d2 = ((P[i0:i1, None, :] - Q[None, :, :]) ** 2).sum(axis=2)
        keep[i0:i1] = (d2.min(axis=1) <= band2)

    return P[keep]

def remove_border_touching_components(low_mask: np.ndarray):
    """
    KEY FIX:
    - label low-density components
    - remove any component that touches the border
    returns:
      low_interior (bool mask),
      lbl_low (labels),
      keep_ids (list)
    """
    lbl, n = label(low_mask)
    if n == 0:
        return low_mask * False, lbl, []

    H, W = low_mask.shape
    border = np.zeros_like(low_mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    border_ids = np.unique(lbl[border & (lbl > 0)])
    keep_ids = [i for i in range(1, n + 1) if i not in set(border_ids)]

    low_interior = np.isin(lbl, keep_ids)
    return low_interior, lbl, keep_ids

# -----------------------------
# Loft helpers (deterministic triangle strip + optional caps)
# -----------------------------
def resample_closed_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a closed loop (Nx3) to exactly n points uniformly by arc length.
    Returns n points (no duplicated endpoint).
    """
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 3:
        raise ValueError("Need >= 3 points for a loop.")

    P_closed = np.vstack([P, P[0]])
    seg = np.linalg.norm(np.diff(P_closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        raise ValueError("Degenerate loop (zero perimeter).")

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

def best_cyclic_shift(prev: np.ndarray, cur: np.ndarray) -> int:
    """
    Find shift k minimizing sum ||prev[i] - cur[(i+k)%n]||^2.
    O(n^2) but n~240 is fine.
    """
    n = prev.shape[0]
    best_k, best_val = 0, np.inf
    for k in range(n):
        D = prev - np.roll(cur, -k, axis=0)
        val = float(np.sum(D * D))
        if val < best_val:
            best_val = val
            best_k = k
    return best_k

def align_loop_to_prev(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """
    Align current loop to previous loop by choosing:
      - direction (normal or reversed)
      - cyclic shift (start index)
    that minimizes L2 distance.
    Assumes both are (n,3).
    """
    cur = np.asarray(cur, dtype=float)
    prev = np.asarray(prev, dtype=float)

    # same direction
    k1 = best_cyclic_shift(prev, cur)
    cur1 = np.roll(cur, -k1, axis=0)
    err1 = float(np.sum((prev - cur1) ** 2))

    # reversed direction
    cur_rev = cur[::-1].copy()
    k2 = best_cyclic_shift(prev, cur_rev)
    cur2 = np.roll(cur_rev, -k2, axis=0)
    err2 = float(np.sum((prev - cur2) ** 2))

    return cur1 if err1 <= err2 else cur2

def loft_between_loops(loop0: np.ndarray, loop1: np.ndarray) -> pv.PolyData:
    """
    Create a triangle surface connecting two closed loops with corresponding vertices.
    loop0, loop1: (n,3)
    """
    A = np.asarray(loop0, dtype=float)
    B = np.asarray(loop1, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"Loop shape mismatch: {A.shape} vs {B.shape}")

    n = A.shape[0]
    pts = np.vstack([A, B])  # (2n,3)

    faces = []
    for i in range(n):
        i0 = i
        i1 = (i + 1) % n
        a0, a1 = i0, i1
        b0, b1 = n + i0, n + i1

        faces.append([3, a0, a1, b1])
        faces.append([3, a0, b1, b0])

    faces = np.array(faces, dtype=np.int64).ravel()
    return pv.PolyData(pts, faces)

def cap_from_loop(loop: np.ndarray) -> pv.PolyData:
    """
    Triangulated planar cap from a loop (n,3).
    Assumes loop is planar (it is: constant CHOSEN_AXIS coordinate).
    """
    P = np.asarray(loop, dtype=float)
    n = P.shape[0]
    poly = pv.PolyData(P)
    poly.faces = np.hstack([[n], np.arange(n, dtype=np.int64)])
    cap = poly.triangulate().clean()
    return cap

def report_watertight(mesh: pv.PolyData):
    feat = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    print(f"[mesh] points={mesh.n_points} cells={mesh.n_cells} triangles={mesh.is_all_triangles}")
    print(f"[mesh] boundary/nonmanifold edge segments = {feat.n_cells}")

# ============================================================
# Main
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)

t0 = timed("Reading VTK...")
mesh = pv.read(vtk_path)
done(t0, "VTK loaded")

points = mesh.points
print(f"Total points: {points.shape[0]}", flush=True)

if DO_MIRROR:
    t_m = timed(f"Mirroring (axis={MIRROR_AXIS}, mode={MIRROR_MODE})...")
    points = mirror_points(points, axis=MIRROR_AXIS, mode=MIRROR_MODE)
    done(t_m, "Mirroring done")
    print(f"Total points after mirror: {points.shape[0]}", flush=True)

fracs = np.linspace(FRAC_START, FRAC_END, N_SLICES)

# Store accepted rings as raw Nx3 arrays + metadata
sections = []
kept = []
last_axes = None
last_center2d = None

for i, frac in enumerate(fracs, start=1):
    # --- slice ---
    slice_pts, axis_center, thickness = extract_slice(points, CHOSEN_AXIS, frac=frac, thickness_frac=THICKNESS_FRAC)
    if slice_pts.shape[0] < 500:
        print(f"[{i:02d}/{N_SLICES}] frac={frac:.3f} SKIP: too few slice points: {slice_pts.shape[0]}", flush=True)
        continue

    A, B, dims = project_to_2d(slice_pts, CHOSEN_AXIS)
    xlabel = ["x", "y", "z"][dims[0]]
    ylabel = ["x", "y", "z"][dims[1]]

    # subsample for density
    A_s, B_s = subsample_xy(A, B, SUBSAMPLE, seed=RNG_SEED)

    # bounds + padding
    a_min, a_max = float(np.min(A)), float(np.max(A))
    b_min, b_max = float(np.min(B)), float(np.max(B))
    pad_a = PAD_FRAC_A * (a_max - a_min + 1e-12)
    pad_b = PAD_FRAC_B * (b_max - b_min + 1e-12)
    a_min_p, a_max_p = a_min - pad_a, a_max + pad_a
    b_min_p, b_max_p = b_min - pad_b, b_max + pad_b

    # density histogram
    H, a_edges, b_edges = np.histogram2d(
        A_s, B_s,
        bins=GRID_N,
        range=[[a_min_p, a_max_p], [b_min_p, b_max_p]]
    )
    D = gaussian_filter(H.T, sigma=SMOOTH_SIGMA)

    aa = 0.5 * (a_edges[:-1] + a_edges[1:])
    bb = 0.5 * (b_edges[:-1] + b_edges[1:])
    AA, BB = np.meshgrid(aa, bb)

    # occupancy gate
    if USE_OCCUPANCY_GATE:
        occ = (H.T > 0)
        occ = binary_dilation(occ, iterations=OCC_DILATE_ITERS)
    else:
        occ = np.ones_like(D, dtype=bool)

    # low-density mask
    thr = float(np.quantile(D[occ], LOW_DENSITY_Q))
    low = (D <= thr) & occ

    if DO_MORPH:
        low = binary_opening(low, iterations=OPEN_ITERS)
        low = binary_closing(low, iterations=CLOSE_ITERS)

    # --------------------------------------------------------
    # KEY FIX: remove border-touching low components
    # --------------------------------------------------------
    low_interior, lbl_low, keep_ids = remove_border_touching_components(low)

    if len(keep_ids) == 0:
        print(f"[{i:02d}/{N_SLICES}] frac={frac:.3f} SKIP: no interior low components", flush=True)

        if SAVE_DEBUG and (i % SAVE_EVERY == 0):
            fig = plt.figure(figsize=(7, 6))
            plt.imshow(D, origin="lower", extent=[a_min_p, a_max_p, b_min_p, b_max_p], aspect="auto")
            plt.scatter(A_s, B_s, s=1, alpha=0.06, label="slice pts (subsample)")
            plt.contour(AA, BB, D, levels=[thr], linewidths=2)
            plt.title(f"slice frac={frac:.3f}  (NO INTERIOR CAVITY)")
            plt.xlabel(xlabel); plt.ylabel(ylabel)
            plt.legend(loc="lower left")
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"debug_slice_{i:03d}_frac{frac:.3f}_nocavity.png"), dpi=180)
            plt.close(fig)
        continue

    # label interior candidates
    lbl_cav, ncomp = label(low_interior)

    # score cavities (compactness + reasonable size)
    best = None
    best_score = -np.inf
    best_boundary_pts = None

    for cid in range(1, ncomp + 1):
        comp = (lbl_cav == cid)
        area = int(comp.sum())
        if area < MIN_COMP_AREA:
            continue

        frac_area = area / float(GRID_N * GRID_N)
        if frac_area > MAX_COMP_FRAC:
            continue

        compact, _, _ = compactness_from_mask(comp)
        if compact <= 1e-8:
            continue

        asp, bbox = bbox_aspect(comp)

        er = binary_erosion(comp)
        boundary_pix = comp & (~er)
        bi, ai = np.nonzero(boundary_pix)
        if len(ai) < 30:
            continue

        boundary_pts = np.column_stack([aa[ai], bb[bi]])

        score = (6.0 * compact) - (2.0 * frac_area) - (0.15 * (asp - 1.0))

        if score > best_score:
            best_score = score
            best = (cid, area, frac_area, compact, asp, bbox)
            best_boundary_pts = boundary_pts

    if best is None:
        print(f"[{i:02d}/{N_SLICES}] frac={frac:.3f} SKIP: no component passed scoring filters", flush=True)
        continue

    # raw candidates near boundary -> ellipse PCA fit (v2)
    cell_a = (a_max_p - a_min_p) / GRID_N
    cell_b = (b_max_p - b_min_p) / GRID_N
    band = 2.0 * max(cell_a, cell_b)

    cand = boundary_band_raw_points(A, B, best_boundary_pts, band=band)
    if cand.shape[0] < 50:
        print(f"[{i:02d}/{N_SLICES}] frac={frac:.3f} SKIP: too few raw candidates near boundary: {cand.shape[0]}", flush=True)
        continue

    C_hat, axes_hat, R_hat = fit_ellipse_pca(cand, q=ELLIPSE_Q)
    a_hat, b_hat = axes_hat
    AR = float(a_hat / (b_hat + 1e-12))

    # continuity checks
    ok = True
    if AR > MAX_AR:
        ok = False

    if ENABLE_CONTINUITY and (last_axes is not None):
        a_prev, b_prev = last_axes
        if (a_hat > MAX_AXIS_JUMP * a_prev) or (b_hat > MAX_AXIS_JUMP * b_prev) or \
           (a_hat < a_prev / MAX_AXIS_JUMP) or (b_hat < b_prev / MAX_AXIS_JUMP):
            ok = False

        if (MAX_CENTER_JUMP is not None) and (last_center2d is not None):
            if np.linalg.norm(C_hat - last_center2d) > MAX_CENTER_JUMP:
                ok = False

    if not ok:
        print(f"[{i:02d}/{N_SLICES}] frac={frac:.3f} REJECT (continuity/AR) a={a_hat:.6g} b={b_hat:.6g} AR={AR:.3f}", flush=True)

        if SAVE_DEBUG and (i % SAVE_EVERY == 0):
            ell2 = ellipse_points(C_hat, axes_hat, R_hat, n=N_RING)
            fig = plt.figure(figsize=(7, 6))
            plt.imshow(D, origin="lower", extent=[a_min_p, a_max_p, b_min_p, b_max_p], aspect="auto")
            plt.scatter(A_s, B_s, s=1, alpha=0.05, label="slice pts (subsample)")
            plt.scatter(cand[:, 0], cand[:, 1], s=2, alpha=0.25, label="candidates (raw near boundary)")
            plt.plot(ell2[:, 0], ell2[:, 1], linewidth=2, label=f"ellipse (REJECT) AR={AR:.3f}")
            plt.title(f"slice frac={frac:.3f}  AR={AR:.3f}  (REJECTED)")
            plt.xlabel(xlabel); plt.ylabel(ylabel)
            plt.legend(loc="lower left")
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"debug_slice_{i:03d}_frac{frac:.3f}_REJECT.png"), dpi=180)
            plt.close(fig)
        continue

    # accepted slice
    last_axes = (a_hat, b_hat)
    last_center2d = C_hat.copy()

    print(f"[{i:02d}/{N_SLICES}] frac={frac:.3f} center={axis_center:.6g} a={a_hat:.6g} b={b_hat:.6g} AR={AR:.3f}", flush=True)

    ell2 = ellipse_points(C_hat, axes_hat, R_hat, n=N_RING)

    # map 2D ellipse back to 3D
    pts3 = np.zeros((ell2.shape[0], 3), dtype=float)
    pts3[:, CHOSEN_AXIS] = axis_center
    pts3[:, dims[0]] = ell2[:, 0]
    pts3[:, dims[1]] = ell2[:, 1]

    sections.append(pts3)
    kept.append((frac, axis_center, C_hat, axes_hat, R_hat))

    # debug image for accepted slice
    if SAVE_DEBUG and (i % SAVE_EVERY == 0):
        fig = plt.figure(figsize=(7, 6))
        plt.imshow(D, origin="lower", extent=[a_min_p, a_max_p, b_min_p, b_max_p], aspect="auto")
        plt.scatter(A_s, B_s, s=1, alpha=0.05, label="slice pts (subsample)")
        plt.contour(AA, BB, D, levels=[thr], linewidths=2)

        plt.imshow(
            low_interior.astype(float),
            origin="lower",
            extent=[a_min_p, a_max_p, b_min_p, b_max_p],
            alpha=0.12,
            aspect="auto",
        )

        plt.scatter(best_boundary_pts[:, 0], best_boundary_pts[:, 1], s=6, alpha=0.8, label="chosen boundary (pixels)")
        plt.scatter(cand[:, 0], cand[:, 1], s=2, alpha=0.25, label="candidates (raw near boundary)")
        plt.plot(ell2[:, 0], ell2[:, 1], linewidth=2, label=f"ellipse AR={AR:.3f}")

        plt.title(f"slice frac={frac:.3f}  AR={AR:.3f}")
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.legend(loc="lower left")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"debug_slice_{i:03d}_frac{frac:.3f}_OK.png"), dpi=180)
        if SHOW_PLOT_LAST and (i == len(fracs)):
            plt.show()
        plt.close(fig)

# ============================================================
# Loft + export (robust, no PolyData.loft)
# ============================================================
if len(sections) < 3:
    raise RuntimeError(f"Not enough accepted slices to loft. Accepted={len(sections)}")

print(f"\nBuilding deterministic loft from {len(sections)} accepted slices...", flush=True)

axis_centers = [k[1] for k in kept]

# Resample (safety) and align
loops = [resample_closed_polyline(L, N_RING) for L in sections]
aligned = [loops[0]]
for i in range(1, len(loops)):
    aligned.append(align_loop_to_prev(loops[i], aligned[-1]))

# Build triangle strips (optionally skip huge gaps)
parts = []
for i in range(len(aligned) - 1):
    if (MAX_AXIS_GAP is not None) and (abs(axis_centers[i + 1] - axis_centers[i]) > MAX_AXIS_GAP):
        print(f"Skipping loft segment {i}->{i+1} due to large axis gap", flush=True)
        continue
    parts.append(loft_between_loops(aligned[i], aligned[i + 1]))

if len(parts) == 0:
    raise RuntimeError("No loft segments were created (all skipped or failed).")

loft_surface = pv.append_polydata(parts).clean().triangulate().clean()

# Cap ends for watertight STL
if CAP_ENDS:
    cap0 = cap_from_loop(aligned[0])
    cap1 = cap_from_loop(aligned[-1])
    loft_surface = pv.append_polydata([loft_surface, cap0, cap1]).clean().triangulate().clean()

report_watertight(loft_surface)

# Save
loft_surface.save(OUT_VTK)
print("Saved:", OUT_VTK, flush=True)

try:
    loft_surface.save(OUT_STL)
    print("Saved:", OUT_STL, flush=True)
except Exception as e:
    print("STL save failed:", e, flush=True)

# View
plotter = pv.Plotter()
plotter.add_mesh(loft_surface, color="lightblue", opacity=1.0)
plotter.add_axes()
plotter.show()