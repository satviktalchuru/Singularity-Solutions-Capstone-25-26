import time
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    gaussian_filter,
    label,
    binary_opening,
    binary_closing,
    binary_erosion,
)
from scipy.ndimage import binary_fill_holes  # used for enclosed-cavity isolation


# ============================================================
# Config (kept minimal + generally transferable)
# ============================================================
vtk_path = "/Users/akhilgorla/Downloads/prestress2800000.vtk"

CHOSEN_AXIS = 0          # 0=x slice -> plot (y,z)
SLICE_FRAC = 0.50
THICKNESS_FRAC = 0.02

# Density field
GRID_N = 250
SUBSAMPLE = 25000
SMOOTH_SIGMA = 4.0

# Define "void-ish" as lowest q of density (quantile is dataset-normalized)
LOW_DENSITY_Q = 0.10

# Grid padding to avoid border artifacts (small + generic)
PAD_FRAC_A = 0.10
PAD_FRAC_B = 0.10

# Mild morphology (avoid fill_holes on low itself; we do topological cavity isolation separately)
DO_MORPH = True
OPEN_ITERS = 1
CLOSE_ITERS = 1

# Occupancy gating (prevents selecting outside empty space far from points)
USE_OCCUPANCY_GATE = True
OCC_DILATE_ITERS = 6  # tune lightly if needed

# Candidate filtering / scoring (generic, not dataset-specific)
MIN_COMP_AREA = 150
MAX_ASPECT = 8.0

# Output
SHOW_PLOT = True
SAVE_BOUNDARY = "boundary_general.png"
SAVE_DENSITY = "density_general.png"
SAVE_NPY = "boundary_pts_axis0.npy"

RNG_SEED = 0


# ============================================================
# Utilities
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

def extract_slice(points: np.ndarray, axis: int, frac=0.5, thickness_frac=0.02):
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

def density_gradient(D: np.ndarray):
    # D indexed [b, a]
    gy, gx = np.gradient(D)  # gy: along rows (b), gx: along cols (a)
    G = np.sqrt(gx * gx + gy * gy)
    return G

def contour_paths(AA, BB, D, level):
    """
    Use matplotlib contouring to extract contour line(s) at a given density level.
    Returns list of (N,2) arrays of xy coords.
    """
    cs = plt.contour(AA, BB, D, levels=[level])
    paths = []
    for coll in cs.collections:
        for p in coll.get_paths():
            v = p.vertices
            if v is not None and len(v) >= 10:
                paths.append(v.copy())
    plt.close()  # prevent display
    return paths

def polygon_area(poly: np.ndarray):
    # poly shape (N,2), closed or not; computes signed area magnitude
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def poly_bbox_aspect(poly: np.ndarray):
    x = poly[:, 0]
    y = poly[:, 1]
    w = (x.max() - x.min()) + 1e-12
    h = (y.max() - y.min()) + 1e-12
    return max(h / w, w / h)

def sample_grid_at_points(aa, bb, G, pts):
    """
    Sample grid image G (indexed [b,a]) at continuous pts (A,B) via nearest-neighbor.
    """
    # Map coordinates to indices
    a_min, a_max = aa[0], aa[-1]
    b_min, b_max = bb[0], bb[-1]
    # normalize to [0, GRID_N-1]
    ai = np.clip(((pts[:, 0] - a_min) / (a_max - a_min) * (len(aa) - 1)).round().astype(int), 0, len(aa) - 1)
    bi = np.clip(((pts[:, 1] - b_min) / (b_max - b_min) * (len(bb) - 1)).round().astype(int), 0, len(bb) - 1)
    return G[bi, ai]


# ============================================================
# Main
# ============================================================
t0 = timed("Reading VTK...")
mesh = pv.read(vtk_path)
done(t0, "VTK loaded")
points = mesh.points
print(f"Total points in mesh: {points.shape[0]}", flush=True)

t0 = timed("Extracting slice...")
slice_pts, center, thickness = extract_slice(points, CHOSEN_AXIS, frac=SLICE_FRAC, thickness_frac=THICKNESS_FRAC)
done(t0, "Slice extracted")
print(f"Slice points: {slice_pts.shape[0]}  center={center:.5f}  thickness={thickness:.5f}", flush=True)

A, B, dims = project_to_2d(slice_pts, CHOSEN_AXIS)
xlabel = ["x", "y", "z"][dims[0]]
ylabel = ["x", "y", "z"][dims[1]]
print("2D axes:", xlabel, "vs", ylabel, flush=True)

A_s, B_s = subsample_xy(A, B, SUBSAMPLE, seed=RNG_SEED)
print(f"Using subsample for density: {len(A_s)} points", flush=True)

# Full-slice bounds + padding
a_min, a_max = float(np.min(A)), float(np.max(A))
b_min, b_max = float(np.min(B)), float(np.max(B))
pad_a = PAD_FRAC_A * (a_max - a_min)
pad_b = PAD_FRAC_B * (b_max - b_min)
a_min_p, a_max_p = a_min - pad_a, a_max + pad_a
b_min_p, b_max_p = b_min - pad_b, b_max + pad_b
print(f"Padded bounds: {xlabel}[{a_min_p:.6g},{a_max_p:.6g}]  {ylabel}[{b_min_p:.6g},{b_max_p:.6g}]", flush=True)

# ------------------------------------------------------------
# Density field = histogram + gaussian blur (KDE-like)
# ------------------------------------------------------------
t0 = timed("Building 2D histogram...")
H, a_edges, b_edges = np.histogram2d(
    A_s, B_s,
    bins=GRID_N,
    range=[[a_min_p, a_max_p], [b_min_p, b_max_p]]
)
done(t0, "Histogram built")

t0 = timed("Smoothing (Gaussian blur)...")
D = gaussian_filter(H.T, sigma=SMOOTH_SIGMA)  # indexed [b,a]
done(t0, "Smoothing done")

# Bin centers
aa = 0.5 * (a_edges[:-1] + a_edges[1:])
bb = 0.5 * (b_edges[:-1] + b_edges[1:])
AA, BB = np.meshgrid(aa, bb)

# Optional occupancy gate (prevents far-outside selection)
if USE_OCCUPANCY_GATE:
    from scipy.ndimage import binary_dilation
    occ = (H.T > 0)
    occ = binary_dilation(occ, iterations=OCC_DILATE_ITERS)
else:
    occ = np.ones_like(D, dtype=bool)

# ------------------------------------------------------------
# Quantile threshold -> low-density candidate (dataset-normalized)
# ------------------------------------------------------------
thr = float(np.quantile(D[occ], LOW_DENSITY_Q))
print(f"Low-density threshold (quantile): q={LOW_DENSITY_Q} thr={thr:.6g}", flush=True)

low = (D <= thr) & occ

if DO_MORPH:
    t0 = timed("Morph cleanup (open/close)...")
    low = binary_opening(low, iterations=OPEN_ITERS)
    low = binary_closing(low, iterations=CLOSE_ITERS)
    done(t0, "Morph cleanup done")

# ------------------------------------------------------------
# Topological cavity isolation:
# Remove outside low-density region by keeping only enclosed low regions.
# holes = (low) ∩ fill_holes(~low)
# ------------------------------------------------------------
t0 = timed("Isolating enclosed cavities (remove outside background)...")
holes = binary_fill_holes(~low) & low
done(t0, "Cavity isolation done")

# Connected components on enclosed cavities
t0 = timed("Connected components on cavities...")
lbl, ncomp = label(holes)
done(t0, "Connected components computed")
print("Enclosed cavity components:", ncomp, flush=True)

if ncomp == 0:
    raise RuntimeError(
        "No enclosed cavities found.\n"
        "Try: increase PAD_FRAC_A/B to 0.15, increase LOW_DENSITY_Q to 0.12–0.18, "
        "or increase SMOOTH_SIGMA slightly."
    )

# ------------------------------------------------------------
# Score cavities: (gradient strength on boundary) + compactness + area sanity
# ------------------------------------------------------------
G = density_gradient(D)

best_id = None
best_score = -np.inf
best_info = None
best_boundary_pts = None

print("\nCavity candidate scores:", flush=True)

for cid in range(1, ncomp + 1):
    comp = (lbl == cid)
    comp_area = int(comp.sum())
    if comp_area < MIN_COMP_AREA:
        continue

    asp, bbox = bbox_aspect(comp)
    if asp > MAX_ASPECT:
        continue

    compact, A_area, P_perim = compactness_from_mask(comp)
    if compact <= 1e-8:
        continue

    # Use contour at thr to get a smoother boundary curve
    # We extract all contours at 'thr' and pick the one with centroid inside this component
    # For simplicity, we'll approximate by taking the component's erosion boundary pixels to score,
    # then later extract a contour for visualization.
    er = binary_erosion(comp)
    boundary_pix = comp & (~er)

    bi, ai = np.nonzero(boundary_pix)
    if len(ai) < 30:
        continue

    boundary_A = aa[ai]
    boundary_B = bb[bi]
    boundary_pts = np.column_stack([boundary_A, boundary_B])

    # Gradient strength along boundary (interface should be strong)
    grad_vals = sample_grid_at_points(aa, bb, G, boundary_pts)
    grad_mean = float(np.mean(grad_vals))
    grad_p75 = float(np.percentile(grad_vals, 75))

    # Penalize cavities that are extremely large relative to grid (likely not tunnel)
    frac = comp_area / float(GRID_N * GRID_N)
    if frac > 0.35:
        continue

    # Combined score (weights are generic; gradient dominates)
    # compactness helps reject frame-like shapes; gradient enforces "real interface"
    score = (2.0 * grad_p75 + 1.0 * grad_mean) + (5.0 * compact) - (2.0 * frac)

    print(
        f"  cid={cid:2d} score={score:8.3f} area={comp_area:5d} frac={frac:5.3f} "
        f"asp={asp:4.2f} compact={compact:5.3f} grad_mean={grad_mean:7.4f} grad_p75={grad_p75:7.4f}",
        flush=True
    )

    if score > best_score:
        best_score = score
        best_id = cid
        best_info = {
            "cid": cid,
            "area": comp_area,
            "frac": frac,
            "aspect": asp,
            "compact": compact,
            "grad_mean": grad_mean,
            "grad_p75": grad_p75,
            "bbox": bbox,
        }
        best_boundary_pts = boundary_pts

if best_id is None:
    raise RuntimeError(
        "No cavity passed the scoring filters.\n"
        "Try: increase LOW_DENSITY_Q, increase PAD_FRAC, reduce OCC_DILATE_ITERS, "
        "or loosen MAX_ASPECT."
    )

print("\nSelected cavity:", best_info, flush=True)

# Save boundary points (pixel-boundary; you can switch to contour-based later)
np.save(SAVE_NPY, best_boundary_pts)
print(f"Saved boundary points: {SAVE_NPY}", flush=True)

# ============================================================
# Plots
# ============================================================
plt.figure(figsize=(8, 6))
plt.imshow(
    D,
    origin="lower",
    extent=[a_min_p, a_max_p, b_min_p, b_max_p],
    aspect="auto",
)
plt.scatter(A_s, B_s, s=1, alpha=0.10, label="slice points (subsample)")
plt.scatter(best_boundary_pts[:, 0], best_boundary_pts[:, 1], s=6, alpha=0.95, label="selected cavity boundary (pixels)")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(
    f"Selected cavity boundary (axis={['x','y','z'][CHOSEN_AXIS]}), "
    f"q={LOW_DENSITY_Q}, sigma={SMOOTH_SIGMA}"
)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(SAVE_BOUNDARY, dpi=200)
print("Saved:", SAVE_BOUNDARY, flush=True)

plt.figure(figsize=(8, 6))
plt.imshow(
    D,
    origin="lower",
    extent=[a_min_p, a_max_p, b_min_p, b_max_p],
    aspect="auto",
)
plt.scatter(A_s, B_s, s=1, alpha=0.12)

# show the threshold contour
plt.contour(AA, BB, D, levels=[thr], linewidths=2)

# show enclosed cavities mask outline (optional visual)
# (Plot as semi-transparent overlay)
plt.imshow(
    holes.astype(float),
    origin="lower",
    extent=[a_min_p, a_max_p, b_min_p, b_max_p],
    alpha=0.18,
    aspect="auto",
)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(f"Density + threshold contour + enclosed cavities overlay (q={LOW_DENSITY_Q})")
plt.tight_layout()
plt.savefig(SAVE_DENSITY, dpi=200)
print("Saved:", SAVE_DENSITY, flush=True)

if SHOW_PLOT:
    plt.show()
else:
    plt.close("all")
