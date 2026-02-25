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
    binary_fill_holes,
)

# ============================================================
# Config
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

# Mild morphology
DO_MORPH = True
OPEN_ITERS = 1
CLOSE_ITERS = 1

# Occupancy gating (prevents selecting outside empty space far from points)
USE_OCCUPANCY_GATE = True
OCC_DILATE_ITERS = 6  # tune lightly if needed

# Candidate filtering / scoring (generic, not dataset-specific)
MIN_COMP_AREA = 150
MAX_ASPECT = 8.0

# Mirroring (x–z plane symmetry => flip y)
DO_MIRROR = True
MIRROR_AXIS = 1          # 0=x, 1=y, 2=z
MIRROR_MODE = "augment"  # "augment" keeps original + mirrored

# Output
SHOW_PLOT = True
RUN_TAG = "mirrored" if DO_MIRROR else "original"
SAVE_BOUNDARY = f"boundary_{RUN_TAG}.png"
SAVE_DENSITY  = f"density_{RUN_TAG}.png"
SAVE_NPY      = f"boundary_pts_{RUN_TAG}_axis{CHOSEN_AXIS}.npy"
SAVE_ELLIPSE  = f"ellipse_fit_{RUN_TAG}_axis{CHOSEN_AXIS}.png"

RNG_SEED = 0


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
    gy, gx = np.gradient(D)  # D indexed [b,a]
    return np.sqrt(gx * gx + gy * gy)

def sample_grid_at_points(aa, bb, G, pts):
    """
    Sample grid image G (indexed [b,a]) at continuous pts (A,B) via nearest-neighbor.
    """
    a_min, a_max = aa[0], aa[-1]
    b_min, b_max = bb[0], bb[-1]
    ai = np.clip(((pts[:, 0] - a_min) / (a_max - a_min) * (len(aa) - 1)).round().astype(int), 0, len(aa) - 1)
    bi = np.clip(((pts[:, 1] - b_min) / (b_max - b_min) * (len(bb) - 1)).round().astype(int), 0, len(bb) - 1)
    return G[bi, ai]

def mirror_points(points: np.ndarray, axis: int = 1, mode: str = "augment") -> np.ndarray:
    """
    Mirror points about the plane where the chosen coordinate is 0.
    For x–z plane symmetry: axis=1 (y), so y -> -y.

    mode:
      - "augment": returns [points; mirrored(points)]
      - "replace": returns mirrored(points)
    """
    mirrored = points.copy()
    mirrored[:, axis] *= -1.0
    if mode == "augment":
        return np.vstack([points, mirrored])
    elif mode == "replace":
        return mirrored
    else:
        raise ValueError("mode must be 'augment' or 'replace'")

# -----------------------------
# V2: candidate raw points near boundary + analytic ellipse fit + metrics
# -----------------------------
def boundary_band_raw_points(A: np.ndarray, B: np.ndarray, boundary_pts: np.ndarray, band: float):
    """
    Select raw slice points within distance 'band' of the extracted boundary point-set.
    Uses chunked nearest-boundary-point distances.
    """
    P = np.column_stack([A, B])     # (N,2)
    Q = boundary_pts                # (M,2)

    N = P.shape[0]
    chunk = 5000
    keep = np.zeros(N, dtype=bool)
    band2 = band * band

    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        d2 = ((P[i0:i1, None, :] - Q[None, :, :]) ** 2).sum(axis=2)  # (chunk, M)
        keep[i0:i1] = (d2.min(axis=1) <= band2)

    return P[keep]

def fit_ellipse_pca(points2d: np.ndarray):
    """
    Analytic-ish ellipse approximation:
      - center = mean
      - orientation from PCA eigenvectors
      - radii from robust percentiles in PCA frame
    Returns center (2,), axes (a,b), R (2x2) where columns are principal axes.
    """
    C = points2d.mean(axis=0)
    X = points2d - C

    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]  # principal axes as columns

    Y = X @ vecs
    a = np.percentile(np.abs(Y[:, 0]), 90)
    b = np.percentile(np.abs(Y[:, 1]), 90)

    # Ensure a >= b
    if b > a:
        a, b = b, a
        vecs = vecs[:, ::-1]

    return C, (a, b), vecs

def ellipse_residuals(points2d: np.ndarray, center: np.ndarray, axes: tuple, R: np.ndarray):
    """
    Residual based on normalized radius:
      u = R^T (p-center)
      rho = sqrt((u_x/a)^2 + (u_y/b)^2)
      residual ≈ |rho - 1| * mean_axis
    """
    a, b = axes
    X = points2d - center
    U = X @ R
    rho = np.sqrt((U[:, 0] / a) ** 2 + (U[:, 1] / b) ** 2)
    mean_axis = 0.5 * (a + b)
    return np.abs(rho - 1.0) * mean_axis

def ellipse_points(center, axes, R, n=300):
    a, b = axes
    t = np.linspace(0, 2*np.pi, n)
    circ = np.column_stack([a*np.cos(t), b*np.sin(t)])  # ellipse frame
    return circ @ R.T + center


# ============================================================
# Main
# ============================================================
t0 = timed("Reading VTK...")
mesh = pv.read(vtk_path)
done(t0, "VTK loaded")
points = mesh.points
print(f"Total points in mesh: {points.shape[0]}", flush=True)
print("y range BEFORE mirror:", points[:, 1].min(), points[:, 1].max(), flush=True)

if DO_MIRROR:
    t_m = timed(f"Mirroring points about axis={MIRROR_AXIS} ({['x','y','z'][MIRROR_AXIS]}) mode={MIRROR_MODE}...")
    points = mirror_points(points, axis=MIRROR_AXIS, mode=MIRROR_MODE)
    done(t_m, "Mirroring done")
    print(f"Total points after mirroring ({MIRROR_MODE}): {points.shape[0]}", flush=True)

print("y range AFTER mirror:", points[:, 1].min(), points[:, 1].max(), flush=True)

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

# Bounds + padding
a_min, a_max = float(np.min(A)), float(np.max(A))
b_min, b_max = float(np.min(B)), float(np.max(B))
pad_a = PAD_FRAC_A * (a_max - a_min)
pad_b = PAD_FRAC_B * (b_max - b_min)
a_min_p, a_max_p = a_min - pad_a, a_max + pad_a
b_min_p, b_max_p = b_min - pad_b, b_max + pad_b
print(f"Padded bounds: {xlabel}[{a_min_p:.6g},{a_max_p:.6g}]  {ylabel}[{b_min_p:.6g},{b_max_p:.6g}]", flush=True)

# Density field: histogram + blur
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

# Occupancy gate
if USE_OCCUPANCY_GATE:
    from scipy.ndimage import binary_dilation
    occ = (H.T > 0)
    occ = binary_dilation(occ, iterations=OCC_DILATE_ITERS)
else:
    occ = np.ones_like(D, dtype=bool)

# Quantile threshold
thr = float(np.quantile(D[occ], LOW_DENSITY_Q))
print(f"Low-density threshold (quantile): q={LOW_DENSITY_Q} thr={thr:.6g}", flush=True)

low = (D <= thr) & occ

if DO_MORPH:
    t0 = timed("Morph cleanup (open/close)...")
    low = binary_opening(low, iterations=OPEN_ITERS)
    low = binary_closing(low, iterations=CLOSE_ITERS)
    done(t0, "Morph cleanup done")

# Enclosed cavities
t0 = timed("Isolating enclosed cavities (remove outside background)...")
holes = binary_fill_holes(~low) & low
done(t0, "Cavity isolation done")

# Components
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

# Score cavities
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

    er = binary_erosion(comp)
    boundary_pix = comp & (~er)

    bi, ai = np.nonzero(boundary_pix)
    if len(ai) < 30:
        continue

    boundary_A = aa[ai]
    boundary_B = bb[bi]
    boundary_pts = np.column_stack([boundary_A, boundary_B])

    grad_vals = sample_grid_at_points(aa, bb, G, boundary_pts)
    grad_mean = float(np.mean(grad_vals))
    grad_p75 = float(np.percentile(grad_vals, 75))

    frac = comp_area / float(GRID_N * GRID_N)
    if frac > 0.35:
        continue

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

# Save boundary points (mask boundary)
np.save(SAVE_NPY, best_boundary_pts)
print(f"Saved boundary points: {SAVE_NPY}", flush=True)

# ============================================================
# V2: Candidate raw points near boundary + ellipse fit + metrics
# ============================================================
cell_a = (a_max_p - a_min_p) / GRID_N
cell_b = (b_max_p - b_min_p) / GRID_N
BAND = 2.0 * max(cell_a, cell_b)
print(f"Using boundary band: {BAND:.6g}", flush=True)

cand = boundary_band_raw_points(A, B, best_boundary_pts, band=BAND)
print("Candidate raw boundary-near points:", cand.shape[0], flush=True)

if cand.shape[0] < 50:
    raise RuntimeError("Too few candidate points near boundary for ellipse fit. Increase BAND or adjust thresholds.")

C_hat, axes_hat, R_hat = fit_ellipse_pca(cand)
a_hat, b_hat = axes_hat
AR_hat = a_hat / b_hat
print(f"Ellipse fit: center=({C_hat[0]:.6g},{C_hat[1]:.6g}) a={a_hat:.6g} b={b_hat:.6g} aspect={AR_hat:.4f}", flush=True)

res = ellipse_residuals(cand, C_hat, axes_hat, R_hat)
print(
    f"Fit residuals (raw): median={np.median(res):.6g}  p90={np.percentile(res,90):.6g}  max={res.max():.6g}",
    flush=True
)

ell = ellipse_points(C_hat, axes_hat, R_hat, n=300)

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
plt.contour(AA, BB, D, levels=[thr], linewidths=2)
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

# Ellipse overlay plot (analytic model + raw candidates)
plt.figure(figsize=(8, 6))
plt.imshow(
    D,
    origin="lower",
    extent=[a_min_p, a_max_p, b_min_p, b_max_p],
    aspect="auto",
)
plt.scatter(A_s, B_s, s=1, alpha=0.08, label="slice points (subsample)")
plt.scatter(cand[:, 0], cand[:, 1], s=2, alpha=0.35, label="candidate raw pts (near boundary)")
plt.plot(ell[:, 0], ell[:, 1], linewidth=2, label=f"ellipse fit (AR={AR_hat:.3f})")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(f"Analytic ellipse fit on raw points (axis={['x','y','z'][CHOSEN_AXIS]})")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(SAVE_ELLIPSE, dpi=200)
print("Saved:", SAVE_ELLIPSE, flush=True)

if SHOW_PLOT:
    plt.show()
else:
    plt.close("all")