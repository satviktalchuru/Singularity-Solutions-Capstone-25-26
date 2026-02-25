import numpy as np
import pyvista as pv
import trimesh

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from skimage.measure import EllipseModel, ransac

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

# -----------------------------
# CONFIG
# -----------------------------

VTK_PATH = r"C:\Users\prasa\Desktop\data-science\horizontal-drilling-analysis\prestress2800000.vtk"
STL_PATH = r"C:\Users\prasa\Desktop\data-science\horizontal-drilling-analysis\drillhead2800000.stl"
OUT_STL  = "borehole_spatial_smooth.stl"


STEP_ALONG = 0.1
MIN_SLICE_PTS = 200
MAX_POINTS = 200_000
RANSAC_RESIDUAL = 2.0
NN_SMOOTH_K = 12   # spatial averaging strength

# FILTER

soil = pv.read("prestress_binary.vtk")
points = np.asarray(soil.points)

tool = trimesh.load_mesh(STL_PATH)
tool_pts = np.asarray(tool.vertices)

print("Total soil points:", points.shape)

# PCA

pca_tool = PCA(n_components=3)
pca_tool.fit(tool_pts)

bore_axis = pca_tool.components_[0]  # dominant direction
bore_axis /= np.linalg.norm(bore_axis)

bore_center = tool_pts.mean(axis=0)

tool_vecs = tool_pts - bore_center
tool_axial = tool_vecs @ bore_axis
tool_proj = bore_center + np.outer(tool_axial, bore_axis)
tool_radial = tool_pts - tool_proj

bore_radius = np.mean(np.linalg.norm(tool_radial, axis=1))

print("Estimated bore radius:", bore_radius)

vecs = points - bore_center
axial_proj = vecs @ bore_axis
proj_pts = bore_center + np.outer(axial_proj, bore_axis)

radial_vec = points - proj_pts
radial_dist = np.linalg.norm(radial_vec, axis=1)

band = 2.0   # thickness tolerance (adjust 1.0–3.0 if needed)

mask = (
    (radial_dist > bore_radius - band) &
    (radial_dist < bore_radius + band)
)

hole_pts = points[mask]

# subsample for performance
if hole_pts.shape[0] > MAX_POINTS:
    idx = np.random.choice(hole_pts.shape[0], MAX_POINTS, replace=False)
    hole_pts = hole_pts[idx]

print("Filtered hole points:", hole_pts.shape)

main_dir = bore_axis
center0 = bore_center

proj = (hole_pts - center0) @ main_dir
s_min, s_max = proj.min(), proj.max()

# Sort once
order = np.argsort(proj)
proj_sorted = proj[order]
hole_sorted = hole_pts[order]

stations = np.arange(s_min, s_max, STEP_ALONG)


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_plane_basis(n):
    n = n / np.linalg.norm(n)
    x = np.array([1, 0, 0])
    if abs(np.dot(x, n)) > 0.9:
        x = np.array([0, 1, 0])
    e1 = x - np.dot(x, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def spatial_average(points2d, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points2d)
    _, idx = nbrs.kneighbors(points2d)
    smoothed = np.array([points2d[i].mean(axis=0) for i in idx])
    return smoothed


# -----------------------------
# SLICE + SPATIAL RANSAC
# -----------------------------

centers = []
radii = []
rings = []

e1, e2 = get_plane_basis(main_dir)

for s in stations:
    left = s - STEP_ALONG/2
    right = s + STEP_ALONG/2

    i1 = np.searchsorted(proj_sorted, left)
    i2 = np.searchsorted(proj_sorted, right)

    chunk = hole_sorted[i1:i2]

    if chunk.shape[0] > 2000:
        idx = np.random.choice(chunk.shape[0], 2000, replace=False)
        chunk = chunk[idx]

    if chunk.shape[0] < MIN_SLICE_PTS:
        continue

    # Project to slice plane
    slice_center = center0 + s * main_dir

    coords2d = np.column_stack([
        (chunk - slice_center) @ e1,
        (chunk - slice_center) @ e2
    ])

    # RANSAC ellipse
    model_robust, inliers = ransac(
        coords2d,
        EllipseModel,
        min_samples=6,
        residual_threshold=RANSAC_RESIDUAL,
        max_trials=100
    )

    model_robust, inliers = ransac(
    coords2d,
    EllipseModel,
    min_samples=6,
    residual_threshold=2.5,
    max_trials=120
)

    if model_robust is None:
        print("FAIL: no model at station", s)
        continue

    if inliers is None:
        print("FAIL: no inliers at station", s)
        continue
    
    inlier_pts = coords2d[inliers]

    # ---- Spatial idea integrated here ----
    inlier_pts = spatial_average(inlier_pts, k=NN_SMOOTH_K)

    model = EllipseModel()
    model.estimate(inlier_pts)

    xc, yc, a, b, theta = model.params

    center3d = center0 + s * main_dir + xc * e1 + yc * e2

    centers.append(center3d)
    radii.append((a + b) / 2)

    # Build clean ring
    t = np.linspace(0, 2*np.pi, 100)
    ring2d = np.column_stack([
        xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta),
        yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    ])

    ring3d = center0 + s * main_dir + ring2d[:,0][:,None]*e1 + ring2d[:,1][:,None]*e2
    rings.append(ring3d)

    
    print("Station:", s, "Slice size:", chunk.shape[0])

print("s_min:", s_min)
print("s_max:", s_max)
print("Range length:", s_max - s_min)
print("Stations count:", len(stations))

# -----------------------------
# SMOOTH CENTERLINE + RADII
# -----------------------------

centers = np.array(centers)
radii = np.array(radii)

centers_smooth = gaussian_filter1d(centers, sigma=2, axis=0)
if len(radii) >= 5:
    win = min(11, len(radii))
    if win % 2 == 0:
        win -= 1  # must be odd
    radii_smooth = savgol_filter(radii, win, 2)
else:
    radii_smooth = radii
print("Number of rings:", len(rings))
# -----------------------------
# LOFT TO MESH
# -----------------------------

vertices = []
faces = []

for i in range(len(rings) - 1):
    r1 = rings[i]
    r2 = rings[i+1]
    n = len(r1)

    base = len(vertices)
    vertices.extend(r1)
    vertices.extend(r2)

    for j in range(n):
        a = base + j
        b = base + (j+1) % n
        c = base + n + j
        d = base + n + (j+1) % n

        faces.append([3, a, b, c])
        faces.append([3, b, d, c])

vertices = np.array(vertices)
faces = np.array(faces, dtype=np.int64).flatten()

mesh = pv.PolyData(vertices, faces)
mesh.save(OUT_STL)

print("Saved:", OUT_STL)