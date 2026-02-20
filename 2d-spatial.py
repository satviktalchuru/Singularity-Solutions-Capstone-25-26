import pyvista as pv 
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.linalg import svd
import random
import matplotlib.pyplot as plt

def load_vtk_points(filename):
    mesh = pv.read(filename)
    print(mesh)
    print("Dataset type:", type(mesh))
    print("Number of points:", mesh.n_points)
    return mesh.points

points3D = load_vtk_points(r"C:\Users\prasa\Desktop\data-science\capstone\prestress2800000.vtk")

# simple PCA approach

def estimate_axis(points):
    if points is None or len(points) == 0:
        raise ValueError("Empty point set passed to estimate_axis")

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Compute covariance matrix (3x3 only!)
    cov = np.dot(centered.T, centered) / centered.shape[0]

    # Eigen decomposition (cheap: 3x3 matrix)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Largest eigenvector = principal axis
    axis = eigvecs[:, np.argmax(eigvals)]

    return centroid, axis

centroid, axis = estimate_axis(points3D)

#slice pt cloud into 2d
def slice_points(points, axis_point, axis_dir, s, thickness):
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    
    projections = np.dot(points - axis_point, axis_dir)
    mask = np.abs(projections - s) < thickness
    slice_pts = points[mask]
    
    # Build 2D basis
    z = axis_dir
    x = np.array([1,0,0])
    if abs(np.dot(x,z)) > 0.9:
        x = np.array([0,1,0])
    x = x - np.dot(x,z)*z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    
    coords2D = np.column_stack([
        np.dot(slice_pts - axis_point, x),
        np.dot(slice_pts - axis_point, y)
    ])
    
    return coords2D

def fit_ellipse(points):
    if points is None or len(points) < 6:
        raise ValueError("At least 6 points are required to fit an ellipse")

    x = points[:, 0]
    y = points[:, 1]

    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    _, _, V = svd(D)
    params = V[-1]
    return params

def ellipse_residual(params, points):
    x = points[:,0]
    y = points[:,1]
    return np.abs(
        params[0]*x*x + params[1]*x*y + params[2]*y*y +
        params[3]*x + params[4]*y + params[5]
    )

def spatial_ransac(points2D, iterations=200, eps=0.5, k=8):
    if points2D is None:
        return None, np.empty((0, 2))

    n = len(points2D)
    if n < 6:
        return None, np.empty((0, 2))

    # If k is larger than number of points, reduce it
    k = min(k, n)

    nbrs = NearestNeighbors(n_neighbors=k).fit(points2D)
    _, indices = nbrs.kneighbors(points2D)

    best_model = None
    best_inliers = []

    for _ in range(iterations):

        # Spatially constrained sampling
        seed = random.randint(0, n - 1)
        neighbors = indices[seed]
        if len(neighbors) < 6:
            continue

        sample_idx = np.random.choice(neighbors, 6, replace=False)
        sample = points2D[sample_idx]

        try:
            model = fit_ellipse(sample)
        except Exception:
            continue

        residuals = ellipse_residual(model, points2D)
        inliers = np.where(residuals < eps)[0]

        if len(inliers) < 20:
            continue

        inlier_set = set(inliers.tolist())

        # Build graph among inliers
        G = nx.Graph()
        for i in inliers:
            G.add_node(i)

        for i in inliers:
            for j in indices[i]:
                if int(j) in inlier_set:
                    G.add_edge(i, j)

        if G.number_of_nodes() == 0:
            continue

        components = list(nx.connected_components(G))
        if not components:
            continue

        largest = max(components, key=len)

        if len(largest) > len(best_inliers):
            best_inliers = list(largest)
            best_model = model

    if best_model is None or len(best_inliers) == 0:
        return None, np.empty((0, 2))

    return best_model, points2D[best_inliers]

def main():
    slice_thickness = 1.0
    s_value = 0.0   # position along axis

    slice_pts = slice_points(points3D, centroid, axis, s_value, slice_thickness)

    if slice_pts is None or len(slice_pts) == 0:
        print("No slice points found at the given position/thickness.")
        return

    model, clean_boundary = spatial_ransac(slice_pts)

    # visual output
    plt.scatter(slice_pts[:, 0], slice_pts[:, 1], s=5, label='Raw')

    if clean_boundary is not None and getattr(clean_boundary, 'size', 0) > 0:
        plt.scatter(clean_boundary[:, 0], clean_boundary[:, 1], s=5, label='Boundary')
    else:
        print('No boundary found by RANSAC; showing raw slice only.')

    plt.legend()
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()