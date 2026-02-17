import pyvista as pv

# TODO: replace with your actual file path
vtk_path = "/Users/akhilgorla/Downloads/prestress2800000.vtk"

mesh = pv.read(vtk_path)

print(mesh)
print("\nPoint data arrays:", list(mesh.point_data.keys()))
print("Cell data arrays:", list(mesh.cell_data.keys()))
print("Number of points:", mesh.n_points)
