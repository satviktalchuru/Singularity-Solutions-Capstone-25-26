import pyvista as pv

mesh = pv.read(r"C:\Users\prasa\Desktop\data-science\horizontal-drilling-analysis\prestress2800000.vtk")
mesh.save("prestress_binary.vtk", binary=True)

print("Saved binary version.")