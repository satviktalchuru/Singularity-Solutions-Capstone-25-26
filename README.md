# Singularity-Solutions-Capstone-25-26

This repository contains code and experiments for reconstructing a borehole from noisy horizontal directional drilling (HDD) simulation data. The goal is to extract the drilled tunnel from a 3D soil point cloud, fit its geometry using statistical methods, and export a clean surface model in STL format for visualization and analysis.
Inputs:
VTK file: Soil particle point cloud
STL file: Drill tool geometry
Output:
STL file: Reconstructed borehole surface
This project compares multiple RANSAC-based modeling approaches implemented by different team members.

This repository contains code and experiments for reconstructing a borehole from noisy horizontal directional drilling (HDD) simulation data. The goal is to extract the drilled tunnel from a 3D soil point cloud, fit its geometry using statistical methods, and export a clean surface model in STL format for visualization and analysis.
  Inputs:
    VTK file: Soil particle point cloud
    STL file: Drill tool geometry
  Output:
    STL file: Reconstructed borehole surface
    
This project compares multiple RANSAC-based modeling approaches implemented by different team members.

Problem Statement
Horizontal directional drilling simulations produce large, noisy point clouds. While the drill path exists in this data, it is not directly usable for engineering analysis.
This project addresses:
- How to locate the borehole inside noisy soil data
- How to estimate its centerline and shape
- How to reconstruct a continuous surface
- How to export the result in a usable format



