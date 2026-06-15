# Singularity-Solutions-Capstone-25-26

# Reconstruction of a Simulated Borehole from Unstructured Particle Data

This repository contains a prototype pipeline for reconstructing an inferred borehole from a noisy 3D VTK point cloud. The borehole is **not explicitly modeled** in the input data; instead, it is inferred from **low-density cavities** observed in 2D cross-sections and then lofted into a conservative, watertight 3D mesh.

The current production workflow lives in `visualization/loft_export_v1.py`. A simpler slice-level diagnostic tool lives in `visualization/probe_and_ransac.py`.

## Overview

Horizontal directional drilling (HDD) simulations can produce large, unstructured particle clouds without an explicit borehole surface. This project reconstructs the borehole by:

1. Slicing the 3D point cloud along the drill path
2. Projecting each slice into 2D
3. Detecting interior low-density cavities
4. Fitting an elliptical cross-section to the cavity boundary
5. Enforcing slice-to-slice continuity
6. Lofting accepted cross-sections into a watertight 3D surface

The result is a CAD-friendly mesh representation of the borehole geometry that is easier to inspect and analyze than raw particle data.

## Current Status

This repository reflects the **current production pipeline**, which includes:

- Border-touching component rejection
- PCA-based ellipse initialization
- Nonlinear ellipse refinement
- Metric-driven candidate selection
- Short-gap interpolation for internal failures
- Deterministic strip lofting
- Watertightness checks via boundary/nonmanifold edge counts
