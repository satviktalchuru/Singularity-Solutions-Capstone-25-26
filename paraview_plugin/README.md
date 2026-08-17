# ParaView Borehole Reconstruction Plugin

This plugin wraps the project's slice-based borehole reconstruction pipeline as a ParaView Python filter.

## Files
- `/Users/akhilgorla/Singularity Solutions/paraview_plugin/borehole_reconstruction_plugin.py`
- `/Users/akhilgorla/Singularity Solutions/paraview_plugin/borehole_core.py`

## What it does
- takes a `vtkPointSet` input
- reconstructs a borehole surface from the point cloud
- outputs a `vtkPolyData` borehole mesh
- stores summary metrics in output field data

Current field-data metrics include:
- `accepted_slice_count`
- `observed_slice_count`
- `interpolated_slice_count`
- `boundary_nonmanifold_edge_segments`
- `resid_med_mean`
- `resid_p90_mean`
- `inlier_mean`
- `center_jump_median`
- `major_pct_abs_median`
- `minor_pct_abs_median`
- `theta_median`
- `iou_median`

Per-slice arrays for observed slices are also attached as field data:
- `slice_axis_center`
- `slice_resid_med`
- `slice_resid_p90`
- `slice_inlier`

## Load in ParaView
1. Open ParaView.
2. Go to `Tools -> Manage Plugins`.
3. Click `Load New`.
4. Select:
   - `/Users/akhilgorla/Singularity Solutions/paraview_plugin/borehole_reconstruction_plugin.py`

After loading, the filter appears as:
- `Borehole Reconstruction`

## Important environment requirement
This is a Python plugin and depends on:
- `numpy`
- `scipy`

Those packages must be available in ParaView's Python environment. If ParaView cannot import `scipy`, the plugin will fail to load or execute.

## Recommended first settings
- `SliceAxis = 0`
- `FractionStart = 0.01`
- `FractionEnd = 0.69`
- `NumberOfSlices = 35`
- `ThicknessFraction = 0.02`
- `MirrorPoints = 1`
- `MirrorAxis = 1`
- `LowDensityQuantile = 0.10`
- `SmoothSigma = 4.0`
- `EnableShortGapFill = 1`
- `MaximumInterpolatedGap = 2`

## Notes
- This plugin is the first ParaView integration pass. It is designed to preserve the current project logic rather than simplify it into a toy filter.
- It does not yet expose every internal parameter from the standalone script.
- It intentionally does not include the removed auto-interval selection logic.
