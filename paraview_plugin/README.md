# ParaView 5.10 Borehole Reconstruction Plugin

This is a compatibility version of the borehole reconstruction plugin for older ParaView environments.

## Main difference from the ParaView 6 plugin

This version avoids `scipy` and uses NumPy/VTK-only replacements for:
- Gaussian smoothing
- binary morphology
- connected components

It also uses a simpler PCA-based final ellipse fit instead of the SciPy optimizer. That makes the plugin easier to load on older ParaView builds, but the output may be slightly less accurate than the full ParaView 6 version.

## Files

- `borehole_reconstruction_510_plugin.py`
- `borehole_core_510.py`

## Load in ParaView 5.10

1. Open ParaView.
2. Go to `Tools -> Manage Plugins`.
3. Click `Load New...`.
4. Select `borehole_reconstruction_510_plugin.py`.
5. Load a point-cloud dataset.
6. Select the point cloud in the Pipeline Browser.
7. Search for `Borehole Reconstruction 5.10` in Filters.
8. Click `Apply`.

## Recommended first settings

- `SliceAxis = 0`
- `FractionStart = 0.01`
- `FractionEnd = 0.69`
- `NumberOfSlices = 35`
- `ThicknessFraction = 0.02`
- `MirrorPoints = 1`
- `MirrorAxis = 1`

## Notes

This version is intended as a compatibility bridge for ParaView 5.10. The full plugin in `paraview_plugin/` remains the preferred version for ParaView 6.1.
