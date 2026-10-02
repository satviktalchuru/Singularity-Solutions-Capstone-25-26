# ParaView 5.10 Borehole Reconstruction Plugin

This is a compatibility version of the borehole reconstruction plugin for older ParaView environments.

The ParaView-facing plugin is intentionally thin. It only imports ParaView/VTK modules that are already part of ParaView, writes the input points to a temporary CSV file, launches a backend Python process, and reads the backend OBJ mesh and JSON metrics back into the ParaView pipeline. NumPy and the reconstruction core are imported only by the backend process.

This avoids coupling the reconstruction code to ParaView's embedded Python package environment. In particular, it lets ParaView 5.10 load the plugin even when the calculation dependencies live in a separate virtual environment.

## Main difference from the ParaView 6 plugin

This version avoids `scipy` and uses NumPy-only replacements for:
- Gaussian smoothing
- binary morphology
- connected components

It also uses a simpler PCA-based final ellipse fit instead of the SciPy optimizer. That makes the plugin easier to load on older ParaView builds, but the output may be slightly less accurate than the full ParaView 6 version.

## Files

- `borehole_reconstruction_510_plugin.py`
- `borehole_backend_510.py`
- `borehole_core_510.py`
- `requirements-backend.txt`

## Backend virtual environment

Create a backend environment with the packages needed by `borehole_core_510.py`. On Ubuntu, one workable starting point is:

```bash
python3 -m venv .venv-borehole
. .venv-borehole/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-backend.txt
```

The backend process only needs `numpy`. It does not import `vtk` or `paraview`. All VTK post-processing uses the VTK already bundled with ParaView, so the system VTK version and backend environment do not need to match ParaView's VTK version.

## Load in ParaView 5.10

1. Open ParaView.
2. Go to `Tools -> Manage Plugins`.
3. Click `Load New...`.
4. Select `borehole_reconstruction_510_plugin.py`.
5. Load a point-cloud dataset.
6. Select the point cloud in the Pipeline Browser.
7. Search for `Borehole Reconstruction 5.10 Portable` in Filters.
8. Set either:
   - `BackendPython` to the full path of the backend interpreter, for example `/path/to/.venv-borehole/bin/python`
   - or `BackendVenvPath` to the venv directory, for example `/path/to/.venv-borehole`
9. Click `Apply`.

Instead of setting these properties in the UI, you can also launch ParaView with one of these environment variables:

```bash
export BOREHOLE_BACKEND_PYTHON=/path/to/.venv-borehole/bin/python
# or
export BOREHOLE_BACKEND_VENV=/path/to/.venv-borehole
```

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

The wrapper still uses ParaView's own VTK because a ParaView filter must receive and return VTK data objects. The external backend does not require VTK: it receives CSV points and returns an OBJ triangle mesh plus JSON metrics. Mesh cleaning, optional hole filling, normals, and ParaView field arrays are handled by the wrapper with ParaView's bundled VTK.

The backend runs synchronously during `Apply`. That keeps the filter behavior deterministic in ParaView's pipeline while still isolating the calculation dependencies from ParaView.
