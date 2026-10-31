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
- `EnableTrajectoryGate = 1`
- `MaximumCenterPredictionError = 0.75`
- `MaximumAxisPredictionError = 0.45`
- `EnableConfidenceGate = 1`
- `MinimumSliceConfidence = 0.35`
- `EnableAdaptiveBoundaryBand = 1`
- `MaximumBoundaryBandCells = 8`

The trajectory gate fits a short linear trend to the recent accepted slice centers and ellipse axes. A new fit is rejected when its center differs by more than `MaximumCenterPredictionError` times the predicted mean radius, or an ellipse axis differs by more than `MaximumAxisPredictionError` of its predicted value. Keep the defaults for initial runs; increase them only when a genuinely curved or rapidly changing borehole is being rejected.

The output field data includes `trajectory_rejected_slice_count` and center/axis prediction-error summaries. Large maximum errors indicate that the detector encountered a cavity candidate inconsistent with the tracked borehole, even when an older continuity rule also rejected it.

The confidence gate combines ellipse inlier ratio, scale-normalized residual, angular boundary coverage, and trajectory agreement. Fits below `MinimumSliceConfidence` are excluded from the loft, and `confidence_rejected_slice_count`, `slice_confidence_mean`, and `slice_confidence_min` are added to the output field data.

The adaptive boundary band begins two histogram cells from the detected cavity boundary and expands only when too few raw points support an ellipse fit. Expansion stops at `MaximumBoundaryBandCells`, preventing an unbounded search into unrelated soil. The output records how many slices expanded and the mean/maximum band width used.

## Physical interpretation of controls

- `SliceAxis`: coordinate direction along which the borehole is sampled.
- `FractionStart` and `FractionEnd`: normalized positions along that axis; `0` and `1` are the point-cloud bounds, not physical distances.
- `NumberOfSlices`: number of cross-sections sampled between the start and end positions.
- `ThicknessFraction`: axial thickness of each cross-section as a fraction of the full point-cloud length.
- `MirrorPoints` and `MirrorAxis`: reconstruct a missing symmetric half of a one-sided point cloud.
- `LowDensityQuantile`: fraction of the occupied density image treated as potential empty cavity.
- `SmoothSigma`: Gaussian smoothing radius measured in histogram cells, not model units.
- `MaximumCenterPredictionError`: allowed center deviation measured in predicted borehole radii.
- `MaximumAxisPredictionError`: allowed fractional change from the predicted ellipse axes.
- `MinimumSliceConfidence`: minimum combined support from inliers, residuals, angular coverage, and trajectory agreement.
- `MaximumInterpolatedGap`: largest number of consecutive missing slices that may be filled between two observed fits.

## Reference-assisted evaluation

`evaluate_new_dataset.py` supports datasets that provide both the soil point cloud and a
separate drillhead mesh. The drillhead is used only to estimate a robust center trajectory
and a local search scale; its surface is not treated as the borehole boundary. From that
trajectory, the core samples the nearest soil points in angular sectors and fits the cavity's
inner wall. Density components and multi-scale DoG proposals remain available as fallbacks.

The reference-assisted path combines a dataset-level sampling regime with per-slice boundary
classification. Both use dimensionless measurements so the decision does not depend on raw
coordinate units. The dataset regime uses median nominal point spacing divided by drillhead
scale:

- `dense` (`<= 0.25`): prefer the five-layer stable fit locally.
- `medium` (`0.25` to `0.55`): use the conservative one-layer baseline unless a displacement
  signal supports a five-layer baseline and global sequence optimization.
- `coarse` (`> 0.55`): use a five-layer protected baseline so sparse sectors contain enough
  wall samples.

Within that regime, each slice uses angular coverage to choose among these routes:

- `partial`: angular coverage below 90%. The center is constrained to the drillhead
  trajectory and a robust quadratic ellipse is fitted to the visible arc.
- `stable`: complete angular coverage and spacing/scale at or below 0.25. Five radial
  layers and outer-band refinement stabilize dense, thick point clouds.
- `inner`: complete angular coverage with coarser sampling. The nearest point in each
  angular sector limits expansion into unrelated soil.

An adaptive fit must exceed the protected baseline score by `0.03`; stable fits also cannot
increase normalized p90 residual by more than 10%. The global optimizer then selects a
continuous sequence using confidence, center motion, axis change, missing-slice gaps, and
route-switch penalties. It runs for signal-supported medium data and whenever the local
baseline accepts fewer than 80% of requested slices.

Density components and multi-scale DoG proposals remain fallback candidates. The metrics
`classified_radial_*_slice_count` report how slices were classified, while
`selected_radial_*_slice_count` report which routes actually supplied the final loft. This
adaptive routing reduces sensitivity to raw coordinate scale; it does not replace the fit
confidence, trajectory, or continuity gates.

Dataset-dependent sampling is inferred automatically when a drillhead reference is supplied.
The core selects the drilling axis from the drillhead's longest dimension, detects a one-sided
soil model that should be mirrored, derives the start/end interval from the drillhead bounds,
and chooses slice count and thickness from point spacing and target points per slice. Manual
arguments remain optional diagnostic overrides:

```bash
python evaluate_new_dataset.py \
  /path/to/prestress1970000.vtk \
  /path/to/drillhead1970000.vtk \
  --output-dir evaluation_output
```

The output includes the reconstructed VTP, JSON metrics, a 3D overlay, and orthographic
side/top/end overlays. Validation results depend on the input dataset and must be read from
the generated JSON rather than treated as fixed expected values. These are internal
consistency metrics, not absolute accuracy against surveyed ground truth.

The reference-assisted path is currently an evaluation API in `borehole_core_510.py`; the
single-input ParaView filter continues to run without a reference. A production UI should
expose the drillhead as an optional second input rather than silently assuming it exists.

## Synthetic ground-truth benchmark

Use `synthetic_benchmark.py` to test absolute geometric accuracy when the original
simulation does not include a ground-truth borehole surface:

```bash
python synthetic_benchmark.py \
  --output-dir synthetic_benchmark_output \
  --points 120000 \
  --seed 7
```

The benchmark generates a straight circular cavity, a curved elliptical cavity, and an
edge-adjacent incomplete cavity. It saves each input, reference, reconstruction, and known
surface as VTP files. It also reports normalized center error, relative major/minor-axis
error, and point-to-analytic-surface error in JSON, CSV, Markdown, and PNG formats.

These are absolute accuracy measurements. Inlier ratio, fit residual, confidence, and
continuity remain useful internal diagnostics, but they cannot establish accuracy without
known geometry. The synthetic cases should therefore be retained as regression tests while
the radial boundary estimator and edge handling are improved.

The current 120,000-point regression benchmark gives mean center errors of 1.1% to 2.7% of
borehole radius. Mean axis errors are 3.4% to 8.8% for complete straight/curved cases; the
incomplete edge case has 5.4% major-axis and 13.2% minor-axis error. The larger incomplete
minor-axis error remains the main known geometric limitation.

## Ground-truth mesh evaluation

Use `evaluate_ground_truth.py` when both a reconstructed surface and its corresponding
ground-truth surface are available. Supply one or more alternating prediction/truth pairs:

```bash
python -m pip install -r requirements-evaluation.txt
python evaluate_ground_truth.py \
  /path/to/reconstruction.vtp /path/to/ground_truth.vtp \
  --output-dir ground_truth_evaluation
```

The evaluator reports centerline error, major/minor-axis error, bidirectional surface
distance, volumetric IoU and Dice, volume error, and start/end extent error. Length-valued
shape errors are also normalized by the ground-truth mean radius so results can be compared
across differently scaled datasets. It does not register the surfaces before comparison;
location and trajectory mistakes therefore remain part of the measured error.

Outputs include JSON and CSV metrics, a Markdown summary, and an accuracy plot. Monte Carlo
occupancy is deterministic for a fixed `--seed`; increase `--volume-samples` when a tighter
IoU or volume estimate is required. The evaluator measures absolute accuracy only when the
provided reference is genuine ground truth.

## Notes

This version is intended as a compatibility bridge for ParaView 5.10. The full plugin in `paraview_plugin/` remains the preferred version for ParaView 6.1.

The wrapper still uses ParaView's own VTK because a ParaView filter must receive and return VTK data objects. The external backend does not require VTK: it receives CSV points and returns an OBJ triangle mesh plus JSON metrics. Mesh cleaning, optional hole filling, normals, and ParaView field arrays are handled by the wrapper with ParaView's bundled VTK.

The backend runs synchronously during `Apply`. That keeps the filter behavior deterministic in ParaView's pipeline while still isolating the calculation dependencies from ParaView.
