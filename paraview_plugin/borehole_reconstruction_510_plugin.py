import json
import os
import shutil
import subprocess
import sys
import tempfile

import vtk

from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smdomain, smproperty, smproxy


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_RUNNER = os.path.join(PLUGIN_DIR, "borehole_backend_510.py")


class BoreholeConfig510:
    def __init__(self):
        self.chosen_axis = 0
        self.frac_start = 0.01
        self.frac_end = 0.69
        self.n_slices = 35
        self.thickness_frac = 0.02
        self.grid_n = 220
        self.subsample = 20000
        self.smooth_sigma = 3.0
        self.low_density_q = 0.10
        self.pad_frac_a = 0.10
        self.pad_frac_b = 0.10
        self.open_iters = 1
        self.close_iters = 1
        self.occ_dilate_iters = 5
        self.min_comp_area = 120
        self.max_comp_frac = 0.20
        self.do_mirror = True
        self.mirror_axis = 1
        self.ellipse_q = 90.0
        self.use_outer_band_refinement = True
        self.outer_band_sectors = 24
        self.outer_band_keep_per_sector = 6
        self.max_ar = 4.0
        self.max_axis_jump = 2.0
        self.enable_short_gap_fill = True
        self.max_interp_gap = 2
        self.n_ring = 240
        self.cap_ends = True
        self.max_gap_factor = 2.5
        self.fill_holes = True
        self.fill_hole_size = 1e6
        self.rng_seed = 0

    def to_dict(self):
        return dict(self.__dict__)


def _iter_datasets(data_object):
    if data_object is None:
        return
    if isinstance(data_object, vtk.vtkCompositeDataSet):
        it = data_object.NewIterator()
        it.UnRegister(None)
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            current = it.GetCurrentDataObject()
            if isinstance(current, vtk.vtkDataSet):
                yield current
            elif isinstance(current, vtk.vtkCompositeDataSet):
                for nested in _iter_datasets(current):
                    yield nested
            it.GoToNextItem()
    elif isinstance(data_object, vtk.vtkDataSet):
        yield data_object


def _write_input_points(data_object, path):
    point_count = 0
    stream = open(path, "w")
    try:
        for dataset in _iter_datasets(data_object):
            input_points = dataset.GetPoints()
            if input_points is None:
                continue
            for idx in range(input_points.GetNumberOfPoints()):
                x, y, z = input_points.GetPoint(idx)
                stream.write("%.17g,%.17g,%.17g\n" % (x, y, z))
                point_count += 1
    finally:
        stream.close()
    if point_count == 0:
        raise RuntimeError("Borehole Reconstruction 5.10 requires an input with point coordinates.")


def _boundary_nonmanifold_edge_count(mesh):
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(mesh)
    feature_edges.BoundaryEdgesOn()
    feature_edges.NonManifoldEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()
    return feature_edges.GetOutput().GetNumberOfCells()


def _add_field_value(poly, name, value):
    if isinstance(value, int):
        array = vtk.vtkIntArray()
    else:
        array = vtk.vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfComponents(1)
    array.InsertNextValue(value)
    poly.GetFieldData().AddArray(array)


def _read_output_surface(path, metrics_path, cfg):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    reader.Update()
    raw_surface = reader.GetOutput()
    if raw_surface is None or raw_surface.GetNumberOfPoints() == 0:
        raise RuntimeError("The borehole backend did not produce a non-empty OBJ surface.")

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(raw_surface)
    clean.Update()
    surface = clean.GetOutput()

    if cfg.fill_holes:
        fill = vtk.vtkFillHolesFilter()
        fill.SetInputData(surface)
        fill.SetHoleSize(cfg.fill_hole_size)
        fill.Update()
        surface = fill.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    result = vtk.vtkPolyData()
    result.ShallowCopy(normals.GetOutput())
    with open(metrics_path, "r") as stream:
        metrics = json.load(stream)
    metrics["boundary_nonmanifold_edge_segments"] = _boundary_nonmanifold_edge_count(result)
    for name in sorted(metrics):
        _add_field_value(result, name, metrics[name])
    return result


def _venv_python(venv_path):
    if not venv_path:
        return ""
    if os.name == "nt":
        return os.path.join(venv_path, "Scripts", "python.exe")
    return os.path.join(venv_path, "bin", "python")


def _resolve_backend_python(explicit_python, venv_path):
    candidates = [
        explicit_python,
        os.environ.get("BOREHOLE_BACKEND_PYTHON", ""),
        _venv_python(venv_path or os.environ.get("BOREHOLE_BACKEND_VENV", "")),
        shutil.which("python3") or "",
        shutil.which("python") or "",
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "No backend Python was found. Set BackendPython, BackendVenvPath, "
        "BOREHOLE_BACKEND_PYTHON, or BOREHOLE_BACKEND_VENV."
    )


def _backend_environment(venv_path):
    env = os.environ.copy()
    active_venv = venv_path or env.get("BOREHOLE_BACKEND_VENV", "")
    if active_venv:
        bin_dir = os.path.dirname(_venv_python(active_venv))
        env["VIRTUAL_ENV"] = active_venv
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def _run_backend(
    input_path,
    output_path,
    metrics_path,
    cfg,
    backend_python,
    backend_venv,
    timeout_seconds,
):
    python = _resolve_backend_python(backend_python, backend_venv)
    cmd = [
        python,
        BACKEND_RUNNER,
        input_path,
        output_path,
        metrics_path,
        json.dumps(cfg.to_dict()),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=PLUGIN_DIR,
            env=_backend_environment(backend_venv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Borehole backend timed out after %s seconds. Partial output:\n%s\n%s"
            % (timeout_seconds, exc.stdout or "", exc.stderr or "")
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "Borehole backend failed with exit code %s using %s.\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (completed.returncode, python, completed.stdout, completed.stderr)
        )


@smproxy.filter(label="Borehole Reconstruction 5.10 Portable")
@smproperty.input(name="Input")
@smdomain.datatype(
    dataTypes=["vtkPolyData", "vtkUnstructuredGrid", "vtkStructuredGrid", "vtkDataSet"],
    composite_data_supported=True,
)
class BoreholeReconstruction510Portable(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkPolyData")
        self._cfg = BoreholeConfig510()
        self._backend_python = ""
        self._backend_venv = ""
        self._backend_timeout_seconds = 0

    def _set_cfg(self, attr, value):
        old = getattr(self._cfg, attr)
        setattr(self._cfg, attr, value)
        if old != value:
            self.Modified()

    def _set_attr(self, attr, value):
        old = getattr(self, attr)
        setattr(self, attr, value)
        if old != value:
            self.Modified()

    @smproperty.stringvector(name="BackendPython", default_values="")
    def SetBackendPython(self, value):
        self._set_attr("_backend_python", str(value).strip())

    @smproperty.stringvector(name="BackendVenvPath", default_values="")
    def SetBackendVenvPath(self, value):
        self._set_attr("_backend_venv", str(value).strip())

    @smproperty.intvector(name="BackendTimeoutSeconds", default_values=0)
    @smdomain.intrange(min=0, max=86400)
    def SetBackendTimeoutSeconds(self, value):
        self._set_attr("_backend_timeout_seconds", int(value))

    @smproperty.intvector(name="SliceAxis", default_values=0)
    @smdomain.intrange(min=0, max=2)
    def SetSliceAxis(self, value):
        self._set_cfg("chosen_axis", int(value))

    @smproperty.doublevector(name="FractionStart", default_values=0.01)
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetFractionStart(self, value):
        self._set_cfg("frac_start", float(value))

    @smproperty.doublevector(name="FractionEnd", default_values=0.69)
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetFractionEnd(self, value):
        self._set_cfg("frac_end", float(value))

    @smproperty.intvector(name="NumberOfSlices", default_values=35)
    @smdomain.intrange(min=4, max=200)
    def SetNumberOfSlices(self, value):
        self._set_cfg("n_slices", int(value))

    @smproperty.doublevector(name="ThicknessFraction", default_values=0.02)
    @smdomain.doublerange(min=0.001, max=0.20)
    def SetThicknessFraction(self, value):
        self._set_cfg("thickness_frac", float(value))

    @smproperty.intvector(name="MirrorPoints", default_values=1)
    @smdomain.intrange(min=0, max=1)
    def SetMirrorPoints(self, value):
        self._set_cfg("do_mirror", bool(value))

    @smproperty.intvector(name="MirrorAxis", default_values=1)
    @smdomain.intrange(min=0, max=2)
    def SetMirrorAxis(self, value):
        self._set_cfg("mirror_axis", int(value))

    @smproperty.doublevector(name="LowDensityQuantile", default_values=0.10)
    @smdomain.doublerange(min=0.001, max=0.5)
    def SetLowDensityQuantile(self, value):
        self._set_cfg("low_density_q", float(value))

    @smproperty.doublevector(name="SmoothSigma", default_values=3.0)
    @smdomain.doublerange(min=0.25, max=20.0)
    def SetSmoothSigma(self, value):
        self._set_cfg("smooth_sigma", float(value))

    @smproperty.intvector(name="EnableShortGapFill", default_values=1)
    @smdomain.intrange(min=0, max=1)
    def SetEnableShortGapFill(self, value):
        self._set_cfg("enable_short_gap_fill", bool(value))

    @smproperty.intvector(name="MaximumInterpolatedGap", default_values=2)
    @smdomain.intrange(min=0, max=10)
    def SetMaximumInterpolatedGap(self, value):
        self._set_cfg("max_interp_gap", int(value))

    def RequestData(self, request, inInfo, outInfo):
        input_data = vtk.vtkDataObject.GetData(inInfo[0], 0)
        output = vtk.vtkPolyData.GetData(outInfo, 0)
        with tempfile.TemporaryDirectory(prefix="borehole_pv510_") as tmpdir:
            input_path = os.path.join(tmpdir, "input_points.csv")
            output_path = os.path.join(tmpdir, "output_surface.obj")
            metrics_path = os.path.join(tmpdir, "output_metrics.json")
            _write_input_points(input_data, input_path)
            _run_backend(
                input_path,
                output_path,
                metrics_path,
                self._cfg,
                self._backend_python,
                self._backend_venv,
                self._backend_timeout_seconds,
            )
            output.ShallowCopy(_read_output_surface(output_path, metrics_path, self._cfg))
        return 1
