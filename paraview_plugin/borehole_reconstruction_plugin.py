import os
import sys

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smdomain, smproperty, smproxy

PLUGIN_DIR = os.path.dirname(__file__)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

_CORE_IMPORT_ERROR = None
try:
    from borehole_core import BoreholeConfig, reconstruct_borehole  # noqa: E402
except Exception as exc:  # pragma: no cover - intended for ParaView runtime diagnostics
    BoreholeConfig = None
    reconstruct_borehole = None
    _CORE_IMPORT_ERROR = exc


def _extract_points_from_data_object(data_object):
    if data_object is None:
        raise RuntimeError("No input data object provided.")

    if isinstance(data_object, vtk.vtkPointSet):
        pts = data_object.GetPoints()
        if pts is None:
            raise RuntimeError("Input point set has no points.")
        return vtk_to_numpy(pts.GetData())

    if isinstance(data_object, vtk.vtkDataSet):
        pts = data_object.GetPoints()
        if pts is None:
            raise RuntimeError("Input dataset has no points.")
        return vtk_to_numpy(pts.GetData())

    if isinstance(data_object, vtk.vtkCompositeDataSet):
        blocks = []
        it = data_object.NewIterator()
        it.UnRegister(None)
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            obj = it.GetCurrentDataObject()
            if isinstance(obj, vtk.vtkDataSet) and obj.GetPoints() is not None:
                blocks.append(vtk_to_numpy(obj.GetPoints().GetData()))
            it.GoToNextItem()
        if not blocks:
            raise RuntimeError("Composite input contains no point coordinates.")
        return np.vstack(blocks)

    raise RuntimeError(f"Unsupported input type: {data_object.GetClassName()}")


@smproxy.filter(label="Borehole Reconstruction")
@smproperty.input(name="Input")
@smdomain.datatype(
    dataTypes=["vtkDataObject", "vtkDataSet", "vtkPolyData", "vtkUnstructuredGrid", "vtkPointSet"],
    composite_data_supported=True,
)
class BoreholeReconstructionFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=1, nOutputPorts=1, outputType="vtkPolyData")
        self._cfg = BoreholeConfig() if BoreholeConfig is not None else None
        self._needs_update = True

    def _mark_modified(self, changed: bool):
        if changed:
            self._needs_update = True
            self.Modified()

    def _ensure_core(self):
        if (_CORE_IMPORT_ERROR is not None) or (BoreholeConfig is None) or (reconstruct_borehole is None):
            raise RuntimeError(
                "Borehole Reconstruction plugin dependencies could not be imported. "
                f"Original import error: {_CORE_IMPORT_ERROR!r}"
            )
        if self._cfg is None:
            self._cfg = BoreholeConfig()

    def _set_cfg(self, attr: str, value):
        self._ensure_core()
        old = getattr(self._cfg, attr)
        setattr(self._cfg, attr, value)
        self._mark_modified(old != getattr(self._cfg, attr))

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
    @smdomain.intrange(min=4, max=500)
    def SetNumberOfSlices(self, value):
        self._set_cfg("n_slices", int(value))

    @smproperty.doublevector(name="ThicknessFraction", default_values=0.02)
    @smdomain.doublerange(min=0.001, max=0.25)
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

    @smproperty.doublevector(name="SmoothSigma", default_values=4.0)
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

    @smproperty.intvector(name="EnableOuterBandRefinement", default_values=1)
    @smdomain.intrange(min=0, max=1)
    def SetEnableOuterBandRefinement(self, value):
        self._set_cfg("use_outer_band_refinement", bool(value))

    @smproperty.intvector(name="EnableEllipseRefinement", default_values=1)
    @smdomain.intrange(min=0, max=1)
    def SetEnableEllipseRefinement(self, value):
        self._set_cfg("use_ellipse_refinement", bool(value))

    def RequestData(self, request, inInfo, outInfo):
        self._ensure_core()
        input_data = vtk.vtkDataObject.GetData(inInfo[0], 0)
        output = vtk.vtkPolyData.GetData(outInfo, 0)
        points = _extract_points_from_data_object(input_data)
        result = reconstruct_borehole(points, self._cfg)
        output.ShallowCopy(result["surface"])
        return 1
