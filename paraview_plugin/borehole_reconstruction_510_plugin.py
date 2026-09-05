import os
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smdomain, smproperty, smproxy

PLUGIN_DIR = os.path.dirname(__file__)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

from borehole_core_510 import BoreholeConfig510, reconstruct_borehole  # noqa: E402


def extract_points(data_object):
    if data_object is None:
        raise RuntimeError("No input dataset.")
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
    if isinstance(data_object, vtk.vtkDataSet) and data_object.GetPoints() is not None:
        return vtk_to_numpy(data_object.GetPoints().GetData())
    raise RuntimeError("Borehole Reconstruction 5.10 requires an input with point coordinates.")


@smproxy.filter(label="Borehole Reconstruction 5.10")
@smproperty.input(name="Input")
@smdomain.datatype(
    dataTypes=["vtkPolyData", "vtkUnstructuredGrid", "vtkStructuredGrid", "vtkDataSet"],
    composite_data_supported=True,
)
class BoreholeReconstruction510(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkPolyData")
        self._cfg = BoreholeConfig510()

    def _set_cfg(self, attr, value):
        old = getattr(self._cfg, attr)
        setattr(self._cfg, attr, value)
        if old != value:
            self.Modified()

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
        points = extract_points(input_data)
        surface = reconstruct_borehole(points, self._cfg)
        output.ShallowCopy(surface)
        return 1
