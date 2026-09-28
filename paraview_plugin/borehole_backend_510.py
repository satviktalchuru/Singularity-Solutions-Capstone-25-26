import json
import os
import sys
import traceback


def _fail(message):
    sys.stderr.write(message.rstrip() + "\n")
    return 1


def main(argv):
    if len(argv) != 4:
        return _fail("Usage: borehole_backend_510.py INPUT_POINTS.vtp OUTPUT_SURFACE.vtp CONFIG_JSON")

    input_path, output_path, config_json = argv[1], argv[2], argv[3]
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        from borehole_core_510 import BoreholeConfig510, reconstruct_borehole

        cfg = BoreholeConfig510()
        for name, value in json.loads(config_json).items():
            if hasattr(cfg, name):
                setattr(cfg, name, value)

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(input_path)
        reader.Update()
        input_poly = reader.GetOutput()
        if input_poly is None or input_poly.GetPoints() is None:
            raise RuntimeError("Backend input has no points.")

        points = vtk_to_numpy(input_poly.GetPoints().GetData())
        surface = reconstruct_borehole(points, cfg)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetDataModeToAscii()
        if hasattr(writer, "SetCompressorTypeToNone"):
            writer.SetCompressorTypeToNone()
        writer.SetInputData(surface)
        if writer.Write() != 1:
            raise RuntimeError("Could not write backend output surface.")
    except Exception:
        return _fail(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
