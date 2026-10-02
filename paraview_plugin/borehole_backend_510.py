import json
import os
import sys
import traceback


def _fail(message):
    sys.stderr.write(message.rstrip() + "\n")
    return 1


def _write_obj(path, points, faces):
    with open(path, "w") as stream:
        stream.write("# Borehole Reconstruction 5.10 NumPy backend\n")
        for x, y, z in points:
            stream.write("v %.17g %.17g %.17g\n" % (x, y, z))
        for i, j, k in faces:
            stream.write("f %d %d %d\n" % (i + 1, j + 1, k + 1))


def main(argv):
    if len(argv) != 5:
        return _fail(
            "Usage: borehole_backend_510.py "
            "INPUT_POINTS.csv OUTPUT_SURFACE.obj OUTPUT_METRICS.json CONFIG_JSON"
        )

    input_path, output_path, metrics_path, config_json = argv[1], argv[2], argv[3], argv[4]
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    try:
        import numpy as np

        from borehole_core_510 import BoreholeConfig510, reconstruct_borehole

        cfg = BoreholeConfig510()
        for name, value in json.loads(config_json).items():
            if hasattr(cfg, name):
                setattr(cfg, name, value)

        points = np.loadtxt(input_path, delimiter=",", dtype=float, ndmin=2)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise RuntimeError("Backend input must contain one X,Y,Z point per CSV row.")

        mesh_points, mesh_faces, metrics = reconstruct_borehole(points, cfg)
        _write_obj(output_path, mesh_points, mesh_faces)
        with open(metrics_path, "w") as stream:
            json.dump(metrics, stream, sort_keys=True)
    except Exception:
        return _fail(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
