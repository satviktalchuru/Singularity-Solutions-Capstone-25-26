import tempfile
import unittest
from types import SimpleNamespace

import pyvista as pv

from evaluate_ground_truth import evaluate_pair


class GroundTruthEvaluatorTests(unittest.TestCase):
    def test_identical_surfaces_have_zero_error(self):
        mesh = pv.Cylinder(
            center=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            radius=0.05,
            height=1.0,
            resolution=96,
            capping=True,
        ).triangulate()
        args = SimpleNamespace(
            axis=0,
            sections=24,
            surface_samples=3000,
            volume_samples=20000,
            seed=17,
        )
        with tempfile.TemporaryDirectory() as directory:
            predicted_path = "%s/identical_reconstruction.vtp" % directory
            truth_path = "%s/identical_ground_truth.vtp" % directory
            mesh.save(predicted_path)
            mesh.save(truth_path)
            metrics = evaluate_pair(predicted_path, truth_path, args)

        self.assertLess(metrics["centerline_error_mean"], 1e-12)
        self.assertLess(metrics["major_axis_relative_error_mean"], 1e-12)
        self.assertLess(metrics["minor_axis_relative_error_mean"], 1e-12)
        self.assertLess(metrics["surface_symmetric_mean"], 1e-10)
        self.assertAlmostEqual(metrics["overlap_iou"], 1.0, places=12)
        self.assertAlmostEqual(metrics["overlap_dice"], 1.0, places=12)
        self.assertAlmostEqual(metrics["volume_relative_error_mc"], 0.0, places=12)
        self.assertEqual(metrics["axis_start_error"], 0.0)
        self.assertEqual(metrics["axis_end_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
