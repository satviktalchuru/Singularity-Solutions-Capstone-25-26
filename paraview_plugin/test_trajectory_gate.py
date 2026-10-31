import unittest

import numpy as np

from borehole_core_510 import (
    adaptive_boundary_candidates,
    apply_adaptive_route_margin,
    BoreholeConfig510,
    build_reference_trajectory,
    characterize_point_cloud,
    classify_radial_mode,
    configure_from_reference_geometry,
    dog_blob_proposals,
    fit_ellipse_fixed_center,
    predict_trajectory,
    radial_inner_envelope,
    reference_at_axis,
    slice_confidence,
    trajectory_errors,
)


class TrajectoryGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = BoreholeConfig510()
        identity = np.eye(2)
        self.kept = []
        for idx, axis_center in enumerate((0.0, 1.0, 2.0, 3.0)):
            center = np.array([0.1 * axis_center, -0.05 * axis_center])
            axes = (1.0 + 0.02 * axis_center, 0.8 + 0.01 * axis_center)
            self.kept.append((0.1 * idx, axis_center, center, axes, identity, idx, "observed", 0.0, 0.0, 1.0))

    def test_prediction_accepts_linear_trajectory(self):
        prediction = predict_trajectory(self.kept, 4.0, self.cfg)
        center_error, axis_error = trajectory_errors(
            np.array([0.4, -0.2]), (1.08, 0.84), prediction
        )
        self.assertLess(center_error, 1e-10)
        self.assertLess(axis_error, 1e-10)

    def test_prediction_rejects_off_trajectory_center(self):
        prediction = predict_trajectory(self.kept, 4.0, self.cfg)
        center_error, axis_error = trajectory_errors(
            np.array([1.4, -0.2]), (1.08, 0.84), prediction
        )
        self.assertGreater(center_error, self.cfg.max_center_prediction_frac)
        self.assertLess(axis_error, self.cfg.max_axis_prediction_frac)

    def test_prediction_rejects_implausible_axis_growth(self):
        prediction = predict_trajectory(self.kept, 4.0, self.cfg)
        center_error, axis_error = trajectory_errors(
            np.array([0.4, -0.2]), (1.7, 0.84), prediction
        )
        self.assertLess(center_error, self.cfg.max_center_prediction_frac)
        self.assertGreater(axis_error, self.cfg.max_axis_prediction_frac)

    def test_confidence_penalizes_weak_partial_fit(self):
        angles = np.linspace(0.0, 0.25 * np.pi, 20)
        partial = np.column_stack([np.cos(angles), 0.8 * np.sin(angles)])
        fit = {
            "points": partial,
            "center": np.zeros(2),
            "axes": (1.0, 0.8),
            "R": np.eye(2),
            "inlier": 0.55,
            "resid_p90": 0.30,
        }
        confidence, coverage, _ = slice_confidence(fit, (0.7, 0.4), self.cfg)
        self.assertLess(coverage, 0.25)
        self.assertLess(confidence, self.cfg.min_slice_confidence)

    def test_characterization_reports_physical_scale(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
        ])
        result = characterize_point_cloud(points, 0)
        self.assertEqual(result["input_point_count"], 2)
        self.assertEqual(result["axis_span"], 2.0)
        self.assertAlmostEqual(result["nominal_point_spacing"], 1.0)

    def test_boundary_band_expands_for_sparse_points(self):
        self.cfg.min_boundary_candidates = 2
        self.cfg.boundary_band_cells = 1.0
        self.cfg.max_boundary_band_cells = 4.0
        boundary = np.array([[0.0, 0.0]])
        A = np.array([0.25, -0.25, 1.0])
        B = np.zeros(3)
        candidates, cells = adaptive_boundary_candidates(A, B, boundary, 0.1, self.cfg)
        self.assertEqual(candidates.shape[0], 2)
        self.assertGreater(cells, self.cfg.boundary_band_cells)

    def test_dog_finds_dark_multiscale_blob(self):
        grid = 81
        yy, xx = np.mgrid[:grid, :grid]
        density = np.ones((grid, grid), dtype=float)
        density[(xx - 40) ** 2 + (yy - 40) ** 2 <= 8 ** 2] = 0.0
        coordinates = np.linspace(-1.0, 1.0, grid)
        proposals = dog_blob_proposals(
            density, np.ones_like(density, dtype=bool), coordinates, coordinates, self.cfg
        )
        self.assertTrue(proposals)
        center = proposals[0]["boundary"].mean(axis=0)
        self.assertLess(np.linalg.norm(center), 0.1)

    def test_reference_trajectory_tracks_binned_centerline(self):
        self.cfg.do_mirror = False
        self.cfg.reference_bins = 5
        self.cfg.reference_min_points_per_bin = 5
        rows = []
        for axis_center in np.linspace(0.0, 1.0, 101):
            for offset in np.linspace(-0.05, 0.05, 7):
                rows.append([axis_center, 0.2 * axis_center + offset, -0.1 * axis_center])
        reference = build_reference_trajectory(np.asarray(rows), self.cfg)
        center, scale = reference_at_axis(reference, 0.5)
        self.assertTrue(np.allclose(center, [0.1, -0.05], atol=0.02))
        self.assertGreater(scale, 0.0)

    def test_reference_trajectory_is_not_extrapolated(self):
        self.cfg.do_mirror = False
        self.cfg.reference_bins = 3
        self.cfg.reference_min_points_per_bin = 1
        points = np.array([
            [0.0, -0.1, 0.0], [0.0, 0.1, 0.0],
            [0.5, -0.1, 0.0], [0.5, 0.1, 0.0],
            [1.0, -0.1, 0.0], [1.0, 0.1, 0.0],
        ])
        reference = build_reference_trajectory(points, self.cfg)
        self.assertIsNone(reference_at_axis(reference, -0.1))

    def test_radial_inner_envelope_selects_cavity_wall(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        inner = np.column_stack([0.5 * np.cos(angles), 0.4 * np.sin(angles)])
        outer = np.column_stack([1.0 * np.cos(angles), 0.9 * np.sin(angles)])
        points = np.vstack([inner, outer])
        selected, coverage = radial_inner_envelope(
            points[:, 0], points[:, 1], np.zeros(2), 0.5, self.cfg
        )
        self.assertGreater(coverage, 0.95)
        self.assertEqual(
            selected.shape[0],
            self.cfg.reference_radial_sectors * self.cfg.reference_radial_min_layers,
        )
        self.assertLess(np.median(np.linalg.norm(selected, axis=1)), 0.55)

    def test_radial_envelope_can_generate_stability_layers(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        layers = [
            np.column_stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
            ])
            for radius in np.linspace(0.5, 0.7, 5)
        ]
        points = np.vstack(layers)
        selected, coverage = radial_inner_envelope(
            points[:, 0], points[:, 1], np.zeros(2), 0.5, self.cfg,
            min_layers=5, max_layers=5,
        )
        self.assertGreater(coverage, 0.95)
        self.assertEqual(selected.shape[0], self.cfg.reference_radial_sectors * 5)

    def test_fixed_center_fit_recovers_ellipse_from_partial_arc(self):
        center = np.array([0.2, -0.1])
        angle = np.linspace(0.35 * np.pi, 1.75 * np.pi, 200)
        points = center + np.column_stack([0.8 * np.cos(angle), 0.4 * np.sin(angle)])
        fitted_center, axes, _ = fit_ellipse_fixed_center(points, center)
        self.assertTrue(np.allclose(fitted_center, center))
        self.assertTrue(np.allclose(axes, [0.8, 0.4], atol=1e-6))

    def test_reference_geometry_infers_sampling_and_mirror_axis(self):
        rng = np.random.default_rng(4)
        soil = np.column_stack([
            rng.uniform(-2.0, 3.0, 2000),
            rng.uniform(0.0, 0.2, 2000),
            rng.uniform(-0.5, 0.5, 2000),
        ])
        reference = np.column_stack([
            rng.uniform(-1.0, 2.0, 500),
            rng.uniform(0.0, 0.05, 500),
            rng.uniform(-0.1, 0.1, 500),
        ])
        cfg = configure_from_reference_geometry(soil, reference, BoreholeConfig510())
        self.assertEqual(cfg.chosen_axis, 0)
        self.assertTrue(cfg.do_mirror)
        self.assertEqual(cfg.mirror_axis, 1)
        self.assertAlmostEqual(cfg.frac_start, 0.2, delta=0.02)
        self.assertAlmostEqual(cfg.frac_end, 0.8, delta=0.02)
        self.assertGreaterEqual(cfg.n_slices, 35)

    def test_radial_mode_uses_coverage_and_spacing(self):
        self.assertEqual(classify_radial_mode(0.80, 1.5, 0.40, self.cfg), "partial")
        self.assertEqual(classify_radial_mode(0.80, 4.0, 0.40, self.cfg), "partial")
        self.assertEqual(classify_radial_mode(0.98, 4.0, 0.20, self.cfg), "stable")
        self.assertEqual(classify_radial_mode(0.98, 0.5, 0.40, self.cfg), "inner")

    def test_adaptive_route_requires_quality_margin(self):
        baseline = {
            "radial_mode": "inner", "selection_score": 0.70,
            "fit": {"resid_p90": 0.10}, "axes": (1.0, 1.0),
            "geometry_failed": False, "confidence_failed": False,
            "reference_failed": False, "route_failed": False,
        }
        weak_partial = {
            "radial_mode": "partial", "selection_score": 0.72,
            "fit": {"resid_p90": 0.10}, "axes": (1.0, 1.0),
            "route_failed": False,
        }
        strong_partial = {
            "radial_mode": "partial", "selection_score": 0.75,
            "fit": {"resid_p90": 0.10}, "axes": (1.0, 1.0),
            "route_failed": False,
        }
        candidates = [baseline, weak_partial, strong_partial]
        apply_adaptive_route_margin(candidates, "partial", self.cfg)
        self.assertTrue(weak_partial["route_failed"])
        self.assertFalse(strong_partial["route_failed"])

    def test_stable_route_cannot_trade_away_residual_accuracy(self):
        baseline = {
            "radial_mode": "inner", "selection_score": 0.70,
            "fit": {"resid_p90": 0.10}, "axes": (1.0, 1.0),
            "geometry_failed": False, "confidence_failed": False,
            "reference_failed": False, "route_failed": False,
        }
        stable = {
            "radial_mode": "stable", "selection_score": 0.90,
            "fit": {"resid_p90": 0.12}, "axes": (1.0, 1.0),
            "route_failed": False,
        }
        apply_adaptive_route_margin([baseline, stable], "stable", self.cfg)
        self.assertTrue(stable["route_failed"])


if __name__ == "__main__":
    unittest.main()
