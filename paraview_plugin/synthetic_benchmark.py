import argparse
import csv
import json
import math
import os

import numpy as np
import pyvista as pv

os.environ.setdefault("MPLCONFIGDIR", "/tmp/borehole_mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from borehole_core_510 import (
    BoreholeConfig510,
    configure_from_reference_geometry,
    reconstruct_borehole,
)


def case_geometry(name, x):
    x = np.asarray(x, dtype=float)
    if name == "straight_circle":
        center_z = np.zeros_like(x)
        axis_y = np.full_like(x, 0.050)
        axis_z = np.full_like(x, 0.050)
    elif name == "curved_ellipse":
        center_z = 0.035 * np.sin(np.pi * (x - 0.25))
        axis_y = 0.043 + 0.005 * np.sin(2.0 * np.pi * x)
        axis_z = 0.062 + 0.006 * np.cos(np.pi * x)
    elif name == "edge_incomplete":
        center_z = 0.092 + 0.010 * np.sin(np.pi * x)
        axis_y = np.full_like(x, 0.045)
        axis_z = np.full_like(x, 0.070)
    else:
        raise ValueError("Unknown synthetic case: %s" % name)
    return center_z, axis_y, axis_z


def generate_case(name, point_count, seed):
    rng = np.random.default_rng(seed)
    bounds = np.array([[-0.20, 2.20], [0.0, 0.15], [-0.15, 0.15]])
    collected = []
    while sum(len(block) for block in collected) < point_count:
        count = max(point_count, 50000)
        points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, 3))
        active = (points[:, 0] >= 0.0) & (points[:, 0] <= 2.0)
        center_z, axis_y, axis_z = case_geometry(name, points[:, 0])
        rho = np.sqrt(
            (points[:, 1] / axis_y) ** 2 +
            ((points[:, 2] - center_z) / axis_z) ** 2
        )
        points = points[~(active & (rho < 1.0))]
        if name == "edge_incomplete":
            points = points[rng.random(len(points)) > 0.15]
            points += rng.normal(0.0, 0.0015, size=points.shape)
        collected.append(points)
    soil = np.vstack(collected)[:point_count]

    ref_x = np.linspace(0.0, 2.0, 96)
    theta = np.linspace(0.0, np.pi, 32, endpoint=True)
    reference = []
    for axis_position in ref_x:
        center_z, axis_y, axis_z = case_geometry(name, np.array([axis_position]))
        reference.append(np.column_stack([
            np.full_like(theta, axis_position),
            0.50 * axis_y[0] * np.sin(theta),
            center_z[0] + 0.50 * axis_z[0] * np.cos(theta),
        ]))
    return soil, np.vstack(reference)


def ground_truth_surface(name, n_axis=160, n_ring=180):
    axis_positions = np.linspace(0.0, 2.0, n_axis)
    theta = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    loops = []
    for axis_position in axis_positions:
        center_z, axis_y, axis_z = case_geometry(name, np.array([axis_position]))
        loops.append(np.column_stack([
            np.full_like(theta, axis_position),
            axis_y[0] * np.cos(theta),
            center_z[0] + axis_z[0] * np.sin(theta),
        ]))
    points = np.vstack(loops)
    faces = []
    for section in range(n_axis - 1):
        for index in range(n_ring):
            next_index = (index + 1) % n_ring
            a = section * n_ring + index
            b = section * n_ring + next_index
            c = (section + 1) * n_ring + next_index
            d = (section + 1) * n_ring + index
            faces.extend(([a, b, c], [a, c, d]))
    return points, np.asarray(faces, dtype=np.int64)


def polydata(points, triangles=None):
    if triangles is None:
        return pv.PolyData(points)
    faces = np.column_stack([
        np.full(triangles.shape[0], 3, dtype=np.int64), triangles
    ]).ravel()
    return pv.PolyData(points, faces)


def absolute_geometry_metrics(name, mesh_points, n_ring):
    positions = np.unique(np.round(mesh_points[:, 0], 10))
    center_errors = []
    major_errors = []
    minor_errors = []
    surface_errors = []
    for position in positions:
        ring = mesh_points[np.isclose(mesh_points[:, 0], position, atol=1e-9)]
        if ring.shape[0] < 0.8 * n_ring:
            continue
        yz = ring[:, 1:3]
        estimated_center = np.mean(yz, axis=0)
        covariance = np.cov((yz - estimated_center).T)
        estimated_axes = np.sqrt(2.0 * np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        true_z, true_y_axis, true_z_axis = case_geometry(name, np.array([position]))
        true_center = np.array([0.0, true_z[0]])
        true_axes = np.sort([true_y_axis[0], true_z_axis[0]])[::-1]
        mean_radius = float(np.mean(true_axes))
        center_errors.append(np.linalg.norm(estimated_center - true_center) / mean_radius)
        major_errors.append(abs(estimated_axes[0] - true_axes[0]) / true_axes[0])
        minor_errors.append(abs(estimated_axes[1] - true_axes[1]) / true_axes[1])
        rho = np.sqrt(
            (ring[:, 1] / true_y_axis[0]) ** 2 +
            ((ring[:, 2] - true_z[0]) / true_z_axis[0]) ** 2
        )
        surface_errors.extend((np.abs(rho - 1.0) * mean_radius).tolist())
    if not center_errors:
        raise RuntimeError("No reconstructed rings were available for absolute validation")
    return {
        "ground_truth_ring_count": len(center_errors),
        "center_error_normalized_mean": float(np.mean(center_errors)),
        "center_error_normalized_max": float(np.max(center_errors)),
        "major_axis_relative_error_mean": float(np.mean(major_errors)),
        "major_axis_relative_error_max": float(np.max(major_errors)),
        "minor_axis_relative_error_mean": float(np.mean(minor_errors)),
        "minor_axis_relative_error_max": float(np.max(minor_errors)),
        "surface_error_mean": float(np.mean(surface_errors)),
        "surface_error_p95": float(np.percentile(surface_errors, 95)),
        "surface_error_max": float(np.max(surface_errors)),
    }


def run_case(name, point_count, seed, output_dir):
    soil, reference = generate_case(name, point_count, seed)
    cfg = configure_from_reference_geometry(soil, reference, BoreholeConfig510())
    mesh_points, mesh_faces, metrics = reconstruct_borehole(
        soil, cfg, reference_points=reference
    )
    metrics.update(absolute_geometry_metrics(name, mesh_points, cfg.n_ring))
    metrics["case"] = name
    metrics["requested_slice_count"] = cfg.n_slices
    metrics["fraction_start"] = cfg.frac_start
    metrics["fraction_end"] = cfg.frac_end
    metrics["thickness_fraction"] = cfg.thickness_frac

    truth_points, truth_faces = ground_truth_surface(name)
    polydata(soil).save(os.path.join(output_dir, "%s_soil.vtp" % name))
    polydata(reference).save(os.path.join(output_dir, "%s_reference.vtp" % name))
    polydata(mesh_points, mesh_faces).save(os.path.join(output_dir, "%s_reconstruction.vtp" % name))
    polydata(truth_points, truth_faces).save(os.path.join(output_dir, "%s_ground_truth.vtp" % name))
    return metrics


def write_outputs(results, output_dir):
    with open(os.path.join(output_dir, "synthetic_metrics.json"), "w") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
    columns = [
        "case", "observed_slice_fraction", "inlier_mean", "slice_confidence_mean",
        "center_error_normalized_mean", "major_axis_relative_error_mean",
        "minor_axis_relative_error_mean", "surface_error_mean", "surface_error_p95",
    ]
    with open(os.path.join(output_dir, "synthetic_metrics.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in columns})

    with open(os.path.join(output_dir, "SYNTHETIC_BENCHMARK.md"), "w") as stream:
        stream.write("# Synthetic Ground-Truth Benchmark\n\n")
        stream.write("| Case | Coverage | Center error | Major-axis error | Minor-axis error | Surface p95 |\n")
        stream.write("|---|---:|---:|---:|---:|---:|\n")
        for result in results:
            stream.write(
                "| {case} | {coverage:.1%} | {center:.2%} | {major:.2%} | "
                "{minor:.2%} | {surface:.5f} |\n".format(
                    case=result["case"],
                    coverage=result["observed_slice_fraction"],
                    center=result["center_error_normalized_mean"],
                    major=result["major_axis_relative_error_mean"],
                    minor=result["minor_axis_relative_error_mean"],
                    surface=result["surface_error_p95"],
                )
            )

    labels = [result["case"].replace("_", " ") for result in results]
    values = {
        "Center": [result["center_error_normalized_mean"] for result in results],
        "Major axis": [result["major_axis_relative_error_mean"] for result in results],
        "Minor axis": [result["minor_axis_relative_error_mean"] for result in results],
    }
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(11, 6))
    for index, (label, data) in enumerate(values.items()):
        axis.bar(x + (index - 1) * width, data, width, label=label)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Relative error")
    axis.set_title("Synthetic absolute-accuracy benchmark")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "synthetic_accuracy.png"), dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Run synthetic borehole ground-truth benchmarks.")
    parser.add_argument("--output-dir", default="synthetic_benchmark_output")
    parser.add_argument("--points", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cases = ("straight_circle", "curved_ellipse", "edge_incomplete")
    results = [
        run_case(name, args.points, args.seed + index, args.output_dir)
        for index, name in enumerate(cases)
    ]
    write_outputs(results, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
