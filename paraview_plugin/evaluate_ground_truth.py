import argparse
import csv
import json
import os

import numpy as np
import pyvista as pv

os.environ.setdefault("MPLCONFIGDIR", "/tmp/borehole_mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_surface(path):
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate().clean()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError("Surface contains no polygons: %s" % path)
    return mesh


def sample_surface(mesh, count, rng):
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    triangles = np.asarray(mesh.points)[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 1e-18
    triangles = triangles[valid]
    areas = areas[valid]
    if not len(triangles):
        raise ValueError("Surface contains no nondegenerate triangles")
    chosen = rng.choice(len(triangles), size=int(count), p=areas / np.sum(areas))
    selected = triangles[chosen]
    u = rng.random(int(count))
    v = rng.random(int(count))
    reflected = u + v > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    return (
        selected[:, 0] +
        u[:, None] * (selected[:, 1] - selected[:, 0]) +
        v[:, None] * (selected[:, 2] - selected[:, 0])
    )


def point_to_surface_distances(points, surface):
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    measured = cloud.compute_implicit_distance(surface, inplace=False)
    return np.abs(np.asarray(measured.point_data["implicit_distance"], dtype=float))


def close_surface(mesh):
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    if boundary.n_cells == 0:
        return mesh
    diagonal = float(np.linalg.norm(np.asarray(mesh.bounds)[1::2] - np.asarray(mesh.bounds)[::2]))
    return mesh.fill_holes(max(2.0 * diagonal, 1e-9)).triangulate().clean()


def enclosed_mask(points, surface):
    selected = pv.PolyData(points).select_enclosed_points(
        surface,
        tolerance=1e-6,
        check_surface=False,
    )
    return np.asarray(selected.point_data["SelectedPoints"], dtype=np.uint8).astype(bool)


def section_geometry(mesh, axis, position):
    origin = np.zeros(3, dtype=float)
    origin[axis] = position
    normal = np.zeros(3, dtype=float)
    normal[axis] = 1.0
    section = mesh.slice(normal=normal, origin=origin)
    if section.n_points < 8:
        return None
    dims = [idx for idx in range(3) if idx != axis]
    points = np.asarray(section.points)[:, dims]
    center = np.mean(points, axis=0)
    covariance = np.cov((points - center).T)
    axes = np.sqrt(2.0 * np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    if np.any(axes <= 1e-12):
        return None
    return center, axes


def cross_section_metrics(predicted, truth, axis, section_count):
    pred_bounds = np.asarray(predicted.bounds, dtype=float)
    truth_bounds = np.asarray(truth.bounds, dtype=float)
    lower = max(pred_bounds[2 * axis], truth_bounds[2 * axis])
    upper = min(pred_bounds[2 * axis + 1], truth_bounds[2 * axis + 1])
    span = upper - lower
    if span <= 1e-12:
        raise ValueError("Prediction and ground truth have no overlapping axial interval")
    margin = min(0.01 * span, 0.25 * span)
    positions = np.linspace(lower + margin, upper - margin, int(section_count))
    center_errors = []
    center_normalized = []
    major_errors = []
    minor_errors = []
    truth_radii = []
    for position in positions:
        pred_geometry = section_geometry(predicted, axis, position)
        truth_geometry = section_geometry(truth, axis, position)
        if pred_geometry is None or truth_geometry is None:
            continue
        pred_center, pred_axes = pred_geometry
        truth_center, truth_axes = truth_geometry
        radius = max(float(np.mean(truth_axes)), 1e-12)
        center_error = float(np.linalg.norm(pred_center - truth_center))
        center_errors.append(center_error)
        center_normalized.append(center_error / radius)
        major_errors.append(abs(pred_axes[0] - truth_axes[0]) / truth_axes[0])
        minor_errors.append(abs(pred_axes[1] - truth_axes[1]) / truth_axes[1])
        truth_radii.append(radius)
    if not center_errors:
        raise RuntimeError("No valid paired cross-sections were found")
    return {
        "paired_section_count": len(center_errors),
        "truth_mean_radius": float(np.mean(truth_radii)),
        "centerline_error_mean": float(np.mean(center_errors)),
        "centerline_error_p95": float(np.percentile(center_errors, 95)),
        "centerline_error_max": float(np.max(center_errors)),
        "centerline_error_normalized_mean": float(np.mean(center_normalized)),
        "centerline_error_normalized_p95": float(np.percentile(center_normalized, 95)),
        "major_axis_relative_error_mean": float(np.mean(major_errors)),
        "major_axis_relative_error_p95": float(np.percentile(major_errors, 95)),
        "minor_axis_relative_error_mean": float(np.mean(minor_errors)),
        "minor_axis_relative_error_p95": float(np.percentile(minor_errors, 95)),
    }


def surface_distance_metrics(predicted, truth, sample_count, rng, scale):
    pred_samples = sample_surface(predicted, sample_count, rng)
    truth_samples = sample_surface(truth, sample_count, rng)
    pred_to_truth = point_to_surface_distances(pred_samples, truth)
    truth_to_pred = point_to_surface_distances(truth_samples, predicted)
    combined = np.concatenate([pred_to_truth, truth_to_pred])
    return {
        "surface_pred_to_truth_mean": float(np.mean(pred_to_truth)),
        "surface_pred_to_truth_p95": float(np.percentile(pred_to_truth, 95)),
        "surface_truth_to_pred_mean": float(np.mean(truth_to_pred)),
        "surface_truth_to_pred_p95": float(np.percentile(truth_to_pred, 95)),
        "surface_symmetric_mean": float(np.mean(combined)),
        "surface_symmetric_p95": float(np.percentile(combined, 95)),
        "surface_hausdorff_approx": float(np.max(combined)),
        "surface_symmetric_mean_normalized": float(np.mean(combined) / scale),
        "surface_symmetric_p95_normalized": float(np.percentile(combined, 95) / scale),
    }


def overlap_metrics(predicted, truth, sample_count, rng):
    predicted_closed = close_surface(predicted)
    truth_closed = close_surface(truth)
    pred_bounds = np.asarray(predicted_closed.bounds, dtype=float)
    truth_bounds = np.asarray(truth_closed.bounds, dtype=float)
    lower = np.minimum(pred_bounds[::2], truth_bounds[::2])
    upper = np.maximum(pred_bounds[1::2], truth_bounds[1::2])
    padding = 0.01 * np.maximum(upper - lower, 1e-12)
    lower -= padding
    upper += padding
    samples = rng.uniform(lower, upper, size=(int(sample_count), 3))
    pred_inside = enclosed_mask(samples, predicted_closed)
    truth_inside = enclosed_mask(samples, truth_closed)
    intersection = int(np.count_nonzero(pred_inside & truth_inside))
    union = int(np.count_nonzero(pred_inside | truth_inside))
    pred_count = int(np.count_nonzero(pred_inside))
    truth_count = int(np.count_nonzero(truth_inside))
    box_volume = float(np.prod(upper - lower))
    pred_volume = box_volume * pred_count / float(sample_count)
    truth_volume = box_volume * truth_count / float(sample_count)
    return {
        "overlap_iou": intersection / float(max(union, 1)),
        "overlap_dice": 2.0 * intersection / float(max(pred_count + truth_count, 1)),
        "predicted_volume_mc": pred_volume,
        "truth_volume_mc": truth_volume,
        "volume_relative_error_mc": abs(pred_volume - truth_volume) / max(truth_volume, 1e-12),
        "predicted_volume_mesh": float(abs(predicted_closed.volume)),
        "truth_volume_mesh": float(abs(truth_closed.volume)),
    }


def evaluate_pair(predicted_path, truth_path, args):
    predicted = load_surface(predicted_path)
    truth = load_surface(truth_path)
    truth_spans = np.asarray(truth.bounds)[1::2] - np.asarray(truth.bounds)[::2]
    axis = int(np.argmax(truth_spans)) if args.axis is None else int(args.axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    rng = np.random.default_rng(args.seed)
    metrics = cross_section_metrics(predicted, truth, axis, args.sections)
    metrics.update(surface_distance_metrics(
        predicted, truth, args.surface_samples, rng, metrics["truth_mean_radius"]
    ))
    metrics.update(overlap_metrics(predicted, truth, args.volume_samples, rng))
    pred_bounds = np.asarray(predicted.bounds, dtype=float)
    truth_bounds = np.asarray(truth.bounds, dtype=float)
    truth_axis_length = max(
        float(truth_bounds[2 * axis + 1] - truth_bounds[2 * axis]), 1e-12
    )
    axis_start_error = float(abs(pred_bounds[2 * axis] - truth_bounds[2 * axis]))
    axis_end_error = float(abs(pred_bounds[2 * axis + 1] - truth_bounds[2 * axis + 1]))
    metrics.update({
        "case": os.path.basename(predicted_path).replace("_reconstruction.vtp", ""),
        "predicted_path": os.path.abspath(predicted_path),
        "truth_path": os.path.abspath(truth_path),
        "axis": axis,
        "truth_axis_length": truth_axis_length,
        "axis_start_error": axis_start_error,
        "axis_end_error": axis_end_error,
        "axis_start_error_normalized": axis_start_error / truth_axis_length,
        "axis_end_error_normalized": axis_end_error / truth_axis_length,
        "axis_extent_error_normalized_max": max(axis_start_error, axis_end_error) / truth_axis_length,
        "surface_sample_count_each_direction": int(args.surface_samples),
        "volume_sample_count": int(args.volume_samples),
    })
    return metrics


def write_outputs(results, output_dir):
    with open(os.path.join(output_dir, "ground_truth_metrics.json"), "w") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
    columns = [
        "case", "centerline_error_normalized_mean",
        "major_axis_relative_error_mean", "minor_axis_relative_error_mean",
        "surface_symmetric_mean_normalized", "surface_symmetric_p95_normalized",
        "overlap_iou", "overlap_dice", "volume_relative_error_mc",
        "axis_start_error", "axis_end_error", "axis_start_error_normalized",
        "axis_end_error_normalized", "axis_extent_error_normalized_max",
    ]
    with open(os.path.join(output_dir, "ground_truth_metrics.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in columns})
    with open(os.path.join(output_dir, "GROUND_TRUTH_REPORT.md"), "w") as stream:
        stream.write("# Ground-Truth Borehole Evaluation\n\n")
        stream.write("No rigid registration was applied. Metrics therefore include location and trajectory error.\n\n")
        stream.write("| Case | Centerline | Major axis | Minor axis | Surface p95 | IoU | Dice | Volume error | Max extent |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for result in results:
            stream.write(
                "| {case} | {center:.2%} | {major:.2%} | {minor:.2%} | {surface:.2%} | "
                "{iou:.3f} | {dice:.3f} | {volume:.2%} | {extent:.2%} |\n".format(
                    case=result["case"],
                    center=result["centerline_error_normalized_mean"],
                    major=result["major_axis_relative_error_mean"],
                    minor=result["minor_axis_relative_error_mean"],
                    surface=result["surface_symmetric_p95_normalized"],
                    iou=result["overlap_iou"],
                    dice=result["overlap_dice"],
                    volume=result["volume_relative_error_mc"],
                    extent=result["axis_extent_error_normalized_max"],
                )
            )
        stream.write("\nSurface Hausdorff values are approximate because they use sampled points. ")
        stream.write("IoU, Dice, and volume use deterministic Monte Carlo occupancy.\n")

    labels = [result["case"].replace("_", " ") for result in results]
    series = {
        "Centerline": [r["centerline_error_normalized_mean"] for r in results],
        "Major axis": [r["major_axis_relative_error_mean"] for r in results],
        "Minor axis": [r["minor_axis_relative_error_mean"] for r in results],
        "Surface p95": [r["surface_symmetric_p95_normalized"] for r in results],
    }
    x = np.arange(len(labels))
    width = 0.20
    figure, axis = plt.subplots(figsize=(11, 6))
    for index, (label, values) in enumerate(series.items()):
        axis.bar(x + (index - 1.5) * width, values, width, label=label)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Relative error")
    axis.set_title("Ground-truth borehole evaluation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "ground_truth_accuracy.png"), dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Compare reconstructed borehole surfaces with ground-truth meshes."
    )
    parser.add_argument(
        "pairs", nargs="+",
        help="Alternating paths: PREDICTED_A TRUTH_A [PREDICTED_B TRUTH_B ...]",
    )
    parser.add_argument("--output-dir", default="ground_truth_evaluation")
    parser.add_argument("--axis", type=int, default=None)
    parser.add_argument("--sections", type=int, default=80)
    parser.add_argument("--surface-samples", type=int, default=50000)
    parser.add_argument("--volume-samples", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if len(args.pairs) < 2 or len(args.pairs) % 2:
        parser.error("Provide one or more PREDICTED GROUND_TRUTH path pairs.")
    os.makedirs(args.output_dir, exist_ok=True)
    results = [
        evaluate_pair(args.pairs[index], args.pairs[index + 1], args)
        for index in range(0, len(args.pairs), 2)
    ]
    write_outputs(results, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
