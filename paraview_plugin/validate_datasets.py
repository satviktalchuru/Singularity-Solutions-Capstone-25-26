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

from borehole_core_510 import (
    BoreholeConfig510,
    configure_from_reference_geometry,
    reconstruct_borehole,
)


REPORT_KEYS = (
    "observed_slice_count",
    "observed_slice_fraction",
    "interpolated_slice_count",
    "interpolated_slice_fraction",
    "inlier_mean",
    "slice_confidence_mean",
    "resid_p90_mean",
    "resid_p90_normalized",
    "center_step_normalized_median",
    "center_step_normalized_max",
    "major_axis_change_median",
    "major_axis_change_max",
    "minor_axis_change_median",
    "minor_axis_change_max",
    "reference_spacing_ratio_median",
    "baseline_radial_layers",
    "global_tracking_used",
    "selected_radial_inner_slice_count",
    "selected_radial_stable_slice_count",
    "selected_radial_partial_slice_count",
)


def point_signal(dataset):
    if "soil_disp" not in dataset.point_data:
        return None
    values = np.asarray(dataset.point_data["soil_disp"])
    return np.linalg.norm(values, axis=1) if values.ndim == 2 else np.abs(values)


def dataset_name(path):
    stem = os.path.splitext(os.path.basename(path))[0].replace(" (1)", "")
    name = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def save_surface(path, points, triangles):
    faces = np.column_stack([
        np.full(triangles.shape[0], 3, dtype=np.int64), triangles
    ]).ravel()
    pv.PolyData(points, faces).save(path)


def run_case(name, soil_path, cfg, output_dir, reference_path=None):
    soil = pv.read(soil_path)
    reference = pv.read(reference_path) if reference_path else None
    points, faces, metrics = reconstruct_borehole(
        np.asarray(soil.points),
        cfg,
        reference_points=np.asarray(reference.points) if reference is not None else None,
        point_signal=point_signal(soil),
    )
    metrics["dataset"] = name
    metrics["requested_slice_count"] = cfg.n_slices
    metrics["fraction_start"] = cfg.frac_start
    metrics["fraction_end"] = cfg.frac_end
    metrics["thickness_fraction"] = cfg.thickness_frac
    save_surface(os.path.join(output_dir, "%s_loft.vtp" % name), points, faces)
    return metrics


def reference_scaled_config(soil_path, reference_path):
    soil = pv.read(soil_path)
    reference = pv.read(reference_path)
    return configure_from_reference_geometry(
        np.asarray(soil.points),
        np.asarray(reference.points),
        BoreholeConfig510(),
    )


def write_report(output_dir, results):
    json_path = os.path.join(output_dir, "validation_metrics.json")
    with open(json_path, "w") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)

    csv_path = os.path.join(output_dir, "validation_metrics.csv")
    columns = ["dataset"] + list(REPORT_KEYS)
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in columns})

    markdown_path = os.path.join(output_dir, "VALIDATION_REPORT.md")
    with open(markdown_path, "w") as stream:
        stream.write("# Borehole Reconstruction Validation\n\n")
        stream.write(
            "These metrics measure internal consistency and point-cloud support. "
            "They do not measure absolute geometric error because surveyed borehole "
            "ground truth is not available.\n\n"
        )
        stream.write(
            "| Dataset | Observed coverage | Interpolated | Inlier ratio | Confidence | "
            "Normalized p90 residual | Median center step |\n"
        )
        stream.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for result in results:
            stream.write(
                "| {dataset} | {coverage:.1%} | {interp} | {inlier:.3f} | "
                "{confidence:.3f} | {residual:.3f} | {center:.3f} |\n".format(
                    dataset=result["dataset"],
                    coverage=result.get("observed_slice_fraction", float("nan")),
                    interp=result.get("interpolated_slice_count", 0),
                    inlier=result.get("inlier_mean", float("nan")),
                    confidence=result.get("slice_confidence_mean", float("nan")),
                    residual=result.get("resid_p90_normalized", float("nan")),
                    center=result.get("center_step_normalized_median", float("nan")),
                )
            )
        stream.write("\n## Interpretation\n\n")
        stream.write("- Higher observed coverage, inlier ratio, and confidence are better.\n")
        stream.write("- Lower interpolation, normalized residual, and abrupt continuity changes are better.\n")
        stream.write("- Reference-assisted results use the drillhead for location and scale only; soil points define the wall.\n")
        stream.write("- Absolute accuracy still requires a known borehole surface or surveyed dimensions.\n")

    labels = [result["dataset"] for result in results]
    series = [
        ("Observed coverage", "observed_slice_fraction"),
        ("Inlier ratio", "inlier_mean"),
        ("Mean confidence", "slice_confidence_mean"),
        ("1 - normalized residual", "resid_p90_normalized"),
    ]
    x = np.arange(len(labels))
    width = 0.18
    figure, axis = plt.subplots(figsize=(11, 6))
    for index, (title, key) in enumerate(series):
        values = [result.get(key, 0.0) for result in results]
        if key == "resid_p90_normalized":
            values = [max(0.0, 1.0 - value) for value in values]
        axis.bar(x + (index - 1.5) * width, values, width, label=title)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Internal quality score")
    axis.set_title("Cross-dataset internal validation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "validation_comparison.png"), dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Validate soil and drillhead dataset pairs.")
    parser.add_argument(
        "pairs", nargs="+",
        help="Alternating paths: SOIL_A REFERENCE_A [SOIL_B REFERENCE_B ...]",
    )
    parser.add_argument("--output-dir", default="validation_output")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if len(args.pairs) < 2 or len(args.pairs) % 2:
        parser.error("Provide one or more SOIL REFERENCE path pairs.")
    results = []
    for index in range(0, len(args.pairs), 2):
        soil_path, reference_path = args.pairs[index:index + 2]
        cfg = reference_scaled_config(soil_path, reference_path)
        results.append(run_case(
            dataset_name(soil_path),
            soil_path,
            cfg,
            args.output_dir,
            reference_path=reference_path,
        ))
    write_report(args.output_dir, results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
