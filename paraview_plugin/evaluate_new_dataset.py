import argparse
import json
import os

import numpy as np
import pyvista as pv

os.environ.setdefault("MPLCONFIGDIR", "/tmp/borehole_mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from borehole_core_510 import (
    BoreholeConfig510,
    configure_from_reference_geometry,
    reconstruct_borehole,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate borehole reconstruction with a drillhead overlay.")
    parser.add_argument("soil")
    parser.add_argument("drillhead")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--slices", type=int)
    parser.add_argument("--thickness", type=float)
    parser.add_argument("--output-dir", default="evaluation_output")
    args = parser.parse_args()

    soil = pv.read(args.soil)
    drillhead = pv.read(args.drillhead)
    cfg = configure_from_reference_geometry(
        np.asarray(soil.points), np.asarray(drillhead.points), BoreholeConfig510()
    )
    if args.start is not None:
        cfg.frac_start = args.start
    if args.end is not None:
        cfg.frac_end = args.end
    if args.slices is not None:
        cfg.n_slices = args.slices
    if args.thickness is not None:
        cfg.thickness_frac = args.thickness
    soil_displacement = None
    if "soil_disp" in soil.point_data:
        values = np.asarray(soil.point_data["soil_disp"])
        soil_displacement = np.linalg.norm(values, axis=1) if values.ndim == 2 else np.abs(values)
    mesh_points, mesh_faces, metrics = reconstruct_borehole(
        np.asarray(soil.points), cfg,
        reference_points=np.asarray(drillhead.points),
        point_signal=soil_displacement,
    )

    faces = np.column_stack([
        np.full(mesh_faces.shape[0], 3, dtype=np.int64),
        mesh_faces,
    ]).ravel()
    loft = pv.PolyData(mesh_points, faces)

    os.makedirs(args.output_dir, exist_ok=True)
    loft.save(os.path.join(args.output_dir, "candidate_loft.vtp"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)

    rng = np.random.default_rng(0)
    count = min(30000, soil.n_points)
    sample = np.asarray(soil.points)[rng.choice(soil.n_points, count, replace=False)]
    drill_faces = drillhead.faces.reshape(-1, 4)[:, 1:]

    figure = plt.figure(figsize=(16, 9))
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.2, c="#5d72b8", alpha=0.10)
    axes.add_collection3d(Poly3DCollection(
        np.asarray(drillhead.points)[drill_faces], facecolor="#d97735", alpha=0.35, edgecolor="none"
    ))
    axes.add_collection3d(Poly3DCollection(
        mesh_points[mesh_faces], facecolor="#19a974", alpha=0.85, edgecolor="none"
    ))
    bounds = np.asarray(soil.bounds).reshape(3, 2)
    axes.set_xlim(bounds[0])
    axes.set_ylim(bounds[1])
    axes.set_zlim(bounds[2])
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title("New dataset: soil, drillhead, and reconstructed candidate loft")
    axes.view_init(elev=22, azim=-65)
    figure.tight_layout()
    figure.savefig(os.path.join(args.output_dir, "overlay.png"), dpi=180)
    plt.close(figure)

    drill_points = np.asarray(drillhead.points)
    views = plt.figure(figsize=(16, 10))
    projections = [
        (0, 2, "Side view (x-z)", "x", "z"),
        (0, 1, "Top view (x-y)", "x", "y"),
        (1, 2, "End view (y-z)", "y", "z"),
    ]
    for panel, (xdim, ydim, title, xlabel, ylabel) in enumerate(projections, start=1):
        axis = views.add_subplot(2, 2, panel)
        axis.scatter(sample[:, xdim], sample[:, ydim], s=0.15, c="#5d72b8", alpha=0.08)
        axis.scatter(drill_points[:, xdim], drill_points[:, ydim], s=0.3, c="#d97735", alpha=0.25)
        axis.scatter(mesh_points[:, xdim], mesh_points[:, ydim], s=0.4, c="#19a974", alpha=0.35)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_aspect("equal", adjustable="box")
    summary = views.add_subplot(2, 2, 4)
    summary.axis("off")
    lines = [
        "Reconstruction metrics",
        "Observed slices: %d" % metrics.get("observed_slice_count", 0),
        "Interpolated slices: %d" % metrics.get("interpolated_slice_count", 0),
        "Mean inlier ratio: %.3f" % metrics.get("inlier_mean", float("nan")),
        "Mean confidence: %.3f" % metrics.get("slice_confidence_mean", float("nan")),
        "Mean p90 residual: %.5f" % metrics.get("resid_p90_mean", float("nan")),
        "Radial slices: %d" % metrics.get("selected_radial_slice_count", 0),
    ]
    summary.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=12)
    views.tight_layout()
    views.savefig(os.path.join(args.output_dir, "overlay_views.png"), dpi=180)
    plt.close(views)


if __name__ == "__main__":
    main()
