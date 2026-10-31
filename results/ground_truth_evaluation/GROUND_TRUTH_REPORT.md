# Ground-Truth Borehole Evaluation

No rigid registration was applied. Metrics therefore include location and trajectory error.

| Case | Centerline | Major axis | Minor axis | Surface p95 | IoU | Dice | Volume error | Max extent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| straight_circle | 1.68% | 3.32% | 2.32% | 8.01% | 0.936 | 0.967 | 3.47% | 0.00% |
| curved_ellipse | 0.88% | 8.26% | 8.53% | 10.65% | 0.895 | 0.945 | 4.51% | 1.92% |
| edge_incomplete | 2.58% | 5.35% | 12.93% | 16.37% | 0.850 | 0.919 | 3.55% | 1.96% |

Surface Hausdorff values are approximate because they use sampled points. IoU, Dice, and volume use deterministic Monte Carlo occupancy.
