# Results

This directory contains lightweight outputs used to support the empirical anchors in the review.

## Included

- `cad_available/outputs/`: CAD-at-test-time and detector-output CSV/JSON summaries.
- `cad_available/yolo_runs/`: selected YOLO training diagnostics, including `results.csv`, run arguments, PR/F1 curves, and confusion matrices.
- `cad_unavailable/`: per-method MVTec AD / VisA anomaly-detection result summaries and normal-reference budget ablations.

## Excluded

The full training directories, Lightning logs, checkpoints, cached features, raw predictions containing image data, and benchmark datasets are not included. Those files are large and are not appropriate for a lightweight GitHub project repository.
