# Experiments

This directory contains the scripts used to prepare datasets, run representative anchors, score outputs, and summarize results for the review.

The scripts are provided for transparency and reuse. They expect external datasets to be downloaded separately, as described in `data/README.md`.

The most important script families are:

- `train_yolo_tless_*.sh` and COCO/YOLO conversion scripts for CAD-as-renderer transfer.
- `render_cad_*`, `evaluate_*cad_mask_iou*`, and `score_*cad*` scripts for CAD-at-test-time diagnostics.
- `run_*_anomalib.py` and `run_*_all.sh` scripts for CAD-unavailable anomaly-detection anchors.
- `summarize_*` scripts for producing lightweight result tables.
