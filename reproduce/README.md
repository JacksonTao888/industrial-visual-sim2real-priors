# Reproduction Notes

The empirical component is designed as a set of review anchors, not a single leaderboard. The scripts in `experiments/` reproduce the major branches used in the paper once the external datasets are available locally.

## Environment

The scripts were developed in a Python environment with common scientific and vision packages, including:

```text
numpy
pandas
opencv-python
Pillow
matplotlib
scikit-learn
torch
torchvision
ultralytics
anomalib
```

Some CAD-at-test-time scripts also require BOP/T-LESS-style metadata and, for MegaPose-style diagnostics, a working MegaPose installation or exported predictions.

## CAD-Guided Anchors

The CAD-as-renderer branch uses T-LESS/BOP synthetic PBR data and held-out real Primesense images. The main scripts are:

```text
experiments/bop_to_coco_detection.py
experiments/coco_to_yolo_detection.py
experiments/train_yolo_tless_*.sh
experiments/summarize_coco_detection.py
```

The CAD-at-test-time branch uses rendered CAD overlays, mask IoU, ROI scoring, and depth-consistency diagnostics:

```text
experiments/render_cad_gt_pose_overlay.py
experiments/render_cad_mask_iou.py
experiments/evaluate_gt_pose_cad_mask_iou_heldout.py
experiments/export_b6_real_predictions.py
experiments/score_b6_predictions_with_oracle_cad.py
experiments/score_megapose_depth_consistency.py
```

Lightweight outputs are in `results/cad_available/`.

## CAD-Unavailable Anchors

The CAD-unavailable branch uses MVTec AD and VisA with representative anomaly-detection families:

```text
experiments/run_patchcore_anomalib.py
experiments/run_efficientad_anomalib.py
experiments/run_winclip_anomalib.py
experiments/run_anomalydino_anomalib.py
experiments/run_supersimplenet_anomalib.py
experiments/run_no_cad_budget_ablation.sh
```

Summary outputs are in `results/cad_unavailable/`.

## Figures

Aggregate, non-dataset figures are in `figures/aggregate/`. Figures or contact sheets containing benchmark images are intentionally excluded from the repository.
