# Dataset Notes

This repository does not redistribute benchmark datasets. Download each dataset from its official source and follow the corresponding license and citation requirements.

## Datasets Used

- **T-LESS/BOP**: used for CAD-guided object detection, pose, and render-and-compare diagnostics.
- **MVTec AD**: used for CAD-unavailable industrial anomaly detection and segmentation.
- **VisA**: used for CAD-unavailable anomaly detection under varied objects and defect categories.
- **DTD / Imagenette**: used only as auxiliary sources for selected synthetic-anomaly or background/texture experiments.

## Expected Local Layout

The experiment scripts were run with a local layout similar to:

```text
datasets/
  tless/
  mvtec_ad/
  visa/
  dtd/
  imagenette/
```

If you use a different layout, update the path arguments or constants in the relevant scripts under `experiments/`.

## Redistribution Boundary

The GitHub repository includes lightweight result summaries and aggregate plots. It excludes:

- raw benchmark images, masks, depth maps, CAD meshes, and archives;
- generated dataset copies or converted COCO/YOLO datasets;
- qualitative contact sheets or overlays containing benchmark images;
- trained checkpoints and full run directories.

This keeps the repository focused on reproducibility without repackaging third-party datasets.
