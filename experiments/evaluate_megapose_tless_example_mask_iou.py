#!/usr/bin/env python3
"""Evaluate one MegaPose T-LESS example by rendering predicted CAD masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from render_cad_mask_iou import (
    blend_mask,
    load_mask,
    mask_iou,
    parse_ply_mesh,
    render_scene_masks,
)


COLORS = np.array(
    [
        [255, 76, 76],
        [76, 179, 255],
        [76, 220, 120],
        [255, 212, 64],
        [207, 107, 255],
        [255, 143, 64],
        [64, 235, 218],
        [245, 105, 180],
    ],
    dtype=np.uint8,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def quat_xyzw_to_rotation(quat: list[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat]
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError(f"Invalid zero quaternion: {quat}")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def obj_id_from_label(label: str) -> int:
    if not label.startswith("obj_"):
        raise ValueError(f"Expected T-LESS label like obj_000001, got {label!r}")
    return int(label.split("_")[-1])


def load_megapose_predictions(path: Path) -> list[dict]:
    predictions = []
    for idx, item in enumerate(load_json(path)):
        label = item["label"]
        quat_xyzw, trans_m = item["TWO"]
        rotation = quat_xyzw_to_rotation(quat_xyzw)
        translation_mm = np.asarray(trans_m, dtype=np.float32) * 1000.0
        predictions.append(
            {
                "pred_index": idx,
                "label": label,
                "obj_id": obj_id_from_label(label),
                "cam_R_m2c": rotation.reshape(-1).tolist(),
                "cam_t_m2c": translation_mm.tolist(),
            }
        )
    return predictions


def load_scene_gt(dataset_root: Path, split: str, scene_id: int, image_id: int):
    scene_dir = dataset_root / split / f"{scene_id:06d}"
    key = str(image_id)
    scene_gt = load_json(scene_dir / "scene_gt.json")
    scene_gt_info = load_json(scene_dir / "scene_gt_info.json")
    gt_instances = []
    for inst_idx, (instance, info) in enumerate(zip(scene_gt[key], scene_gt_info[key])):
        obj_id = int(instance["obj_id"])
        image_stem = f"{image_id:06d}"
        gt_instances.append(
            {
                "instance_index": inst_idx,
                "obj_id": obj_id,
                "visib_fract": float(info.get("visib_fract", 0.0)),
                "full_mask": load_mask(scene_dir / "mask" / f"{image_stem}_{inst_idx:06d}.png"),
                "visible_mask": load_mask(
                    scene_dir / "mask_visib" / f"{image_stem}_{inst_idx:06d}.png"
                ),
            }
        )
    return gt_instances


def make_overlay(
    rgb: np.ndarray,
    rows: list[dict],
    pred_visible: list[np.ndarray],
    gt_instances: list[dict],
) -> np.ndarray:
    overlay = rgb.copy()
    for row, pred_mask in zip(rows, pred_visible):
        color = COLORS[int(row["pred_index"]) % len(COLORS)]
        overlay = blend_mask(overlay, pred_mask, color, 0.45)
        gt_idx = int(row["best_gt_instance_index"])
        if gt_idx >= 0:
            gt_mask = gt_instances[gt_idx]["visible_mask"]
            misses = np.logical_and(gt_mask, ~pred_mask)
            extras = np.logical_and(pred_mask, ~gt_mask)
            overlay[misses] = np.array([255, 255, 255], dtype=np.uint8)
            overlay[extras] = np.array([255, 0, 255], dtype=np.uint8)
    return overlay


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def mask_fraction(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    denominator = int(denominator_mask.sum())
    if denominator == 0:
        return 0.0
    return float(np.logical_and(numerator_mask, denominator_mask).sum() / denominator)


def full_mask_fraction(full_mask: np.ndarray, visible_mask: np.ndarray) -> float:
    full_area = int(full_mask.sum())
    if full_area == 0:
        return 0.0
    return float(visible_mask.sum() / full_area)


def match_same_class_gt_by_full_mask(
    full_mask: np.ndarray,
    same_class_gt: list[dict],
) -> tuple[dict | None, float]:
    best_gt = None
    best_full_iou = -1.0
    for gt in same_class_gt:
        full_iou = mask_iou(full_mask, gt["full_mask"])
        if full_iou > best_full_iou:
            best_full_iou = full_iou
            best_gt = gt
    return best_gt, best_full_iou


def compute_prediction_metrics(
    *,
    full_mask: np.ndarray,
    visible_mask: np.ndarray,
    best_gt: dict | None,
) -> dict:
    pred_full_area = int(full_mask.sum())
    pred_visible_area = int(visible_mask.sum())
    if best_gt is None:
        return {
            "best_gt_instance_index": -1,
            "best_gt_visib_fract": 0.0,
            "pred_full_area": pred_full_area,
            "pred_visible_area": pred_visible_area,
            "gt_full_area": 0,
            "gt_visible_area": 0,
            "full_mask_iou": 0.0,
            "visible_mask_iou": 0.0,
            "scene_visible_mask_iou": 0.0,
            "full_mask_vs_gt_visible_iou": 0.0,
            "gt_visible_coverage_by_full_mask": 0.0,
            "pred_full_precision_wrt_gt_visible": 0.0,
            "pred_visible_fraction": full_mask_fraction(full_mask, visible_mask),
        }

    gt_full = best_gt["full_mask"]
    gt_visible = best_gt["visible_mask"]
    scene_visible_iou = mask_iou(visible_mask, gt_visible)
    return {
        "best_gt_instance_index": int(best_gt["instance_index"]),
        "best_gt_visib_fract": float(best_gt["visib_fract"]),
        "pred_full_area": pred_full_area,
        "pred_visible_area": pred_visible_area,
        "gt_full_area": int(gt_full.sum()),
        "gt_visible_area": int(gt_visible.sum()),
        "full_mask_iou": mask_iou(full_mask, gt_full),
        "visible_mask_iou": scene_visible_iou,
        "scene_visible_mask_iou": scene_visible_iou,
        "full_mask_vs_gt_visible_iou": mask_iou(full_mask, gt_visible),
        "gt_visible_coverage_by_full_mask": mask_fraction(full_mask, gt_visible),
        "pred_full_precision_wrt_gt_visible": mask_fraction(gt_visible, full_mask),
        "pred_visible_fraction": full_mask_fraction(full_mask, visible_mask),
    }


METRIC_FIELDNAMES = [
    "best_gt_instance_index",
    "best_gt_visib_fract",
    "pred_full_area",
    "pred_visible_area",
    "gt_full_area",
    "gt_visible_area",
    "full_mask_iou",
    "visible_mask_iou",
    "scene_visible_mask_iou",
    "full_mask_vs_gt_visible_iou",
    "gt_visible_coverage_by_full_mask",
    "pred_full_precision_wrt_gt_visible",
    "pred_visible_fraction",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--example-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to experiments/outputs/megapose_examples/"
            "tless_scene_000001_image_000001."
        ),
    )
    parser.add_argument("--megapose-output", type=Path, default=None)
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--face-stride", type=int, default=1)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root
    dataset_root = project_root / "datasets" / "tless"
    models_dir = dataset_root / "models_eval"
    example_dir = args.example_dir or (
        project_root
        / "experiments"
        / "outputs"
        / "megapose_examples"
        / "tless_scene_000001_image_000001"
    )
    megapose_output = args.megapose_output or (example_dir / "outputs" / "object_data.json")
    manifest_path = example_dir / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    scene_id = args.scene_id if args.scene_id is not None else int(manifest.get("scene_id", 1))
    image_id = args.image_id if args.image_id is not None else int(manifest.get("image_id", 1))
    out_path = args.out or (
        project_root
        / "experiments"
        / "outputs"
        / f"megapose_tless_scene_{scene_id:06d}_image_{image_id:06d}_mask_iou.csv"
    )
    summary_path = out_path.with_suffix(".summary.json")
    overlay_path = out_path.with_suffix(".overlay.png")

    scene_dir = dataset_root / args.split / f"{scene_id:06d}"
    image_stem = f"{image_id:06d}"
    rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{image_stem}.png").convert("RGB"))
    height, width = rgb.shape[:2]
    scene_camera = load_json(scene_dir / "scene_camera.json")
    camera_matrix = np.asarray(scene_camera[str(image_id)]["cam_K"], dtype=np.float32).reshape(3, 3)

    predictions = load_megapose_predictions(megapose_output)
    if not predictions:
        raise RuntimeError(f"No MegaPose predictions found: {megapose_output}")

    mesh_cache = {}
    for prediction in predictions:
        obj_id = int(prediction["obj_id"])
        if obj_id not in mesh_cache:
            mesh_cache[obj_id] = parse_ply_mesh(models_dir / f"obj_{obj_id:06d}.ply")

    _, pred_full, pred_visible = render_scene_masks(
        instances=predictions,
        mesh_cache=mesh_cache,
        camera_matrix=camera_matrix,
        image_shape=(height, width),
        face_stride=args.face_stride,
    )
    gt_instances = load_scene_gt(dataset_root, args.split, scene_id, image_id)

    rows = []
    for prediction, full_mask, visible_mask in zip(predictions, pred_full, pred_visible):
        same_class_gt = [
            gt for gt in gt_instances if int(gt["obj_id"]) == int(prediction["obj_id"])
        ]
        best_gt, _ = match_same_class_gt_by_full_mask(full_mask, same_class_gt)
        metrics = compute_prediction_metrics(
            full_mask=full_mask,
            visible_mask=visible_mask,
            best_gt=best_gt,
        )

        rows.append(
            {
                "scene_id": scene_id,
                "image_id": image_id,
                "pred_index": int(prediction["pred_index"]),
                "label": prediction["label"],
                "obj_id": int(prediction["obj_id"]),
                **metrics,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_id",
        "image_id",
        "pred_index",
        "label",
        "obj_id",
        *METRIC_FIELDNAMES,
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    saved_overlay = None
    if args.save_overlay:
        overlay = make_overlay(rgb, rows, pred_visible, gt_instances)
        Image.fromarray(overlay).save(overlay_path)
        saved_overlay = str(overlay_path)

    summary = {
        "method_family": "Render-and-Compare / Test-Time CAD Use",
        "backbone_paper": "MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare, CoRL 2022",
        "note": (
            "Single-image MegaPose predicted poses rendered with the local CAD mask "
            "renderer and matched to same-class BOP GT masks by best visible-mask IoU."
        ),
        "example_dir": str(example_dir),
        "megapose_output": str(megapose_output),
        "split": args.split,
        "scene_id": scene_id,
        "image_id": image_id,
        "face_stride": args.face_stride,
        "predictions": len(rows),
        "full_mask_iou": summarize([float(row["full_mask_iou"]) for row in rows]),
        "visible_mask_iou": summarize([float(row["visible_mask_iou"]) for row in rows]),
        "scene_visible_mask_iou": summarize(
            [float(row["scene_visible_mask_iou"]) for row in rows]
        ),
        "full_mask_vs_gt_visible_iou": summarize(
            [float(row["full_mask_vs_gt_visible_iou"]) for row in rows]
        ),
        "gt_visible_coverage_by_full_mask": summarize(
            [float(row["gt_visible_coverage_by_full_mask"]) for row in rows]
        ),
        "pred_visible_fraction": summarize(
            [float(row["pred_visible_fraction"]) for row in rows]
        ),
        "csv": str(out_path),
        "overlay": saved_overlay,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"scene: {scene_id:06d}")
    print(f"image: {image_id:06d}")
    print(f"predictions: {len(rows)}")
    print(f"mean_full_mask_iou: {summary['full_mask_iou']['mean']:.4f}")
    print(f"mean_scene_visible_mask_iou: {summary['scene_visible_mask_iou']['mean']:.4f}")
    print(
        "mean_gt_visible_coverage_by_full_mask: "
        f"{summary['gt_visible_coverage_by_full_mask']['mean']:.4f}"
    )
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")
    if saved_overlay:
        print(f"overlay: {saved_overlay}")
    for row in rows:
        print(
            "  "
            f"pred_{row['pred_index']:02d} {row['label']} "
            f"gt_inst={row['best_gt_instance_index']} "
            f"full_iou={row['full_mask_iou']:.4f} "
            f"scene_visible_iou={row['scene_visible_mask_iou']:.4f} "
            f"gt_visible_coverage={row['gt_visible_coverage_by_full_mask']:.4f}"
        )


if __name__ == "__main__":
    main()
