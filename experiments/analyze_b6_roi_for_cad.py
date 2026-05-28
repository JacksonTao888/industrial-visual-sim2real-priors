#!/usr/bin/env python3
"""Analyze whether B6 detections are good ROIs for CAD-at-test-time methods."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box]
    return x, y, x + w, y + h


def pred_xyxy(row: dict) -> tuple[float, float, float, float]:
    return float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = box_intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_coverage(target: tuple[float, float, float, float], proposal: tuple[float, float, float, float]) -> float:
    area = box_area(target)
    return box_intersection(target, proposal) / area if area > 0 else 0.0


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def mask_coverage_by_box(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    x1, y1, x2, y2 = box
    inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    return float(inside.sum() / len(xs))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--coco",
        type=Path,
        default=None,
        help="Defaults to the 900-image held-out real validation COCO file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/b6_tless_real_predictions.csv.",
    )
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/b6_roi_for_cad_analysis.csv.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    coco_path = args.coco or (
        project_root / "datasets" / "derived" / "coco" / "tless_real_val_heldout_900_detection.json"
    )
    pred_path = args.predictions or (project_root / "experiments" / "outputs" / "b6_tless_real_predictions.csv")
    out_path = args.out or (project_root / "experiments" / "outputs" / "b6_roi_for_cad_analysis.csv")
    summary_path = out_path.with_suffix(".summary.json")

    coco = load_json(coco_path)
    preds = read_csv(pred_path)

    images = {int(image["id"]): image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    filtered_preds = []
    for pred in preds:
        pred["confidence_float"] = float(pred["confidence"])
        if pred["confidence_float"] >= args.conf_threshold:
            filtered_preds.append(pred)
            preds_by_image[int(pred["image_id"])].append(pred)

    rows = []
    matched_pred_keys = set()
    for image_id, anns in annotations_by_image.items():
        image = images[image_id]
        image_preds = preds_by_image.get(image_id, [])
        scene_id = int(image["scene_id"])
        bop_image_id = int(image["bop_image_id"])
        for ann in anns:
            gt_box = xywh_to_xyxy(ann["bbox"])
            obj_id = int(ann["category_id"])
            same_class = [pred for pred in image_preds if int(pred["obj_id"]) == obj_id]
            any_class = image_preds

            best_same = max(same_class, key=lambda pred: box_iou(gt_box, pred_xyxy(pred)), default=None)
            best_any = max(any_class, key=lambda pred: box_iou(gt_box, pred_xyxy(pred)), default=None)

            best_same_box = pred_xyxy(best_same) if best_same else None
            best_any_box = pred_xyxy(best_any) if best_any else None
            same_iou = box_iou(gt_box, best_same_box) if best_same_box else 0.0
            any_iou = box_iou(gt_box, best_any_box) if best_any_box else 0.0
            same_coverage = box_coverage(gt_box, best_same_box) if best_same_box else 0.0
            same_conf = float(best_same["confidence"]) if best_same else 0.0
            same_pred_index = int(best_same["pred_index"]) if best_same else -1

            mask_path = (
                project_root
                / "datasets"
                / "tless"
                / "test_primesense"
                / f"{scene_id:06d}"
                / "mask_visib"
                / f"{bop_image_id:06d}_{int(ann['bop_instance_id']):06d}.png"
            )
            if best_same_box and mask_path.exists():
                visible_mask_coverage = mask_coverage_by_box(load_mask(mask_path), best_same_box)
            else:
                visible_mask_coverage = 0.0

            matched = same_iou >= args.match_iou
            if matched and best_same is not None:
                matched_pred_keys.add((int(best_same["image_id"]), int(best_same["pred_index"])))

            rows.append(
                {
                    "image_id": image_id,
                    "scene_id": scene_id,
                    "bop_image_id": bop_image_id,
                    "ann_id": int(ann["id"]),
                    "bop_instance_id": int(ann["bop_instance_id"]),
                    "obj_id": obj_id,
                    "visib_fract": float(ann.get("visib_fract", 0.0)),
                    "best_same_class_iou": same_iou,
                    "best_same_class_conf": same_conf,
                    "best_same_class_pred_index": same_pred_index,
                    "best_same_class_gt_bbox_coverage": same_coverage,
                    "best_same_class_visible_mask_coverage": visible_mask_coverage,
                    "best_any_class_iou": any_iou,
                    "best_any_class_obj_id": int(best_any["obj_id"]) if best_any else -1,
                    "matched_iou_threshold": matched,
                }
            )

    false_positive_preds = []
    for pred in filtered_preds:
        key = (int(pred["image_id"]), int(pred["pred_index"]))
        if key not in matched_pred_keys:
            false_positive_preds.append(pred)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "scene_id",
        "bop_image_id",
        "ann_id",
        "bop_instance_id",
        "obj_id",
        "visib_fract",
        "best_same_class_iou",
        "best_same_class_conf",
        "best_same_class_pred_index",
        "best_same_class_gt_bbox_coverage",
        "best_same_class_visible_mask_coverage",
        "best_any_class_iou",
        "best_any_class_obj_id",
        "matched_iou_threshold",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    same_ious = [row["best_same_class_iou"] for row in rows]
    same_coverages = [row["best_same_class_gt_bbox_coverage"] for row in rows]
    visible_coverages = [row["best_same_class_visible_mask_coverage"] for row in rows]
    any_ious = [row["best_any_class_iou"] for row in rows]
    matched_rows = [row for row in rows if row["matched_iou_threshold"]]
    wrong_class_best = [row for row in rows if row["best_any_class_iou"] >= args.match_iou and row["best_any_class_obj_id"] != row["obj_id"]]

    summary = {
        "method": "B6 detector ROI quality for CAD-at-test-time",
        "coco": str(coco_path),
        "predictions": str(pred_path),
        "conf_threshold": args.conf_threshold,
        "match_iou": args.match_iou,
        "images": len(images),
        "gt_instances": len(rows),
        "predictions_after_conf": len(filtered_preds),
        "matched_gt_instances_same_class_iou": len(matched_rows),
        "recall_same_class_iou": len(matched_rows) / len(rows) if rows else 0.0,
        "false_positive_predictions": len(false_positive_preds),
        "false_positive_per_image": len(false_positive_preds) / len(images) if images else 0.0,
        "wrong_class_best_iou50_instances": len(wrong_class_best),
        "mean_best_same_class_iou": mean(same_ious),
        "median_best_same_class_iou": percentile(same_ious, 50),
        "p10_best_same_class_iou": percentile(same_ious, 10),
        "mean_gt_bbox_coverage_by_same_class_roi": mean(same_coverages),
        "mean_visible_mask_coverage_by_same_class_roi": mean(visible_coverages),
        "median_visible_mask_coverage_by_same_class_roi": percentile(visible_coverages, 50),
        "mean_best_any_class_iou": mean(any_ious),
        "csv": str(out_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"images: {summary['images']}")
    print(f"gt_instances: {summary['gt_instances']}")
    print(f"predictions_after_conf: {summary['predictions_after_conf']}")
    print(f"recall_same_class_iou@{args.match_iou}: {summary['recall_same_class_iou']:.4f}")
    print(f"false_positive_predictions: {summary['false_positive_predictions']}")
    print(f"false_positive_per_image: {summary['false_positive_per_image']:.4f}")
    print(f"mean_best_same_class_iou: {summary['mean_best_same_class_iou']:.4f}")
    print(f"mean_gt_bbox_coverage_by_same_class_roi: {summary['mean_gt_bbox_coverage_by_same_class_roi']:.4f}")
    print(f"mean_visible_mask_coverage_by_same_class_roi: {summary['mean_visible_mask_coverage_by_same_class_roi']:.4f}")
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
