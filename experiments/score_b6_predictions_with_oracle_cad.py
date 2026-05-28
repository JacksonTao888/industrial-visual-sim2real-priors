#!/usr/bin/env python3
"""Score B6 detections with an oracle CAD/GT-pose visible-mask signal."""

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


def area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def mask_coverage_by_box(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    x1, y1, x2, y2 = box
    inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    return float(inside.sum() / len(xs))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC with average ranks for ties."""

    labels_arr = np.asarray(labels, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float64)
    pos = int(labels_arr.sum())
    neg = int(len(labels_arr) - pos)
    if pos == 0 or neg == 0:
        return 0.0

    order = np.argsort(scores_arr)
    sorted_scores = scores_arr[order]
    ranks = np.empty(len(scores_arr), dtype=np.float64)
    start = 0
    while start < len(scores_arr):
        end = start + 1
        while end < len(scores_arr) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = ranks[labels_arr == 1].sum()
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def precision_recall_at_threshold(labels: list[int], scores: list[float], threshold: float) -> tuple[float, float, int]:
    selected = [idx for idx, score in enumerate(scores) if score >= threshold]
    if not selected:
        return 0.0, 0.0, 0
    tp = sum(labels[idx] for idx in selected)
    total_pos = sum(labels)
    precision = tp / len(selected)
    recall = tp / total_pos if total_pos else 0.0
    return float(precision), float(recall), len(selected)


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
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--cad-score-threshold", type=float, default=0.70)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/b6_oracle_cad_scores.csv.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    coco_path = args.coco or (
        project_root / "datasets" / "derived" / "coco" / "tless_real_val_heldout_900_detection.json"
    )
    pred_path = args.predictions or (project_root / "experiments" / "outputs" / "b6_tless_real_predictions.csv")
    out_path = args.out or (project_root / "experiments" / "outputs" / "b6_oracle_cad_scores.csv")
    summary_path = out_path.with_suffix(".summary.json")

    coco = load_json(coco_path)
    preds = [row for row in read_csv(pred_path) if float(row["confidence"]) >= args.conf_threshold]

    images = {int(image["id"]): image for image in coco["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        ann = dict(ann)
        ann["gt_box"] = xywh_to_xyxy(ann["bbox"])
        anns_by_image[int(ann["image_id"])].append(ann)

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in preds:
        pred["confidence_float"] = float(pred["confidence"])
        pred["pred_box"] = pred_xyxy(pred)
        preds_by_image[int(pred["image_id"])].append(pred)

    # Greedy COCO-style TP/FP labeling for detections at the selected confidence threshold.
    gt_matched: set[tuple[int, int]] = set()
    pred_tp: dict[tuple[int, int], bool] = {}
    for image_id, image_preds in preds_by_image.items():
        sorted_preds = sorted(image_preds, key=lambda row: row["confidence_float"], reverse=True)
        anns = anns_by_image.get(image_id, [])
        for pred in sorted_preds:
            obj_id = int(pred["obj_id"])
            candidates = [
                ann
                for ann in anns
                if int(ann["category_id"]) == obj_id and (image_id, int(ann["id"])) not in gt_matched
            ]
            best_ann = max(candidates, key=lambda ann: box_iou(pred["pred_box"], ann["gt_box"]), default=None)
            best_iou = box_iou(pred["pred_box"], best_ann["gt_box"]) if best_ann else 0.0
            key = (image_id, int(pred["pred_index"]))
            is_tp = best_ann is not None and best_iou >= args.match_iou
            pred_tp[key] = is_tp
            if is_tp:
                gt_matched.add((image_id, int(best_ann["id"])))

    mask_cache: dict[Path, tuple[np.ndarray, tuple[float, float, float, float] | None]] = {}

    rows = []
    for pred in preds:
        image_id = int(pred["image_id"])
        image = images[image_id]
        scene_id = int(image["scene_id"])
        bop_image_id = int(image["bop_image_id"])
        obj_id = int(pred["obj_id"])
        pred_box = pred["pred_box"]
        anns = anns_by_image.get(image_id, [])
        same_class = [ann for ann in anns if int(ann["category_id"]) == obj_id]
        best_ann = max(same_class, key=lambda ann: box_iou(pred_box, ann["gt_box"]), default=None)

        best_iou = box_iou(pred_box, best_ann["gt_box"]) if best_ann else 0.0
        oracle_mask_coverage = 0.0
        oracle_mask_bbox_iou = 0.0
        oracle_gt_obj_id = -1
        oracle_ann_id = -1
        oracle_bop_instance_id = -1
        oracle_visib_fract = 0.0
        if best_ann is not None:
            oracle_gt_obj_id = int(best_ann["category_id"])
            oracle_ann_id = int(best_ann["id"])
            oracle_bop_instance_id = int(best_ann["bop_instance_id"])
            oracle_visib_fract = float(best_ann.get("visib_fract", 0.0))
            mask_path = (
                project_root
                / "datasets"
                / "tless"
                / "test_primesense"
                / f"{scene_id:06d}"
                / "mask_visib"
                / f"{bop_image_id:06d}_{oracle_bop_instance_id:06d}.png"
            )
            if mask_path not in mask_cache:
                mask = load_mask(mask_path)
                mask_cache[mask_path] = (mask, mask_bbox(mask))
            mask, bbox = mask_cache[mask_path]
            oracle_mask_coverage = mask_coverage_by_box(mask, pred_box)
            oracle_mask_bbox_iou = box_iou(pred_box, bbox) if bbox else 0.0

        oracle_cad_score = 0.7 * oracle_mask_coverage + 0.3 * oracle_mask_bbox_iou
        pred_key = (image_id, int(pred["pred_index"]))
        is_tp = bool(pred_tp.get(pred_key, False))
        rows.append(
            {
                "image_id": image_id,
                "scene_id": scene_id,
                "bop_image_id": bop_image_id,
                "pred_index": int(pred["pred_index"]),
                "pred_obj_id": obj_id,
                "confidence": float(pred["confidence"]),
                "x1": float(pred["x1"]),
                "y1": float(pred["y1"]),
                "x2": float(pred["x2"]),
                "y2": float(pred["y2"]),
                "is_tp_iou50_greedy": int(is_tp),
                "best_same_class_iou": best_iou,
                "oracle_ann_id": oracle_ann_id,
                "oracle_bop_instance_id": oracle_bop_instance_id,
                "oracle_gt_obj_id": oracle_gt_obj_id,
                "oracle_visib_fract": oracle_visib_fract,
                "oracle_visible_mask_coverage": oracle_mask_coverage,
                "oracle_mask_bbox_iou": oracle_mask_bbox_iou,
                "oracle_cad_score": oracle_cad_score,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "scene_id",
        "bop_image_id",
        "pred_index",
        "pred_obj_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "is_tp_iou50_greedy",
        "best_same_class_iou",
        "oracle_ann_id",
        "oracle_bop_instance_id",
        "oracle_gt_obj_id",
        "oracle_visib_fract",
        "oracle_visible_mask_coverage",
        "oracle_mask_bbox_iou",
        "oracle_cad_score",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels = [int(row["is_tp_iou50_greedy"]) for row in rows]
    yolo_scores = [float(row["confidence"]) for row in rows]
    cad_scores = [float(row["oracle_cad_score"]) for row in rows]
    tp_rows = [row for row in rows if row["is_tp_iou50_greedy"]]
    fp_rows = [row for row in rows if not row["is_tp_iou50_greedy"]]
    cad_precision, cad_recall, cad_selected = precision_recall_at_threshold(
        labels, cad_scores, args.cad_score_threshold
    )

    summary = {
        "method": "Oracle CAD verification score for B6 detections",
        "note": "Uses BOP mask_visib as a GT-pose visible CAD mask oracle; this is diagnostic, not a deployable score.",
        "coco": str(coco_path),
        "predictions": str(pred_path),
        "conf_threshold": args.conf_threshold,
        "match_iou": args.match_iou,
        "cad_score": "0.7 * visible_mask_coverage_by_pred_box + 0.3 * IoU(pred_box, visible_mask_bbox)",
        "predictions_scored": len(rows),
        "tp_iou50_greedy": int(sum(labels)),
        "fp_iou50_greedy": int(len(labels) - sum(labels)),
        "tp_rate_among_predictions": float(sum(labels) / len(labels)) if labels else 0.0,
        "mean_confidence_tp": mean([row["confidence"] for row in tp_rows]),
        "mean_confidence_fp": mean([row["confidence"] for row in fp_rows]),
        "mean_cad_score_tp": mean([row["oracle_cad_score"] for row in tp_rows]),
        "mean_cad_score_fp": mean([row["oracle_cad_score"] for row in fp_rows]),
        "median_cad_score_tp": median([row["oracle_cad_score"] for row in tp_rows]),
        "median_cad_score_fp": median([row["oracle_cad_score"] for row in fp_rows]),
        "mean_visible_mask_coverage_tp": mean([row["oracle_visible_mask_coverage"] for row in tp_rows]),
        "mean_visible_mask_coverage_fp": mean([row["oracle_visible_mask_coverage"] for row in fp_rows]),
        "auroc_confidence_for_tp": roc_auc(labels, yolo_scores),
        "auroc_oracle_cad_score_for_tp": roc_auc(labels, cad_scores),
        "cad_score_threshold": args.cad_score_threshold,
        "cad_score_precision_at_threshold": cad_precision,
        "cad_score_recall_at_threshold": cad_recall,
        "cad_score_selected_predictions": cad_selected,
        "csv": str(out_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"predictions_scored: {summary['predictions_scored']}")
    print(f"tp_iou50_greedy: {summary['tp_iou50_greedy']}")
    print(f"fp_iou50_greedy: {summary['fp_iou50_greedy']}")
    print(f"mean_confidence_tp: {summary['mean_confidence_tp']:.4f}")
    print(f"mean_confidence_fp: {summary['mean_confidence_fp']:.4f}")
    print(f"mean_cad_score_tp: {summary['mean_cad_score_tp']:.4f}")
    print(f"mean_cad_score_fp: {summary['mean_cad_score_fp']:.4f}")
    print(f"auroc_confidence_for_tp: {summary['auroc_confidence_for_tp']:.4f}")
    print(f"auroc_oracle_cad_score_for_tp: {summary['auroc_oracle_cad_score_for_tp']:.4f}")
    print(
        f"cad_score>={args.cad_score_threshold}: "
        f"precision={cad_precision:.4f} recall={cad_recall:.4f} selected={cad_selected}"
    )
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
