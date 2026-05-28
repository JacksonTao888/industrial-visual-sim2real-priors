#!/usr/bin/env python3
"""Score B6+MegaPose proposals with deployable CAD/bbox consistency signals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_megapose_tless_example_mask_iou import load_megapose_predictions
from render_cad_mask_iou import parse_ply_mesh, render_scene_masks


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def box_area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def mask_inside_box_fraction(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    x1, y1, x2, y2 = box
    inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    return float(inside.sum() / len(xs))


def abs_log_area_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("inf")
    return abs(math.log(a / b))


def roc_auc(labels: list[int], scores: list[float]) -> float:
    labels_arr = np.asarray(labels, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores_arr)
    labels_arr = labels_arr[finite]
    scores_arr = scores_arr[finite]
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


def precision_recall_at_threshold(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> tuple[float, float, int]:
    selected = [idx for idx, score in enumerate(scores) if score >= threshold]
    if not selected:
        return 0.0, 0.0, 0
    tp = sum(labels[idx] for idx in selected)
    total_pos = sum(labels)
    precision = tp / len(selected)
    recall = tp / total_pos if total_pos else 0.0
    return float(precision), float(recall), len(selected)


def load_metric_rows(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for row in read_csv(path):
        rows[(row["example_name"], int(row["pred_index"]))] = row
    return rows


def load_failure_rows(path: Path | None) -> dict[tuple[str, int], dict]:
    if path is None or not path.exists():
        return {}
    rows = {}
    for row in read_csv(path):
        rows[(row["example_name"], int(row["pred_index"]))] = row
    return rows


def load_original_predictions(manifest: dict) -> list[dict]:
    source_predictions = Path(manifest["source_predictions"])
    scene_id = int(manifest["scene_id"])
    image_id = int(manifest["image_id"])
    conf_threshold = float(manifest["conf_threshold"])
    max_detections = int(manifest["max_detections"])
    rows = [
        row
        for row in read_csv(source_predictions)
        if int(row["scene_id"]) == scene_id
        and int(row["bop_image_id"]) == image_id
        and float(row["confidence"]) >= conf_threshold
    ]
    return sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)[:max_detections]


def load_input_objects(example_dir: Path) -> list[dict]:
    return load_json(example_dir / "inputs" / "object_data.json")


def boxes_close(a: list[float], b: list[float], tol: float = 1e-3) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def make_input_lookup(example_dir: Path, manifest: dict) -> dict[str, list[dict]]:
    input_objects = load_input_objects(example_dir)
    original_predictions = load_original_predictions(manifest)
    if len(input_objects) != len(original_predictions):
        raise RuntimeError(
            f"Input/original prediction count mismatch for {example_dir}: "
            f"{len(input_objects)} input vs {len(original_predictions)} original"
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for idx, (input_obj, original) in enumerate(zip(input_objects, original_predictions)):
        input_box = [float(v) for v in input_obj["bbox_modal"]]
        original_box = [
            float(original["x1"]),
            float(original["y1"]),
            float(original["x2"]),
            float(original["y2"]),
        ]
        if input_obj["label"] != original["class_name"] or not boxes_close(input_box, original_box):
            raise RuntimeError(
                f"Input/original prediction mismatch for {example_dir} index {idx}: "
                f"{input_obj} vs {original}"
            )
        grouped[input_obj["label"]].append(
            {
                "input_index": idx,
                "label": input_obj["label"],
                "bbox_modal": input_box,
                "confidence": float(original["confidence"]),
                "original_pred_index": int(original["pred_index"]),
            }
        )
    return grouped


def find_example_dirs(examples_root: Path) -> list[Path]:
    return sorted(
        path
        for path in examples_root.glob("tless_scene_*_image_*")
        if path.is_dir() and (path / "manifest.json").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "outputs"
            / "megapose_batches"
            / "conf_0p25_maxdet_10_scene_round_robin_per_scene_5_n100_offset0"
        ),
    )
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--face-stride", type=int, default=1)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--failure-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.70)
    args = parser.parse_args()

    project_root = args.project_root
    dataset_root = project_root / "datasets" / "tless"
    models_dir = dataset_root / "models_eval"
    examples_root = args.examples_root
    metrics_csv = args.metrics_csv or (examples_root / "megapose_batch_mask_iou.csv")
    failure_csv = args.failure_csv or (examples_root / "megapose_failure_analysis.csv")
    out_path = args.out or (examples_root / "megapose_cad_consistency_scores.csv")
    summary_path = out_path.with_suffix(".summary.json")

    metric_rows = load_metric_rows(metrics_csv)
    failure_rows = load_failure_rows(failure_csv)
    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scored_rows = []

    example_dirs = find_example_dirs(examples_root)
    for example_idx, example_dir in enumerate(example_dirs, start=1):
        manifest = load_json(example_dir / "manifest.json")
        scene_id = int(manifest["scene_id"])
        image_id = int(manifest["image_id"])
        megapose_output = example_dir / "outputs" / "object_data.json"
        if not megapose_output.exists():
            continue

        input_lookup = make_input_lookup(example_dir, manifest)
        input_cursor = defaultdict(int)
        megapose_predictions = load_megapose_predictions(megapose_output)
        if sum(len(v) for v in input_lookup.values()) != len(megapose_predictions):
            raise RuntimeError(
                f"Prediction count mismatch for {example_dir}: "
                f"{sum(len(v) for v in input_lookup.values())} input vs "
                f"{len(megapose_predictions)} MegaPose"
            )

        scene_dir = dataset_root / args.split / f"{scene_id:06d}"
        image_stem = f"{image_id:06d}"
        rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{image_stem}.png").convert("RGB"))
        height, width = rgb.shape[:2]
        scene_camera = load_json(scene_dir / "scene_camera.json")
        camera_matrix = np.asarray(scene_camera[str(image_id)]["cam_K"], dtype=np.float32).reshape(3, 3)

        for prediction in megapose_predictions:
            obj_id = int(prediction["obj_id"])
            if obj_id not in mesh_cache:
                mesh_cache[obj_id] = parse_ply_mesh(models_dir / f"obj_{obj_id:06d}.ply")

        _, pred_full, pred_visible = render_scene_masks(
            instances=megapose_predictions,
            mesh_cache=mesh_cache,
            camera_matrix=camera_matrix,
            image_shape=(height, width),
            face_stride=args.face_stride,
        )

        for idx, (pose_pred, full_mask, visible_mask) in enumerate(
            zip(megapose_predictions, pred_full, pred_visible)
        ):
            label = pose_pred["label"]
            label_items = input_lookup[label]
            cursor = input_cursor[label]
            if cursor >= len(label_items):
                raise RuntimeError(f"No remaining input bbox for {example_dir} label {label}")
            matched_input = label_items[cursor]
            input_cursor[label] += 1
            det_box = (
                float(matched_input["bbox_modal"][0]),
                float(matched_input["bbox_modal"][1]),
                float(matched_input["bbox_modal"][2]),
                float(matched_input["bbox_modal"][3]),
            )
            full_bbox = mask_bbox(full_mask)
            visible_bbox = mask_bbox(visible_mask)
            det_area = box_area(det_box)
            full_bbox_area = box_area(full_bbox)
            visible_bbox_area = box_area(visible_bbox)
            full_bbox_iou = box_iou(det_box, full_bbox)
            visible_bbox_iou = box_iou(det_box, visible_bbox)
            full_inside_det = mask_inside_box_fraction(full_mask, det_box)
            visible_inside_det = mask_inside_box_fraction(visible_mask, det_box)
            full_area_ratio_penalty = abs_log_area_ratio(det_area, full_bbox_area)
            visible_area_ratio_penalty = abs_log_area_ratio(det_area, visible_bbox_area)

            # This score uses only deployable quantities: detector box and
            # rendered CAD mask from the estimated pose.
            cad_bbox_score = (
                0.45 * full_bbox_iou
                + 0.35 * full_inside_det
                + 0.20 * math.exp(-min(full_area_ratio_penalty, 10.0))
            )
            cad_visible_bbox_score = (
                0.45 * visible_bbox_iou
                + 0.35 * visible_inside_det
                + 0.20 * math.exp(-min(visible_area_ratio_penalty, 10.0))
            )

            key = (example_dir.name, idx)
            metric = metric_rows.get(key, {})
            failure = failure_rows.get(key, {})
            is_good_pose = int(failure.get("failure_category", "") == "good_pose_silhouette")
            has_same_class_gt = int(int(metric.get("best_gt_instance_index", -1)) >= 0)
            same_class_nonzero = int(
                has_same_class_gt and float(metric.get("full_mask_iou", 0.0)) > 0.0
            )

            scored_rows.append(
                {
                    "example_name": example_dir.name,
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "coco_image_id": int(manifest.get("coco_image_id", -1)),
                    "pred_index": idx,
                    "input_index": int(matched_input["input_index"]),
                    "original_pred_index": int(matched_input["original_pred_index"]),
                    "label": pose_pred["label"],
                    "obj_id": int(pose_pred["obj_id"]),
                    "detector_confidence": float(matched_input["confidence"]),
                    "detector_box_area": det_area,
                    "render_full_bbox_area": full_bbox_area,
                    "render_visible_bbox_area": visible_bbox_area,
                    "full_bbox_iou_with_detector": full_bbox_iou,
                    "visible_bbox_iou_with_detector": visible_bbox_iou,
                    "full_mask_inside_detector_box": full_inside_det,
                    "visible_mask_inside_detector_box": visible_inside_det,
                    "full_area_ratio_penalty": full_area_ratio_penalty,
                    "visible_area_ratio_penalty": visible_area_ratio_penalty,
                    "cad_bbox_score": cad_bbox_score,
                    "cad_visible_bbox_score": cad_visible_bbox_score,
                    "failure_category": failure.get("failure_category", ""),
                    "is_good_pose_silhouette": is_good_pose,
                    "has_same_class_gt": has_same_class_gt,
                    "same_class_nonzero_alignment": same_class_nonzero,
                    "full_mask_iou": float(metric.get("full_mask_iou", 0.0)),
                    "scene_visible_mask_iou": float(metric.get("scene_visible_mask_iou", 0.0)),
                    "gt_visible_coverage_by_full_mask": float(
                        metric.get("gt_visible_coverage_by_full_mask", 0.0)
                    ),
                }
            )

        if example_idx == 1 or example_idx % 10 == 0 or example_idx == len(example_dirs):
            print(
                f"processed_examples: {example_idx}/{len(example_dirs)} "
                f"scored_rows: {len(scored_rows)}",
                flush=True,
            )

    fieldnames = list(scored_rows[0].keys()) if scored_rows else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    score_names = [
        "detector_confidence",
        "full_bbox_iou_with_detector",
        "visible_bbox_iou_with_detector",
        "full_mask_inside_detector_box",
        "visible_mask_inside_detector_box",
        "cad_bbox_score",
        "cad_visible_bbox_score",
    ]
    label_specs = {
        "good_pose_silhouette": [int(row["is_good_pose_silhouette"]) for row in scored_rows],
        "has_same_class_gt": [int(row["has_same_class_gt"]) for row in scored_rows],
        "same_class_nonzero_alignment": [
            int(row["same_class_nonzero_alignment"]) for row in scored_rows
        ],
    }
    auroc = {
        label_name: {
            score_name: roc_auc(labels, [float(row[score_name]) for row in scored_rows])
            for score_name in score_names
        }
        for label_name, labels in label_specs.items()
    }

    threshold_metrics = {}
    for label_name, labels in label_specs.items():
        threshold_metrics[label_name] = {}
        for score_name in ["cad_bbox_score", "cad_visible_bbox_score", "detector_confidence"]:
            precision, recall, selected = precision_recall_at_threshold(
                labels,
                [float(row[score_name]) for row in scored_rows],
                args.score_threshold,
            )
            threshold_metrics[label_name][score_name] = {
                "threshold": args.score_threshold,
                "precision": precision,
                "recall": recall,
                "selected": selected,
            }

    fusion_weight_sweep = {}
    for label_name, labels in label_specs.items():
        fusion_weight_sweep[label_name] = {}
        for weight_confidence in [i / 10 for i in range(11)]:
            scores = [
                weight_confidence * float(row["detector_confidence"])
                + (1.0 - weight_confidence) * float(row["cad_visible_bbox_score"])
                for row in scored_rows
            ]
            fusion_weight_sweep[label_name][f"confidence_{weight_confidence:.1f}"] = roc_auc(
                labels, scores
            )

    summary = {
        "method_family": "Practical CAD consistency / proposal verification",
        "note": (
            "Scores use only detector bbox/confidence plus rendered CAD masks from "
            "MegaPose estimated poses; GT labels are used only for evaluation."
        ),
        "examples_root": str(examples_root),
        "metrics_csv": str(metrics_csv),
        "failure_csv": str(failure_csv),
        "rows": len(scored_rows),
        "score_summaries": {
            score_name: summarize([float(row[score_name]) for row in scored_rows])
            for score_name in score_names
        },
        "positive_counts": {name: int(sum(labels)) for name, labels in label_specs.items()},
        "auroc": auroc,
        "fusion_weight_sweep": fusion_weight_sweep,
        "threshold_metrics": threshold_metrics,
        "csv": str(out_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"rows: {len(scored_rows)}")
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")
    print("AUROC:")
    for label_name, scores in auroc.items():
        print(f"  label={label_name}")
        for score_name, value in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {score_name}: {value:.4f}")


if __name__ == "__main__":
    main()
