#!/usr/bin/env python3
"""Score MegaPose CAD silhouettes against real RGB image edges."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluate_megapose_tless_example_mask_iou import load_megapose_predictions
from render_cad_mask_iou import parse_ply_mesh, render_scene_masks
from score_megapose_cad_consistency import (
    find_example_dirs,
    load_json,
    make_input_lookup,
    read_csv,
    roc_auc,
    summarize,
)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.sum() == 0:
        return np.zeros_like(mask_u8, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (mask_u8 > 0) & (eroded == 0)


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def crop_mask_for_box(shape: tuple[int, int], box: tuple[float, float, float, float], pad: int) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = box
    x0 = max(0, int(np.floor(x1)) - pad)
    y0 = max(0, int(np.floor(y1)) - pad)
    x3 = min(width, int(np.ceil(x2)) + pad)
    y3 = min(height, int(np.ceil(y2)) + pad)
    mask = np.zeros((height, width), dtype=bool)
    if x3 > x0 and y3 > y0:
        mask[y0:y3, x0:x3] = True
    return mask


def edge_support_scores(
    *,
    rendered_mask: np.ndarray,
    image_edges: np.ndarray,
    detector_box: tuple[float, float, float, float],
    edge_tolerance: int,
    bbox_pad: int,
) -> dict[str, float]:
    boundary = mask_boundary(rendered_mask)
    roi = crop_mask_for_box(rendered_mask.shape, detector_box, bbox_pad)
    boundary_roi = boundary & roi
    boundary_count = int(boundary_roi.sum())
    if boundary_count == 0:
        return {
            "cad_edge_pixels": 0.0,
            "cad_edge_support": 0.0,
            "real_edge_precision_near_cad": 0.0,
            "real_edge_density_in_roi": 0.0,
            "edge_chamfer_mean": 999.0,
        }

    dilated_edges = dilate_bool(image_edges, edge_tolerance)
    supported = boundary_roi & dilated_edges
    cad_edge_support = float(supported.sum() / boundary_count)

    dilated_boundary = dilate_bool(boundary_roi, edge_tolerance)
    real_edges_roi = image_edges & roi
    real_edge_count = int(real_edges_roi.sum())
    real_edge_precision = (
        float((real_edges_roi & dilated_boundary).sum() / real_edge_count)
        if real_edge_count
        else 0.0
    )
    roi_area = int(roi.sum())
    real_edge_density = float(real_edge_count / roi_area) if roi_area else 0.0

    distance_to_edge = cv2.distanceTransform((~image_edges).astype(np.uint8), cv2.DIST_L2, 3)
    edge_chamfer_mean = float(distance_to_edge[boundary_roi].mean())

    return {
        "cad_edge_pixels": float(boundary_count),
        "cad_edge_support": cad_edge_support,
        "real_edge_precision_near_cad": real_edge_precision,
        "real_edge_density_in_roi": real_edge_density,
        "edge_chamfer_mean": edge_chamfer_mean,
    }


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
    return {(row["example_name"], int(row["pred_index"])): row for row in read_csv(path)}


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
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--edge-tolerance", type=int, default=3)
    parser.add_argument("--bbox-pad", type=int, default=12)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--failure-csv", type=Path, default=None)
    parser.add_argument("--bbox-score-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    args = parser.parse_args()

    project_root = args.project_root
    dataset_root = project_root / "datasets" / "tless"
    models_dir = dataset_root / "models_eval"
    examples_root = args.examples_root
    metrics_csv = args.metrics_csv or (examples_root / "megapose_batch_mask_iou.csv")
    failure_csv = args.failure_csv or (examples_root / "megapose_failure_analysis.csv")
    bbox_score_csv = args.bbox_score_csv or (examples_root / "megapose_cad_consistency_scores.csv")
    out_path = args.out or (examples_root / "megapose_edge_consistency_scores.csv")
    summary_path = out_path.with_suffix(".summary.json")

    metric_rows = load_metric_rows(metrics_csv)
    failure_rows = load_metric_rows(failure_csv) if failure_csv.exists() else {}
    bbox_score_rows = load_metric_rows(bbox_score_csv) if bbox_score_csv.exists() else {}
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

        scene_dir = dataset_root / args.split / f"{scene_id:06d}"
        image_stem = f"{image_id:06d}"
        rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{image_stem}.png").convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        image_edges = cv2.Canny(gray, args.canny_low, args.canny_high).astype(bool)
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

            full_edge = edge_support_scores(
                rendered_mask=full_mask,
                image_edges=image_edges,
                detector_box=det_box,
                edge_tolerance=args.edge_tolerance,
                bbox_pad=args.bbox_pad,
            )
            visible_edge = edge_support_scores(
                rendered_mask=visible_mask,
                image_edges=image_edges,
                detector_box=det_box,
                edge_tolerance=args.edge_tolerance,
                bbox_pad=args.bbox_pad,
            )

            key = (example_dir.name, idx)
            metric = metric_rows.get(key, {})
            failure = failure_rows.get(key, {})
            bbox_score = bbox_score_rows.get(key, {})
            detector_conf = float(matched_input["confidence"])
            cad_visible_bbox_score = float(bbox_score.get("cad_visible_bbox_score", 0.0))

            # The edge term is intentionally modest because low-texture T-LESS
            # objects often have weak RGB edges.
            fused_edge_score = (
                0.75 * detector_conf
                + 0.15 * cad_visible_bbox_score
                + 0.10 * visible_edge["cad_edge_support"]
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
                    "detector_confidence": detector_conf,
                    "cad_visible_bbox_score": cad_visible_bbox_score,
                    "full_cad_edge_pixels": full_edge["cad_edge_pixels"],
                    "full_cad_edge_support": full_edge["cad_edge_support"],
                    "full_real_edge_precision_near_cad": full_edge[
                        "real_edge_precision_near_cad"
                    ],
                    "full_real_edge_density_in_roi": full_edge["real_edge_density_in_roi"],
                    "full_edge_chamfer_mean": full_edge["edge_chamfer_mean"],
                    "visible_cad_edge_pixels": visible_edge["cad_edge_pixels"],
                    "visible_cad_edge_support": visible_edge["cad_edge_support"],
                    "visible_real_edge_precision_near_cad": visible_edge[
                        "real_edge_precision_near_cad"
                    ],
                    "visible_real_edge_density_in_roi": visible_edge[
                        "real_edge_density_in_roi"
                    ],
                    "visible_edge_chamfer_mean": visible_edge["edge_chamfer_mean"],
                    "fused_edge_score": fused_edge_score,
                    "failure_category": failure.get("failure_category", ""),
                    "is_good_pose_silhouette": int(
                        failure.get("failure_category", "") == "good_pose_silhouette"
                    ),
                    "has_same_class_gt": int(int(metric.get("best_gt_instance_index", -1)) >= 0),
                    "same_class_nonzero_alignment": int(
                        int(metric.get("best_gt_instance_index", -1)) >= 0
                        and float(metric.get("full_mask_iou", 0.0)) > 0.0
                    ),
                    "full_mask_iou": float(metric.get("full_mask_iou", 0.0)),
                    "scene_visible_mask_iou": float(
                        metric.get("scene_visible_mask_iou", 0.0)
                    ),
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
        "cad_visible_bbox_score",
        "full_cad_edge_support",
        "visible_cad_edge_support",
        "full_real_edge_precision_near_cad",
        "visible_real_edge_precision_near_cad",
        "fused_edge_score",
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
        for score_name in ["visible_cad_edge_support", "fused_edge_score", "detector_confidence"]:
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

    summary = {
        "method_family": "Practical CAD edge/render consistency",
        "note": (
            "Scores compare rendered CAD silhouette edges from MegaPose pose to "
            "Canny edges in the real RGB image. GT is used only for evaluation."
        ),
        "examples_root": str(examples_root),
        "rows": len(scored_rows),
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "edge_tolerance": args.edge_tolerance,
        "bbox_pad": args.bbox_pad,
        "score_summaries": {
            score_name: summarize([float(row[score_name]) for row in scored_rows])
            for score_name in score_names
        },
        "positive_counts": {name: int(sum(labels)) for name, labels in label_specs.items()},
        "auroc": auroc,
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
