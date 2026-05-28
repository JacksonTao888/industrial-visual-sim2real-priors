#!/usr/bin/env python3
"""Score MegaPose CAD poses against real T-LESS depth images."""

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
from render_cad_mask_iou import (
    parse_ply_mesh,
    project_points,
    transform_points,
    triangle_depth,
)
from score_megapose_cad_consistency import (
    find_example_dirs,
    load_json,
    make_input_lookup,
    read_csv,
    roc_auc,
    summarize,
)


def render_scene_depths(
    *,
    instances: list[dict],
    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    camera_matrix: np.ndarray,
    image_shape: tuple[int, int],
    face_stride: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    height, width = image_shape
    scene_depth = np.full((height, width), np.inf, dtype=np.float32)
    instance_map = np.full((height, width), -1, dtype=np.int16)
    full_depths = [np.full((height, width), np.inf, dtype=np.float32) for _ in instances]

    for inst_idx, instance in enumerate(instances):
        obj_id = int(instance["obj_id"])
        vertices, faces = mesh_cache[obj_id]
        rotation = np.asarray(instance["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        translation = np.asarray(instance["cam_t_m2c"], dtype=np.float32)
        vertices_c = transform_points(vertices, rotation, translation)
        vertices_uv = project_points(vertices_c, camera_matrix)

        for face in faces[:: max(1, face_stride)]:
            tri_uv = vertices_uv[face]
            tri_z = vertices_c[face, 2]
            if not np.isfinite(tri_uv).all() or np.any(tri_z <= 1e-6):
                continue

            min_x = int(np.floor(tri_uv[:, 0].min()))
            max_x = int(np.ceil(tri_uv[:, 0].max()))
            min_y = int(np.floor(tri_uv[:, 1].min()))
            max_y = int(np.ceil(tri_uv[:, 1].max()))
            if max_x < 0 or max_y < 0 or min_x >= width or min_y >= height:
                continue
            min_x = max(0, min_x)
            max_x = min(width - 1, max_x)
            min_y = max(0, min_y)
            max_y = min(height - 1, max_y)
            if max_x <= min_x or max_y <= min_y:
                continue

            result = triangle_depth(tri_uv, tri_z, min_x, min_y, max_x, max_y)
            if len(result) == 2:
                continue
            xs_local, ys_local, tri_depth = result
            xs = xs_local + min_x
            ys = ys_local + min_y

            closer_to_instance = tri_depth < full_depths[inst_idx][ys, xs]
            if closer_to_instance.any():
                ys_i = ys[closer_to_instance]
                xs_i = xs[closer_to_instance]
                full_depths[inst_idx][ys_i, xs_i] = tri_depth[closer_to_instance]

            closer_to_scene = tri_depth < scene_depth[ys, xs]
            if closer_to_scene.any():
                ys_s = ys[closer_to_scene]
                xs_s = xs[closer_to_scene]
                scene_depth[ys_s, xs_s] = tri_depth[closer_to_scene]
                instance_map[ys_s, xs_s] = inst_idx

    full_masks = [np.isfinite(depth) for depth in full_depths]
    visible_masks = [(instance_map == idx) for idx in range(len(instances))]
    visible_depths = [
        np.where(visible_masks[idx], scene_depth, np.inf).astype(np.float32)
        for idx in range(len(instances))
    ]
    return instance_map, scene_depth, full_masks, visible_masks, visible_depths


def crop_mask_for_box(
    shape: tuple[int, int],
    box: tuple[float, float, float, float],
    pad: int,
) -> np.ndarray:
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


def depth_consistency_scores(
    *,
    rendered_depth_mm: np.ndarray,
    rendered_mask: np.ndarray,
    real_depth_mm: np.ndarray,
    detector_box: tuple[float, float, float, float],
    bbox_pad: int,
    tolerance_mm: float,
    scale_mm: float,
) -> dict[str, float]:
    roi = crop_mask_for_box(rendered_mask.shape, detector_box, bbox_pad)
    rendered_pixels = rendered_mask & roi & np.isfinite(rendered_depth_mm)
    rendered_count = int(rendered_pixels.sum())
    if rendered_count == 0:
        return {
            "rendered_depth_pixels": 0.0,
            "valid_real_depth_pixels": 0.0,
            "valid_real_depth_fraction": 0.0,
            "depth_abs_error_mean_mm": 9999.0,
            "depth_abs_error_median_mm": 9999.0,
            "depth_abs_error_p90_mm": 9999.0,
            "depth_signed_error_mean_mm": 9999.0,
            "depth_support_5mm": 0.0,
            "depth_support_10mm": 0.0,
            "depth_support_20mm": 0.0,
            "depth_support_50mm": 0.0,
            "depth_support_tolerance": 0.0,
            "real_closer_fraction_20mm": 0.0,
            "real_farther_fraction_20mm": 0.0,
            "depth_consistency_score": 0.0,
        }

    valid = rendered_pixels & (real_depth_mm > 0)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return {
            "rendered_depth_pixels": float(rendered_count),
            "valid_real_depth_pixels": 0.0,
            "valid_real_depth_fraction": 0.0,
            "depth_abs_error_mean_mm": 9999.0,
            "depth_abs_error_median_mm": 9999.0,
            "depth_abs_error_p90_mm": 9999.0,
            "depth_signed_error_mean_mm": 9999.0,
            "depth_support_5mm": 0.0,
            "depth_support_10mm": 0.0,
            "depth_support_20mm": 0.0,
            "depth_support_50mm": 0.0,
            "depth_support_tolerance": 0.0,
            "real_closer_fraction_20mm": 0.0,
            "real_farther_fraction_20mm": 0.0,
            "depth_consistency_score": 0.0,
        }

    signed_error = real_depth_mm[valid] - rendered_depth_mm[valid]
    abs_error = np.abs(signed_error)
    support_5 = float((abs_error <= 5.0).mean())
    support_10 = float((abs_error <= 10.0).mean())
    support_20 = float((abs_error <= 20.0).mean())
    support_50 = float((abs_error <= 50.0).mean())
    support_tolerance = float((abs_error <= tolerance_mm).mean())
    median_error = float(np.median(abs_error))

    # Positive signed error means the measured surface is behind the rendered CAD
    # surface; negative means real depth is closer, often caused by occlusion or a
    # pose rendered behind foreground clutter.
    real_closer = float((signed_error < -20.0).mean())
    real_farther = float((signed_error > 20.0).mean())
    depth_score = float((valid_count / rendered_count) * math.exp(-median_error / scale_mm))

    return {
        "rendered_depth_pixels": float(rendered_count),
        "valid_real_depth_pixels": float(valid_count),
        "valid_real_depth_fraction": float(valid_count / rendered_count),
        "depth_abs_error_mean_mm": float(abs_error.mean()),
        "depth_abs_error_median_mm": median_error,
        "depth_abs_error_p90_mm": float(np.percentile(abs_error, 90)),
        "depth_signed_error_mean_mm": float(signed_error.mean()),
        "depth_support_5mm": support_5,
        "depth_support_10mm": support_10,
        "depth_support_20mm": support_20,
        "depth_support_50mm": support_50,
        "depth_support_tolerance": support_tolerance,
        "real_closer_fraction_20mm": real_closer,
        "real_farther_fraction_20mm": real_farther,
        "depth_consistency_score": depth_score,
    }


def load_metric_rows(path: Path) -> dict[tuple[str, int], dict]:
    return {(row["example_name"], int(row["pred_index"])): row for row in read_csv(path)}


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
    parser.add_argument("--bbox-pad", type=int, default=12)
    parser.add_argument("--tolerance-mm", type=float, default=20.0)
    parser.add_argument("--score-scale-mm", type=float, default=50.0)
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
    out_path = args.out or (examples_root / "megapose_depth_consistency_scores.csv")
    summary_path = out_path.with_suffix(".summary.json")
    fusion_path = out_path.with_name("megapose_depth_bbox_confidence_fusion_sweep.json")

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
        real_depth_raw = np.asarray(Image.open(scene_dir / "depth" / f"{image_stem}.png"))
        scene_camera = load_json(scene_dir / "scene_camera.json")
        camera_info = scene_camera[str(image_id)]
        depth_scale = float(camera_info.get("depth_scale", 1.0))
        real_depth_mm = real_depth_raw.astype(np.float32) * depth_scale
        height, width = real_depth_mm.shape
        camera_matrix = np.asarray(camera_info["cam_K"], dtype=np.float32).reshape(3, 3)

        for prediction in megapose_predictions:
            obj_id = int(prediction["obj_id"])
            if obj_id not in mesh_cache:
                mesh_cache[obj_id] = parse_ply_mesh(models_dir / f"obj_{obj_id:06d}.ply")

        _, _, full_masks, visible_masks, visible_depths = render_scene_depths(
            instances=megapose_predictions,
            mesh_cache=mesh_cache,
            camera_matrix=camera_matrix,
            image_shape=(height, width),
            face_stride=args.face_stride,
        )

        for idx, (pose_pred, full_mask, visible_mask, visible_depth) in enumerate(
            zip(megapose_predictions, full_masks, visible_masks, visible_depths)
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

            depth_scores = depth_consistency_scores(
                rendered_depth_mm=visible_depth,
                rendered_mask=visible_mask,
                real_depth_mm=real_depth_mm,
                detector_box=det_box,
                bbox_pad=args.bbox_pad,
                tolerance_mm=args.tolerance_mm,
                scale_mm=args.score_scale_mm,
            )

            key = (example_dir.name, idx)
            metric = metric_rows.get(key, {})
            failure = failure_rows.get(key, {})
            bbox_score = bbox_score_rows.get(key, {})
            detector_conf = float(matched_input["confidence"])
            cad_visible_bbox_score = float(bbox_score.get("cad_visible_bbox_score", 0.0))
            depth_consistency = float(depth_scores["depth_consistency_score"])
            depth_support_20 = float(depth_scores["depth_support_20mm"])
            fused_depth_score = (
                0.75 * detector_conf
                + 0.15 * cad_visible_bbox_score
                + 0.10 * depth_consistency
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
                    **depth_scores,
                    "fused_depth_score": fused_depth_score,
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
                    "depth_support_20mm_times_valid_fraction": (
                        depth_support_20 * float(depth_scores["valid_real_depth_fraction"])
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
        "valid_real_depth_fraction",
        "depth_support_10mm",
        "depth_support_20mm",
        "depth_support_50mm",
        "depth_support_tolerance",
        "depth_consistency_score",
        "depth_support_20mm_times_valid_fraction",
        "fused_depth_score",
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
        for score_name in [
            "depth_consistency_score",
            "fused_depth_score",
            "detector_confidence",
        ]:
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
    weights = [i / 10 for i in range(11)]
    for label_name, labels in label_specs.items():
        fusion_weight_sweep[label_name] = {}
        best = {"auroc": -1.0, "confidence": 0.0, "bbox": 0.0, "depth": 0.0}
        for w_conf in weights:
            for w_bbox in weights:
                if w_conf + w_bbox > 1.0:
                    continue
                w_depth = 1.0 - w_conf - w_bbox
                scores = [
                    w_conf * float(row["detector_confidence"])
                    + w_bbox * float(row["cad_visible_bbox_score"])
                    + w_depth * float(row["depth_consistency_score"])
                    for row in scored_rows
                ]
                value = roc_auc(labels, scores)
                key = f"confidence_{w_conf:.1f}_bbox_{w_bbox:.1f}_depth_{w_depth:.1f}"
                fusion_weight_sweep[label_name][key] = value
                if value > best["auroc"]:
                    best = {
                        "auroc": value,
                        "confidence": w_conf,
                        "bbox": w_bbox,
                        "depth": w_depth,
                    }
        fusion_weight_sweep[label_name]["best"] = best

    summary = {
        "method_family": "Practical CAD depth/render consistency",
        "note": (
            "Scores compare rendered CAD depth from MegaPose estimated poses to "
            "real T-LESS depth images. GT is used only for evaluation labels."
        ),
        "examples_root": str(examples_root),
        "rows": len(scored_rows),
        "depth_scale_source": "scene_camera.json depth_scale; depth PNG values are scaled to mm",
        "face_stride": args.face_stride,
        "bbox_pad": args.bbox_pad,
        "tolerance_mm": args.tolerance_mm,
        "score_scale_mm": args.score_scale_mm,
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
        "fusion_sweep": str(fusion_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fusion_path.write_text(json.dumps(fusion_weight_sweep, indent=2), encoding="utf-8")

    print(f"rows: {len(scored_rows)}")
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")
    print("AUROC:")
    for label_name, scores in auroc.items():
        print(f"  label={label_name}")
        for score_name, value in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {score_name}: {value:.4f}")
        best = fusion_weight_sweep[label_name]["best"]
        print(
            "    best_grid_fusion: "
            f"{best['auroc']:.4f} "
            f"(confidence={best['confidence']:.1f}, "
            f"bbox={best['bbox']:.1f}, depth={best['depth']:.1f})"
        )


if __name__ == "__main__":
    main()
