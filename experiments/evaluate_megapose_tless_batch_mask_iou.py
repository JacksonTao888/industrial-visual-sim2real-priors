#!/usr/bin/env python3
"""Evaluate a batch of MegaPose T-LESS outputs by rendered CAD mask IoU."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_megapose_tless_example_mask_iou import (
    METRIC_FIELDNAMES,
    compute_prediction_metrics,
    load_megapose_predictions,
    load_scene_gt,
    match_same_class_gt_by_full_mask,
    make_overlay,
    summarize,
)
from render_cad_mask_iou import (
    parse_ply_mesh,
    render_scene_masks,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_example_dirs(examples_root: Path) -> list[Path]:
    return sorted(
        path
        for path in examples_root.glob("tless_scene_*_image_*")
        if path.is_dir() and (path / "manifest.json").exists()
    )


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "predictions": len(rows),
        "full_mask_iou": summarize([float(row["full_mask_iou"]) for row in rows]),
        "scene_visible_mask_iou": summarize(
            [float(row["scene_visible_mask_iou"]) for row in rows]
        ),
        "gt_visible_coverage_by_full_mask": summarize(
            [float(row["gt_visible_coverage_by_full_mask"]) for row in rows]
        ),
        "full_mask_vs_gt_visible_iou": summarize(
            [float(row["full_mask_vs_gt_visible_iou"]) for row in rows]
        ),
        "pred_visible_fraction": summarize(
            [float(row["pred_visible_fraction"]) for row in rows]
        ),
    }


def evaluate_example(
    *,
    project_root: Path,
    example_dir: Path,
    split: str,
    face_stride: int,
    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    save_overlay: bool,
    overlay_dir: Path,
) -> tuple[list[dict], dict]:
    dataset_root = project_root / "datasets" / "tless"
    models_dir = dataset_root / "models_eval"
    manifest = load_json(example_dir / "manifest.json")
    scene_id = int(manifest["scene_id"])
    image_id = int(manifest["image_id"])
    megapose_output = example_dir / "outputs" / "object_data.json"
    if not megapose_output.exists():
        return [], {
            "example_dir": str(example_dir),
            "scene_id": scene_id,
            "image_id": image_id,
            "status": "missing_output",
            "predictions": 0,
        }

    scene_dir = dataset_root / split / f"{scene_id:06d}"
    image_stem = f"{image_id:06d}"
    rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{image_stem}.png").convert("RGB"))
    height, width = rgb.shape[:2]
    scene_camera = load_json(scene_dir / "scene_camera.json")
    camera_matrix = np.asarray(scene_camera[str(image_id)]["cam_K"], dtype=np.float32).reshape(3, 3)

    predictions = load_megapose_predictions(megapose_output)
    for prediction in predictions:
        obj_id = int(prediction["obj_id"])
        if obj_id not in mesh_cache:
            mesh_cache[obj_id] = parse_ply_mesh(models_dir / f"obj_{obj_id:06d}.ply")

    _, pred_full, pred_visible = render_scene_masks(
        instances=predictions,
        mesh_cache=mesh_cache,
        camera_matrix=camera_matrix,
        image_shape=(height, width),
        face_stride=face_stride,
    )
    gt_instances = load_scene_gt(dataset_root, split, scene_id, image_id)

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
                "example_name": example_dir.name,
                "scene_id": scene_id,
                "image_id": image_id,
                "coco_image_id": int(manifest.get("coco_image_id", -1)),
                "pred_index": int(prediction["pred_index"]),
                "label": prediction["label"],
                "obj_id": int(prediction["obj_id"]),
                **metrics,
            }
        )

    overlay_path = None
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay = make_overlay(rgb, rows, pred_visible, gt_instances)
        overlay_path = overlay_dir / f"{example_dir.name}_megapose_mask_iou_overlay.png"
        Image.fromarray(overlay).save(overlay_path)

    return rows, {
        "example_dir": str(example_dir),
        "scene_id": scene_id,
        "image_id": image_id,
        "status": "evaluated",
        "predictions": len(rows),
        "mean_full_mask_iou": float(np.mean([row["full_mask_iou"] for row in rows])) if rows else 0.0,
        "mean_visible_mask_iou": float(np.mean([row["visible_mask_iou"] for row in rows])) if rows else 0.0,
        "mean_gt_visible_coverage_by_full_mask": (
            float(np.mean([row["gt_visible_coverage_by_full_mask"] for row in rows]))
            if rows
            else 0.0
        ),
        "overlay": str(overlay_path) if overlay_path is not None else None,
    }


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
        required=True,
        help="Directory containing prepared tless_scene_*_image_* example directories.",
    )
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--face-stride", type=int, default=1)
    parser.add_argument("--save-overlays", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root
    examples_root = args.examples_root
    out_path = args.out or (examples_root / "megapose_batch_mask_iou.csv")
    summary_path = out_path.with_suffix(".summary.json")
    overlay_dir = out_path.parent / "megapose_batch_mask_iou_overlays"

    example_dirs = find_example_dirs(examples_root)
    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_rows = []
    example_summaries = []
    overlays_left = args.save_overlays

    for idx, example_dir in enumerate(example_dirs, start=1):
        rows, example_summary = evaluate_example(
            project_root=project_root,
            example_dir=example_dir,
            split=args.split,
            face_stride=args.face_stride,
            mesh_cache=mesh_cache,
            save_overlay=overlays_left > 0,
            overlay_dir=overlay_dir,
        )
        if example_summary.get("overlay"):
            overlays_left -= 1
        all_rows.extend(rows)
        example_summaries.append(example_summary)
        if idx == 1 or idx % 10 == 0 or idx == len(example_dirs):
            print(
                f"processed_examples: {idx}/{len(example_dirs)} "
                f"evaluated_predictions: {len(all_rows)}",
                flush=True,
            )

    fieldnames = [
        "example_name",
        "scene_id",
        "image_id",
        "coco_image_id",
        "pred_index",
        "label",
        "obj_id",
        *METRIC_FIELDNAMES,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    rows_by_image = defaultdict(list)
    rows_by_scene = defaultdict(list)
    rows_by_object = defaultdict(list)
    for row in all_rows:
        rows_by_image[(int(row["scene_id"]), int(row["image_id"]))].append(row)
        rows_by_scene[int(row["scene_id"])].append(row)
        rows_by_object[int(row["obj_id"])].append(row)

    evaluated_examples = [s for s in example_summaries if s["status"] == "evaluated"]
    missing_outputs = [s for s in example_summaries if s["status"] == "missing_output"]
    visible_ious = [float(row["visible_mask_iou"]) for row in all_rows]
    full_ious = [float(row["full_mask_iou"]) for row in all_rows]
    gt_visible_coverages = [
        float(row["gt_visible_coverage_by_full_mask"]) for row in all_rows
    ]
    full_vs_gt_visible_ious = [
        float(row["full_mask_vs_gt_visible_iou"]) for row in all_rows
    ]
    pred_visible_fractions = [float(row["pred_visible_fraction"]) for row in all_rows]
    summary = {
        "method_family": "Render-and-Compare / Test-Time CAD Use",
        "backbone_paper": "MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare, CoRL 2022",
        "note": (
            "MegaPose predicted poses rendered with the local CAD mask renderer and "
            "matched to same-class BOP GT masks by best visible-mask IoU."
        ),
        "examples_root": str(examples_root),
        "split": args.split,
        "face_stride": args.face_stride,
        "examples_total": len(example_dirs),
        "examples_evaluated": len(evaluated_examples),
        "examples_missing_outputs": len(missing_outputs),
        "images_evaluated": len(rows_by_image),
        "predictions_evaluated": len(all_rows),
        "full_mask_iou": summarize(full_ious),
        "visible_mask_iou": summarize(visible_ious),
        "scene_visible_mask_iou": summarize(visible_ious),
        "full_mask_vs_gt_visible_iou": summarize(full_vs_gt_visible_ious),
        "gt_visible_coverage_by_full_mask": summarize(gt_visible_coverages),
        "pred_visible_fraction": summarize(pred_visible_fractions),
        "csv": str(out_path),
        "summary": str(summary_path),
        "missing_outputs": missing_outputs,
        "per_scene": {
            f"{scene_id:06d}": {
                "images": len({int(row["image_id"]) for row in scene_rows}),
                **summarize_rows(scene_rows),
            }
            for scene_id, scene_rows in sorted(rows_by_scene.items())
        },
        "per_object": {
            f"obj_{obj_id:06d}": summarize_rows(object_rows)
            for obj_id, object_rows in sorted(rows_by_object.items())
        },
        "per_example": example_summaries,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"examples_total: {summary['examples_total']}")
    print(f"examples_evaluated: {summary['examples_evaluated']}")
    print(f"examples_missing_outputs: {summary['examples_missing_outputs']}")
    print(f"predictions_evaluated: {summary['predictions_evaluated']}")
    print(f"mean_full_mask_iou: {summary['full_mask_iou']['mean']:.4f}")
    print(f"mean_scene_visible_mask_iou: {summary['scene_visible_mask_iou']['mean']:.4f}")
    print(
        "mean_gt_visible_coverage_by_full_mask: "
        f"{summary['gt_visible_coverage_by_full_mask']['mean']:.4f}"
    )
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
