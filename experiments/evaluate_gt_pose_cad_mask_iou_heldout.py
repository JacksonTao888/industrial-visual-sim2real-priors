#!/usr/bin/env python3
"""Evaluate GT-pose CAD rendered mask IoU on the held-out T-LESS real split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from render_cad_mask_iou import evaluate_image, make_overlay


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


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
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--face-stride", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-overlays", type=int, default=0, help="Save overlays for the first N images.")
    parser.add_argument(
        "--include-unannotated-scene-gt",
        action="store_true",
        help=(
            "By default, only COCO held-out annotations are included in metrics. "
            "Set this to also include scene_gt instances filtered out of COCO."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/tless_heldout_gt_pose_cad_mask_iou.csv.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    dataset_root = project_root / "datasets" / "tless"
    models_dir = dataset_root / "models_eval"
    coco_path = args.coco or (
        project_root / "datasets" / "derived" / "coco" / "tless_real_val_heldout_900_detection.json"
    )
    out_path = args.out or (project_root / "experiments" / "outputs" / "tless_heldout_gt_pose_cad_mask_iou.csv")
    summary_path = out_path.with_suffix(".summary.json")
    overlay_dir = out_path.parent / "tless_heldout_gt_pose_cad_mask_iou_overlays"

    coco = load_json(coco_path)
    images = sorted(coco["images"], key=lambda image: image["id"])
    if args.max_images is not None:
        images = images[: args.max_images]
    image_ids = {int(image["id"]) for image in images}
    coco_annotations_by_image = defaultdict(dict)
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        if image_id in image_ids:
            key = (int(ann["bop_instance_id"]), int(ann["category_id"]))
            coco_annotations_by_image[image_id][key] = int(ann["id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_overlays > 0:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    mesh_cache = {}
    all_rows = []
    saved_overlays = []
    scene_counts = defaultdict(int)

    for idx, image in enumerate(images, start=1):
        scene_id = int(image["scene_id"])
        bop_image_id = int(image["bop_image_id"])
        rows, rgb, rendered_visible, gt_visible = evaluate_image(
            dataset_root=dataset_root,
            split=args.split,
            scene_id=scene_id,
            image_id=bop_image_id,
            models_dir=models_dir,
            mesh_cache=mesh_cache,
            face_stride=args.face_stride,
        )
        image_id = int(image["id"])
        allowed = coco_annotations_by_image.get(image_id, {})
        for row in rows:
            key = (int(row["instance_index"]), int(row["obj_id"]))
            if not args.include_unannotated_scene_gt and key not in allowed:
                continue
            row["coco_image_id"] = image_id
            row["coco_ann_id"] = allowed.get(key, -1)
            all_rows.append(row)
        scene_counts[scene_id] += 1

        if len(saved_overlays) < args.save_overlays:
            overlay = make_overlay(rgb, rows, rendered_visible, gt_visible)
            overlay_path = (
                overlay_dir
                / f"scene_{scene_id:06d}_image_{bop_image_id:06d}_gt_pose_cad_mask_iou_overlay.png"
            )
            Image.fromarray(overlay).save(overlay_path)
            saved_overlays.append(str(overlay_path))

        if idx == 1 or idx % 50 == 0 or idx == len(images):
            print(f"processed_images: {idx}/{len(images)} instances: {len(all_rows)}", flush=True)

    fieldnames = [
        "coco_image_id",
        "coco_ann_id",
        "scene_id",
        "image_id",
        "instance_index",
        "obj_id",
        "visib_fract",
        "render_full_area",
        "render_visible_area",
        "gt_full_area",
        "gt_visible_area",
        "full_mask_iou",
        "visible_mask_iou",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    full_ious = [float(row["full_mask_iou"]) for row in all_rows]
    visible_ious = [float(row["visible_mask_iou"]) for row in all_rows]

    per_scene = {}
    per_object = {}
    for scene_id in sorted(scene_counts):
        scene_rows = [row for row in all_rows if int(row["scene_id"]) == scene_id]
        per_scene[f"{scene_id:06d}"] = {
            "images": scene_counts[scene_id],
            "instances": len(scene_rows),
            "full_mask_iou": summarize([float(row["full_mask_iou"]) for row in scene_rows]),
            "visible_mask_iou": summarize([float(row["visible_mask_iou"]) for row in scene_rows]),
        }

    obj_ids = sorted({int(row["obj_id"]) for row in all_rows})
    for obj_id in obj_ids:
        obj_rows = [row for row in all_rows if int(row["obj_id"]) == obj_id]
        per_object[f"{obj_id:06d}"] = {
            "instances": len(obj_rows),
            "full_mask_iou": summarize([float(row["full_mask_iou"]) for row in obj_rows]),
            "visible_mask_iou": summarize([float(row["visible_mask_iou"]) for row in obj_rows]),
        }

    visible_by_visib = {
        "visib_ge_0.75": summarize(
            [float(row["visible_mask_iou"]) for row in all_rows if float(row["visib_fract"]) >= 0.75]
        ),
        "visib_0.50_to_0.75": summarize(
            [
                float(row["visible_mask_iou"])
                for row in all_rows
                if 0.50 <= float(row["visib_fract"]) < 0.75
            ]
        ),
        "visib_lt_0.50": summarize(
            [float(row["visible_mask_iou"]) for row in all_rows if float(row["visib_fract"]) < 0.50]
        ),
    }

    summary = {
        "method_family": "Render-and-Compare / Test-Time CAD Use oracle upper bound",
        "backbone_paper": "MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare, CoRL 2022",
        "note": "GT pose + CAD mesh render compared to BOP full and visible masks.",
        "metric_instance_filter": (
            "all scene_gt instances"
            if args.include_unannotated_scene_gt
            else "COCO held-out annotations only"
        ),
        "coco_annotations_in_selected_images": sum(len(v) for v in coco_annotations_by_image.values()),
        "coco": str(coco_path),
        "split": args.split,
        "face_stride": args.face_stride,
        "images": len(images),
        "instances": len(all_rows),
        "full_mask_iou": summarize(full_ious),
        "visible_mask_iou": summarize(visible_ious),
        "visible_mask_iou_by_visibility": visible_by_visib,
        "per_scene": per_scene,
        "per_object": per_object,
        "csv": str(out_path),
        "saved_overlays": saved_overlays,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"images: {len(images)}")
    print(f"instances: {len(all_rows)}")
    print(f"mean_full_mask_iou: {summary['full_mask_iou']['mean']:.4f}")
    print(f"mean_visible_mask_iou: {summary['visible_mask_iou']['mean']:.4f}")
    print(f"median_visible_mask_iou: {summary['visible_mask_iou']['median']:.4f}")
    print(f"p10_visible_mask_iou: {summary['visible_mask_iou']['p10']:.4f}")
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")
    if saved_overlays:
        print("saved_overlays:")
        for path in saved_overlays:
            print(f"  {path}")


if __name__ == "__main__":
    main()
