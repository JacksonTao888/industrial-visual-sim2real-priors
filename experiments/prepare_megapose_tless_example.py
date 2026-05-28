#!/usr/bin/env python3
"""Prepare one T-LESS image + B6 detections in MegaPose example format."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def symlink_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        import shutil

        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/b6_tless_real_predictions.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/megapose_examples/tless_scene_<scene>_image_<image>.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    dataset_root = project_root / "datasets" / "tless"
    scene_dir = dataset_root / "test_primesense" / f"{args.scene_id:06d}"
    image_stem = f"{args.image_id:06d}"
    predictions_path = args.predictions or (project_root / "experiments" / "outputs" / "b6_tless_real_predictions.csv")
    out_dir = args.out_dir or (
        project_root
        / "experiments"
        / "outputs"
        / "megapose_examples"
        / f"tless_scene_{args.scene_id:06d}_image_{args.image_id:06d}"
    )

    rgb_path = scene_dir / "rgb" / f"{image_stem}.png"
    camera_path = scene_dir / "scene_camera.json"
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    preds = [
        row
        for row in read_predictions(predictions_path)
        if int(row["scene_id"]) == args.scene_id
        and int(row["bop_image_id"]) == args.image_id
        and float(row["confidence"]) >= args.conf_threshold
    ]
    preds = sorted(preds, key=lambda row: float(row["confidence"]), reverse=True)[: args.max_detections]
    if not preds:
        raise RuntimeError(
            f"No predictions found for scene={args.scene_id} image={args.image_id} "
            f"at conf>={args.conf_threshold}"
        )

    scene_camera = load_json(camera_path)
    camera = scene_camera[str(args.image_id)]
    camera_data = {
        "K": [
            camera["cam_K"][0:3],
            camera["cam_K"][3:6],
            camera["cam_K"][6:9],
        ],
        "resolution": [540, 720],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(exist_ok=True)
    (out_dir / "meshes").mkdir(exist_ok=True)
    symlink_or_copy(rgb_path.resolve(), out_dir / "image_rgb.png")
    (out_dir / "camera_data.json").write_text(json.dumps(camera_data, indent=2), encoding="utf-8")

    object_data = []
    labels_used = set()
    for pred in preds:
        label = pred["class_name"]
        labels_used.add(label)
        object_data.append(
            {
                "label": label,
                "bbox_modal": [
                    round(float(pred["x1"]), 4),
                    round(float(pred["y1"]), 4),
                    round(float(pred["x2"]), 4),
                    round(float(pred["y2"]), 4),
                ],
            }
        )
    (out_dir / "inputs" / "object_data.json").write_text(
        json.dumps(object_data, indent=2), encoding="utf-8"
    )

    for label in sorted(labels_used):
        obj_id = int(label.split("_")[-1])
        mesh_source = dataset_root / "models_eval" / f"obj_{obj_id:06d}.ply"
        mesh_dir = out_dir / "meshes" / label
        mesh_dir.mkdir(parents=True, exist_ok=True)
        symlink_or_copy(mesh_source.resolve(), mesh_dir / f"{label}.ply")

    manifest = {
        "format": "MegaPose example input",
        "scene_id": args.scene_id,
        "image_id": args.image_id,
        "source_rgb": str(rgb_path),
        "source_predictions": str(predictions_path),
        "conf_threshold": args.conf_threshold,
        "detections": len(object_data),
        "labels": sorted(labels_used),
        "out_dir": str(out_dir),
        "notes": [
            "Labels match mesh directory names and object_data.json labels.",
            "Meshes use T-LESS models_eval PLY files in millimeters.",
            "Depth is intentionally omitted for the RGB MegaPose model.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"out_dir: {out_dir}")
    print(f"detections: {len(object_data)}")
    print(f"labels: {', '.join(sorted(labels_used))}")
    print(f"object_data: {out_dir / 'inputs' / 'object_data.json'}")
    print(f"camera_data: {out_dir / 'camera_data.json'}")
    print(f"manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
