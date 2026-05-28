#!/usr/bin/env python3
"""Prepare a batch of T-LESS images in MegaPose example format."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
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


def safe_symlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def example_name(scene_id: int, image_id: int) -> str:
    return f"tless_scene_{scene_id:06d}_image_{image_id:06d}"


def make_camera_data(scene_camera: dict, image_id: int, height: int, width: int) -> dict:
    camera = scene_camera[str(image_id)]
    return {
        "K": [
            camera["cam_K"][0:3],
            camera["cam_K"][3:6],
            camera["cam_K"][6:9],
        ],
        "resolution": [height, width],
    }


def select_images(
    images: list[dict],
    preds_by_image_id: dict[int, list[dict]],
    sampling: str,
    max_images: int,
    offset: int,
    images_per_scene: int | None,
) -> list[dict]:
    eligible = [image for image in images if int(image["id"]) in preds_by_image_id]
    if sampling == "sequential":
        return eligible[offset : offset + max_images]

    if sampling != "scene_round_robin":
        raise ValueError(f"Unknown sampling mode: {sampling}")

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for image in eligible:
        by_scene[int(image["scene_id"])].append(image)

    scene_ids = sorted(by_scene)
    selected = []
    per_scene_counts = defaultdict(int)
    depth = 0
    while len(selected) < max_images:
        added_this_round = False
        for scene_id in scene_ids:
            if len(selected) >= max_images:
                break
            if images_per_scene is not None and per_scene_counts[scene_id] >= images_per_scene:
                continue
            scene_images = by_scene[scene_id]
            if depth >= len(scene_images):
                continue
            selected.append(scene_images[depth])
            per_scene_counts[scene_id] += 1
            added_this_round = True
        if not added_this_round:
            break
        depth += 1

    if offset:
        selected = selected[offset:]
    return selected[:max_images]


def write_example(
    *,
    project_root: Path,
    image: dict,
    preds: list[dict],
    predictions_path: Path,
    out_root: Path,
    megapose_examples_root: Path | None,
    conf_threshold: float,
    max_detections: int,
) -> dict:
    dataset_root = project_root / "datasets" / "tless"
    split = "test_primesense"
    scene_id = int(image["scene_id"])
    image_id = int(image["bop_image_id"])
    name = example_name(scene_id, image_id)
    scene_dir = dataset_root / split / f"{scene_id:06d}"
    image_stem = f"{image_id:06d}"
    rgb_path = scene_dir / "rgb" / f"{image_stem}.png"
    scene_camera = load_json(scene_dir / "scene_camera.json")
    out_dir = out_root / name

    selected = sorted(preds, key=lambda row: float(row["confidence"]), reverse=True)[
        :max_detections
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(exist_ok=True)
    (out_dir / "meshes").mkdir(exist_ok=True)
    symlink_or_copy(rgb_path.resolve(), out_dir / "image_rgb.png")
    (out_dir / "camera_data.json").write_text(
        json.dumps(make_camera_data(scene_camera, image_id, int(image["height"]), int(image["width"])), indent=2),
        encoding="utf-8",
    )

    object_data = []
    labels_used = set()
    for pred in selected:
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

    linked_path = None
    if megapose_examples_root is not None:
        megapose_examples_root.mkdir(parents=True, exist_ok=True)
        linked_path = megapose_examples_root / name
        safe_symlink(out_dir.resolve(), linked_path)

    manifest = {
        "format": "MegaPose example input",
        "scene_id": scene_id,
        "image_id": image_id,
        "coco_image_id": int(image["id"]),
        "source_rgb": str(rgb_path),
        "source_predictions": str(predictions_path),
        "conf_threshold": conf_threshold,
        "max_detections": max_detections,
        "detections": len(object_data),
        "labels": sorted(labels_used),
        "out_dir": str(out_dir),
        "megapose_example_name": name,
        "megapose_examples_symlink": str(linked_path) if linked_path is not None else None,
        "notes": [
            "Labels match mesh directory names and object_data.json labels.",
            "Meshes use T-LESS models_eval PLY files in millimeters.",
            "Depth is intentionally omitted for the RGB MegaPose model.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


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
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=["sequential", "scene_round_robin"],
        default="sequential",
        help="Image selection strategy among held-out images with detections.",
    )
    parser.add_argument(
        "--images-per-scene",
        type=int,
        default=None,
        help="Cap images per scene when using scene_round_robin.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/megapose_batches/conf_<threshold>_n<max-images>.",
    )
    parser.add_argument(
        "--megapose-examples-root",
        default=os.environ.get("MEGAPOSE_EXAMPLES_ROOT", ""),
        help="Symlink prepared examples here. Defaults to MEGAPOSE_EXAMPLES_ROOT; use an empty string to disable.",
    )
    parser.add_argument(
        "--megapose-root",
        default=os.environ.get("MEGAPOSE_ROOT", ""),
        help="MegaPose checkout used in the generated command file. Defaults to MEGAPOSE_ROOT.",
    )
    parser.add_argument(
        "--megapose-data-dir",
        default=os.environ.get("MEGAPOSE_DATA_DIR", ""),
        help="MegaPose data directory written into the generated command file. Defaults to MEGAPOSE_DATA_DIR.",
    )
    parser.add_argument(
        "--model",
        default="megapose-1.0-RGB-multi-hypothesis",
        help="Model name written into the generated command file.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    coco_path = args.coco or (
        project_root / "datasets" / "derived" / "coco" / "tless_real_val_heldout_900_detection.json"
    )
    predictions_path = args.predictions or (
        project_root / "experiments" / "outputs" / "b6_tless_real_predictions.csv"
    )
    threshold_label = str(args.conf_threshold).replace(".", "p")
    sampling_label = (
        args.sampling
        if args.images_per_scene is None
        else f"{args.sampling}_per_scene_{args.images_per_scene}"
    )
    out_root = args.out_root or (
        project_root
        / "experiments"
        / "outputs"
        / "megapose_batches"
        / (
            f"conf_{threshold_label}_maxdet_{args.max_detections}_"
            f"{sampling_label}_n{args.max_images}_offset{args.offset}"
        )
    )
    megapose_examples_root = (
        None if args.megapose_examples_root == "" else Path(args.megapose_examples_root)
    )
    if megapose_examples_root is None:
        megapose_examples_root = None

    coco = load_json(coco_path)
    predictions = [
        row for row in read_predictions(predictions_path) if float(row["confidence"]) >= args.conf_threshold
    ]
    preds_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for row in predictions:
        preds_by_image_id[int(row["image_id"])].append(row)

    images = sorted(coco["images"], key=lambda image: int(image["id"]))
    selected_images = select_images(
        images=images,
        preds_by_image_id=preds_by_image_id,
        sampling=args.sampling,
        max_images=args.max_images,
        offset=args.offset,
        images_per_scene=args.images_per_scene,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for image in selected_images:
        manifests.append(
            write_example(
                project_root=project_root,
                image=image,
                preds=preds_by_image_id[int(image["id"])],
                predictions_path=predictions_path,
                out_root=out_root,
                megapose_examples_root=megapose_examples_root,
                conf_threshold=args.conf_threshold,
                max_detections=args.max_detections,
            )
        )

    commands_path = out_root / "run_megapose_inference_commands.sh"
    with commands_path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        if args.megapose_root:
            f.write(f"cd {args.megapose_root}\n")
        if args.megapose_data_dir:
            f.write(f"export MEGAPOSE_DATA_DIR={args.megapose_data_dir}\n")
        f.write("\n")
        for manifest in manifests:
            f.write(
                "python -m megapose.scripts.run_inference_on_example "
                f"{manifest['megapose_example_name']} "
                f"--run-inference --model {args.model}\n"
            )
    commands_path.chmod(0o755)

    summary = {
        "format": "MegaPose T-LESS batch preparation summary",
        "coco": str(coco_path),
        "predictions": str(predictions_path),
        "conf_threshold": args.conf_threshold,
        "max_detections": args.max_detections,
        "max_images": args.max_images,
        "offset": args.offset,
        "sampling": args.sampling,
        "images_per_scene": args.images_per_scene,
        "selected_images": len(manifests),
        "total_detections": sum(int(m["detections"]) for m in manifests),
        "scenes": sorted({int(m["scene_id"]) for m in manifests}),
        "images_by_scene": {
            f"{scene_id:06d}": sum(1 for m in manifests if int(m["scene_id"]) == scene_id)
            for scene_id in sorted({int(m["scene_id"]) for m in manifests})
        },
        "out_root": str(out_root),
        "megapose_examples_root": str(megapose_examples_root) if megapose_examples_root else None,
        "commands": str(commands_path),
        "examples": manifests,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"out_root: {out_root}")
    print(f"examples: {len(manifests)}")
    print(f"detections: {summary['total_detections']}")
    print(f"commands: {commands_path}")
    print(f"summary: {summary_path}")
    for manifest in manifests[:20]:
        print(
            "  "
            f"{manifest['megapose_example_name']} "
            f"detections={manifest['detections']} "
            f"labels={','.join(manifest['labels'])}"
        )


if __name__ == "__main__":
    main()
