#!/usr/bin/env python3
"""Create empty-label real-background crops for marker/background suppression."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bbox_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def expand_bbox(
    bbox: list[float],
    margin: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (
        max(0.0, x - margin),
        max(0.0, y - margin),
        min(float(image_width), x + w + margin),
        min(float(image_height), y + h + margin),
    )


def sample_crop(
    rng: random.Random,
    image_width: int,
    image_height: int,
    crop_sizes: list[int],
) -> tuple[int, int, int, int]:
    crop_size = rng.choice(crop_sizes)
    crop_w = min(crop_size, image_width)
    crop_h = min(crop_size, image_height)
    x = rng.randint(0, image_width - crop_w)
    y = rng.randint(0, image_height - crop_h)
    return x, y, x + crop_w, y + crop_h


def crop_overlap_fraction(
    crop: tuple[int, int, int, int],
    expanded_boxes: list[tuple[float, float, float, float]],
) -> float:
    crop_area = float((crop[2] - crop[0]) * (crop[3] - crop[1]))
    if crop_area <= 0:
        return 1.0
    overlap = sum(bbox_intersection_area(crop, box) for box in expanded_boxes)
    return overlap / crop_area


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "derived"
        / "coco"
        / "tless_real_train_10pct_100_detection.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "tless",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "derived"
        / "real_background_negatives"
        / "tless_100real_marker_patches",
    )
    parser.add_argument("--num-crops", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-overlap", type=float, default=0.005)
    parser.add_argument("--box-margin", type=float, default=24.0)
    parser.add_argument("--max-attempts-per-crop", type=int, default=200)
    args = parser.parse_args()

    coco = load_json(args.annotation)
    images = coco["images"]
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)

    rng = random.Random(args.seed)
    crop_sizes = [192, 224, 256, 320, 384]
    image_dir = args.out_dir / "images"
    label_dir = args.out_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    attempts = 0
    while created < args.num_crops:
        image_info = rng.choice(images)
        source_path = args.dataset_root / image_info["file_name"]
        image = Image.open(source_path).convert("RGB")
        width, height = image.size
        expanded_boxes = [
            expand_bbox(ann["bbox"], args.box_margin, width, height)
            for ann in annotations_by_image.get(int(image_info["id"]), [])
        ]

        accepted_crop = None
        for _ in range(args.max_attempts_per_crop):
            attempts += 1
            crop = sample_crop(rng, width, height, crop_sizes)
            if crop_overlap_fraction(crop, expanded_boxes) <= args.max_overlap:
                accepted_crop = crop
                break

        if accepted_crop is None:
            continue

        crop_image = image.crop(accepted_crop)
        out_stem = f"real_bg_{created:06d}"
        crop_image.save(image_dir / f"{out_stem}.jpg", quality=92)
        (label_dir / f"{out_stem}.txt").write_text("", encoding="utf-8")
        created += 1

    metadata = {
        "source_annotation": str(args.annotation),
        "source_dataset_root": str(args.dataset_root),
        "num_crops": created,
        "seed": args.seed,
        "max_overlap": args.max_overlap,
        "box_margin": args.box_margin,
        "attempts": attempts,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"out_dir: {args.out_dir}")
    print(f"created_crops: {created}")
    print(f"attempts: {attempts}")


if __name__ == "__main__":
    main()

