#!/usr/bin/env python3
"""Create a reproducible image-level subset of a COCO detection annotation file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coco = load_json(args.annotation)
    images = list(coco["images"])
    rng = random.Random(args.seed)
    if args.num_images < len(images):
        selected_images = sorted(rng.sample(images, args.num_images), key=lambda item: item["id"])
    else:
        selected_images = images

    selected_image_ids = {image["id"] for image in selected_images}
    selected_annotations = [
        ann for ann in coco["annotations"] if ann["image_id"] in selected_image_ids
    ]

    subset = {
        **coco,
        "info": {
            **coco.get("info", {}),
            "subset_source": str(args.annotation),
            "subset_num_images": len(selected_images),
            "subset_seed": args.seed,
        },
        "images": selected_images,
        "annotations": selected_annotations,
    }
    write_json(args.out, subset)

    print(f"source: {args.annotation}")
    print(f"out: {args.out}")
    print(f"images: {len(selected_images)}")
    print(f"annotations: {len(selected_annotations)}")
    print(f"seed: {args.seed}")


if __name__ == "__main__":
    main()

