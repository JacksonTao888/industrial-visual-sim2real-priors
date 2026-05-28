#!/usr/bin/env python3
"""Summarize and visualize a COCO detection annotation file."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_preview(dataset_root: Path, coco: dict[str, Any], out_path: Path, image_index: int) -> None:
    images = coco["images"]
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    image_info = images[image_index % len(images)]
    image_path = dataset_root / image_info["file_name"]
    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(image)
    ax.axis("off")

    for ann in annotations_by_image[image_info["id"]]:
        x, y, w, h = ann["bbox"]
        color = plt.cm.tab20((ann["category_id"] - 1) % 20)
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x,
            max(0, y - 2),
            f"{ann['category_id']}",
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_distribution(category_counts: Counter[int], out_path: Path) -> None:
    ids = np.array(sorted(category_counts.keys()))
    counts = np.array([category_counts[int(obj_id)] for obj_id in ids])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(ids, counts, color="#3b82f6")
    ax.set_xlabel("Object ID")
    ax.set_ylabel("Annotations")
    ax.set_title("T-LESS Object Annotation Distribution")
    ax.set_xticks(ids)
    ax.grid(axis="y", alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "tless",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "outputs",
    )
    parser.add_argument("--preview-index", type=int, default=0)
    args = parser.parse_args()

    coco = load_json(args.annotation)
    annotation_name = args.annotation.stem
    category_counts = Counter(int(ann["category_id"]) for ann in coco["annotations"])
    ann_counts_by_image = Counter(int(ann["image_id"]) for ann in coco["annotations"])

    distribution_path = args.out_dir / f"{annotation_name}_class_distribution.png"
    preview_path = args.out_dir / f"{annotation_name}_preview.png"
    draw_distribution(category_counts, distribution_path)
    draw_preview(args.dataset_root, coco, preview_path, args.preview_index)

    print(f"annotation: {args.annotation}")
    print(f"images: {len(coco['images'])}")
    print(f"annotations: {len(coco['annotations'])}")
    print(f"categories: {len(coco['categories'])}")
    print(f"images_with_annotations: {len(ann_counts_by_image)}")
    print(f"annotations_per_image_mean: {np.mean(list(ann_counts_by_image.values())):.2f}")
    print(f"annotations_per_image_max: {max(ann_counts_by_image.values())}")
    print(f"min_category_count: {min(category_counts.values())}")
    print(f"max_category_count: {max(category_counts.values())}")
    print(f"distribution: {distribution_path}")
    print(f"preview: {preview_path}")


if __name__ == "__main__":
    main()

