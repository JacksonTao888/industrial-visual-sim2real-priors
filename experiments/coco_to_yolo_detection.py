#!/usr/bin/env python3
"""Export COCO detection annotations to a YOLO-style dataset with symlinked images."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_stem(file_name: str) -> str:
    return file_name.replace("/", "__").rsplit(".", 1)[0]


def normalize_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    return cx, cy, w / width, h / height


def symlink_or_keep(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def export_split(dataset_root: Path, coco_path: Path, out_root: Path, split_name: str) -> int:
    coco = load_json(coco_path)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)

    image_dir = out_root / "images" / split_name
    label_dir = out_root / "labels" / split_name
    label_dir.mkdir(parents=True, exist_ok=True)

    for image in coco["images"]:
        src = (dataset_root / image["file_name"]).resolve()
        suffix = src.suffix.lower()
        stem = safe_stem(image["file_name"])
        dst = image_dir / f"{stem}{suffix}"
        symlink_or_keep(src, dst)

        label_path = label_dir / f"{stem}.txt"
        lines = []
        for ann in annotations_by_image.get(int(image["id"]), []):
            class_id = int(ann["category_id"]) - 1
            cx, cy, w, h = normalize_bbox(ann["bbox"], int(image["width"]), int(image["height"]))
            lines.append(f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return len(coco["images"])


def write_yaml(out_root: Path, split_names: list[str], names: list[str]) -> None:
    lines = [f"path: {out_root.resolve()}"]
    if "train" in split_names:
        lines.append("train: images/train")
    if "val" in split_names:
        lines.append("val: images/val")
    if "test" in split_names:
        lines.append("test: images/test")
    lines.append("names:")
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "tless",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        help="Split spec in the form name:/path/to/coco.json, e.g. train:train.json",
    )
    args = parser.parse_args()

    split_specs: list[tuple[str, Path]] = []
    for spec in args.split:
        if ":" not in spec:
            raise ValueError(f"Invalid split spec: {spec}")
        name, path = spec.split(":", 1)
        split_specs.append((name, Path(path)))

    names = [f"obj_{obj_id:06d}" for obj_id in range(1, 31)]
    exported_names = []
    for split_name, coco_path in split_specs:
        count = export_split(args.dataset_root, coco_path, args.out_root, split_name)
        exported_names.append(split_name)
        print(f"{split_name}: {count} images")

    write_yaml(args.out_root, exported_names, names)
    print(f"data_yaml: {args.out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()

