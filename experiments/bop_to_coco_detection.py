#!/usr/bin/env python3
"""Convert BOP scene annotations to COCO-style object detection JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def find_image(scene_dir: Path, image_key: str) -> Path:
    image_stem = f"{int(image_key):06d}"
    for ext in IMAGE_EXTENSIONS:
        path = scene_dir / "rgb" / f"{image_stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No RGB image found for {scene_dir.name}/{image_stem}")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def make_categories(models_info_path: Path) -> list[dict[str, Any]]:
    models_info = load_json(models_info_path)
    categories = []
    for obj_id in sorted(int(key) for key in models_info.keys()):
        categories.append(
            {
                "id": obj_id,
                "name": f"obj_{obj_id:06d}",
                "supercategory": "tless_object",
            }
        )
    return categories


def valid_bbox(bbox: list[float], width: int, height: int) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + w)
    y2 = min(float(height), y + h)
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return [x1, y1, clipped_w, clipped_h]


def convert_split(
    dataset_root: Path,
    split_name: str,
    out_path: Path,
    *,
    min_visible_area: int,
    min_visib_fract: float,
) -> dict[str, int]:
    split_dir = dataset_root / split_name
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    coco: dict[str, Any] = {
        "info": {
            "description": f"T-LESS {split_name} converted from BOP to COCO detection",
            "source_dataset": "BOP T-LESS",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": make_categories(dataset_root / "models_cad" / "models_info.json"),
    }

    annotation_id = 1
    image_id = 1
    skipped_invisible = 0
    skipped_missing_info = 0

    scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    for scene_dir in scene_dirs:
        scene_gt = load_json(scene_dir / "scene_gt.json")
        scene_gt_info_path = scene_dir / "scene_gt_info.json"
        scene_gt_info = load_json(scene_gt_info_path) if scene_gt_info_path.exists() else {}

        for image_key in sorted(scene_gt.keys(), key=lambda value: int(value)):
            image_path = find_image(scene_dir, image_key)
            width, height = image_size(image_path)
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": str(image_path.relative_to(dataset_root)),
                    "width": width,
                    "height": height,
                    "scene_id": int(scene_dir.name),
                    "bop_image_id": int(image_key),
                }
            )

            gt_entries = scene_gt[image_key]
            info_entries = scene_gt_info.get(image_key)
            if info_entries is None:
                skipped_missing_info += len(gt_entries)
                info_entries = [{} for _ in gt_entries]

            for inst_idx, gt in enumerate(gt_entries):
                info = info_entries[inst_idx] if inst_idx < len(info_entries) else {}
                if info.get("px_count_visib", 0) < min_visible_area:
                    skipped_invisible += 1
                    continue
                if info.get("visib_fract", 0.0) < min_visib_fract:
                    skipped_invisible += 1
                    continue

                bbox = valid_bbox(info.get("bbox_visib", [-1, -1, -1, -1]), width, height)
                if bbox is None:
                    skipped_invisible += 1
                    continue

                area = float(info.get("px_count_visib", bbox[2] * bbox[3]))
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(gt["obj_id"]),
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                        "bop_instance_id": inst_idx,
                        "visib_fract": float(info.get("visib_fract", 0.0)),
                    }
                )
                annotation_id += 1

            image_id += 1

    write_json(out_path, coco)
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories": len(coco["categories"]),
        "skipped_invisible": skipped_invisible,
        "skipped_missing_info": skipped_missing_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "tless",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "derived" / "coco",
    )
    parser.add_argument("--min-visible-area", type=int, default=16)
    parser.add_argument("--min-visib-fract", type=float, default=0.0)
    args = parser.parse_args()

    splits = {
        "train_pbr": args.out_dir / "tless_train_pbr_detection.json",
        "test_primesense": args.out_dir / "tless_test_primesense_detection.json",
    }
    for split_name, out_path in splits.items():
        stats = convert_split(
            args.dataset_root,
            split_name,
            out_path,
            min_visible_area=args.min_visible_area,
            min_visib_fract=args.min_visib_fract,
        )
        print(f"{split_name}: {out_path}")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

