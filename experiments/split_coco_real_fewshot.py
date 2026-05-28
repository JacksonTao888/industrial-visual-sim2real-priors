#!/usr/bin/env python3
"""Create fixed real-image few-shot train splits plus a held-out real validation split."""

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


def image_subset(coco: dict[str, Any], selected_ids: set[int], source: Path, name: str) -> dict[str, Any]:
    annotations = [ann for ann in coco["annotations"] if int(ann["image_id"]) in selected_ids]
    images = [image for image in coco["images"] if int(image["id"]) in selected_ids]
    return {
        **coco,
        "info": {
            **coco.get("info", {}),
            "split_source": str(source),
            "split_name": name,
            "split_images": len(images),
        },
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "derived"
        / "coco"
        / "tless_test_primesense_detection.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "derived" / "coco",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--real-val-images", type=int, default=900)
    args = parser.parse_args()

    coco = load_json(args.annotation)
    images = sorted(coco["images"], key=lambda item: int(item["id"]))
    rng = random.Random(args.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    val_images = sorted(shuffled[: args.real_val_images], key=lambda item: int(item["id"]))
    real_pool = sorted(shuffled[args.real_val_images :], key=lambda item: int(item["id"]))

    val_ids = {int(image["id"]) for image in val_images}
    pool_ids = [int(image["id"]) for image in real_pool]

    split_specs = {
        "real_train_1pct_10": pool_ids[:10],
        "real_train_5pct_50": pool_ids[:50],
        "real_train_10pct_100": pool_ids[:100],
        "real_val_heldout_900": sorted(val_ids),
    }

    for name, ids in split_specs.items():
        selected_ids = set(ids)
        split = image_subset(coco, selected_ids, args.annotation, name)
        out_path = args.out_dir / f"tless_{name}_detection.json"
        write_json(out_path, split)
        print(f"{name}: {out_path}")
        print(f"  images: {len(split['images'])}")
        print(f"  annotations: {len(split['annotations'])}")


if __name__ == "__main__":
    main()

