#!/usr/bin/env python3
"""Quick sanity check for the local T-LESS BOP dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_images(path: Path) -> int:
    return sum(1 for pattern in ("*.png", "*.jpg", "*.jpeg") for _ in path.glob(pattern))


def choose_scene_and_image(test_dir: Path) -> tuple[Path, str]:
    scenes = sorted(p for p in test_dir.iterdir() if p.is_dir())
    if not scenes:
        raise FileNotFoundError(f"No scene directories under {test_dir}")

    for scene in scenes:
        rgb_dir = scene / "rgb"
        images = sorted(rgb_dir.glob("*.png"))
        if images:
            return scene, images[0].stem

    raise FileNotFoundError(f"No RGB images found under {test_dir}")


def build_overlay(rgb: Image.Image, mask_paths: list[Path]) -> np.ndarray:
    image = np.asarray(rgb.convert("RGB")).astype(np.float32) / 255.0
    overlay = image.copy()
    colors = np.array(
        [
            [1.0, 0.15, 0.10],
            [0.10, 0.65, 1.0],
            [0.20, 0.90, 0.35],
            [1.0, 0.85, 0.10],
            [0.80, 0.30, 1.0],
            [1.0, 0.45, 0.10],
        ],
        dtype=np.float32,
    )

    for idx, mask_path in enumerate(mask_paths):
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        color = colors[idx % len(colors)]
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color

    return np.clip(overlay, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "tless",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "outputs",
    )
    args = parser.parse_args()

    root = args.dataset_root
    test_dir = root / "test_primesense"
    train_pbr_dir = root / "train_pbr"
    model_dir = root / "models_cad"
    if not root.exists():
        raise FileNotFoundError(root)

    model_count = len(list(model_dir.glob("obj_*.ply")))
    scenes = sorted(p for p in test_dir.iterdir() if p.is_dir())
    pbr_scenes = sorted(p for p in train_pbr_dir.iterdir() if p.is_dir()) if train_pbr_dir.exists() else []
    rgb_count = sum(count_images(scene / "rgb") for scene in scenes)
    depth_count = sum(count_images(scene / "depth") for scene in scenes)
    mask_count = sum(count_images(scene / "mask_visib") for scene in scenes)
    pbr_rgb_count = sum(count_images(scene / "rgb") for scene in pbr_scenes)

    scene, image_id = choose_scene_and_image(test_dir)
    scene_gt = load_json(scene / "scene_gt.json")
    scene_camera = load_json(scene / "scene_camera.json")
    instances = scene_gt.get(str(int(image_id)), [])
    camera = scene_camera.get(str(int(image_id)), {})

    rgb_path = scene / "rgb" / f"{image_id}.png"
    mask_paths = sorted((scene / "mask_visib").glob(f"{image_id}_*.png"))
    rgb = Image.open(rgb_path)
    overlay = build_overlay(rgb, mask_paths)

    args.out.mkdir(parents=True, exist_ok=True)
    preview_path = args.out / f"tless_scene_{scene.name}_image_{image_id}_overlay.png"
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(preview_path, dpi=160)
    plt.close()

    obj_ids = sorted({item["obj_id"] for item in instances})
    print(f"dataset_root: {root}")
    print(f"cad_models: {model_count}")
    print(f"test_scenes: {len(scenes)}")
    print(f"test_rgb_images: {rgb_count}")
    print(f"test_depth_images: {depth_count}")
    print(f"visible_masks: {mask_count}")
    print(f"pbr_scenes: {len(pbr_scenes)}")
    print(f"pbr_rgb_images: {pbr_rgb_count}")
    print(f"sample_scene: {scene.name}")
    print(f"sample_image: {image_id}")
    print(f"sample_size: {rgb.width}x{rgb.height}")
    print(f"sample_instances: {len(instances)}")
    print(f"sample_obj_ids: {obj_ids}")
    print(f"sample_camera_keys: {sorted(camera.keys())}")
    print(f"preview: {preview_path}")


if __name__ == "__main__":
    main()
