#!/usr/bin/env python3
"""Sanity-check the no-CAD industrial anomaly datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def load_mask_overlay(
    image_path: Path,
    mask_path: Path | None,
    size: tuple[int, int],
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    if mask_path is not None and mask_path.exists():
        mask = Image.open(mask_path).convert("L").resize(original_size, Image.Resampling.NEAREST)
        red = Image.new("RGB", original_size, (255, 40, 40))
        image = Image.composite(Image.blend(image, red, 0.45), image, mask)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_grid(
    items: list[tuple[Path, str, Path | None]],
    out_path: Path,
    *,
    cols: int = 4,
    thumb_size: tuple[int, int] = (220, 170),
    label_height: int = 36,
) -> None:
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    width = cols * thumb_size[0]
    height = rows * (thumb_size[1] + label_height)
    grid = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for idx, (image_path, label, mask_path) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = col * thumb_size[0]
        y = row * (thumb_size[1] + label_height)
        thumb = load_mask_overlay(image_path, mask_path, thumb_size)
        grid.paste(thumb, (x, y))
        draw.rectangle(
            [x, y + thumb_size[1], x + thumb_size[0], y + thumb_size[1] + label_height],
            fill=(30, 30, 30),
        )
        draw.text(
            (x + 6, y + thumb_size[1] + 6),
            label[:36],
            fill=(255, 255, 255),
            font=font,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def summarize_mvtec(root: Path) -> dict:
    categories = sorted(path for path in root.iterdir() if path.is_dir())
    category_rows = []
    preview_items: list[tuple[Path, str, Path | None]] = []

    for category in categories:
        train_good = sorted((category / "train" / "good").glob("*.png"))
        test_good = sorted((category / "test" / "good").glob("*.png"))
        defect_dirs = sorted(
            path
            for path in (category / "test").iterdir()
            if path.is_dir() and path.name != "good"
        )
        anomaly_images = []
        masks = []
        defect_counts = {}
        for defect_dir in defect_dirs:
            defect_images = sorted(path for path in defect_dir.glob("*.png") if is_image(path))
            anomaly_images.extend(defect_images)
            defect_counts[defect_dir.name] = len(defect_images)
            gt_dir = category / "ground_truth" / defect_dir.name
            masks.extend(sorted(path for path in gt_dir.glob("*.png") if is_image(path)))

        category_rows.append(
            {
                "category": category.name,
                "train_good_images": len(train_good),
                "test_good_images": len(test_good),
                "test_anomaly_images": len(anomaly_images),
                "ground_truth_masks": len(masks),
                "defect_types": defect_counts,
            }
        )

        if train_good:
            preview_items.append((train_good[0], f"{category.name} train/good", None))
        if test_good:
            preview_items.append((test_good[0], f"{category.name} test/good", None))
        if anomaly_images:
            anomaly = anomaly_images[0]
            mask = (
                category
                / "ground_truth"
                / anomaly.parent.name
                / f"{anomaly.stem}_mask{anomaly.suffix}"
            )
            preview_items.append((anomaly, f"{category.name} {anomaly.parent.name}", mask))

    return {
        "root": str(root),
        "categories": len(category_rows),
        "category_names": [row["category"] for row in category_rows],
        "totals": {
            "train_good_images": sum(row["train_good_images"] for row in category_rows),
            "test_good_images": sum(row["test_good_images"] for row in category_rows),
            "test_anomaly_images": sum(row["test_anomaly_images"] for row in category_rows),
            "ground_truth_masks": sum(row["ground_truth_masks"] for row in category_rows),
            "total_images": sum(
                row["train_good_images"] + row["test_good_images"] + row["test_anomaly_images"]
                for row in category_rows
            ),
        },
        "per_category": category_rows,
        "preview_items": preview_items,
    }


def summarize_visa(root: Path) -> dict:
    categories = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name != "split_csv"
    )
    category_rows = []
    preview_items: list[tuple[Path, str, Path | None]] = []

    for category in categories:
        images_dir = category / "Data" / "Images"
        masks_dir = category / "Data" / "Masks"
        normal_images = sorted(
            path for path in (images_dir / "Normal").glob("*") if is_image(path)
        )
        anomaly_images = sorted(
            path for path in (images_dir / "Anomaly").glob("*") if is_image(path)
        )
        masks = sorted(path for path in masks_dir.rglob("*") if is_image(path))
        category_rows.append(
            {
                "category": category.name,
                "normal_images": len(normal_images),
                "anomaly_images": len(anomaly_images),
                "ground_truth_masks": len(masks),
                "annotation_csv": str(category / "image_anno.csv"),
            }
        )

        if normal_images:
            preview_items.append((normal_images[0], f"{category.name} normal", None))
        if anomaly_images:
            anomaly = anomaly_images[0]
            mask_candidates = [
                masks_dir / "Anomaly" / f"{anomaly.stem}.png",
                masks_dir / "Anomaly" / anomaly.name,
                masks_dir / f"{anomaly.stem}.png",
            ]
            mask = next((path for path in mask_candidates if path.exists()), None)
            preview_items.append((anomaly, f"{category.name} anomaly", mask))

    split_csv = sorted(path.name for path in (root / "split_csv").glob("*.csv"))
    return {
        "root": str(root),
        "categories": len(category_rows),
        "category_names": [row["category"] for row in category_rows],
        "split_csv_files": split_csv,
        "totals": {
            "normal_images": sum(row["normal_images"] for row in category_rows),
            "anomaly_images": sum(row["anomaly_images"] for row in category_rows),
            "ground_truth_masks": sum(row["ground_truth_masks"] for row in category_rows),
            "total_images": sum(
                row["normal_images"] + row["anomaly_images"] for row in category_rows
            ),
        },
        "per_category": category_rows,
        "preview_items": preview_items,
    }


def strip_preview_items(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "preview_items"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--mvtec-root", type=Path, default=None)
    parser.add_argument("--visa-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root
    mvtec_root = args.mvtec_root or project_root / "datasets" / "mvtec_ad"
    visa_root = args.visa_root or project_root / "datasets" / "visa"
    out_dir = args.out_dir or project_root / "experiments" / "outputs" / "no_cad_dataset_sanity"
    preview_dir = out_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    mvtec = summarize_mvtec(mvtec_root)
    visa = summarize_visa(visa_root)

    make_grid(
        mvtec["preview_items"][:36],
        preview_dir / "mvtec_ad_preview_grid.jpg",
        cols=3,
    )
    make_grid(
        visa["preview_items"][:36],
        preview_dir / "visa_preview_grid.jpg",
        cols=4,
    )

    summary = {
        "method_family": "CAD-unavailable dataset sanity check",
        "datasets": {
            "mvtec_ad": strip_preview_items(mvtec),
            "visa": strip_preview_items(visa),
        },
        "previews": {
            "mvtec_ad": str(preview_dir / "mvtec_ad_preview_grid.jpg"),
            "visa": str(preview_dir / "visa_preview_grid.jpg"),
        },
    }
    summary_path = out_dir / "no_cad_dataset_sanity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("MVTec AD")
    print(f"  categories: {mvtec['categories']}")
    for key, value in mvtec["totals"].items():
        print(f"  {key}: {value}")
    print("VisA")
    print(f"  categories: {visa['categories']}")
    for key, value in visa["totals"].items():
        print(f"  {key}: {value}")
    print(f"summary: {summary_path}")
    print(f"mvtec_preview: {preview_dir / 'mvtec_ad_preview_grid.jpg'}")
    print(f"visa_preview: {preview_dir / 'visa_preview_grid.jpg'}")


if __name__ == "__main__":
    main()
