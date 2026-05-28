#!/usr/bin/env python3
"""Combine an existing YOLO dataset with empty-label real background negatives."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def symlink_or_keep(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def link_tree_files(src_dir: Path, dst_dir: Path, suffixes: tuple[str, ...]) -> int:
    count = 0
    for src in sorted(src_dir.iterdir()):
        if src.suffix.lower() not in suffixes:
            continue
        symlink_or_keep(src, dst_dir / src.name)
        count += 1
    return count


def copy_labels(src_dir: Path, dst_dir: Path) -> int:
    count = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.txt")):
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        count += 1
    return count


def write_yaml(out_root: Path, source_yaml: Path) -> None:
    lines = source_yaml.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("path:"):
            new_lines.append(f"path: {out_root.resolve()}")
        else:
            new_lines.append(line)
    (out_root / "data.yaml").write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-yolo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "derived"
        / "yolo"
        / "tless_pbr_full_to_real",
    )
    parser.add_argument(
        "--background-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "derived"
        / "real_background_negatives"
        / "tless_100real_marker_patches",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "derived"
        / "yolo"
        / "tless_pbr_full_plus_real_bg_negatives",
    )
    args = parser.parse_args()

    base = args.base_yolo_root
    out = args.out_root
    bg = args.background_root

    train_img_count = link_tree_files(base / "images" / "train", out / "images" / "train", (".jpg", ".jpeg", ".png"))
    val_img_count = link_tree_files(base / "images" / "val", out / "images" / "val", (".jpg", ".jpeg", ".png"))
    train_label_count = copy_labels(base / "labels" / "train", out / "labels" / "train")
    val_label_count = copy_labels(base / "labels" / "val", out / "labels" / "val")

    bg_img_count = 0
    bg_label_count = 0
    for image_path in sorted((bg / "images").glob("*.jpg")):
        dst_name = f"bgneg__{image_path.name}"
        symlink_or_keep(image_path, out / "images" / "train" / dst_name)
        (out / "labels" / "train" / f"{Path(dst_name).stem}.txt").write_text("", encoding="utf-8")
        bg_img_count += 1
        bg_label_count += 1

    write_yaml(out, base / "data.yaml")

    print(f"out_root: {out}")
    print(f"base_train_images: {train_img_count}")
    print(f"base_train_labels: {train_label_count}")
    print(f"background_negative_images: {bg_img_count}")
    print(f"background_negative_labels: {bg_label_count}")
    print(f"val_images: {val_img_count}")
    print(f"val_labels: {val_label_count}")
    print(f"data_yaml: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()

