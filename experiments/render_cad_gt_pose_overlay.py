#!/usr/bin/env python3
"""Project T-LESS CAD models into real images using ground-truth BOP poses."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


COLORS = [
    (255, 76, 76),
    (76, 179, 255),
    (76, 220, 120),
    (255, 212, 64),
    (207, 107, 255),
    (255, 143, 64),
    (64, 235, 218),
    (245, 105, 180),
]

BBOX_EDGES = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

PLY_TYPES = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "float": ("f", 4),
    "double": ("d", 8),
    "float32": ("f", 4),
    "float64": ("d", 8),
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_ply_vertices(path: Path) -> np.ndarray:
    """Read vertex xyz coordinates from simple ASCII or binary-little-endian PLY."""

    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        if header_lines[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt = None
        vertex_count = 0
        vertex_props: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[:1] == ["element"] and parts[1] != "vertex":
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"] and parts[1] != "list":
                vertex_props.append((parts[2], parts[1]))

        if vertex_count <= 0:
            raise ValueError(f"No vertices found in {path}")
        prop_names = [name for name, _ in vertex_props]
        try:
            xyz_indices = [prop_names.index(axis) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise ValueError(f"PLY is missing xyz vertex properties: {path}") from exc

        if fmt == "ascii":
            vertices = np.empty((vertex_count, 3), dtype=np.float32)
            for idx in range(vertex_count):
                values = f.readline().decode("ascii", errors="replace").split()
                vertices[idx] = [float(values[i]) for i in xyz_indices]
            return vertices

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")

        struct_fmt = "<" + "".join(PLY_TYPES[prop_type][0] for _, prop_type in vertex_props)
        vertex_size = struct.calcsize(struct_fmt)
        vertices = np.empty((vertex_count, 3), dtype=np.float32)
        for idx in range(vertex_count):
            chunk = f.read(vertex_size)
            values = struct.unpack(struct_fmt, chunk)
            vertices[idx] = [values[i] for i in xyz_indices]
        return vertices


def bbox_corners(model_info: dict) -> np.ndarray:
    min_x = float(model_info["min_x"])
    min_y = float(model_info["min_y"])
    min_z = float(model_info["min_z"])
    max_x = min_x + float(model_info["size_x"])
    max_y = min_y + float(model_info["size_y"])
    max_z = min_z + float(model_info["size_z"])
    return np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [min_x, max_y, min_z],
            [max_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [min_x, max_y, max_z],
            [max_x, max_y, max_z],
        ],
        dtype=np.float32,
    )


def transform_points(points_m: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points_m @ rotation.T + translation[None, :]


def project_points(points_c: np.ndarray, camera_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = points_c[:, 2] > 1e-6
    projected = np.full((points_c.shape[0], 2), np.nan, dtype=np.float32)
    if valid.any():
        pts = points_c[valid]
        xy = pts[:, :2] / pts[:, 2:3]
        uv = xy @ camera_matrix[:2, :2].T + camera_matrix[:2, 2]
        projected[valid] = uv.astype(np.float32)
    return projected, valid


def bbox_xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box]
    return x, y, x + w, y + h


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clipped_projected_bbox(points_uv: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    finite = np.isfinite(points_uv).all(axis=1)
    if not finite.any():
        return None
    pts = points_uv[finite]
    in_frame = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < width)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < height)
    )
    if not in_frame.any():
        return None
    pts = pts[in_frame]
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def draw_bbox(draw: ImageDraw.ImageDraw, xyxy: tuple[float, float, float, float], color, width: int = 2) -> None:
    x1, y1, x2, y2 = xyxy
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def draw_projected_points(draw: ImageDraw.ImageDraw, uv: np.ndarray, image_size: tuple[int, int], color) -> int:
    width, height = image_size
    finite = np.isfinite(uv).all(axis=1)
    pts = uv[finite]
    in_frame = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < width)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < height)
    )
    pts = pts[in_frame]
    for x, y in pts:
        draw.point((float(x), float(y)), fill=color)
    return int(len(pts))


def draw_projected_cuboid(draw: ImageDraw.ImageDraw, uv: np.ndarray, color, width: int = 3) -> None:
    if not np.isfinite(uv).all():
        return
    pts = [(float(x), float(y)) for x, y in uv]
    for a, b in BBOX_EDGES:
        draw.line([pts[a], pts[b]], fill=color, width=width)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "tless",
    )
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--scene-id", type=int, default=1)
    parser.add_argument(
        "--image-id",
        type=int,
        default=None,
        help="Defaults to the first RGB image found in the scene.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/models_eval.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "outputs",
    )
    parser.add_argument("--max-points-per-object", type=int, default=2500)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    scene_dir = dataset_root / args.split / f"{args.scene_id:06d}"
    models_dir = args.models_dir or dataset_root / "models_eval"
    if args.image_id is None:
        rgb_images = sorted((scene_dir / "rgb").glob("*.png"))
        if not rgb_images:
            raise FileNotFoundError(f"No RGB images found under {scene_dir / 'rgb'}")
        args.image_id = int(rgb_images[0].stem)
    image_stem = f"{args.image_id:06d}"
    key = str(args.image_id)

    rgb_path = scene_dir / "rgb" / f"{image_stem}.png"
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)

    scene_gt = load_json(scene_dir / "scene_gt.json")
    scene_gt_info = load_json(scene_dir / "scene_gt_info.json")
    scene_camera = load_json(scene_dir / "scene_camera.json")
    models_info = load_json(models_dir / "models_info.json")

    instances = scene_gt[key]
    instance_infos = scene_gt_info[key]
    camera = scene_camera[key]
    camera_matrix = np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3)

    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    mesh_cache: dict[int, np.ndarray] = {}
    diagnostics = []

    for idx, (instance, info) in enumerate(zip(instances, instance_infos)):
        obj_id = int(instance["obj_id"])
        color = COLORS[idx % len(COLORS)]
        rotation = np.asarray(instance["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        translation = np.asarray(instance["cam_t_m2c"], dtype=np.float32)

        if obj_id not in mesh_cache:
            vertices = parse_ply_vertices(models_dir / f"obj_{obj_id:06d}.ply")
            if args.max_points_per_object > 0 and len(vertices) > args.max_points_per_object:
                step = int(np.ceil(len(vertices) / args.max_points_per_object))
                vertices = vertices[::step]
            mesh_cache[obj_id] = vertices

        vertices_c = transform_points(mesh_cache[obj_id], rotation, translation)
        vertices_uv, _ = project_points(vertices_c, camera_matrix)
        points_in_frame = draw_projected_points(draw, vertices_uv, image.size, color)

        corners = bbox_corners(models_info[str(obj_id)])
        corners_c = transform_points(corners, rotation, translation)
        corners_uv, _ = project_points(corners_c, camera_matrix)
        draw_projected_cuboid(draw, corners_uv, color, width=3)

        gt_bbox = bbox_xywh_to_xyxy(info["bbox_obj"])
        draw_bbox(draw, gt_bbox, (255, 255, 255), width=2)

        projected_bbox = clipped_projected_bbox(vertices_uv, image.width, image.height)
        iou = bbox_iou(projected_bbox, gt_bbox) if projected_bbox else 0.0
        diagnostics.append(
            {
                "instance_index": idx,
                "obj_id": obj_id,
                "points_in_frame": points_in_frame,
                "gt_bbox_obj": [round(v, 2) for v in gt_bbox],
                "projected_vertex_bbox": [round(v, 2) for v in projected_bbox] if projected_bbox else None,
                "projected_bbox_iou_with_gt_bbox_obj": round(iou, 4),
                "visib_fract": round(float(info.get("visib_fract", 0.0)), 4),
            }
        )
        draw.text((gt_bbox[0], max(0, gt_bbox[1] - 14)), f"obj_{obj_id:06d}", fill=color)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"tless_scene_{args.scene_id:06d}_image_{args.image_id:06d}_cad_gt_pose_overlay.png"
    diag_path = out_path.with_suffix(".json")
    image.save(out_path)
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method_family": "Render-and-Compare / Test-Time CAD Use diagnostic",
                "backbone_paper": "MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare, CoRL 2022",
                "dataset_root": str(dataset_root),
                "split": args.split,
                "scene_id": args.scene_id,
                "image_id": args.image_id,
                "rgb_path": str(rgb_path),
                "models_dir": str(models_dir),
                "camera_matrix": camera_matrix.tolist(),
                "instances": diagnostics,
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(9, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout(pad=0)
    preview_path = out_path.with_name(out_path.stem + "_preview.png")
    plt.savefig(preview_path, dpi=160)
    plt.close()

    print(f"scene: {args.scene_id:06d}")
    print(f"image: {args.image_id:06d}")
    print(f"instances: {len(instances)}")
    print(f"overlay: {out_path}")
    print(f"preview: {preview_path}")
    print(f"diagnostics: {diag_path}")
    for item in diagnostics:
        print(
            "  "
            f"obj_{item['obj_id']:06d} points={item['points_in_frame']} "
            f"bbox_iou={item['projected_bbox_iou_with_gt_bbox_obj']:.4f} "
            f"visib={item['visib_fract']:.4f}"
        )


if __name__ == "__main__":
    main()
