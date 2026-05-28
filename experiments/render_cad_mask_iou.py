#!/usr/bin/env python3
"""Render T-LESS CAD masks from GT pose and compare them with BOP masks."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


COLORS = np.array(
    [
        [255, 76, 76],
        [76, 179, 255],
        [76, 220, 120],
        [255, 212, 64],
        [207, 107, 255],
        [255, 143, 64],
        [64, 235, 218],
        [245, 105, 180],
    ],
    dtype=np.uint8,
)

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


def parse_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertex xyz and triangular faces from simple ASCII/binary PLY."""

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
        face_count = 0
        vertex_props: list[tuple[str, str]] = []
        in_vertex = False
        in_face = False
        face_list_prop: tuple[str, str] | None = None

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
                in_face = False
            elif parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
                in_vertex = False
                in_face = True
            elif parts[:1] == ["element"]:
                in_vertex = False
                in_face = False
            elif in_vertex and parts[:1] == ["property"] and parts[1] != "list":
                vertex_props.append((parts[2], parts[1]))
            elif in_face and parts[:2] == ["property", "list"]:
                face_list_prop = (parts[2], parts[3])

        if vertex_count <= 0 or face_count <= 0:
            raise ValueError(f"Expected vertices and faces in {path}")
        if face_list_prop is None:
            raise ValueError(f"Expected list face property in {path}")

        prop_names = [name for name, _ in vertex_props]
        xyz_indices = [prop_names.index(axis) for axis in ("x", "y", "z")]

        if fmt == "ascii":
            vertices = np.empty((vertex_count, 3), dtype=np.float32)
            for idx in range(vertex_count):
                values = f.readline().decode("ascii", errors="replace").split()
                vertices[idx] = [float(values[i]) for i in xyz_indices]

            faces: list[list[int]] = []
            for _ in range(face_count):
                values = f.readline().decode("ascii", errors="replace").split()
                n = int(values[0])
                indices = [int(v) for v in values[1 : 1 + n]]
                if n == 3:
                    faces.append(indices)
                elif n > 3:
                    first = indices[0]
                    for j in range(1, n - 1):
                        faces.append([first, indices[j], indices[j + 1]])
            return vertices, np.asarray(faces, dtype=np.int32)

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")

        vertex_struct = "<" + "".join(PLY_TYPES[prop_type][0] for _, prop_type in vertex_props)
        vertex_size = struct.calcsize(vertex_struct)
        vertices = np.empty((vertex_count, 3), dtype=np.float32)
        for idx in range(vertex_count):
            values = struct.unpack(vertex_struct, f.read(vertex_size))
            vertices[idx] = [values[i] for i in xyz_indices]

        count_type, index_type = face_list_prop
        count_fmt, count_size = PLY_TYPES[count_type]
        index_fmt, index_size = PLY_TYPES[index_type]
        faces = []
        for _ in range(face_count):
            n = struct.unpack("<" + count_fmt, f.read(count_size))[0]
            raw = f.read(index_size * n)
            indices = list(struct.unpack("<" + index_fmt * n, raw))
            if n == 3:
                faces.append(indices)
            elif n > 3:
                first = indices[0]
                for j in range(1, n - 1):
                    faces.append([first, indices[j], indices[j + 1]])
        return vertices, np.asarray(faces, dtype=np.int32)


def transform_points(points_m: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points_m @ rotation.T + translation[None, :]


def project_points(points_c: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    uv = np.full((points_c.shape[0], 2), np.nan, dtype=np.float32)
    valid = points_c[:, 2] > 1e-6
    pts = points_c[valid]
    xy = pts[:, :2] / pts[:, 2:3]
    uv[valid] = (xy @ camera_matrix[:2, :2].T + camera_matrix[:2, 2]).astype(np.float32)
    return uv


def triangle_depth(uv: np.ndarray, z: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    patch_w = x1 - x0 + 1
    patch_h = y1 - y0 + 1
    tri = np.round(uv - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
    tri_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cv2.fillConvexPoly(tri_mask, tri, 1)
    ys, xs = np.nonzero(tri_mask)
    if len(xs) == 0:
        return xs, ys

    px = xs.astype(np.float32) + x0 + 0.5
    py = ys.astype(np.float32) + y0 + 0.5
    a, b, c = uv
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(denom)) < 1e-8:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denom
    w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denom
    w2 = 1.0 - w0 - w1
    depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
    return xs, ys, depth


def render_scene_masks(
    instances: list[dict],
    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    camera_matrix: np.ndarray,
    image_shape: tuple[int, int],
    face_stride: int = 1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = image_shape
    depth = np.full((height, width), np.inf, dtype=np.float32)
    instance_map = np.full((height, width), -1, dtype=np.int16)
    full_masks = [np.zeros((height, width), dtype=bool) for _ in instances]

    for inst_idx, instance in enumerate(instances):
        obj_id = int(instance["obj_id"])
        vertices, faces = mesh_cache[obj_id]
        rotation = np.asarray(instance["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        translation = np.asarray(instance["cam_t_m2c"], dtype=np.float32)
        vertices_c = transform_points(vertices, rotation, translation)
        vertices_uv = project_points(vertices_c, camera_matrix)
        face_iter = faces[:: max(1, face_stride)]

        for face in face_iter:
            tri_uv = vertices_uv[face]
            tri_z = vertices_c[face, 2]
            if not np.isfinite(tri_uv).all() or np.any(tri_z <= 1e-6):
                continue
            min_x = int(np.floor(tri_uv[:, 0].min()))
            max_x = int(np.ceil(tri_uv[:, 0].max()))
            min_y = int(np.floor(tri_uv[:, 1].min()))
            max_y = int(np.ceil(tri_uv[:, 1].max()))
            if max_x < 0 or max_y < 0 or min_x >= width or min_y >= height:
                continue
            min_x = max(0, min_x)
            max_x = min(width - 1, max_x)
            min_y = max(0, min_y)
            max_y = min(height - 1, max_y)
            if max_x <= min_x or max_y <= min_y:
                continue

            result = triangle_depth(tri_uv, tri_z, min_x, min_y, max_x, max_y)
            if len(result) == 2:
                continue
            xs_local, ys_local, tri_depth = result
            xs = xs_local + min_x
            ys = ys_local + min_y
            full_masks[inst_idx][ys, xs] = True
            closer = tri_depth < depth[ys, xs]
            if closer.any():
                ys_c = ys[closer]
                xs_c = xs[closer]
                depth[ys_c, xs_c] = tri_depth[closer]
                instance_map[ys_c, xs_c] = inst_idx

    visible_masks = [(instance_map == idx) for idx in range(len(instances))]
    return instance_map, full_masks, visible_masks


def mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(inter / union) if union else 0.0


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def blend_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    out = image.copy()
    out[mask] = (out[mask].astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    return out


def make_overlay(rgb: np.ndarray, rows: list[dict], rendered_visible: list[np.ndarray], gt_visible: list[np.ndarray]) -> np.ndarray:
    overlay = rgb.copy()
    for row, pred, gt in zip(rows, rendered_visible, gt_visible):
        color = COLORS[row["instance_index"] % len(COLORS)]
        overlay = blend_mask(overlay, pred, color, 0.45)
        misses = np.logical_and(gt, ~pred)
        extras = np.logical_and(pred, ~gt)
        overlay[misses] = np.array([255, 255, 255], dtype=np.uint8)
        overlay[extras] = np.array([255, 0, 255], dtype=np.uint8)
    return overlay


def find_image_ids(scene_dir: Path) -> list[int]:
    return [int(path.stem) for path in sorted((scene_dir / "rgb").glob("*.png"))]


def evaluate_image(
    dataset_root: Path,
    split: str,
    scene_id: int,
    image_id: int,
    models_dir: Path,
    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    face_stride: int,
) -> tuple[list[dict], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    scene_dir = dataset_root / split / f"{scene_id:06d}"
    image_stem = f"{image_id:06d}"
    key = str(image_id)

    rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{image_stem}.png").convert("RGB"))
    height, width = rgb.shape[:2]
    scene_gt = load_json(scene_dir / "scene_gt.json")
    scene_gt_info = load_json(scene_dir / "scene_gt_info.json")
    scene_camera = load_json(scene_dir / "scene_camera.json")
    instances = scene_gt[key]
    instance_infos = scene_gt_info[key]
    camera_matrix = np.asarray(scene_camera[key]["cam_K"], dtype=np.float32).reshape(3, 3)

    for instance in instances:
        obj_id = int(instance["obj_id"])
        if obj_id not in mesh_cache:
            mesh_cache[obj_id] = parse_ply_mesh(models_dir / f"obj_{obj_id:06d}.ply")

    _, full_masks, visible_masks = render_scene_masks(
        instances=instances,
        mesh_cache=mesh_cache,
        camera_matrix=camera_matrix,
        image_shape=(height, width),
        face_stride=face_stride,
    )

    rows = []
    gt_visible_masks = []
    for inst_idx, (instance, info) in enumerate(zip(instances, instance_infos)):
        obj_id = int(instance["obj_id"])
        mask_path = scene_dir / "mask" / f"{image_stem}_{inst_idx:06d}.png"
        mask_visib_path = scene_dir / "mask_visib" / f"{image_stem}_{inst_idx:06d}.png"
        gt_full = load_mask(mask_path)
        gt_visib = load_mask(mask_visib_path)
        gt_visible_masks.append(gt_visib)
        rows.append(
            {
                "scene_id": scene_id,
                "image_id": image_id,
                "instance_index": inst_idx,
                "obj_id": obj_id,
                "visib_fract": float(info.get("visib_fract", 0.0)),
                "render_full_area": int(full_masks[inst_idx].sum()),
                "render_visible_area": int(visible_masks[inst_idx].sum()),
                "gt_full_area": int(gt_full.sum()),
                "gt_visible_area": int(gt_visib.sum()),
                "full_mask_iou": mask_iou(full_masks[inst_idx], gt_full),
                "visible_mask_iou": mask_iou(visible_masks[inst_idx], gt_visib),
            }
        )
    return rows, rgb, visible_masks, gt_visible_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "tless",
    )
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--scene-id", type=int, default=1)
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--face-stride", type=int, default=1, help="Use >1 for faster approximate rendering.")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "outputs",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    models_dir = args.models_dir or dataset_root / "models_eval"
    scene_dir = dataset_root / args.split / f"{args.scene_id:06d}"
    if args.image_id is None:
        image_ids = find_image_ids(scene_dir)[: args.max_images]
    else:
        image_ids = [args.image_id]

    args.out.mkdir(parents=True, exist_ok=True)
    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_rows = []
    last_overlay_path = None

    for image_id in image_ids:
        rows, rgb, rendered_visible, gt_visible = evaluate_image(
            dataset_root=dataset_root,
            split=args.split,
            scene_id=args.scene_id,
            image_id=image_id,
            models_dir=models_dir,
            mesh_cache=mesh_cache,
            face_stride=args.face_stride,
        )
        all_rows.extend(rows)
        overlay = make_overlay(rgb, rows, rendered_visible, gt_visible)
        overlay_path = args.out / f"tless_scene_{args.scene_id:06d}_image_{image_id:06d}_cad_mask_iou_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        last_overlay_path = overlay_path

    csv_path = args.out / f"tless_scene_{args.scene_id:06d}_cad_mask_iou.csv"
    fieldnames = [
        "scene_id",
        "image_id",
        "instance_index",
        "obj_id",
        "visib_fract",
        "render_full_area",
        "render_visible_area",
        "gt_full_area",
        "gt_visible_area",
        "full_mask_iou",
        "visible_mask_iou",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    full_ious = np.array([row["full_mask_iou"] for row in all_rows], dtype=np.float32)
    visible_ious = np.array([row["visible_mask_iou"] for row in all_rows], dtype=np.float32)
    summary = {
        "method_family": "Render-and-Compare / Test-Time CAD Use diagnostic",
        "backbone_paper": "MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare, CoRL 2022",
        "split": args.split,
        "scene_id": args.scene_id,
        "image_ids": image_ids,
        "face_stride": args.face_stride,
        "instances": len(all_rows),
        "mean_full_mask_iou": float(full_ious.mean()) if len(full_ious) else 0.0,
        "mean_visible_mask_iou": float(visible_ious.mean()) if len(visible_ious) else 0.0,
        "min_visible_mask_iou": float(visible_ious.min()) if len(visible_ious) else 0.0,
        "csv": str(csv_path),
        "last_overlay": str(last_overlay_path) if last_overlay_path else None,
    }
    summary_path = csv_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"scene: {args.scene_id:06d}")
    print(f"images: {len(image_ids)}")
    print(f"instances: {len(all_rows)}")
    print(f"mean_full_mask_iou: {summary['mean_full_mask_iou']:.4f}")
    print(f"mean_visible_mask_iou: {summary['mean_visible_mask_iou']:.4f}")
    print(f"min_visible_mask_iou: {summary['min_visible_mask_iou']:.4f}")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")
    if last_overlay_path:
        print(f"last_overlay: {last_overlay_path}")
    for row in all_rows[:20]:
        print(
            "  "
            f"{row['image_id']:06d}_{row['instance_index']:06d} "
            f"obj_{row['obj_id']:06d} "
            f"full_iou={row['full_mask_iou']:.4f} "
            f"vis_iou={row['visible_mask_iou']:.4f} "
            f"visib={row['visib_fract']:.4f}"
        )


if __name__ == "__main__":
    main()
