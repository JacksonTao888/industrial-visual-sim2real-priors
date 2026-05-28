#!/usr/bin/env python3
"""Export B6 YOLO detections on T-LESS real held-out images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Defaults to B6 best checkpoint.",
    )
    parser.add_argument(
        "--coco",
        type=Path,
        default=None,
        help="Defaults to the 900-image held-out real validation COCO file.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/b6_tless_real_predictions.csv.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    model_path = args.model or (
        project_root
        / "experiments"
        / "runs"
        / "yolo"
        / "tless_yolov8s_dr_pretrained_real_5pct_finetune"
        / "weights"
        / "best.pt"
    )
    coco_path = args.coco or (
        project_root / "datasets" / "derived" / "coco" / "tless_real_val_heldout_900_detection.json"
    )
    out_path = args.out or (project_root / "experiments" / "outputs" / "b6_tless_real_predictions.csv")
    summary_path = out_path.with_suffix(".summary.json")

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not coco_path.exists():
        raise FileNotFoundError(coco_path)

    coco = load_json(coco_path)
    images = sorted(coco["images"], key=lambda item: item["id"])
    if args.max_images is not None:
        images = images[: args.max_images]

    model = YOLO(str(model_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "scene_id",
        "bop_image_id",
        "file_name",
        "width",
        "height",
        "pred_index",
        "class_id",
        "obj_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
    ]

    prediction_count = 0
    image_count_with_predictions = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image in images:
            image_path = project_root / "datasets" / "tless" / image["file_name"]
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            image_count_with_predictions += 1
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for pred_index, (box, confidence, class_id) in enumerate(zip(xyxy, confs, classes)):
                x1, y1, x2, y2 = [float(v) for v in box]
                obj_id = int(class_id) + 1
                writer.writerow(
                    {
                        "image_id": image["id"],
                        "scene_id": image["scene_id"],
                        "bop_image_id": image["bop_image_id"],
                        "file_name": image["file_name"],
                        "width": image["width"],
                        "height": image["height"],
                        "pred_index": pred_index,
                        "class_id": int(class_id),
                        "obj_id": obj_id,
                        "class_name": f"obj_{obj_id:06d}",
                        "confidence": f"{float(confidence):.8f}",
                        "x1": f"{x1:.4f}",
                        "y1": f"{y1:.4f}",
                        "x2": f"{x2:.4f}",
                        "y2": f"{y2:.4f}",
                        "w": f"{x2 - x1:.4f}",
                        "h": f"{y2 - y1:.4f}",
                    }
                )
                prediction_count += 1

    summary = {
        "method": "B6 YOLOv8s + DR + 5% real fine-tune predictions",
        "model": str(model_path),
        "coco": str(coco_path),
        "images": len(images),
        "images_with_predictions": image_count_with_predictions,
        "predictions": prediction_count,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "csv": str(out_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"images: {len(images)}")
    print(f"images_with_predictions: {image_count_with_predictions}")
    print(f"predictions: {prediction_count}")
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
