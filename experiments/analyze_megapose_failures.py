#!/usr/bin/env python3
"""Categorize MegaPose T-LESS proposal-level mask-IoU failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def categorize(row: dict, args: argparse.Namespace) -> tuple[str, str]:
    full_iou = float(row["full_mask_iou"])
    scene_visible_iou = float(row["scene_visible_mask_iou"])
    coverage = float(row["gt_visible_coverage_by_full_mask"])
    pred_visible_fraction = float(row["pred_visible_fraction"])
    has_same_class_gt = int(row["best_gt_instance_index"]) >= 0

    if not has_same_class_gt:
        return "detector_false_positive_no_same_class_gt", "No same-class GT instance exists."
    if full_iou <= args.zero_iou_epsilon:
        return "same_class_zero_alignment", "Same-class GT exists, but rendered full-mask IoU is zero."
    if full_iou >= args.good_full_iou and scene_visible_iou < args.bad_scene_visible_iou:
        return (
            "good_pose_bad_predicted_occlusion",
            "Full-mask pose is good, but z-buffered predicted visible mask is poor.",
        )
    if coverage >= args.good_coverage and scene_visible_iou < args.bad_scene_visible_iou:
        return (
            "good_coverage_bad_predicted_occlusion",
            "Predicted full mask covers GT visible object, but predicted visible mask is poor.",
        )
    if full_iou < args.bad_full_iou or coverage < args.bad_coverage:
        return "weak_pose_or_silhouette", "Full-mask pose or GT-visible coverage is weak."
    if full_iou >= args.good_full_iou and coverage >= args.good_coverage:
        return "good_pose_silhouette", "Pose/silhouette alignment is good."
    if pred_visible_fraction < args.low_pred_visible_fraction:
        return (
            "moderate_pose_low_predicted_visibility",
            "Pose is not bad, but predicted object is mostly occluded in the predicted scene.",
        )
    return "moderate_pose_silhouette", "Pose/silhouette alignment is moderate."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "outputs"
            / "megapose_batches"
            / "conf_0p25_maxdet_10_scene_round_robin_per_scene_5_n100_offset0"
            / "megapose_batch_mask_iou.csv"
        ),
    )
    parser.add_argument("--zero-iou-epsilon", type=float, default=1e-9)
    parser.add_argument("--bad-full-iou", type=float, default=0.5)
    parser.add_argument("--bad-coverage", type=float, default=0.5)
    parser.add_argument("--bad-scene-visible-iou", type=float, default=0.5)
    parser.add_argument("--good-full-iou", type=float, default=0.8)
    parser.add_argument("--good-coverage", type=float, default=0.8)
    parser.add_argument("--low-pred-visible-fraction", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = read_csv(args.csv)
    out_path = args.out or args.csv.with_name("megapose_failure_analysis.csv")
    summary_path = out_path.with_suffix(".summary.json")

    analyzed_rows = []
    for row in rows:
        category, reason = categorize(row, args)
        out_row = dict(row)
        out_row["failure_category"] = category
        out_row["failure_reason"] = reason
        analyzed_rows.append(out_row)

    fieldnames = list(analyzed_rows[0].keys()) if analyzed_rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analyzed_rows)

    category_counts = Counter(row["failure_category"] for row in analyzed_rows)
    category_summaries = {}
    for category in sorted(category_counts):
        cat_rows = [row for row in analyzed_rows if row["failure_category"] == category]
        category_summaries[category] = {
            "count": len(cat_rows),
            "fraction": len(cat_rows) / len(analyzed_rows) if analyzed_rows else 0.0,
            "full_mask_iou": summarize([float(row["full_mask_iou"]) for row in cat_rows]),
            "scene_visible_mask_iou": summarize(
                [float(row["scene_visible_mask_iou"]) for row in cat_rows]
            ),
            "gt_visible_coverage_by_full_mask": summarize(
                [float(row["gt_visible_coverage_by_full_mask"]) for row in cat_rows]
            ),
            "examples": [
                {
                    "example_name": row["example_name"],
                    "label": row["label"],
                    "best_gt_instance_index": int(row["best_gt_instance_index"]),
                    "full_mask_iou": float(row["full_mask_iou"]),
                    "scene_visible_mask_iou": float(row["scene_visible_mask_iou"]),
                    "gt_visible_coverage_by_full_mask": float(
                        row["gt_visible_coverage_by_full_mask"]
                    ),
                    "pred_visible_fraction": float(row["pred_visible_fraction"]),
                }
                for row in sorted(
                    cat_rows,
                    key=lambda r: (
                        float(r["full_mask_iou"]),
                        float(r["gt_visible_coverage_by_full_mask"]),
                    ),
                )[: args.top_k]
            ],
        }

    per_scene = defaultdict(Counter)
    per_object = defaultdict(Counter)
    for row in analyzed_rows:
        per_scene[f"{int(row['scene_id']):06d}"][row["failure_category"]] += 1
        per_object[row["label"]][row["failure_category"]] += 1

    worst_scene_failure_rates = []
    for scene_id, counts in sorted(per_scene.items()):
        total = sum(counts.values())
        bad = total - counts.get("good_pose_silhouette", 0)
        worst_scene_failure_rates.append(
            {
                "scene_id": scene_id,
                "total": total,
                "non_good": bad,
                "non_good_fraction": bad / total if total else 0.0,
                "categories": dict(counts),
            }
        )
    worst_scene_failure_rates = sorted(
        worst_scene_failure_rates,
        key=lambda row: (row["non_good_fraction"], row["non_good"]),
        reverse=True,
    )

    worst_object_failure_rates = []
    for label, counts in sorted(per_object.items()):
        total = sum(counts.values())
        bad = total - counts.get("good_pose_silhouette", 0)
        worst_object_failure_rates.append(
            {
                "label": label,
                "total": total,
                "non_good": bad,
                "non_good_fraction": bad / total if total else 0.0,
                "categories": dict(counts),
            }
        )
    worst_object_failure_rates = sorted(
        worst_object_failure_rates,
        key=lambda row: (row["non_good_fraction"], row["non_good"]),
        reverse=True,
    )

    summary = {
        "input_csv": str(args.csv),
        "output_csv": str(out_path),
        "rows": len(analyzed_rows),
        "thresholds": {
            "zero_iou_epsilon": args.zero_iou_epsilon,
            "bad_full_iou": args.bad_full_iou,
            "bad_coverage": args.bad_coverage,
            "bad_scene_visible_iou": args.bad_scene_visible_iou,
            "good_full_iou": args.good_full_iou,
            "good_coverage": args.good_coverage,
            "low_pred_visible_fraction": args.low_pred_visible_fraction,
        },
        "category_counts": dict(category_counts),
        "category_summaries": category_summaries,
        "worst_scene_failure_rates": worst_scene_failure_rates[: args.top_k],
        "worst_object_failure_rates": worst_object_failure_rates[: args.top_k],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"rows: {len(analyzed_rows)}")
    print(f"csv: {out_path}")
    print(f"summary: {summary_path}")
    print("categories:")
    for category, count in category_counts.most_common():
        fraction = count / len(analyzed_rows) if analyzed_rows else 0.0
        print(f"  {category}: {count} ({fraction:.3f})")
    print("worst scenes:")
    for row in worst_scene_failure_rates[:5]:
        print(
            f"  {row['scene_id']}: non_good={row['non_good']}/{row['total']} "
            f"({row['non_good_fraction']:.3f}) {row['categories']}"
        )
    print("worst objects:")
    for row in worst_object_failure_rates[:5]:
        print(
            f"  {row['label']}: non_good={row['non_good']}/{row['total']} "
            f"({row['non_good_fraction']:.3f}) {row['categories']}"
        )


if __name__ == "__main__":
    main()
