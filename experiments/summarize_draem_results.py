#!/usr/bin/env python3
"""Summarize per-category DRAEM metrics written by run_draem_anomalib.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

METRIC_KEYS = [
    "image_AUROC",
    "image_F1Score",
    "pixel_AUROC",
    "pixel_F1Score",
]


def expected_categories(dataset: str) -> list[str]:
    if dataset == "mvtec_ad":
        return MVTEC_CATEGORIES
    if dataset == "visa":
        return VISA_CATEGORIES
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_dir.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("category", "")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float):
            values.append((row.get("category", ""), float(value)))
    if not values:
        return None

    sorted_values = sorted(values, key=lambda item: item[1])
    numeric = [value for _, value in sorted_values]
    return {
        "mean": mean(numeric),
        "median": median(numeric),
        "min": {"category": sorted_values[0][0], "value": sorted_values[0][1]},
        "max": {"category": sorted_values[-1][0], "value": sorted_values[-1][1]},
        "ranked_low_to_high": [
            {"category": category, "value": value} for category, value in sorted_values
        ],
    }


def summarize(dataset: str, result_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = [str(row.get("category", "")) for row in rows]
    expected = expected_categories(dataset)
    complete_categories = [category for category in expected if category in categories]
    unexpected_categories = [category for category in categories if category not in expected]
    missing_categories = [category for category in expected if category not in categories]

    metrics = {
        key: stats
        for key in METRIC_KEYS
        if (stats := metric_stats(rows, key)) is not None
    }

    configs = {}
    for key in [
        "dtd_dir",
        "enable_sspcab",
        "sspcab_lambda",
        "beta_min",
        "beta_max",
        "max_epochs",
        "max_steps",
        "check_val_every_n_epoch",
        "train_batch_size",
        "eval_batch_size",
        "num_workers",
        "seed",
    ]:
        values = sorted({str(row.get(key)) for row in rows if key in row})
        if values:
            configs[key] = values

    return {
        "method_family": "Synthetic anomaly generation / DRAEM",
        "dataset": dataset,
        "result_dir": str(result_dir),
        "expected_category_count": len(expected),
        "completed_category_count": len(complete_categories),
        "completed_categories": complete_categories,
        "missing_categories": missing_categories,
        "unexpected_categories": unexpected_categories,
        "is_complete": not missing_categories,
        "config_values": configs,
        "metrics": metrics,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--dataset", choices=["mvtec_ad", "visa"], required=True)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.result_dir or (
        args.project_root / "experiments" / "runs" / "no_cad" / "draem" / args.dataset
    )
    csv_path = args.csv or result_dir / "draem_results.csv"
    summary_path = args.summary or result_dir / "draem_results.summary.json"

    rows = load_rows(result_dir)
    write_csv(csv_path, rows)
    summary = summarize(args.dataset, result_dir, rows)
    summary["csv"] = str(csv_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"dataset: {args.dataset}")
    print(f"completed: {summary['completed_category_count']}/{summary['expected_category_count']}")
    if summary["missing_categories"]:
        print(f"missing: {', '.join(summary['missing_categories'])}")
    for key in METRIC_KEYS:
        stats = summary["metrics"].get(key)
        if stats:
            print(f"{key}_mean: {stats['mean']:.4f}")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
