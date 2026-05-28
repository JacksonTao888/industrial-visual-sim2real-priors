#!/usr/bin/env python3
"""Summarize per-category SuperSimpleNet metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_draem_results import (
    METRIC_KEYS,
    expected_categories,
    load_rows,
    metric_stats,
    write_csv,
)


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
        "perlin_threshold",
        "backbone",
        "layers",
        "adapt_cls_features",
        "max_epochs",
        "max_steps",
        "train_steps_per_epoch",
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
        "method_family": "Feature-level synthetic anomaly generation / SuperSimpleNet",
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
        args.project_root / "experiments" / "runs" / "no_cad" / "supersimplenet" / args.dataset
    )
    csv_path = args.csv or result_dir / "supersimplenet_results.csv"
    summary_path = args.summary or result_dir / "supersimplenet_results.summary.json"

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
