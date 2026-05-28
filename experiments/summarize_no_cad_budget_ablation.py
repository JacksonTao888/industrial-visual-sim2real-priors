#!/usr/bin/env python3
"""Summarize no-CAD normal-reference budget ablation results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "runs" / "no_cad" / "budget_ablation"

METHODS = {
    "PatchCore": {
        "budget_root": OUT_DIR / "patchcore",
        "summary_name": "patchcore_results.summary.json",
        "full": {
            "mvtec_ad": PROJECT_ROOT
            / "experiments"
            / "runs"
            / "no_cad"
            / "patchcore"
            / "mvtec_ad"
            / "patchcore_results.summary.json",
            "visa": PROJECT_ROOT
            / "experiments"
            / "runs"
            / "no_cad"
            / "patchcore"
            / "visa"
            / "patchcore_results.summary.json",
        },
    },
    "AnomalyDINO-S": {
        "budget_root": OUT_DIR / "anomalydino",
        "summary_name": "anomalydino_results.summary.json",
        "full": {
            "mvtec_ad": PROJECT_ROOT
            / "experiments"
            / "runs"
            / "no_cad"
            / "anomalydino"
            / "mvtec_ad"
            / "anomalydino_results.summary.json",
            "visa": PROJECT_ROOT
            / "experiments"
            / "runs"
            / "no_cad"
            / "anomalydino"
            / "visa"
            / "anomalydino_results.summary.json",
        },
    },
}

BUDGETS = [
    ("budget_005", 0.05),
    ("budget_010", 0.10),
    ("budget_025", 0.25),
]

SELECTED_CATEGORIES = {
    "mvtec_ad": ["toothbrush", "capsule", "cable"],
    "visa": ["macaroni2", "capsules", "pcb1"],
}

METRICS = ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data["rows"]


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


def summarize_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["dataset"], float(row["train_budget_ratio"]))].append(row)

    summary: dict[str, Any] = {}
    for (method, dataset, ratio), group_rows in sorted(grouped.items()):
        key = f"{method}|{dataset}|{ratio:.2f}"
        metric_summary = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows]
            weakest = min(group_rows, key=lambda row: float(row[metric]))
            strongest = max(group_rows, key=lambda row: float(row[metric]))
            metric_summary[metric] = {
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "weakest_category": weakest["category"],
                "strongest_category": strongest["category"],
            }
        summary[key] = {
            "method": method,
            "dataset": dataset,
            "train_budget_ratio": ratio,
            "categories": [row["category"] for row in group_rows],
            "metric_summary": metric_summary,
        }
    return summary


def main() -> None:
    all_rows: list[dict[str, Any]] = []

    for method, config in METHODS.items():
        for tag, ratio in BUDGETS:
            for dataset, categories in SELECTED_CATEGORIES.items():
                summary_path = config["budget_root"] / tag / dataset / config["summary_name"]
                if not summary_path.exists():
                    raise FileNotFoundError(summary_path)
                for row in load_rows(summary_path):
                    row = dict(row)
                    row["method"] = method
                    row["budget_tag"] = tag
                    row["train_budget_ratio"] = ratio
                    all_rows.append(row)

        for dataset, categories in SELECTED_CATEGORIES.items():
            full_summary_path = config["full"][dataset]
            for row in load_rows(full_summary_path):
                if row["category"] not in categories:
                    continue
                row = dict(row)
                row["method"] = method
                row["budget_tag"] = "budget_100"
                row["train_budget_ratio"] = 1.0
                row.setdefault("train_budget_original_count", None)
                row.setdefault("train_budget_count", None)
                all_rows.append(row)

    csv_path = OUT_DIR / "no_cad_budget_ablation_results.csv"
    summary_path = OUT_DIR / "no_cad_budget_ablation_results.summary.json"
    write_csv(csv_path, all_rows)
    summary = {
        "description": "Normal-reference budget ablation for selected no-CAD categories.",
        "selected_categories": SELECTED_CATEGORIES,
        "budgets": [0.05, 0.10, 0.25, 1.0],
        "rows": all_rows,
        "group_summary": summarize_groups(all_rows),
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
