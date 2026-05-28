#!/usr/bin/env python3
"""Run DRAEM no-CAD synthetic-anomaly baselines with anomalib."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

import anomalib
from anomalib.data import MVTecAD, Visa
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.datasets.image.visa import VisaDataset
from anomalib.engine import Engine
from anomalib.models import Draem


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


class MVTecADStringSplit(MVTecAD):
    """MVTecAD datamodule workaround for anomalib 2.4.2 Split enum filtering."""

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = MVTecADDataset(
            split="train",
            root=self.root,
            category=self.category,
        )
        self.test_data = MVTecADDataset(
            split="test",
            root=self.root,
            category=self.category,
        )


class VisaStringSplit(Visa):
    """Visa datamodule workaround for anomalib 2.4.2 Split enum filtering."""

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = VisaDataset(
            split="train",
            root=self.split_root,
            category=self.category,
        )
        self.test_data = VisaDataset(
            split="test",
            root=self.split_root,
            category=self.category,
        )


def as_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): as_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_serializable(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def flatten_metrics(metrics: list[dict[str, Any]] | dict[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    if isinstance(metrics, list):
        merged: dict[str, Any] = {}
        for item in metrics:
            merged.update(item)
        metrics = merged
    return {key: as_serializable(value) for key, value in metrics.items()}


def category_list(dataset: str, categories: list[str]) -> list[str]:
    if len(categories) == 1 and categories[0] == "all":
        return MVTEC_CATEGORIES if dataset == "mvtec_ad" else VISA_CATEGORIES
    return categories


def make_datamodule(
    *,
    dataset: str,
    root: Path,
    category: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
):
    common = {
        "root": root,
        "category": category,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "seed": seed,
    }
    if dataset == "mvtec_ad":
        return MVTecADStringSplit(**common)
    if dataset == "visa":
        return VisaStringSplit(**common)
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_category(
    *,
    dataset: str,
    root: Path,
    category: str,
    out_root: Path,
    dtd_dir: Path,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    enable_sspcab: bool,
    sspcab_lambda: float,
    beta_min: float,
    beta_max: float,
    max_epochs: int,
    max_steps: int | None,
    check_val_every_n_epoch: int | None,
    accelerator: str,
    devices: int | str,
) -> dict[str, Any]:
    run_dir = out_root / dataset / category
    run_dir.mkdir(parents=True, exist_ok=True)

    datamodule = make_datamodule(
        dataset=dataset,
        root=root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    model = Draem(
        dtd_dir=dtd_dir,
        enable_sspcab=enable_sspcab,
        sspcab_lambda=sspcab_lambda,
        beta=(beta_min, beta_max),
        visualizer=False,
    )
    engine_kwargs: dict[str, Any] = {
        "default_root_dir": run_dir,
        "logger": False,
        "accelerator": accelerator,
        "devices": devices,
        "max_epochs": max_epochs,
    }
    if max_steps is not None:
        engine_kwargs["max_steps"] = max_steps
    if check_val_every_n_epoch is not None:
        engine_kwargs["check_val_every_n_epoch"] = check_val_every_n_epoch
    engine = Engine(**engine_kwargs)

    engine.fit(model=model, datamodule=datamodule)
    metrics = flatten_metrics(engine.test(model=model, datamodule=datamodule))

    row = {
        "dataset": dataset,
        "category": category,
        "run_dir": str(run_dir),
        "dtd_dir": str(dtd_dir),
        "enable_sspcab": enable_sspcab,
        "sspcab_lambda": sspcab_lambda,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "seed": seed,
        **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--dataset", choices=["mvtec_ad", "visa"], default="mvtec_ad")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["toothbrush"],
        help='Use one or more category names, or "all".',
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--dtd-dir", type=Path, default=None)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-sspcab", action="store_true")
    parser.add_argument("--sspcab-lambda", type=float, default=0.1)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    project_root = args.project_root
    dataset_root = args.root or (
        project_root / "datasets" / ("mvtec_ad" if args.dataset == "mvtec_ad" else "visa")
    )
    dtd_dir = args.dtd_dir or project_root / "datasets" / "dtd"
    out_root = args.out_root or project_root / "experiments" / "runs" / "no_cad" / "draem"
    categories = category_list(args.dataset, args.categories)

    rows = []
    for idx, category in enumerate(categories, start=1):
        print(f"[{idx}/{len(categories)}] {args.dataset}/{category}", flush=True)
        row = run_category(
            dataset=args.dataset,
            root=dataset_root,
            category=category,
            out_root=out_root,
            dtd_dir=dtd_dir,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            enable_sspcab=args.enable_sspcab,
            sspcab_lambda=args.sspcab_lambda,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            max_epochs=args.max_epochs,
            max_steps=args.max_steps,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    result_dir = out_root / args.dataset
    csv_path = result_dir / "draem_results.csv"
    summary_path = result_dir / "draem_results.summary.json"
    write_csv(csv_path, rows)
    summary = {
        "method_family": "Synthetic anomaly generation / DRAEM",
        "library": "anomalib",
        "anomalib_version": anomalib.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "dtd_dir": str(dtd_dir),
        "categories": categories,
        "rows": rows,
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
