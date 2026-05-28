#!/usr/bin/env python3
"""Run AnomalyDINO no-CAD DINOv2 memory-bank baselines with anomalib."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

import anomalib
from anomalib.data import MVTecAD, Visa
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.datasets.image.visa import VisaDataset
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO


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

ANOMALYDINO_MASKED_MVTEC_CATEGORIES = {
    "capsule",
    "hazelnut",
    "pill",
    "screw",
    "toothbrush",
}


class MVTecADStringSplit(MVTecAD):
    """MVTecAD datamodule workaround for anomalib 2.4.2 Split enum filtering."""

    def __init__(
        self,
        *args: Any,
        train_budget_ratio: float = 1.0,
        train_budget_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.train_budget_ratio = train_budget_ratio
        self.train_budget_seed = train_budget_seed if train_budget_seed is not None else kwargs.get("seed", 42)
        self.train_budget_original_count: int | None = None
        self.train_budget_count: int | None = None
        super().__init__(*args, **kwargs)

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
        self.train_data = apply_train_budget(
            self.train_data,
            ratio=self.train_budget_ratio,
            seed=self.train_budget_seed,
        )
        self.train_budget_original_count = getattr(self.train_data, "train_budget_original_count", len(self.train_data))
        self.train_budget_count = len(self.train_data)


class VisaStringSplit(Visa):
    """Visa datamodule workaround for anomalib 2.4.2 Split enum filtering."""

    def __init__(
        self,
        *args: Any,
        train_budget_ratio: float = 1.0,
        train_budget_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.train_budget_ratio = train_budget_ratio
        self.train_budget_seed = train_budget_seed if train_budget_seed is not None else kwargs.get("seed", 42)
        self.train_budget_original_count: int | None = None
        self.train_budget_count: int | None = None
        super().__init__(*args, **kwargs)

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
        self.train_data = apply_train_budget(
            self.train_data,
            ratio=self.train_budget_ratio,
            seed=self.train_budget_seed,
        )
        self.train_budget_original_count = getattr(self.train_data, "train_budget_original_count", len(self.train_data))
        self.train_budget_count = len(self.train_data)


def apply_train_budget(dataset: Any, *, ratio: float, seed: int | None) -> Any:
    """Deterministically subsample normal training images for budget ablations."""
    original_count = len(dataset)
    dataset.train_budget_original_count = original_count
    if ratio >= 1.0:
        dataset.train_budget_count = original_count
        return dataset
    if ratio <= 0:
        raise ValueError(f"train_budget_ratio must be positive, got {ratio}")

    budget_count = max(1, min(original_count, math.ceil(original_count * ratio)))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(original_count), budget_count))
    dataset = dataset.subsample(indices, inplace=True)
    dataset.train_budget_original_count = original_count
    dataset.train_budget_count = budget_count
    return dataset


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


def should_mask(dataset: str, category: str, auto_masking: bool, masking: bool) -> bool:
    if masking:
        return True
    if auto_masking and dataset == "mvtec_ad":
        return category in ANOMALYDINO_MASKED_MVTEC_CATEGORIES
    return False


def make_datamodule(
    *,
    dataset: str,
    root: Path,
    category: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    train_budget_ratio: float,
):
    common = {
        "root": root,
        "category": category,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "train_budget_ratio": train_budget_ratio,
        "train_budget_seed": seed,
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
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    train_budget_ratio: float,
    num_neighbours: int,
    encoder_name: str,
    masking: bool,
    auto_masking: bool,
    coreset_subsampling: bool,
    sampling_ratio: float,
    precision: str,
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
        train_budget_ratio=train_budget_ratio,
    )
    category_masking = should_mask(dataset, category, auto_masking, masking)
    model = AnomalyDINO(
        num_neighbours=num_neighbours,
        encoder_name=encoder_name,
        masking=category_masking,
        coreset_subsampling=coreset_subsampling,
        sampling_ratio=sampling_ratio,
        precision=precision,
        visualizer=False,
    )
    engine = Engine(
        default_root_dir=run_dir,
        logger=False,
        accelerator=accelerator,
        devices=devices,
        max_epochs=1,
    )

    engine.fit(model=model, datamodule=datamodule)
    metrics = flatten_metrics(engine.test(model=model, datamodule=datamodule))

    row = {
        "dataset": dataset,
        "category": category,
        "run_dir": str(run_dir),
        "num_neighbours": num_neighbours,
        "encoder_name": encoder_name,
        "masking": category_masking,
        "auto_masking": auto_masking,
        "coreset_subsampling": coreset_subsampling,
        "sampling_ratio": sampling_ratio,
        "precision": precision,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "train_budget_ratio": train_budget_ratio,
        "train_budget_original_count": getattr(datamodule, "train_budget_original_count", None),
        "train_budget_count": getattr(datamodule, "train_budget_count", None),
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
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-budget-ratio",
        type=float,
        default=1.0,
        help="Deterministically subsample this fraction of normal training images.",
    )
    parser.add_argument("--num-neighbours", type=int, default=1)
    parser.add_argument("--encoder-name", default="dinov2_vit_small_14")
    parser.add_argument("--masking", action="store_true")
    parser.add_argument("--auto-masking", action="store_true")
    parser.add_argument("--coreset-subsampling", action="store_true")
    parser.add_argument("--sampling-ratio", type=float, default=0.1)
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    project_root = args.project_root
    dataset_root = args.root or (
        project_root / "datasets" / ("mvtec_ad" if args.dataset == "mvtec_ad" else "visa")
    )
    out_root = args.out_root or project_root / "experiments" / "runs" / "no_cad" / "anomalydino"
    categories = category_list(args.dataset, args.categories)

    rows = []
    for idx, category in enumerate(categories, start=1):
        print(f"[{idx}/{len(categories)}] {args.dataset}/{category}", flush=True)
        row = run_category(
            dataset=args.dataset,
            root=dataset_root,
            category=category,
            out_root=out_root,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            train_budget_ratio=args.train_budget_ratio,
            num_neighbours=args.num_neighbours,
            encoder_name=args.encoder_name,
            masking=args.masking,
            auto_masking=args.auto_masking,
            coreset_subsampling=args.coreset_subsampling,
            sampling_ratio=args.sampling_ratio,
            precision=args.precision,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    result_dir = out_root / args.dataset
    csv_path = result_dir / "anomalydino_results.csv"
    summary_path = result_dir / "anomalydino_results.summary.json"
    write_csv(csv_path, rows)
    summary = {
        "method_family": "Vision foundation features / AnomalyDINO",
        "library": "anomalib",
        "anomalib_version": anomalib.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "categories": categories,
        "train_budget_ratio": args.train_budget_ratio,
        "num_neighbours": args.num_neighbours,
        "encoder_name": args.encoder_name,
        "auto_masking": args.auto_masking,
        "masking": args.masking,
        "coreset_subsampling": args.coreset_subsampling,
        "sampling_ratio": args.sampling_ratio,
        "precision": args.precision,
        "rows": rows,
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
