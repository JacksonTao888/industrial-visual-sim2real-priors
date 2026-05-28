#!/usr/bin/env python3
"""Run SuperSimpleNet no-CAD synthetic-anomaly baselines with anomalib."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler

import anomalib
from anomalib.engine import Engine
from anomalib.models import Supersimplenet

from run_draem_anomalib import (
    category_list,
    flatten_metrics,
    make_datamodule,
    write_csv,
)


def enable_repeat_train_loader(datamodule: Any, steps_per_epoch: int) -> None:
    """Replace train_dataloader with replacement sampling for tiny categories."""

    def train_dataloader(self: Any) -> DataLoader:
        sampler = RandomSampler(
            self.train_data,
            replacement=True,
            num_samples=steps_per_epoch * self.train_batch_size,
        )
        return DataLoader(
            dataset=self.train_data,
            sampler=sampler,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.external_collate_fn or self.train_data.collate_fn,
        )

    datamodule.train_dataloader = types.MethodType(train_dataloader, datamodule)


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
    perlin_threshold: float,
    backbone: str,
    layers: list[str],
    adapt_cls_features: bool,
    max_epochs: int,
    max_steps: int | None,
    train_steps_per_epoch: int | None,
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
    if train_steps_per_epoch is not None:
        enable_repeat_train_loader(datamodule, train_steps_per_epoch)

    model = Supersimplenet(
        perlin_threshold=perlin_threshold,
        backbone=backbone,
        layers=layers,
        supervised=False,
        adapt_cls_features=adapt_cls_features,
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
        "perlin_threshold": perlin_threshold,
        "backbone": backbone,
        "layers": ",".join(layers),
        "adapt_cls_features": adapt_cls_features,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "train_steps_per_epoch": train_steps_per_epoch,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "seed": seed,
        **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


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
    parser.add_argument("--perlin-threshold", type=float, default=0.2)
    parser.add_argument("--backbone", default="wide_resnet50_2.tv_in1k")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--adapt-cls-features", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-steps-per-epoch", type=int, default=None)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    project_root = args.project_root
    dataset_root = args.root or (
        project_root / "datasets" / ("mvtec_ad" if args.dataset == "mvtec_ad" else "visa")
    )
    out_root = args.out_root or project_root / "experiments" / "runs" / "no_cad" / "supersimplenet"
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
            perlin_threshold=args.perlin_threshold,
            backbone=args.backbone,
            layers=args.layers,
            adapt_cls_features=args.adapt_cls_features,
            max_epochs=args.max_epochs,
            max_steps=args.max_steps,
            train_steps_per_epoch=args.train_steps_per_epoch,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    result_dir = out_root / args.dataset
    csv_path = result_dir / "supersimplenet_results.csv"
    summary_path = result_dir / "supersimplenet_results.summary.json"
    write_csv(csv_path, rows)
    summary = {
        "method_family": "Feature-level synthetic anomaly generation / SuperSimpleNet",
        "library": "anomalib",
        "anomalib_version": anomalib.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "categories": categories,
        "rows": rows,
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
