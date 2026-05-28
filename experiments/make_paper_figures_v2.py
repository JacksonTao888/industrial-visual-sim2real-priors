#!/usr/bin/env python3
"""Create publication-style paper figures from existing experiment outputs.

This script is intentionally post-processing only. It reads existing CSV/JSON
metrics and generated visual assets, then writes publication-oriented panels.
It does not train, evaluate, or regenerate model predictions.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs" / "paper_figures_v2"
MAIN_OUT = OUT / "main"
SUPP_OUT = OUT / "supplement"

# Style follows the installed scientific-visualization skill:
# compact journal dimensions, sans-serif type, colorblind-safe colors, vector
# exports, and no large title text inside figures.
COLORS = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "ink": "#1F2933",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "panel_bg": "#F8FAFC",
}

METHOD_COLORS = {
    "PatchCore": COLORS["blue"],
    "EfficientAD-S": COLORS["orange"],
    "WinCLIP": COLORS["purple"],
    "AnomalyDINO-S": COLORS["green"],
    "AnomalyDINO-L": "#44AA99",
}

DATASET_COLORS = {"MVTec AD": COLORS["blue"], "VisA": COLORS["vermillion"]}


YOLO_RUNS = [
    ("B0", "Synthetic 5k", "tless_pbr_5k_to_real_yolov8n"),
    ("B1", "Synthetic 50k", "tless_pbr_full_to_real_yolov8n"),
    ("B2", "Domain rand.", "tless_pbr_full_to_real_yolov8n_domain_randomization"),
    ("B3", "DR + 5% real", "tless_dr_pretrained_real_5pct_finetune"),
    ("B4", "DR + bg neg.", "tless_pbr_full_dr_plus_real_bg_negatives"),
    ("B5", "YOLOv8s + DR", "tless_pbr_full_to_real_yolov8s_domain_randomization"),
    ("B6", "B5 + 5% real", "tless_yolov8s_dr_pretrained_real_5pct_finetune"),
]

NO_CAD_RUNS = [
    ("PatchCore", "patchcore", "Normal memory", "patchcore_results.summary.json"),
    ("EfficientAD-S", "efficientad", "Teacher-student", "efficientad_results.summary.json"),
    ("WinCLIP", "winclip", "Zero-shot VLM", "winclip_results.summary.json"),
    ("AnomalyDINO-S", "anomalydino", "Dense DINOv2", "anomalydino_results.summary.json"),
    ("AnomalyDINO-L", "anomalydino_large", "DINOv2 scale", "anomalydino_results.summary.json"),
]


def ensure_dirs() -> None:
    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    SUPP_OUT.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.prop_cycle": mpl.cycler(
                color=[
                    COLORS["orange"],
                    COLORS["sky"],
                    COLORS["green"],
                    COLORS["yellow"],
                    COLORS["blue"],
                    COLORS["vermillion"],
                    COLORS["purple"],
                    COLORS["black"],
                ]
            ),
        }
    )


def save_figure(fig: plt.Figure, stem: str, subdir: Path = MAIN_OUT) -> list[str]:
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = subdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.04)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def read_last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[-1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def crop_grid_tile(path: Path, row: int = 1, col: int = 1, rows: int = 4, cols: int = 4) -> np.ndarray:
    """Crop one tile from an Ultralytics validation grid for a readable example."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        tile_w = im.width // cols
        tile_h = im.height // rows
        left = col * tile_w
        upper = row * tile_h
        tile = im.crop((left, upper, left + tile_w, upper + tile_h))
        return np.asarray(tile)


def fit_image(path: Path, size: tuple[int, int], bg=(255, 255, 255)) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, bg)
        xy = ((size[0] - im.width) // 2, (size[1] - im.height) // 2)
        canvas.paste(im, xy)
        return canvas


def wrapped_preview_array(path: Path, segments: int, max_height: int | None = None) -> np.ndarray:
    """Wrap a tall existing preview grid into horizontal bands for readability."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        band_h = im.height // segments
        bands = []
        for i in range(segments):
            y0 = i * band_h
            y1 = im.height if i == segments - 1 else (i + 1) * band_h
            bands.append(im.crop((0, y0, im.width, y1)))
        out = Image.new("RGB", (im.width * segments, max(b.height for b in bands)), (255, 255, 255))
        x = 0
        for band in bands:
            out.paste(band, (x, 0))
            x += im.width
        if max_height and out.height > max_height:
            scale = max_height / out.height
            out = out.resize((int(out.width * scale), max_height), Image.Resampling.LANCZOS)
        return np.asarray(out)


def make_contact_sheet(
    paths: list[Path],
    out_path: Path,
    *,
    tile_size=(360, 270),
    cols=4,
    labels: list[str] | None = None,
) -> None:
    rows = math.ceil(len(paths) / cols)
    label_h = 34 if labels else 0
    pad = 14
    sheet = Image.new(
        "RGB",
        (cols * tile_size[0] + (cols + 1) * pad, rows * (tile_size[1] + label_h) + (rows + 1) * pad),
        (255, 255, 255),
    )
    for i, path in enumerate(paths):
        row, col = divmod(i, cols)
        x = pad + col * (tile_size[0] + pad)
        y = pad + row * (tile_size[1] + label_h + pad)
        sheet.paste(fit_image(path, tile_size), (x, y + label_h))
        if labels:
            import PIL.ImageDraw as ImageDraw
            import PIL.ImageFont as ImageFont

            draw = ImageDraw.Draw(sheet)
            font = ImageFont.load_default()
            draw.text((x, y), labels[i], fill=(25, 35, 45), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def subtle_grid(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=COLORS["light_gray"], linewidth=0.45)
    ax.set_axisbelow(True)


def make_taxonomy() -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 1.75))
    ax.set_axis_off()

    boxes = [
        (0.03, 0.25, 0.29, 0.52, COLORS["blue"], "CAD available", "Meshes, camera geometry\nrendering, pose/depth"),
        (0.355, 0.25, 0.29, 0.52, COLORS["green"], "Reference available", "Normal images, few-shot views\nappearance memory, features"),
        (0.68, 0.25, 0.29, 0.52, COLORS["vermillion"], "Weak-prior boundary", "Language/foundation priors\nreasoning, interaction"),
    ]
    for x, y, w, h, color, title, subtitle in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor=color,
            facecolor="#FFFFFF",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=6.7, color=COLORS["ink"], transform=ax.transAxes)

    ax.annotate(
        "",
        xy=(0.35, 0.51),
        xytext=(0.325, 0.51),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=COLORS["gray"]),
    )
    ax.annotate(
        "",
        xy=(0.675, 0.51),
        xytext=(0.65, 0.51),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=COLORS["gray"]),
    )
    ax.text(0.335, 0.08, "Empirical anchors in this review", ha="center", va="center", fontsize=6.8, color=COLORS["gray"], transform=ax.transAxes)
    ax.text(0.825, 0.08, "Literature boundary / future work", ha="center", va="center", fontsize=6.8, color=COLORS["gray"], transform=ax.transAxes)
    return save_figure(fig, "fig1_prior_availability_taxonomy")


def yolo_metrics() -> list[dict]:
    rows = []
    for run_id, label, run_dir in YOLO_RUNS:
        row = read_last_csv_row(ROOT / "experiments" / "runs" / "yolo" / run_dir / "results.csv")
        rows.append(
            {
                "id": run_id,
                "label": label,
                "run_dir": run_dir,
                "map50": float(row["metrics/mAP50(B)"]),
                "map5095": float(row["metrics/mAP50-95(B)"]),
            }
        )
    return rows


def draw_yolo_bar(ax: plt.Axes) -> None:
    rows = yolo_metrics()
    y = np.arange(len(rows))
    map5095 = [r["map5095"] for r in rows]
    map50 = [r["map50"] for r in rows]
    labels = [f"{r['id']}  {r['label']}" for r in rows]
    ax.barh(y, map5095, height=0.58, color=COLORS["blue"], alpha=0.88, label="mAP50-95")
    ax.scatter(map50, y, marker="D", s=18, color=COLORS["orange"], label="mAP50", zorder=3)
    for yi, v in zip(y, map5095):
        ax.text(v + 0.018, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.3, color=COLORS["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("T-LESS real validation mAP")
    ax.invert_yaxis()
    subtle_grid(ax, axis="x")
    ax.legend(frameon=False, loc="lower right", handlelength=1.5)
    clean_axis(ax)


def make_cad_renderer() -> list[str]:
    fig = plt.figure(figsize=(7.2, 4.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.55], hspace=0.32, wspace=0.12)
    examples = [
        ("B0", "Synthetic only", "tless_pbr_5k_to_real_yolov8n"),
        ("B2", "Domain randomized", "tless_pbr_full_to_real_yolov8n_domain_randomization"),
        ("B6", "5% real calibration", "tless_yolov8s_dr_pretrained_real_5pct_finetune"),
    ]
    for idx, (run_id, title, run_dir) in enumerate(examples):
        ax = fig.add_subplot(gs[0, idx])
        path = ROOT / "experiments" / "runs" / "yolo" / run_dir / "val_batch0_pred.jpg"
        ax.imshow(crop_grid_tile(path, row=1, col=1))
        ax.set_title(f"{run_id}: {title}", pad=2.5)
        ax.set_axis_off()
        panel_label(ax, chr(ord("A") + idx))
    ax = fig.add_subplot(gs[1, :])
    draw_yolo_bar(ax)
    panel_label(ax, "D")
    paths = save_figure(fig, "fig2_cad_as_renderer_transfer")

    fig2, ax2 = plt.subplots(figsize=(4.8, 2.65))
    draw_yolo_bar(ax2)
    paths += save_figure(fig2, "fig2b_yolo_b0_b6_map_bars")
    return paths


def best_depth_fusion_value() -> float:
    path = ROOT / "experiments" / "outputs" / "megapose_batches" / "conf_0p25_maxdet_10_scene_round_robin_per_scene_5_n100_offset0" / "megapose_depth_bbox_confidence_fusion_sweep.json"
    data = read_json(path)
    scores = data["good_pose_silhouette"]
    return max(float(v) for v in scores.values() if isinstance(v, (int, float)))


def make_cad_geometry() -> list[str]:
    base = ROOT / "experiments" / "outputs"
    mp_base = base / "megapose_batches" / "conf_0p25_maxdet_10_scene_round_robin_per_scene_5_n100_offset0"
    gt_path = base / "tless_heldout_gt_pose_cad_mask_iou_overlays" / "scene_000001_image_000001_gt_pose_cad_mask_iou_overlay.png"
    mp_success = mp_base / "megapose_batch_mask_iou_overlays" / "tless_scene_000001_image_000001_megapose_mask_iou_overlay.png"
    mp_imperfect = mp_base / "megapose_batch_mask_iou_overlays" / "tless_scene_000002_image_000024_megapose_mask_iou_overlay.png"
    if not mp_imperfect.exists():
        mp_imperfect = mp_success

    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78], hspace=0.28, wspace=0.10)
    for idx, (path, title) in enumerate(
        [
            (gt_path, "GT-pose CAD oracle"),
            (mp_success, "MegaPose predicted pose"),
            (mp_imperfect, "MegaPose cross-scene example"),
        ]
    ):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(image_array(path))
        ax.set_axis_off()
        ax.set_title(title, pad=2.5)
        panel_label(ax, chr(ord("A") + idx))

    ax = fig.add_subplot(gs[1, :])
    summary = read_json(mp_base / "megapose_depth_consistency_scores.summary.json")
    auroc = summary["auroc"]["good_pose_silhouette"]
    values = [
        auroc["detector_confidence"],
        auroc["depth_consistency_score"],
        best_depth_fusion_value(),
    ]
    labels = ["Detector confidence", "Depth consistency", "Best confidence + depth fusion"]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["green"]]
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.68, 0.91)
    ax.set_xlabel("Good-pose selection AUROC")
    ax.invert_yaxis()
    subtle_grid(ax, axis="x")
    for yi, v in zip(y, values):
        ax.text(v + 0.004, yi, f"{v:.3f}", ha="left", va="center", fontsize=6.8, color=COLORS["ink"])
    clean_axis(ax)
    panel_label(ax, "D")
    paths = save_figure(fig, "fig3_cad_at_test_time_geometry")

    fig2, ax2 = plt.subplots(figsize=(3.35, 1.45))
    ax2.barh(y, values, color=colors, height=0.55)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0.68, 0.91)
    ax2.set_xlabel("Good-pose selection AUROC")
    ax2.invert_yaxis()
    subtle_grid(ax2, axis="x")
    for yi, v in zip(y, values):
        ax2.text(v + 0.004, yi, f"{v:.3f}", ha="left", va="center", fontsize=6.8, color=COLORS["ink"])
    clean_axis(ax2)
    paths += save_figure(fig2, "fig3b_depth_fusion_auroc")
    return paths


def no_cad_rows() -> list[dict]:
    rows = []
    for method, run_dir, family, file_name in NO_CAD_RUNS:
        for dataset in ("mvtec_ad", "visa"):
            data = read_json(ROOT / "experiments" / "runs" / "no_cad" / run_dir / dataset / file_name)
            metrics = data.get("metrics", data.get("metric_summary"))
            if metrics is None:
                raise KeyError(f"No metric summary found in {run_dir}/{dataset}/{file_name}")
            rows.append(
                {
                    "method": method,
                    "run_dir": run_dir,
                    "family": family,
                    "dataset": "MVTec AD" if dataset == "mvtec_ad" else "VisA",
                    "image_auroc": float(metrics["image_AUROC"]["mean"]),
                    "image_f1": float(metrics["image_F1Score"]["mean"]),
                    "pixel_auroc": float(metrics["pixel_AUROC"]["mean"]),
                    "pixel_f1": float(metrics["pixel_F1Score"]["mean"]),
                }
            )
    return rows


def draw_no_cad_grouped(ax: plt.Axes, metric: str, title: str) -> None:
    rows = no_cad_rows()
    methods = [m[0] for m in NO_CAD_RUNS]
    datasets = ["MVTec AD", "VisA"]
    y = np.arange(len(methods))
    for j, dataset in enumerate(datasets):
        values = [next(r[metric] for r in rows if r["method"] == method and r["dataset"] == dataset) for method in methods]
        offset = (j - 0.5) * 0.18
        ax.scatter(values, y + offset, s=26, color=DATASET_COLORS[dataset], label=dataset, zorder=3)
        for yi, v in zip(y + offset, values):
            ax.text(v + 0.008, yi, f"{v:.2f}", va="center", ha="left", fontsize=5.9, color=COLORS["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel(title)
    ax.invert_yaxis()
    subtle_grid(ax, axis="x")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2)
    clean_axis(ax)


def make_no_cad_full() -> list[str]:
    fig = plt.figure(figsize=(7.2, 4.95))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0], hspace=0.30, wspace=0.32)
    previews = [
        ("MVTec AD examples", ROOT / "experiments" / "outputs" / "no_cad_dataset_sanity" / "previews" / "mvtec_ad_preview_grid.jpg", 3),
        ("VisA examples", ROOT / "experiments" / "outputs" / "no_cad_dataset_sanity" / "previews" / "visa_preview_grid.jpg", 2),
    ]
    for idx, (title, path, segments) in enumerate(previews):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(wrapped_preview_array(path, segments, max_height=310))
        ax.set_axis_off()
        ax.set_title(title, pad=2.5)
        panel_label(ax, chr(ord("A") + idx))

    ax = fig.add_subplot(gs[1, 0])
    draw_no_cad_grouped(ax, "image_auroc", "Image AUROC")
    panel_label(ax, "C")
    ax = fig.add_subplot(gs[1, 1])
    draw_no_cad_grouped(ax, "pixel_auroc", "Pixel AUROC")
    panel_label(ax, "D")
    paths = save_figure(fig, "fig4_no_cad_full_baselines")

    fig2, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    draw_no_cad_grouped(axes[0], "image_auroc", "Image AUROC")
    draw_no_cad_grouped(axes[1], "pixel_auroc", "Pixel AUROC")
    paths += save_figure(fig2, "fig4b_no_cad_auroc_bars")
    return paths


def budget_aggregates() -> dict[tuple[str, str], list[dict]]:
    path = ROOT / "experiments" / "runs" / "no_cad" / "budget_ablation" / "no_cad_budget_ablation_results.csv"
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["method"], row["dataset"], float(row["train_budget_ratio"]))
            grouped.setdefault(key, []).append(row)
    result: dict[tuple[str, str], list[dict]] = {}
    for (method, dataset, budget), rows in grouped.items():
        result.setdefault((method, dataset), []).append(
            {
                "budget": budget,
                "image_auroc": float(np.mean([float(r["image_AUROC"]) for r in rows])),
                "pixel_auroc": float(np.mean([float(r["pixel_AUROC"]) for r in rows])),
            }
        )
    for values in result.values():
        values.sort(key=lambda r: r["budget"])
    return result


def make_budget_ablation() -> list[str]:
    agg = budget_aggregates()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True)
    style = {
        ("PatchCore", "mvtec_ad"): (COLORS["blue"], "-", "o"),
        ("PatchCore", "visa"): (COLORS["blue"], "--", "s"),
        ("AnomalyDINO-S", "mvtec_ad"): (COLORS["green"], "-", "o"),
        ("AnomalyDINO-S", "visa"): (COLORS["green"], "--", "s"),
    }
    labels = {
        ("PatchCore", "mvtec_ad"): "PatchCore / MVTec",
        ("PatchCore", "visa"): "PatchCore / VisA",
        ("AnomalyDINO-S", "mvtec_ad"): "AnomalyDINO-S / MVTec",
        ("AnomalyDINO-S", "visa"): "AnomalyDINO-S / VisA",
    }
    for key, rows in agg.items():
        color, ls, marker = style[key]
        x = [r["budget"] * 100 for r in rows]
        image_values = [r["image_auroc"] for r in rows]
        pixel_values = [r["pixel_auroc"] for r in rows]
        axes[0].plot(x, image_values, marker=marker, color=color, linestyle=ls, label=labels[key], linewidth=1.35, markersize=3.6)
        axes[1].plot(x, pixel_values, marker=marker, color=color, linestyle=ls, label=labels[key], linewidth=1.35, markersize=3.6)
    for ax, title in zip(axes, ["Image AUROC", "Pixel AUROC"]):
        ax.set_title(title)
        ax.set_xlabel("Normal-reference budget (%)")
        ax.set_xticks([5, 10, 25, 100])
        ax.set_xlim(3, 105)
        subtle_grid(ax, axis="both")
        clean_axis(ax)
    axes[0].set_ylabel("Selected-category mean AUROC")
    axes[0].set_ylim(0.70, 1.01)
    axes[1].set_ylim(0.94, 0.995)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.24, wspace=0.24)
    return save_figure(fig, "fig5_normal_reference_budget")


def make_supplement_contact_sheets() -> list[str]:
    written = []
    gt_dir = ROOT / "experiments" / "outputs" / "tless_heldout_gt_pose_cad_mask_iou_overlays"
    gt_paths = sorted(gt_dir.glob("*.png"))
    if gt_paths:
        out = SUPP_OUT / "supp_gt_pose_cad_overlay_contact_sheet.jpg"
        make_contact_sheet(gt_paths, out, cols=4, labels=[p.stem.replace("_gt_pose_cad_mask_iou_overlay", "") for p in gt_paths])
        written.append(str(out.relative_to(ROOT)))

    for name, rel in [
        ("supp_megapose_n20_overlay_contact_sheet.jpg", "conf_0p25_maxdet_10_n20_offset0"),
        ("supp_megapose_cross_scene_overlay_contact_sheet.jpg", "conf_0p25_maxdet_10_scene_round_robin_per_scene_5_n100_offset0"),
    ]:
        mp_dir = ROOT / "experiments" / "outputs" / "megapose_batches" / rel / "megapose_batch_mask_iou_overlays"
        paths = sorted(mp_dir.glob("*.png"))
        if paths:
            out = SUPP_OUT / name
            make_contact_sheet(paths, out, cols=4, labels=[p.stem.replace("_megapose_mask_iou_overlay", "") for p in paths])
            written.append(str(out.relative_to(ROOT)))

    yolo_paths = []
    yolo_labels = []
    for run_id, _, run_dir in YOLO_RUNS:
        run_path = ROOT / "experiments" / "runs" / "yolo" / run_dir
        for file_name in ("results.png", "BoxPR_curve.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg"):
            path = run_path / file_name
            if path.exists():
                yolo_paths.append(path)
                yolo_labels.append(f"{run_id} {file_name}")
    if yolo_paths:
        out = SUPP_OUT / "supp_yolo_visual_pack_contact_sheet.jpg"
        make_contact_sheet(yolo_paths, out, tile_size=(320, 245), cols=4, labels=yolo_labels)
        written.append(str(out.relative_to(ROOT)))

    ssn_base = (
        ROOT
        / "experiments"
        / "runs"
        / "no_cad"
        / "supersimplenet_official_cli"
        / "probe"
        / "mvtec_ad"
        / "toothbrush"
        / "Supersimplenet"
        / "MVTecADStringSplit"
        / "toothbrush"
        / "v0"
        / "images"
    )
    ssn_paths = sorted((ssn_base / "good").glob("*.png"))[:4] + sorted((ssn_base / "defective").glob("*.png"))[:8]
    ssn_labels = [f"good/{p.name}" if "good" in p.parts else f"defective/{p.name}" for p in ssn_paths]
    if ssn_paths:
        out = SUPP_OUT / "supp_supersimplenet_probe_contact_sheet.jpg"
        make_contact_sheet(ssn_paths, out, tile_size=(300, 220), cols=4, labels=ssn_labels)
        written.append(str(out.relative_to(ROOT)))
    return written


def write_manifest(paths: list[str]) -> None:
    manifest = {
        "purpose": "Publication-style v2 figure package for the industrial sim2real review.",
        "script": "experiments/make_paper_figures_v2.py",
        "style_basis": [
            "Rules applied: compact journal sizing, sans-serif typography, colorblind-safe palette, vector PDF/SVG export, no figure-level titles.",
        ],
        "note": "Generated from existing CSV/JSON metrics and already-produced images only; no experiments were rerun.",
        "main_figure_plan": [
            "Fig. 1: prior-availability taxonomy.",
            "Fig. 2: CAD-as-renderer qualitative progression and B0-B6 mAP bars.",
            "Fig. 3: CAD-at-test-time GT/MegaPose overlays and depth-fusion AUROC.",
            "Fig. 4: no-CAD dataset previews and full baseline AUROC bars.",
            "Fig. 5: normal-reference budget ablation.",
        ],
        "supplement_plan": [
            "YOLO visual-pack contact sheet.",
            "GT-pose CAD overlay contact sheet.",
            "MegaPose overlay contact sheets.",
            "SuperSimpleNet qualitative probe contact sheet.",
            "Full per-category tables remain in the source CSV/JSON files.",
        ],
        "generated_files": paths,
    }
    (OUT / "paper_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    apply_style()
    ensure_dirs()
    paths = []
    paths += make_taxonomy()
    paths += make_cad_renderer()
    paths += make_cad_geometry()
    paths += make_no_cad_full()
    paths += make_budget_ablation()
    paths += make_supplement_contact_sheets()
    write_manifest(paths)
    print(f"Wrote {len(paths)} files under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
