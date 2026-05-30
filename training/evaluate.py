"""Test-set evaluation + metrics for all 4 HEMT-CLIP ablation variants.

Runs each variant's best checkpoint on the requested split (default: test),
computes the Blueprint §9.1 metrics (accuracy, precision/recall/F1 per-class,
AUC-ROC, confusion matrix), and saves the report-ready figures:

    - 4 confusion-matrix PNGs (one per variant)
    - 1 ROC overlay PNG (all 4 variants on one axes)
    - 1 ablation F1 bar PNG
    - 1 per-class precision/recall grouped bar PNG

plus a comparison table (CSV + Markdown) and per-variant prediction arrays
(.npz: logits, probs, preds, labels) for downstream XAI/SHAP use.

Auto-discovers `*_{variant}_*_best.pt` in cfg.checkpointing.dir, preferring
non-`_seed*` files (so hemt_clip picks the canonical v4 seed=42 ckpt, not
the seed=7/123 robustness runs). Override with --checkpoints (JSON mapping).

Example:
    !python -m training.evaluate
    !python -m training.evaluate --split val
    !python -m training.evaluate --checkpoints configs/eval_checkpoints.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader

from data.dataset import HEMTClipDataset
from models.hemt_clip import VARIANTS, build_from_config

LOG = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=Path("configs/base.yaml"), type=Path)
    p.add_argument("--ckpt-dir", default=None, type=Path,
                   help="Where to auto-discover *_best.pt. Defaults to cfg.checkpointing.dir.")
    p.add_argument("--checkpoints", default=None, type=Path,
                   help="JSON mapping variant -> ckpt path. Overrides auto-discovery.")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--out-dir", default=Path("outputs/eval"), type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def discover_checkpoints(ckpt_dir: Path) -> dict[str, Path]:
    """For each variant, pick the canonical best.pt:
       1) prefer non-`_seed*` files (skip seed-robustness runs),
       2) among those, pick the most recently modified."""
    out: dict[str, Path] = {}
    for variant in VARIANTS:
        candidates = list(ckpt_dir.glob(f"hemt_{variant}_*_best.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint matching hemt_{variant}_*_best.pt in {ckpt_dir}"
            )
        no_seed = [c for c in candidates if "_seed" not in c.name]
        chosen = sorted(no_seed or candidates, key=lambda p: p.stat().st_mtime)[-1]
        out[variant] = chosen
    return out


def load_checkpoints_json(path: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text())
    return {v: Path(p) for v, p in raw.items()}


def load_model(variant: str, ckpt_path: Path, cfg: dict, device: torch.device) -> tuple[torch.nn.Module, float]:
    """Build model for `variant` from cfg, load weights from ckpt_path, return (model, val_f1_at_save)."""
    model = build_from_config(cfg, variant=variant).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOG.warning("  missing keys (%d): %s", len(missing), missing[:3])
    if unexpected:
        LOG.warning("  unexpected keys (%d): %s", len(unexpected), unexpected[:3])
    val_f1 = float(payload.get("state", {}).get("best_val_f1", 0.0)) if isinstance(payload, dict) else 0.0
    return model, val_f1


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (logits, probs, preds, labels) — all np arrays of length N_split."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model(b)
        all_logits.append(out["logits"].float().cpu().numpy())
        all_labels.append(b["label"].cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = logits.argmax(axis=-1).astype(np.int64)
    return logits, probs, preds, labels


def compute_metrics(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    report = classification_report(
        labels, preds, labels=[0, 1], target_names=["real", "fake"],
        output_dict=True, zero_division=0,
    )
    return {
        "n":                int(len(labels)),
        "accuracy":         float(accuracy_score(labels, preds)),
        "f1_binary_pos1":   float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_macro":         float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_pos1":   float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_pos1":      float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "auc_roc":          float(roc_auc_score(labels, probs[:, 1])),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "real": {k: float(v) for k, v in report["real"].items()},
            "fake": {k: float(v) for k, v in report["fake"].items()},
        },
    }


def plot_confusion_matrix(cm: list[list[int]], variant: str, out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_arr, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real (0)", "fake (1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["real (0)", "fake (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {variant}")
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_arr[i, j]:,}", ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


COLOR_BY_VARIANT = {
    "text_only":     "#999999",
    "image_only":    "#bbbbbb",
    "concat_fusion": "#4477aa",
    "hemt_clip":     "#cc6677",
}


def plot_roc_overlay(per_variant: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for variant, data in per_variant.items():
        labels = data["labels"]; probs = data["probs"]
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        auc = roc_auc_score(labels, probs[:, 1])
        ax.plot(fpr, tpr, lw=2,
                color=COLOR_BY_VARIANT.get(variant, "k"),
                label=f"{variant}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="chance")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    n = len(next(iter(per_variant.values()))["labels"])
    ax.set_title(f"ROC curves — test split (n={n:,})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_f1_bar(metrics_by_variant: dict[str, dict], out_path: Path) -> None:
    variants = list(metrics_by_variant.keys())
    f1s = [m["f1_binary_pos1"] for m in metrics_by_variant.values()]
    colors = [COLOR_BY_VARIANT.get(v, "#888") for v in variants]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(variants, f1s, color=colors)
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, f1 + 0.005,
                f"{f1:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Test F1 (fake = positive class)")
    ax.set_title("Test-set F1 by variant")
    ax.set_ylim(min(f1s) - 0.05, max(f1s) + 0.03)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_pr(metrics_by_variant: dict[str, dict], out_path: Path) -> None:
    variants = list(metrics_by_variant.keys())
    x = np.arange(len(variants))
    w = 0.2
    real_p = [m["per_class"]["real"]["precision"] for m in metrics_by_variant.values()]
    real_r = [m["per_class"]["real"]["recall"]    for m in metrics_by_variant.values()]
    fake_p = [m["per_class"]["fake"]["precision"] for m in metrics_by_variant.values()]
    fake_r = [m["per_class"]["fake"]["recall"]    for m in metrics_by_variant.values()]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 1.5 * w, real_p, w, color="#4477aa", label="real precision")
    ax.bar(x - 0.5 * w, real_r, w, color="#88aacc", label="real recall")
    ax.bar(x + 0.5 * w, fake_p, w, color="#cc6677", label="fake precision")
    ax.bar(x + 1.5 * w, fake_r, w, color="#ee99aa", label="fake recall")
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall by variant (test split)")
    ax.legend(ncol=4, loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_summary_table(
    metrics_by_variant: dict[str, dict],
    checkpoints: dict[str, Path],
    val_f1_by_variant: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for variant, m in metrics_by_variant.items():
        rows.append({
            "variant":    variant,
            "n":          m["n"],
            "val_f1":     round(val_f1_by_variant[variant], 4),
            "test_acc":   round(m["accuracy"], 4),
            "test_f1":    round(m["f1_binary_pos1"], 4),
            "test_prec":  round(m["precision_pos1"], 4),
            "test_rec":   round(m["recall_pos1"], 4),
            "test_auc":   round(m["auc_roc"], 4),
            "checkpoint": checkpoints[variant].name,
        })
    return pd.DataFrame(rows).set_index("variant")


def main() -> int:
    setup_logging()
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    LOG.info("device=%s | gpu=%s", device,
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—")

    # ---- Resolve checkpoints ------------------------------------------------
    if args.checkpoints is not None:
        ckpts = load_checkpoints_json(args.checkpoints)
        LOG.info("loaded checkpoints from %s", args.checkpoints)
    else:
        ckpt_dir = args.ckpt_dir or Path(cfg["checkpointing"]["dir"])
        ckpts = discover_checkpoints(ckpt_dir)
        LOG.info("auto-discovered checkpoints in %s:", ckpt_dir)
    for v, p in ckpts.items():
        LOG.info("  %-14s : %s", v, p.name)

    # ---- Shared dataset / loader -------------------------------------------
    tok_name = cfg["model"]["text"]["name"]
    max_len = cfg["data"]["max_text_len"]
    hdf5 = cfg["data"]["hdf5_path"]
    eval_ds = HEMTClipDataset(hdf5, args.split, tok_name, max_len)
    LOG.info("dataset: split=%s n=%d  hdf5=%s", args.split, len(eval_ds), hdf5)
    loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    metrics_by_variant: dict[str, dict[str, Any]] = {}
    preds_by_variant: dict[str, dict[str, np.ndarray]] = {}
    val_f1_by_variant: dict[str, float] = {}

    # ---- Per-variant inference ---------------------------------------------
    for variant, ckpt_path in ckpts.items():
        LOG.info("============================================================")
        LOG.info("[%s] loading %s", variant, ckpt_path.name)
        model, val_f1 = load_model(variant, ckpt_path, cfg, device)
        val_f1_by_variant[variant] = val_f1
        LOG.info("[%s] val F1 at checkpoint: %.4f", variant, val_f1)

        LOG.info("[%s] running inference on %s split…", variant, args.split)
        logits, probs, preds, labels = collect_predictions(model, loader, device)
        m = compute_metrics(probs, preds, labels)
        LOG.info("[%s] test acc=%.4f f1=%.4f prec=%.4f rec=%.4f auc=%.4f",
                 variant, m["accuracy"], m["f1_binary_pos1"],
                 m["precision_pos1"], m["recall_pos1"], m["auc_roc"])

        metrics_by_variant[variant] = m
        preds_by_variant[variant] = {"labels": labels, "probs": probs, "preds": preds}

        # Per-variant artefacts
        np.savez_compressed(args.out_dir / f"preds_{variant}.npz",
                            logits=logits, probs=probs, preds=preds, labels=labels)
        plot_confusion_matrix(m["confusion_matrix"], variant,
                              args.out_dir / f"cm_{variant}.png")
        with open(args.out_dir / f"metrics_{variant}.json", "w") as f:
            json.dump({"variant": variant,
                       "checkpoint": str(ckpt_path),
                       "split": args.split,
                       "val_f1_at_ckpt": val_f1,
                       **m}, f, indent=2)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Cross-variant artefacts -------------------------------------------
    plot_roc_overlay(preds_by_variant, args.out_dir / f"roc_overlay_{args.split}.png")
    plot_f1_bar(metrics_by_variant, args.out_dir / f"f1_bar_{args.split}.png")
    plot_per_class_pr(metrics_by_variant, args.out_dir / f"per_class_pr_{args.split}.png")

    table = build_summary_table(metrics_by_variant, ckpts, val_f1_by_variant)
    table.to_csv(args.out_dir / f"summary_{args.split}.csv")
    (args.out_dir / f"summary_{args.split}.md").write_text(
        table.to_markdown() + "\n", encoding="utf-8")

    LOG.info("============================================================")
    LOG.info("Summary (split=%s):", args.split)
    for line in table.to_string().splitlines():
        LOG.info("  %s", line)
    LOG.info("wrote: %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
