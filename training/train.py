"""Main training loop with two-stage fine-tuning, TensorBoard, and resume.

Stages (Blueprint §8.1):
    Stage 1 (head warmup): both encoders FULLY frozen, lr=1e-4, 1 epoch.
              Trains projections + fusion + classifier only.
    Stage 2 (encoder fine-tune): last N layers/blocks of each encoder unfrozen,
              lr=2e-5, up to N epochs with early-stop on val F1 (patience from cfg).

Engineering (Blueprint §14):
    fp16 autocast + GradScaler, AdamW + linear warmup (per stage),
    gradient clipping, gradient checkpointing on backbones,
    per-epoch atomic checkpoints to Drive (keep_last_n), separate best.pt on
    val-F1 improvement, full resume (model + optim + scaler + scheduler +
    epoch + RNG state).

Tracking: TensorBoard event files → cfg.logging.tensorboard_dir/{run_name}/.

CLI:
    python -m training.train --variant hemt_clip
    python -m training.train --variant hemt_clip --resume <ckpt.pt>
    python -m training.train --variant text_only --config configs/debug.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from data.dataset import HEMTClipDataset
from models.hemt_clip import VARIANTS, build_from_config

LOG = logging.getLogger("train")


# ----------------------------- CLI + setup ------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/base.yaml", type=Path)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--resume", default=None, type=Path,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--run-name", default=None,
                   help="Override auto-generated run name (default: {prefix}_{variant}_{YYYYMMDD-HHMM}).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override cfg.seed for this run. When set, run_name auto-gets _seed{N} suffix "
                        "(unless --run-name is also explicit). Used for multi-seed robustness reporting.")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_batch_size(cfg: dict) -> int:
    """Pick batch size from cfg.training.batch_size by detected GPU class."""
    if not torch.cuda.is_available():
        return 4
    name = torch.cuda.get_device_name(0).upper()
    bs = cfg["training"]["batch_size"]
    if "A100" in name:
        return bs["A100"]
    if "L4" in name:
        return bs["L4"]
    return bs["T4"]


# ----------------------------- Stage management -------------------------------

def apply_stage(model: nn.Module, stage: int, cfg: dict) -> None:
    """Stage 1: freeze both encoder backbones entirely (head warmup).
    Stage 2: re-apply the per-encoder freezing from __init__ (last N trainable)."""
    if stage == 1:
        for sub in (model.text_encoder, model.image_encoder):
            if sub is None:
                continue
            for p in sub.backbone.parameters():
                p.requires_grad = False
    elif stage == 2:
        if model.text_encoder is not None:
            model.text_encoder._freeze(cfg["model"]["text"]["trainable_layers"])
        if model.image_encoder is not None:
            model.image_encoder._freeze(cfg["model"]["image"]["trainable_blocks"])
    else:
        raise ValueError(f"unknown stage: {stage}")


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


# ----------------------------- Train / eval -----------------------------------

def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(
    model, loader, optim, scheduler, criterion, scaler,
    device, cfg, writer, state,
) -> dict[str, float]:
    model.train()
    n = 0
    loss_sum = 0.0
    corr = 0
    grad_clip = cfg["training"]["grad_clip"]
    log_every = cfg["logging"]["log_every_n_steps"]
    accum = max(1, cfg["training"].get("grad_accum_steps", 1))

    optim.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        batch = move(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model(batch)
            loss = criterion(out["logits"], batch["label"]) / accum
        scaler.scale(loss).backward()

        bs = batch["label"].size(0)
        loss_sum += loss.item() * accum * bs   # un-scale for reporting
        n += bs
        corr += (out["logits"].argmax(-1) == batch["label"]).sum().item()

        if (i + 1) % accum == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip,
            )
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            state.global_step += 1

            if state.global_step % log_every == 0:
                writer.add_scalar("train/loss", loss.item() * accum, state.global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], state.global_step)

    return {"loss": loss_sum / n, "acc": corr / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model(batch)
            loss = criterion(out["logits"], batch["label"])
        loss_sum += loss.item() * batch["label"].size(0)
        n += batch["label"].size(0)
        all_preds.append(out["logits"].argmax(-1).cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {
        "loss": loss_sum / n,
        "acc": float((y_pred == y_true).mean()),
        "f1":   float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "prec": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "rec":  float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }


# ----------------------------- Checkpointing ----------------------------------

@dataclass
class TrainState:
    stage: int = 1
    epoch: int = 0            # last completed epoch within current stage
    global_step: int = 0
    best_val_f1: float = 0.0
    epochs_since_improve: int = 0


def save_checkpoint(path: Path, model, optim, scaler, scheduler, state: TrainState) -> None:
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict() if optim is not None else None,
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "state": state.__dict__,
        "rng": {
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, scaler, device) -> tuple[TrainState, dict]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    scaler.load_state_dict(payload["scaler"])
    torch.set_rng_state(payload["rng"]["torch"])
    if torch.cuda.is_available() and payload["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    np.random.set_state(payload["rng"]["numpy"])
    random.setstate(payload["rng"]["python"])
    return TrainState(**payload["state"]), payload


def cleanup_old_checkpoints(ckpt_dir: Path, run_name: str, keep: int) -> None:
    pattern = f"{run_name}_stage*_epoch*.pt"
    ckpts = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep]:
        try:
            old.unlink()
        except OSError as e:
            LOG.warning("could not delete old ckpt %s: %s", old, e)


def make_run_name(prefix: str, variant: str) -> str:
    return f"{prefix}_{variant}_{datetime.now().strftime('%Y%m%d-%H%M')}"


# ----------------------------- Main -------------------------------------------

def main() -> int:
    setup_logging()
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = detect_batch_size(cfg)
    LOG.info("device=%s | gpu=%s | batch=%d",
             device,
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—",
             batch_size)

    # --- Data
    tok_name = cfg["model"]["text"]["name"]
    max_len = cfg["data"]["max_text_len"]
    hdf5 = cfg["data"]["hdf5_path"]

    train_ds = HEMTClipDataset(hdf5, "train", tok_name, max_len)
    val_ds   = HEMTClipDataset(hdf5, "val",   tok_name, max_len)
    LOG.info("dataset: train=%d val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # --- Model
    model = build_from_config(cfg, variant=args.variant).to(device)
    if cfg["training"].get("gradient_checkpointing"):
        if model.text_encoder is not None:
            model.text_encoder.backbone.gradient_checkpointing_enable()
        if model.image_encoder is not None:
            model.image_encoder.backbone.gradient_checkpointing_enable()
        LOG.info("gradient checkpointing enabled on backbones")

    # --- Run name + output dirs
    run_name = args.run_name or make_run_name(cfg["logging"]["run_name_prefix"], args.variant)
    if args.seed is not None and args.run_name is None:
        run_name = f"{run_name}_seed{args.seed}"
    tb_dir   = Path(cfg["logging"]["tensorboard_dir"]) / run_name
    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    LOG.info("run=%s\n  tb_dir   = %s\n  ckpt_dir = %s", run_name, tb_dir, ckpt_dir)

    # --- Loss + scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Resume
    state = TrainState()
    resume_payload: dict | None = None
    if args.resume is not None:
        LOG.info("resuming from %s", args.resume)
        state, resume_payload = load_checkpoint(args.resume, model, scaler, device)
        LOG.info("resumed: stage=%d epoch=%d step=%d best_f1=%.4f",
                 state.stage, state.epoch, state.global_step, state.best_val_f1)

    # --- Stage loop
    stages = [
        {"stage": 1, "epochs": cfg["training"]["stage1"]["epochs"], "lr": cfg["training"]["stage1"]["lr"]},
        {"stage": 2, "epochs": cfg["training"]["stage2"]["epochs"], "lr": cfg["training"]["stage2"]["lr"]},
    ]
    stop_training = False
    for s_cfg in stages:
        # Skip stages already completed in the resumed run.
        if state.stage > s_cfg["stage"]:
            continue

        LOG.info("=" * 60)
        LOG.info("Stage %d — lr=%g, max_epochs=%d", s_cfg["stage"], s_cfg["lr"], s_cfg["epochs"])
        apply_stage(model, s_cfg["stage"], cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOG.info("trainable params: %.2fM", trainable / 1e6)

        # Fresh optim + scheduler per stage.
        accum = max(1, cfg["training"].get("grad_accum_steps", 1))
        steps_per_epoch = max(1, len(train_loader) // accum)
        total_steps = steps_per_epoch * s_cfg["epochs"]
        optim = build_optimizer(model, s_cfg["lr"], cfg["training"]["weight_decay"])
        scheduler = build_scheduler(optim, total_steps, cfg["training"]["warmup_ratio"])

        # If resumed mid-stage, restore optim/scheduler/scaler state for this stage.
        if resume_payload is not None and state.stage == s_cfg["stage"]:
            if resume_payload["optim"]:
                optim.load_state_dict(resume_payload["optim"])
            if resume_payload["scheduler"]:
                scheduler.load_state_dict(resume_payload["scheduler"])
            resume_payload = None
        else:
            # Stage transition (fresh entry into this stage) → reset within-stage counters.
            state.epoch = 0
            state.epochs_since_improve = 0

        state.stage = s_cfg["stage"]

        for ep in range(state.epoch + 1, s_cfg["epochs"] + 1):
            ep_t0 = time.time()
            tr = train_one_epoch(
                model, train_loader, optim, scheduler, criterion, scaler,
                device, cfg, writer, state,
            )
            val = evaluate(model, val_loader, criterion, device)
            state.epoch = ep

            for k, v in tr.items():
                writer.add_scalar(f"train_epoch/{k}", v, state.global_step)
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, state.global_step)

            LOG.info(
                "stage=%d epoch=%d  train_loss=%.4f train_acc=%.3f  "
                "val_loss=%.4f val_acc=%.3f val_f1=%.4f val_prec=%.3f val_rec=%.3f  (%.0fs)",
                s_cfg["stage"], ep, tr["loss"], tr["acc"],
                val["loss"], val["acc"], val["f1"], val["prec"], val["rec"],
                time.time() - ep_t0,
            )

            # Histograms (cheap, useful for debugging fine-tune dynamics).
            if ep % cfg["logging"].get("hist_every_n_epochs", 1) == 0:
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    writer.add_histogram(f"weights/{name}", p, state.global_step)
                    if p.grad is not None:
                        writer.add_histogram(f"grads/{name}", p.grad, state.global_step)

            # Per-epoch checkpoint (atomic).
            ckpt_path = ckpt_dir / f"{run_name}_stage{s_cfg['stage']}_epoch{ep}.pt"
            save_checkpoint(ckpt_path, model, optim, scaler, scheduler, state)
            cleanup_old_checkpoints(ckpt_dir, run_name, cfg["checkpointing"]["keep_last_n"])

            # Best checkpoint + early stop (stage 2 only — stage 1 is short).
            if val["f1"] > state.best_val_f1:
                state.best_val_f1 = val["f1"]
                state.epochs_since_improve = 0
                best_path = ckpt_dir / f"{run_name}_best.pt"
                save_checkpoint(best_path, model, optim, scaler, scheduler, state)
                LOG.info("  → new best val F1=%.4f, saved %s", state.best_val_f1, best_path.name)
            else:
                state.epochs_since_improve += 1

            patience = cfg["training"]["early_stopping_patience"]
            if s_cfg["stage"] == 2 and state.epochs_since_improve >= patience:
                LOG.info("early stop: val F1 plateaued for %d epochs", patience)
                stop_training = True
                break

        if stop_training:
            break

    writer.close()
    LOG.info("training complete. best val F1 = %.4f", state.best_val_f1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
