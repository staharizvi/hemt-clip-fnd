"""Run all four ablation variants end-to-end.

Variants (Blueprint §7):
    A: text_only       — RoBERTa + classifier
    B: image_only      — CLIP ViT + classifier
    C: concat_fusion   — [text, image, alpha] -> classifier
    D: hemt_clip       — cross-attention, alpha concatenated as a feature
    E: gated_fusion    — cross-attention, alpha as a CLIP-similarity gate (ablation
                         vs D: tests "alpha as gate" vs "alpha as learned feature")

Calls `python -m training.train` for each variant in its own subprocess so
GPU memory / random state are fully reset between runs. Per-variant logs
stream live AND get appended to a per-variant logfile in the run dir.

After all variants finish, reads each variant's `best.pt`, extracts
`best_val_f1` from the embedded TrainState, and writes a comparison table
to `outputs/ablation_summary_{YYYYMMDD-HHMM}.{csv,md}`.

Resume / re-run policy:
    Default: skips a variant if a recent `*_best.pt` for it already exists
             in cfg.checkpointing.dir (looks at runs from the last 24h).
    --force: re-trains every variant from scratch.
    --variants V1 V2: train only the named variants.

CLI:
    python -m training.ablation_runner
    python -m training.ablation_runner --variants text_only image_only
    python -m training.ablation_runner --force
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import yaml

from models.hemt_clip import VARIANTS

LOG = logging.getLogger("ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/base.yaml", type=Path)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=None,
                   help="Subset of variants to run (default: cfg.ablation.variants).")
    p.add_argument("--force", action="store_true",
                   help="Re-train even variants that already have a recent best.pt.")
    p.add_argument("--skip-summary", action="store_true",
                   help="Don't write the summary table at the end (useful for partial runs).")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def find_recent_best(ckpt_dir: Path, variant: str, within_hours: int = 24) -> Path | None:
    """Most recent *_best.pt for this variant whose mtime is within window."""
    cutoff = datetime.now() - timedelta(hours=within_hours)
    candidates = sorted(
        ckpt_dir.glob(f"*_{variant}_*_best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if datetime.fromtimestamp(c.stat().st_mtime) >= cutoff:
            return c
    return None


def run_variant(variant: str, config_path: Path, log_path: Path) -> int:
    """Launch train.py as a subprocess; stream stdout/stderr to console AND logfile."""
    cmd = [sys.executable, "-m", "training.train",
           "--variant", variant,
           "--config", str(config_path)]
    LOG.info("launching: %s", " ".join(cmd))
    LOG.info("  logfile: %s", log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        lf.write(f"# {datetime.now().isoformat()}  {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,  # line-buffered
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")        # live to console
            lf.write(line); lf.flush() # and to disk
        rc = proc.wait()
    return rc


def read_best_val_f1(best_ckpt: Path) -> dict:
    payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    state = payload.get("state", {})
    return {
        "ckpt": str(best_ckpt),
        "best_val_f1": float(state.get("best_val_f1", float("nan"))),
        "stage": int(state.get("stage", -1)),
        "epoch": int(state.get("epoch", -1)),
        "global_step": int(state.get("global_step", -1)),
    }


def write_summary(rows: list[dict], outputs_dir: Path) -> tuple[Path, Path]:
    """Write CSV + Markdown summary tables."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    csv_path = outputs_dir / f"ablation_summary_{ts}.csv"
    md_path  = outputs_dir / f"ablation_summary_{ts}.md"

    fields = ["variant", "best_val_f1", "stage", "epoch", "global_step", "ckpt"]
    with open(csv_path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in fields) + "\n")

    with open(md_path, "w") as f:
        f.write(f"# Ablation Summary — {ts}\n\n")
        f.write("| Variant | Best val F1 | Stage | Epoch | Steps | Checkpoint |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| `{r['variant']}` "
                f"| {r['best_val_f1']:.4f} "
                f"| {r['stage']} "
                f"| {r['epoch']} "
                f"| {r['global_step']} "
                f"| `{Path(r['ckpt']).name}` |\n"
            )

    return csv_path, md_path


def main() -> int:
    setup_logging()
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    variants = args.variants or cfg["ablation"]["variants"]
    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    log_root = Path(cfg["logging"]["tensorboard_dir"]).parent / "ablation_logs"

    LOG.info("variants to consider: %s", variants)
    LOG.info("ckpt_dir: %s", ckpt_dir)

    summary_rows = []
    t0 = time.time()
    for variant in variants:
        existing = None if args.force else find_recent_best(ckpt_dir, variant)
        if existing is not None:
            LOG.info("[%s] SKIP — found recent best: %s", variant, existing.name)
        else:
            LOG.info("=" * 70)
            LOG.info("[%s] training…", variant)
            log_path = log_root / f"{variant}_{datetime.now().strftime('%Y%m%d-%H%M')}.log"
            rc = run_variant(variant, args.config, log_path)
            if rc != 0:
                LOG.error("[%s] train.py exited rc=%d — skipping this variant in summary", variant, rc)
                continue
            existing = find_recent_best(ckpt_dir, variant)
            if existing is None:
                LOG.error("[%s] no best.pt produced — train.py may have failed silently", variant)
                continue

        try:
            row = read_best_val_f1(existing)
            row["variant"] = variant
            summary_rows.append(row)
            LOG.info("[%s] best val F1=%.4f (stage %d epoch %d)",
                     variant, row["best_val_f1"], row["stage"], row["epoch"])
        except Exception as e:
            LOG.error("[%s] could not read %s: %s", variant, existing, e)

    LOG.info("=" * 70)
    LOG.info("all variants done in %.1f min", (time.time() - t0) / 60.0)

    if summary_rows and not args.skip_summary:
        outputs_dir = Path("outputs")
        csv_path, md_path = write_summary(summary_rows, outputs_dir)
        LOG.info("wrote summary: %s", md_path)
        LOG.info("wrote summary: %s", csv_path)

        # Echo the markdown table to console for instant gratification.
        print("\n" + "=" * 70)
        print(md_path.read_text())

    return 0


if __name__ == "__main__":
    sys.exit(main())
