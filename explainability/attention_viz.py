"""Cross-attention heatmap visualization for the report (Blueprint §10.1).

Loads the trained hemt_clip model + its test-set predictions (from
training/evaluate.py), picks a stratified sample of examples — by correctness
× confidence — runs each through the model to capture the cross-attention
weights from the fusion module, reshapes them onto the 14×14 patch grid
(B/16 backbone), and overlays a heatmap on the original image.

Sample selection (default 3 per bucket = 12 total):
    correct_hi  : model right, most confident   — heatmaps "working as intended"
    correct_lo  : model right, least confident  — borderline successes
    wrong_hi    : model wrong, most confident   — most informative for error analysis
    wrong_lo    : model wrong, least confident  — boundary cases

Outputs:
    outputs/xai/attention/{bucket}_NN_predX_trueY.png   — per-example side-by-side
    outputs/xai/attention/attention_grid.png            — composite N-cell figure
    outputs/xai/attention/attention_manifest.json       — picks + bucket → idx map

Example (in Colab, after running training/evaluate.py):
    !python -m explainability.attention_viz \\
        --checkpoint /content/drive/MyDrive/hemt-clip-fnd/checkpoints/hemt_hemt_clip_20260530-0223_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt

from data.dataset import HEMTClipDataset
from models.hemt_clip import build_from_config

LOG = logging.getLogger("attention-viz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=Path("configs/base.yaml"), type=Path)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to hemt_clip best.pt.")
    p.add_argument("--preds-npz", default=Path("outputs/eval/preds_hemt_clip.npz"), type=Path,
                   help="Predictions npz from training.evaluate (preds_hemt_clip.npz).")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--n-per-bucket", type=int, default=3,
                   help="Examples per bucket; 4 buckets × N = total. Default 3 → 12 examples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=Path("outputs/xai/attention"), type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_model(ckpt_path: Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_from_config(cfg, variant="hemt_clip").to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def pick_examples(
    preds: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    n_per_bucket: int,
) -> dict[str, list[int]]:
    """Pick N samples per bucket: {correct,wrong} × {hi,lo} confidence.
       Confidence = max class probability."""
    correct = preds == labels
    confidence = probs.max(axis=1)

    buckets: dict[str, list[int]] = {}
    for is_correct, name in [(True, "correct"), (False, "wrong")]:
        mask = correct if is_correct else ~correct
        pool = np.nonzero(mask)[0]
        if len(pool) == 0:
            buckets[f"{name}_hi"] = []
            buckets[f"{name}_lo"] = []
            continue
        sorted_by_conf = pool[np.argsort(confidence[pool])]
        take = min(n_per_bucket, len(pool) // 2 if len(pool) >= 2 * n_per_bucket else len(pool))
        buckets[f"{name}_lo"] = sorted_by_conf[:take].tolist()
        buckets[f"{name}_hi"] = sorted_by_conf[-take:][::-1].tolist()
    return buckets


def extract_attention(
    model: torch.nn.Module,
    sample: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model forward on a single sample. Returns (attn_grid [P,P], probs [2])."""
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()
             if isinstance(v, torch.Tensor)}
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda",
    ):
        out = model(batch)
    logits = out["logits"].float().cpu().numpy()[0]
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    # attn shape: (B=1, H=8, Q=1, P)  — keep per-head, then mean across heads
    attn = out["attention_weights"].float().cpu().numpy()[0]  # (H, 1, P)
    attn = attn.mean(axis=0).squeeze(0)                        # (P,)
    p = int(round(np.sqrt(len(attn))))
    return attn.reshape(p, p), probs


def upsample_heatmap(attn_grid: np.ndarray, size: int = 224) -> np.ndarray:
    """Bilinear upsample (P,P) → (size,size). Normalise to [0,1]."""
    t = torch.from_numpy(attn_grid).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(size, size), mode="bilinear",
                       align_corners=False).squeeze().numpy()
    lo, hi = float(up.min()), float(up.max())
    return (up - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(up)


def plot_per_example(
    image_u8: np.ndarray,
    attn_grid: np.ndarray,
    text: str,
    pred: int, label: int,
    confidence: float, alpha_val: float, bucket: str,
    out_path: Path,
) -> None:
    img_hwc = image_u8.transpose(1, 2, 0)
    heatmap = upsample_heatmap(attn_grid, size=img_hwc.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    axes[0].imshow(img_hwc)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("Original image", fontsize=11)

    axes[1].imshow(img_hwc)
    axes[1].imshow(heatmap, cmap="hot", alpha=0.5)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title(f"Cross-attention overlay ({attn_grid.shape[0]}×{attn_grid.shape[1]} → 224×224)",
                      fontsize=11)

    pred_name = "FAKE" if pred == 1 else "REAL"
    true_name = "FAKE" if label == 1 else "REAL"
    pred_color = "tab:red" if pred == 1 else "tab:blue"
    status_str = "✓ correct" if pred == label else "✗ wrong"
    snippet = text if len(text) <= 100 else text[:97] + "…"

    fig.suptitle(
        f"[{bucket}]  pred={pred_name} (conf={confidence:.3f})  |  "
        f"true={true_name}  |  {status_str}  |  α={alpha_val:.3f}\n"
        f"\"{snippet}\"",
        fontsize=10, color=pred_color, y=0.99,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.83)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_grid_figure(rows: list[dict], out_path: Path) -> None:
    n = len(rows)
    ncols = min(4, n) if n > 0 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.9 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for i, row in enumerate(rows):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        img_hwc = row["image_u8"].transpose(1, 2, 0)
        heatmap = upsample_heatmap(row["attn"], size=img_hwc.shape[0])
        ax.imshow(img_hwc)
        ax.imshow(heatmap, cmap="hot", alpha=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        pred_name = "F" if row["pred"] == 1 else "R"
        true_name = "F" if row["label"] == 1 else "R"
        mark = "✓" if row["pred"] == row["label"] else "✗"
        col = "tab:red" if row["pred"] == 1 else "tab:blue"
        ax.set_title(f"[{row['bucket']}] {mark} pred={pred_name} true={true_name}\n"
                     f"conf={row['conf']:.2f}  α={row['alpha']:.2f}",
                     fontsize=9, color=col)

    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    fig.suptitle("Cross-attention heatmaps — test predictions sampled by bucket",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    setup_logging()
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    LOG.info("device=%s", device)

    LOG.info("loading preds from %s", args.preds_npz)
    npz = np.load(args.preds_npz)
    preds, probs, labels = npz["preds"], npz["probs"], npz["labels"]
    LOG.info("preds: n=%d  acc=%.4f", len(preds), float((preds == labels).mean()))

    buckets = pick_examples(preds, probs, labels, args.n_per_bucket)
    LOG.info("picks:")
    for name, idxs in buckets.items():
        LOG.info("  %-12s : %s", name, idxs)

    LOG.info("loading hemt_clip from %s", args.checkpoint)
    model = load_model(args.checkpoint, cfg, device)

    ds = HEMTClipDataset(
        cfg["data"]["hdf5_path"], args.split,
        tokenizer_name=cfg["model"]["text"]["name"],
        max_text_len=cfg["data"]["max_text_len"],
    )
    f_h5 = h5py.File(cfg["data"]["hdf5_path"], "r")

    rows_for_grid: list[dict] = []
    manifest: list[dict] = []

    for bucket_name, idxs in buckets.items():
        for i, ds_idx in enumerate(idxs):
            hdf5_row = int(ds.indices[ds_idx])
            image_u8 = f_h5["images"][hdf5_row]
            raw_text = f_h5["texts"][hdf5_row]
            text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)

            sample = ds[ds_idx]
            # IMPORTANT: use the *cached* prediction from preds_npz for labeling, not the
            # fresh forward's argmax. Borderline samples can flip under fp16 + batch=1 vs
            # evaluate.py's batched fp16 run — keeping the cached pred makes the bucket
            # label ("correct_lo" etc.) consistent with what's actually displayed.
            attn_grid, _ = extract_attention(model, sample, device)
            pred = int(preds[ds_idx])
            label = int(labels[ds_idx])
            confidence = float(probs[ds_idx].max())
            alpha_val = float(sample["alpha"].item())

            stem = f"{bucket_name}_{i:02d}_pred-{['real','fake'][pred]}_true-{['real','fake'][label]}"
            out_path = args.out_dir / f"{stem}.png"
            plot_per_example(image_u8, attn_grid, text, pred, label,
                              confidence, alpha_val, bucket_name, out_path)
            LOG.info("  saved %s", out_path.name)

            rows_for_grid.append({
                "image_u8": image_u8, "attn": attn_grid, "text": text,
                "pred": pred, "label": label, "conf": confidence,
                "alpha": alpha_val, "bucket": bucket_name,
            })
            manifest.append({
                "bucket": bucket_name,
                "ds_idx": int(ds_idx),
                "hdf5_row": hdf5_row,
                "pred": pred, "label": label,
                "confidence": confidence, "alpha": alpha_val,
                "file": stem + ".png",
                "text": text,
            })

    grid_path = args.out_dir / "attention_grid.png"
    make_grid_figure(rows_for_grid, grid_path)
    LOG.info("saved composite grid: %s", grid_path.name)

    with open(args.out_dir / "attention_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    f_h5.close()
    LOG.info("done. wrote: %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
