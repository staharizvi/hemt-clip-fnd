"""Cross-modal explainability — per-sample modality contribution (HEMT-CLIP).

The text SHAP/LIME and the image attention each explain *one* modality. This method
asks the genuinely multimodal question: for a given verdict, **how much did each
modality move the decision?** It does so by occluding one modality at a time on the
headline `gated_fusion` model and measuring the shift in the predicted-class logit:

    full     : real text + real image + real alpha
    no_image : pixel_values = 0, alpha = 0   (the gate collapses to the text path)
    no_text  : input_ids = tokenizer("") = <s></s>, real image + alpha

For the predicted class c:
    dlogit_image = logit_full[c] - logit_no_image[c]   (support the image lends)
    dlogit_text  = logit_full[c] - logit_no_text[c]    (support the text lends)
plus probability deltas and a verdict-flip check (does masking a modality flip argmax?).

This evidences that the model's decisions are actually multimodal — and, per sample,
which modality carried the verdict. Caveat (shared with all occlusion/perturbation
methods): masked inputs are off-distribution, so deltas are directional, not exact
Shapley credit. We report them as relative magnitudes, not calibrated attributions.

Outputs (outputs/xai/modality/):
    modality_contrib.png      — per-sample grouped bars: dlogit_image vs dlogit_text
    modality_contrib.csv      — per-sample logits / deltas / flip flags
    modality_manifest.json    — per-sample records (text, pred, deltas, flips)

Example (Colab, after training.evaluate ran):
    !python -m explainability.modality_contrib \\
        --checkpoint /content/drive/MyDrive/hemt-clip-fnd/checkpoints/hemt_gated_fusion_..._best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

from data.dataset import HEMTClipDataset
from explainability.mm_common import pick_samples
from models.hemt_clip import build_from_config

LOG = logging.getLogger("modality-contrib")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=Path("configs/base.yaml"), type=Path)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to the gated_fusion (multimodal HEMT-CLIP) best.pt.")
    p.add_argument("--preds-npz", default=Path("outputs/eval/preds_gated_fusion.npz"), type=Path,
                   help="Predictions npz from training.evaluate (drives sample picks).")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--n-samples", type=int, default=30,
                   help="Same default as shap/lime so the sample set is comparable.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=Path("outputs/xai/modality"), type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@torch.no_grad()
def _logits(model, batch, device):
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        out = model(batch)
    return out["logits"].float().cpu().numpy()[0]   # (2,)


def plot_contributions(records: list[dict], out_path: Path) -> None:
    """Per-sample grouped horizontal bars: how much each modality supports the verdict."""
    if not records:
        return
    recs = sorted(records, key=lambda r: r["dlogit_image"] + r["dlogit_text"])
    y = np.arange(len(recs))
    d_img = [r["dlogit_image"] for r in recs]
    d_txt = [r["dlogit_text"] for r in recs]
    labels = [f"[{r['pred_name']}] {r['text'][:40]}" for r in recs]

    fig, ax = plt.subplots(figsize=(11, max(4, len(recs) * 0.42)))
    h = 0.4
    ax.barh(y + h / 2, d_img, height=h, color="#e6863c", label="image (Δ logit when image removed)")
    ax.barh(y - h / 2, d_txt, height=h, color="#4477aa", label="text (Δ logit when text removed)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("support for the predicted class  (larger → that modality mattered more)")
    ax.set_title("Per-sample modality contribution — HEMT-CLIP (gated_fusion)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
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

    from transformers import AutoTokenizer

    LOG.info("loading preds from %s", args.preds_npz)
    npz = np.load(args.preds_npz)
    preds, probs, labels = npz["preds"], npz["probs"], npz["labels"]
    LOG.info("preds: n=%d  acc=%.4f", len(preds), float((preds == labels).mean()))

    LOG.info("loading gated_fusion from %s", args.checkpoint)
    model = build_from_config(cfg, variant="gated_fusion").to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text"]["name"])
    max_len = cfg["data"]["max_text_len"]

    ds = HEMTClipDataset(cfg["data"]["hdf5_path"], args.split,
                         tokenizer_name=cfg["model"]["text"]["name"],
                         max_text_len=max_len)
    f_h5 = h5py.File(cfg["data"]["hdf5_path"], "r")

    rng = np.random.default_rng(args.seed)
    chosen = pick_samples(preds, probs, labels, args.n_samples, rng)
    LOG.info("picked %d samples", len(chosen))

    # Empty-text input (shared across samples) for the no_text counterfactual.
    empty_enc = tokenizer("", padding="max_length", truncation=True,
                          max_length=max_len, return_tensors="pt").to(device)

    records: list[dict] = []
    n_flip_image = n_flip_text = 0
    LOG.info("computing modality contributions on %d samples...", len(chosen))

    for i, ds_idx in enumerate(chosen):
        hdf5_row = int(ds.indices[ds_idx])
        s = ds[ds_idx]
        raw_text = f_h5["texts"][hdf5_row]
        text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)
        label = int(labels[ds_idx])

        input_ids = s["input_ids"].unsqueeze(0).to(device)
        attn_mask = s["attention_mask"].unsqueeze(0).to(device)
        pixel_values = s["pixel_values"].unsqueeze(0).to(device)
        alpha = s["alpha"].view(1).to(device)
        zero_label = torch.zeros(1, dtype=torch.long, device=device)

        full = {"input_ids": input_ids, "attention_mask": attn_mask,
                "pixel_values": pixel_values, "alpha": alpha, "label": zero_label}
        no_image = {"input_ids": input_ids, "attention_mask": attn_mask,
                    "pixel_values": torch.zeros_like(pixel_values),
                    "alpha": torch.zeros_like(alpha), "label": zero_label}
        no_text = {"input_ids": empty_enc["input_ids"], "attention_mask": empty_enc["attention_mask"],
                   "pixel_values": pixel_values, "alpha": alpha, "label": zero_label}

        lg_full = _logits(model, full, device)
        lg_noimg = _logits(model, no_image, device)
        lg_notxt = _logits(model, no_text, device)

        def softmax(z):
            e = np.exp(z - z.max())
            return e / e.sum()
        p_full, p_noimg, p_notxt = softmax(lg_full), softmax(lg_noimg), softmax(lg_notxt)

        c = int(lg_full.argmax())
        pred_name = "fake" if c == 1 else "real"
        true_name = "fake" if label == 1 else "real"
        flip_image = int(lg_noimg.argmax()) != c
        flip_text = int(lg_notxt.argmax()) != c
        n_flip_image += int(flip_image)
        n_flip_text += int(flip_text)

        rec = {
            "i": i, "ds_idx": int(ds_idx), "hdf5_row": hdf5_row,
            "text": text, "pred": c, "pred_name": pred_name,
            "label": label, "true_name": true_name,
            "confidence": float(p_full[c]),
            "logit_full": float(lg_full[c]),
            "logit_no_image": float(lg_noimg[c]),
            "logit_no_text": float(lg_notxt[c]),
            "dlogit_image": float(lg_full[c] - lg_noimg[c]),
            "dlogit_text": float(lg_full[c] - lg_notxt[c]),
            "dprob_image": float(p_full[c] - p_noimg[c]),
            "dprob_text": float(p_full[c] - p_notxt[c]),
            "flip_image": flip_image,
            "flip_text": flip_text,
        }
        records.append(rec)
        LOG.info("  [%02d] ds_idx=%d %s d_img=%+.3f d_txt=%+.3f%s%s",
                 i, ds_idx, pred_name, rec["dlogit_image"], rec["dlogit_text"],
                 "  [img-flip]" if flip_image else "", "  [txt-flip]" if flip_text else "")

    f_h5.close()

    if not records:
        LOG.warning("no records produced.")
        return 1

    df = pd.DataFrame(records)
    df.to_csv(args.out_dir / "modality_contrib.csv", index=False)
    plot_contributions(records, args.out_dir / "modality_contrib.png")
    with open(args.out_dir / "modality_manifest.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    n = len(records)
    img_dom = int((df["dlogit_image"].abs() > df["dlogit_text"].abs()).sum())
    LOG.info("—— summary ——")
    LOG.info("samples: %d", n)
    LOG.info("mean |Δlogit| — image: %.3f | text: %.3f",
             df["dlogit_image"].abs().mean(), df["dlogit_text"].abs().mean())
    LOG.info("verdict flips when image removed: %d/%d (%.0f%%)",
             n_flip_image, n, 100 * n_flip_image / n)
    LOG.info("verdict flips when text removed : %d/%d (%.0f%%)",
             n_flip_text, n, 100 * n_flip_text / n)
    LOG.info("image-dominant samples (|Δ_img| > |Δ_txt|): %d/%d", img_dom, n)
    LOG.info("done. wrote: %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
