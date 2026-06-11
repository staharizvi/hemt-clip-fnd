"""Post-hoc explainability — SHAP for text tokens.

Runs `shap.Explainer` (Partition explainer over a `shap.maskers.Text`) attributing
each test-set title's prediction to its individual tokens. Two modes via --model:

  text_only    : attribute the text_only variant — the cleanest "which words, in
                 isolation?" baseline (verdict is a pure function of text).
  gated_fusion : attribute the **headline multimodal HEMT-CLIP** with the sample's
                 real image + alpha held fixed (see explainability.mm_common). The
                 perturbations vary only the text, so the attribution explains the
                 actual multimodal verdict — not a separate text-only proxy.

Running both on the same sample set (drive --pick-from-npz from the same npz) lets
Chapter 6.4 compare them: does adding the image change which words matter?
(This supersedes the old Blueprint §10.2 "skip multimodal SHAP" stance.)

Sample selection (default 30): stratified across {correct, wrong} × {real,
fake} with three confidence quantiles per cell — gives a mix of confident
successes, confident errors, and borderline calls.

Outputs:
    outputs/xai/shap/shap_NN_<status>_pred<X>_true<Y>.png   — per-sample bar plots
    outputs/xai/shap/shap_top_tokens.png                    — aggregate top-tokens figure
    outputs/xai/shap/shap_token_records.csv                 — per-token SHAP values
    outputs/xai/shap/shap_manifest.json                     — picks + bucket map

Example (Colab, after `training/evaluate.py` ran):
    !python -m explainability.shap_text \\
        --checkpoint /content/drive/MyDrive/hemt-clip-fnd/checkpoints/hemt_text_only_20260530-0103_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

from data.dataset import HEMTClipDataset
from explainability.mm_common import make_mm_text_predict_fn, pick_samples
from models.hemt_clip import build_from_config

LOG = logging.getLogger("shap-text")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=Path("configs/base.yaml"), type=Path)
    p.add_argument("--model", default="text_only", choices=("text_only", "gated_fusion"),
                   help="Which model to attribute. gated_fusion = multimodal HEMT-CLIP "
                        "with each sample's image + alpha held fixed.")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to the best.pt matching --model.")
    p.add_argument("--preds-npz", default=Path("outputs/eval/preds_text_only.npz"), type=Path,
                   help="Predictions npz from training.evaluate (drives sample picks unless "
                        "--pick-from-npz is given).")
    p.add_argument("--pick-from-npz", default=None, type=Path,
                   help="Optional: stratify sample picks from this npz instead of --preds-npz, "
                        "so different --model runs can explain the SAME samples for comparison.")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--n-samples", type=int, default=30,
                   help="Total samples to explain (Blueprint §10.5 asks for 30+).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=Path("outputs/xai/shap"), type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_predict_fn(model, tokenizer, device, max_len):
    """Returns f(texts: list[str]) -> probs (N, 2). Used as the model callable for SHAP.

    text_only's forward signature still expects pixel_values/alpha/label keys —
    we pass zero-filled dummies; the model ignores them for this variant."""
    @torch.no_grad()
    def f(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]
        enc = tokenizer(list(texts), padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        B = enc["input_ids"].shape[0]
        batch = {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pixel_values":   torch.zeros(B, 3, 224, 224, device=device),
            "alpha":          torch.zeros(B, device=device),
            "label":          torch.zeros(B, dtype=torch.long, device=device),
        }
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model(batch)
        return torch.softmax(out["logits"].float(), dim=-1).cpu().numpy()
    return f


def plot_token_importance(values: np.ndarray, tokens: list[str], pred_label: int,
                          out_path: Path, title: str) -> None:
    """Per-sample horizontal bar of token contributions to the predicted class."""
    if values.ndim == 2:
        vals = values[:, pred_label]
    else:
        vals = values
    keep = [i for i, t in enumerate(tokens) if t.strip()]
    tokens = [tokens[i] for i in keep]
    vals = vals[keep] if len(keep) else vals
    if len(tokens) == 0:
        return

    if len(tokens) > 20:
        top = np.argsort(np.abs(vals))[::-1][:20]
        top.sort()
        tokens = [tokens[i] for i in top]
        vals = vals[top]

    colors = ["#cc6677" if v > 0 else "#4477aa" for v in vals]
    fig, ax = plt.subplots(figsize=(10, max(3, len(tokens) * 0.32)))
    y = np.arange(len(tokens))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(tokens, fontsize=9)
    ax.invert_yaxis()
    pushed_toward = "fake" if pred_label == 1 else "real"
    ax.set_xlabel(f"SHAP value (positive → pushes toward {pushed_toward})")
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# True stop words only — words that carry no domain signal regardless of context.
# Conservatively trimmed: avoid filtering "way", "made", "looks", "two" etc. since
# on Fakeddit's short titles they're often content (e.g. "two-headed snake" is
# fake-leaning content). Per-sample plots are unfiltered; this set only applies
# to the cross-sample aggregate, where common fillers accumulate spurious means.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those",
    "i", "me", "my", "we", "us", "our",
    "you", "your", "he", "his", "she", "her",
    "it", "its", "they", "them", "their",
    "as", "than", "then", "such", "also", "too",
}


def plot_aggregate(token_df: pd.DataFrame, out_path: Path,
                   min_count: int = 3, drop_stopwords: bool = True) -> bool:
    """Top tokens by mean SHAP value toward fake (right) and real (left).

    `min_count` and `drop_stopwords` together suppress noise from common English
    fillers. For ~30 short titles, min_count=3 is the right floor — most content
    tokens appear 2–4 times across the sample. Falls back to min_count=2 (no
    stop-word filter) if the strict filter empties the pool, so the figure
    always renders something rather than silently dropping the aggregate.

    Returns True if the file was written, False otherwise."""
    base = token_df.groupby("token").agg(
        mean_shap_fake=("shap_fake", "mean"),
        count=("token", "count"),
    ).reset_index()
    agg = base[base["count"] >= min_count]
    if drop_stopwords:
        agg = agg[~agg["token"].isin(STOP_WORDS)]
    if len(agg) == 0:
        LOG.warning("aggregate: no tokens passed min_count=%d + drop_stopwords=%s; "
                    "falling back to min_count=2 (no stop-word filter).",
                    min_count, drop_stopwords)
        agg = base[base["count"] >= 2]
    if len(agg) == 0:
        LOG.warning("aggregate: still no tokens after fallback — skipping plot.")
        return False

    top_fake = agg.nlargest(15, "mean_shap_fake")
    top_real = agg.nsmallest(15, "mean_shap_fake")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(top_real["token"], -top_real["mean_shap_fake"], color="#4477aa")
    axes[0].set_title("Top tokens → REAL prediction")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("|mean SHAP value| (negative on fake-class axis)")
    axes[0].axvline(0, color="k", lw=0.5)
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(top_fake["token"], top_fake["mean_shap_fake"], color="#cc6677")
    axes[1].set_title("Top tokens → FAKE prediction")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean SHAP value (fake class)")
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(f"Aggregate token importance (tokens with ≥ {min_count} occurrences)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def main() -> int:
    setup_logging()
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    LOG.info("device=%s", device)

    # SHAP imports here so the script can at least parse if shap is missing
    try:
        import shap
    except ImportError as e:
        LOG.error("shap not installed (`pip install shap>=0.46.0`): %s", e)
        return 1
    from transformers import AutoTokenizer

    pick_npz_path = args.pick_from_npz or args.preds_npz
    LOG.info("loading picks/labels from %s", pick_npz_path)
    npz = np.load(pick_npz_path)
    preds, probs, labels = npz["preds"], npz["probs"], npz["labels"]
    LOG.info("pick-npz: n=%d  acc=%.4f", len(preds), float((preds == labels).mean()))

    LOG.info("loading %s from %s", args.model, args.checkpoint)
    model = build_from_config(cfg, variant=args.model).to(device)
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

    # text_only: one global predict-fn. gated_fusion: a per-sample fn (this sample's
    # image + alpha held fixed), built inside the loop. The masker is shared.
    global_predict_fn = (build_predict_fn(model, tokenizer, device, max_len)
                         if args.model == "text_only" else None)
    masker = shap.maskers.Text(tokenizer)

    token_records: list[dict] = []
    manifest: list[dict] = []
    LOG.info("running SHAP (%s) on %d samples (~5-10 min on T4)...", args.model, len(chosen))

    for i, ds_idx in enumerate(chosen):
        hdf5_row = int(ds.indices[ds_idx])
        raw_text = f_h5["texts"][hdf5_row]
        text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)
        label = int(labels[ds_idx])

        if args.model == "gated_fusion":
            s = ds[ds_idx]
            predict_fn = make_mm_text_predict_fn(
                model, tokenizer, device, max_len,
                s["pixel_values"].unsqueeze(0), s["alpha"].view(1))
        else:
            predict_fn = global_predict_fn

        # pred/conf from the model actually being explained (not the pick npz),
        # so titles/status are correct even when picks come from another model.
        base = predict_fn([text])[0]
        pred = int(np.argmax(base))
        conf = float(base.max())

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = shap.Explainer(predict_fn, masker, output_names=["real", "fake"])
                shap_values = explainer([text])
        except Exception as e:
            LOG.warning("  [%02d] ds_idx=%d SHAP failed: %s", i, ds_idx, e)
            continue

        tokens = list(shap_values.data[0])
        values = np.asarray(shap_values.values[0])  # (n_tokens, 2)

        status = "correct" if pred == label else "wrong"
        pred_name = "fake" if pred == 1 else "real"
        true_name = "fake" if label == 1 else "real"
        stem = f"shap_{i:02d}_{status}_pred-{pred_name}_true-{true_name}"
        snippet = text if len(text) <= 80 else text[:77] + "…"
        title = (f"Sample {ds_idx:>4d}  pred={pred_name} (conf={conf:.3f})  "
                 f"true={true_name}  [{status}]\n\"{snippet}\"")

        plot_token_importance(values, tokens, pred, args.out_dir / f"{stem}.png", title)
        LOG.info("  [%02d] ds_idx=%d %s pred=%s true=%s saved %s.png",
                 i, ds_idx, status, pred_name, true_name, stem)

        manifest.append({
            "i": i, "ds_idx": int(ds_idx), "hdf5_row": hdf5_row,
            "model": args.model,
            "pred": pred, "label": label, "confidence": conf,
            "status": status, "file": stem + ".png", "text": text,
        })

        for tok, v_real, v_fake in zip(tokens, values[:, 0], values[:, 1]):
            tok_clean = tok.strip().lower()
            if len(tok_clean) > 1:  # drop pads and 1-char tokens
                token_records.append({
                    "sample": int(ds_idx),
                    "token": tok_clean,
                    "shap_real": float(v_real),
                    "shap_fake": float(v_fake),
                    "pred": pred_name,
                    "true": true_name,
                })

    if token_records:
        df = pd.DataFrame(token_records)
        df.to_csv(args.out_dir / "shap_token_records.csv", index=False)
        wrote = plot_aggregate(df, args.out_dir / "shap_top_tokens.png")
        if wrote:
            LOG.info("saved aggregate: shap_top_tokens.png")
        else:
            LOG.warning("aggregate not written — see prior warnings; downstream "
                        "notebook cell should guard for missing shap_top_tokens.png.")
    else:
        LOG.warning("no token records — SHAP failed for every sample.")

    with open(args.out_dir / "shap_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    f_h5.close()
    LOG.info("done. wrote: %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
