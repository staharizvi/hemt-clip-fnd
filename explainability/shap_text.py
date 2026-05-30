"""Post-hoc explainability — SHAP for text tokens (Blueprint §10.2).

Runs `shap.Explainer` (Partition explainer over a `shap.maskers.Text`) on the
text_only variant of the trained model, attributing each test-set title's
prediction to its individual tokens. Why text_only and not hemt_clip's text
branch: SHAP attributes a model's predictions, so the cleanest answer to
"which words pushed the verdict?" comes from a text-only model whose
verdict is a function of text alone — clean question, clean answer.
(Multimodal SHAP is intentionally skipped per Blueprint §10.2: marginal
value for FYP scope, considerable complexity.)

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
from models.hemt_clip import build_from_config

LOG = logging.getLogger("shap-text")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=Path("configs/base.yaml"), type=Path)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to text_only best.pt.")
    p.add_argument("--preds-npz", default=Path("outputs/eval/preds_text_only.npz"), type=Path,
                   help="Predictions npz from training.evaluate (preds_text_only.npz).")
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


def pick_samples(preds: np.ndarray, probs: np.ndarray, labels: np.ndarray,
                 n: int, rng: np.random.Generator) -> list[int]:
    """Stratify across {correct, wrong} × {real, fake} with low/mid/high confidence
       within each cell. Tops up with random picks if n exceeds the strata."""
    correct = preds == labels
    confidence = probs.max(axis=1)
    picks: list[int] = []

    for is_correct in (True, False):
        for label_val in (0, 1):
            mask = (correct == is_correct) & (labels == label_val)
            pool = np.nonzero(mask)[0]
            if len(pool) == 0:
                continue
            sorted_by_conf = pool[np.argsort(confidence[pool])]
            n_pick_per_cell = max(1, n // 8)  # ~8 quantile slots
            for q in np.linspace(0.1, 0.9, n_pick_per_cell):
                idx = int(q * (len(sorted_by_conf) - 1))
                cand = int(sorted_by_conf[idx])
                if cand not in picks:
                    picks.append(cand)

    if len(picks) < n:
        remaining = np.setdiff1d(np.arange(len(preds)), picks)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(n - len(picks), len(remaining)), replace=False)
            picks.extend(int(x) for x in extra)
    return picks[:n]


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


def plot_aggregate(token_df: pd.DataFrame, out_path: Path, min_count: int = 2) -> None:
    """Top tokens by mean SHAP value toward fake (right) and real (left)."""
    agg = token_df.groupby("token").agg(
        mean_shap_fake=("shap_fake", "mean"),
        count=("token", "count"),
    ).reset_index()
    agg = agg[agg["count"] >= min_count]
    if len(agg) == 0:
        LOG.warning("aggregate: no tokens passed min_count=%d filter", min_count)
        return

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

    LOG.info("loading preds from %s", args.preds_npz)
    npz = np.load(args.preds_npz)
    preds, probs, labels = npz["preds"], npz["probs"], npz["labels"]
    LOG.info("preds: n=%d  acc=%.4f", len(preds), float((preds == labels).mean()))

    LOG.info("loading text_only from %s", args.checkpoint)
    model = build_from_config(cfg, variant="text_only").to(device)
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

    predict_fn = build_predict_fn(model, tokenizer, device, max_len)
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker, output_names=["real", "fake"])

    token_records: list[dict] = []
    manifest: list[dict] = []
    LOG.info("running SHAP on %d samples (~5-10 min on T4)...", len(chosen))

    for i, ds_idx in enumerate(chosen):
        hdf5_row = int(ds.indices[ds_idx])
        raw_text = f_h5["texts"][hdf5_row]
        text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)
        pred = int(preds[ds_idx])
        label = int(labels[ds_idx])
        conf = float(probs[ds_idx].max())

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
        plot_aggregate(df, args.out_dir / "shap_top_tokens.png")
        LOG.info("saved aggregate: shap_top_tokens.png")
    else:
        LOG.warning("no token records — SHAP failed for every sample.")

    with open(args.out_dir / "shap_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    f_h5.close()
    LOG.info("done. wrote: %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
