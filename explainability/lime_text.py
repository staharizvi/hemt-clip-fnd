"""Post-hoc explainability — LIME for text tokens.

Complementary to SHAP (shap_text.py). LIME fits a *local linear surrogate*
around each prediction using ~1000 word-deletion perturbations. Output is
per-word importance — coarser than SHAP's token-level (which sees BPE subword
pieces), but directly readable as "this word pushed the decision."

We run SHAP and LIME together so Chapter 6.4 can argue token attribution by
**agreement between two independent perturbation procedures**, a stronger claim
than relying on either alone.

Two modes via --model (mirrors shap_text.py):
  text_only    : attribute the text_only baseline.
  gated_fusion : attribute the **multimodal HEMT-CLIP** with the sample's image +
                 alpha held fixed (explainability.mm_common) — explains the headline
                 model's actual verdict.

Uses the shared `pick_samples` (explainability.mm_common) with the same `--seed`,
so per-sample plots line up with shap_text.py when driven from the same npz.

Outputs:
    outputs/xai/lime/lime_NN_{status}_pred-X_true-Y.png   — per-sample bars
    outputs/xai/lime/lime_word_records.csv                — long-form weights
    outputs/xai/lime/lime_manifest.json                   — picks + bucket map

Example:
    !python -m explainability.lime_text \\
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

LOG = logging.getLogger("lime-text")


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
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--num-perturbations", type=int, default=1000,
                   help="LIME's `num_samples` parameter — number of word-deletion "
                        "perturbations per local linear fit. Default 1000 ≈ 1–2 s/sample.")
    p.add_argument("--num-features", type=int, default=15,
                   help="Top N word weights kept in each per-sample explanation.")
    p.add_argument("--seed", type=int, default=42,
                   help="Same default as shap_text.py so samples are comparable.")
    p.add_argument("--out-dir", default=Path("outputs/xai/lime"), type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_predict_fn(model, tokenizer, device, max_len):
    """list[str] -> probs (N, 2). Same signature as shap_text.py's predict fn.
       text_only ignores image/alpha keys but the forward still requires them."""
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


def plot_word_importance(word_weights: list[tuple[str, float]], pred_label: int,
                         out_path: Path, title: str) -> None:
    """Per-sample horizontal bars from LIME's local linear coefficients."""
    if not word_weights:
        return
    words, weights = zip(*word_weights)
    colors = ["#cc6677" if w > 0 else "#4477aa" for w in weights]
    fig, ax = plt.subplots(figsize=(10, max(3, len(words) * 0.35)))
    y = np.arange(len(words))
    ax.barh(y, weights, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    pushed_toward = "fake" if pred_label == 1 else "real"
    ax.set_xlabel(f"LIME weight (positive → pushes toward {pushed_toward})")
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color="k", lw=0.5)
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

    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as e:
        LOG.error("lime not installed (`pip install lime>=0.2.0`): %s", e)
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
    LOG.info("picked %d samples (same seed as shap_text — comparable)", len(chosen))

    # text_only: one global predict-fn. gated_fusion: a per-sample fn (image held fixed).
    global_predict_fn = (build_predict_fn(model, tokenizer, device, max_len)
                         if args.model == "text_only" else None)
    explainer = LimeTextExplainer(
        class_names=["real", "fake"],
        bow=False,                    # respect word order — Fakeddit titles are short
        random_state=args.seed,
    )

    word_records: list[dict] = []
    manifest: list[dict] = []
    LOG.info("running LIME (%s) on %d samples (~%d s estimated, %d perturbations each)...",
             args.model, len(chosen), len(chosen) * 2, args.num_perturbations)

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

        # pred/conf from the model actually being explained (not the pick npz)
        base = predict_fn([text])[0]
        pred = int(np.argmax(base))
        conf = float(base.max())

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp = explainer.explain_instance(
                    text, predict_fn,
                    num_features=args.num_features,
                    num_samples=args.num_perturbations,
                    labels=[pred],
                )
            word_weights = exp.as_list(label=pred)
        except Exception as e:
            LOG.warning("  [%02d] ds_idx=%d LIME failed: %s", i, ds_idx, e)
            continue

        status = "correct" if pred == label else "wrong"
        pred_name = "fake" if pred == 1 else "real"
        true_name = "fake" if label == 1 else "real"
        stem = f"lime_{i:02d}_{status}_pred-{pred_name}_true-{true_name}"
        snippet = text if len(text) <= 80 else text[:77] + "…"
        title = (f"Sample {ds_idx:>4d}  pred={pred_name} (conf={conf:.3f})  "
                 f"true={true_name}  [{status}]\n\"{snippet}\"")

        plot_word_importance(word_weights, pred, args.out_dir / f"{stem}.png", title)
        LOG.info("  [%02d] ds_idx=%d %s pred=%s true=%s saved %s.png",
                 i, ds_idx, status, pred_name, true_name, stem)

        manifest.append({
            "i": i, "ds_idx": int(ds_idx), "hdf5_row": hdf5_row,
            "model": args.model,
            "pred": pred, "label": label, "confidence": conf,
            "status": status, "file": stem + ".png", "text": text,
            "word_weights": word_weights,
        })

        for word, weight in word_weights:
            word_records.append({
                "sample": int(ds_idx),
                "word": word.lower(),
                "weight": float(weight),
                "pred": pred_name,
                "true": true_name,
            })

    if word_records:
        df = pd.DataFrame(word_records)
        df.to_csv(args.out_dir / "lime_word_records.csv", index=False)

    with open(args.out_dir / "lime_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    f_h5.close()
    LOG.info("done. wrote: %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
