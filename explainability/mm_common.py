"""Shared helpers for the explainability scripts.

Two things live here so SHAP and LIME stay in lock-step:

1. `pick_samples` — the stratified sample selector (was copy-pasted identically in
   shap_text.py and lime_text.py). Importing it from one place guarantees every
   method explains the *same* rows when driven from the same npz + seed.

2. `make_mm_text_predict_fn` — a per-sample text predict-fn for the **multimodal**
   `gated_fusion` model with the sample's image held fixed. This is what makes the
   word-attribution explain the headline model rather than a separate text_only one:
   SHAP/LIME perturb only the text while the real image + alpha stay constant, so the
   attribution answers "which words moved *this multimodal* verdict?".

   The image is encoded **once** per sample and its patch tokens are cached; each
   perturbation then runs only the text encoder + fusion + classifier — bypassing the
   expensive CLIP ViT on every forward. The forward is an exact replay of
   `HEMTCLIP.forward` for the gated_fusion variant (models/hemt_clip.py:118-121).
"""

from __future__ import annotations

import numpy as np
import torch


def pick_samples(preds: np.ndarray, probs: np.ndarray, labels: np.ndarray,
                 n: int, rng: np.random.Generator) -> list[int]:
    """Stratify across {correct, wrong} x {real, fake} with low/mid/high confidence
    within each cell. Tops up with random picks if n exceeds the strata.

    Deterministic given (preds, probs, labels, seed) — so two methods pointed at the
    same npz with the same --seed explain an identical sample set."""
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


def make_mm_text_predict_fn(model, tokenizer, device, max_len,
                            pixel_values: torch.Tensor, alpha: torch.Tensor):
    """Return f(texts) -> probs (N, 2) for the multimodal `gated_fusion` model with
    THIS sample's image + alpha held fixed. Used as the SHAP/LIME model callable so
    the word attribution explains the real multimodal verdict.

    `pixel_values` is (1, 3, 224, 224); `alpha` is (1,). The image patches are encoded
    once here and reused across every perturbation — only the text branch + fusion +
    classifier re-run per call (mirrors HEMTCLIP.forward, gated_fusion branch).

    Runs in full precision (no autocast): the cached patches and the per-call text
    features must share a dtype, and mixing fp32 patches with fp16 autocast activations
    makes nn.MultiheadAttention raise. fp32 is plenty fast for ~30 samples (matches the
    nb06 demo's _mm_text_predict)."""
    model.eval()
    with torch.no_grad():
        patches = model.image_encoder(pixel_values.to(device)).patches  # (1, P, 512), fp32
    alpha = alpha.to(device).view(1)

    @torch.no_grad()
    def f(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out_probs = []
        for i in range(0, len(texts), 16):              # chunk to bound memory
            chunk = texts[i:i + 16]
            enc = tokenizer(chunk, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            b = len(chunk)
            text_feats = model.text_encoder(enc["input_ids"], enc["attention_mask"])
            fused, _ = model.fusion(text_feats,
                                    patches.expand(b, -1, -1),
                                    alpha=alpha.expand(b))
            logits = model.classifier(fused)
            out_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out_probs, axis=0)

    return f
