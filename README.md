# HEMT-CLIP — Multimodal Fake News Detection

**XAI for Multimodal Fake News Detection — Final Year Project, UMT Sialkot, 2022–2026**

Team: Syed Taha Faiz-ul-Hassan Rizvi, Ayesha Bukhari, Ali Mosa Raza
Advisor: Ma'am Hina Tufail

## Overview

HEMT-CLIP takes a social-media post (text + image) and predicts Real vs Fake with a confidence score, explained via cross-attention heatmaps and SHAP. The architecture combines:

- **RoBERTa-base** (text, last 4 layers fine-tuned)
- **CLIP ViT-B/16** (vision tower, last 4 blocks fine-tuned)
- **8-head cross-attention fusion** (text [CLS] query attends over image patches)
- **CLIP-similarity gate α** — the attended vector is gated by the text–image cosine similarity: `fused = α·attended + (1−α)·text`
- **2-layer MLP classifier** → P(Real), P(Fake)

On the held-out test split (n = 2,573) the α-gated model is the best variant: **F1 0.839 / AUC 0.912** — beating plain concatenation and an α-as-feature ablation, while also providing intrinsic attention-heatmap explainability.

See [HEMT_CLIP_Implementation_Blueprint.md](HEMT_CLIP_Implementation_Blueprint.md) for the full specification.

## Setup

```bash
pip install -r requirements.txt
```

## Demo

The interactive demo is **`notebooks/06_demo.ipynb`** — open it in Colab, run the bootstrap
cell, then call `analyze(title, image)` to get a Real/Fake prediction with confidence, the CLIP
text–image alignment α, and a live cross-attention heatmap. It serves the headline α-gated
HEMT-CLIP (`gated_fusion`), discovers the latest checkpoint automatically, and ships with
curated test examples plus a cell for trying custom headline + image inputs.

## Repository Layout

```
configs/         hyperparameters (base.yaml, debug.yaml)
data/            Fakeddit download, HDF5 packer, PyTorch Dataset, splits
models/          text/image encoders, fusion, classifier, full HEMT-CLIP
explainability/  cross-attention viz + SHAP/LIME for text
training/        training loop, evaluation, ablation runner
notebooks/       data exploration, smoke test, full training, eval, XAI, demo
checkpoints/     trained weights (kept on Google Drive)
outputs/         plots and figures for the report
```

## Ablation Variants

| Variant | Description |
|---|---|
| A | Text-only baseline |
| B | Image-only baseline |
| C | Concat fusion `[text, image, α]` |
| D | **HEMT-CLIP** (α-gated cross-attention) |
