# HEMT-CLIP — Multimodal Fake News Detection

**XAI for Multimodal Fake News Detection — Final Year Project, UMT Sialkot, 2022–2026**

Team: Syed Taha Faiz-ul-Hassan Rizvi, Ayesha Bukhari, Ali Mosa Raza
Advisor: Ma'am Hina Tufail

## Overview

HEMT-CLIP takes a paired social-media title and image and estimates the corresponding binary Fakeddit label. The interface reports softmax class scores and model-behaviour diagnostics derived from cross-attention, SHAP, and LIME. These outputs characterize the fitted classifier and do not constitute independent factual verification. The architecture combines:

- **RoBERTa-base** (text, last 4 layers fine-tuned)
- **CLIP ViT-B/16** (vision tower, last 4 blocks fine-tuned)
- **8-head cross-attention fusion** (text [CLS] query attends over image patches)
- **CLIP-similarity gate α** — the attended vector is gated by the text–image cosine similarity: `fused = α·attended + (1−α)·text`
- **2-layer MLP classifier** → P(Real), P(Fake)

On the held-out test split (n = 2,573), the α-gated variant produces the highest observed F1 and ROC AUC among the five controlled variants: **F1 0.839 / AUC 0.912**. Cross-attention weights provide an intrinsic image-side diagnostic; they are not interpreted as causal explanations.

See [HEMT_CLIP_Implementation_Blueprint.md](HEMT_CLIP_Implementation_Blueprint.md) for the full specification.

## Setup

```bash
pip install -r requirements.txt
```

## Demo

The Streamlit frontend showcases the held-out findings, ablation study, and explainability
artifacts, and includes a live headline + image demo using the trained alpha-gated checkpoint:

```bash
pip install -r requirements.txt
run_app.cmd
```

The findings dashboard works directly from the tracked `outputs/` artifacts. The live model
downloads the configured RoBERTa and CLIP backbones on first use. To enable the curated
test-row picker, place `fakeddit.h5` at `data/fakeddit.h5` or set `HEMT_HDF5` to its path.

The Colab-friendly notebook demo remains available at **`notebooks/06_demo.ipynb`**.

To verify the live inference path independently of the UI, run the curated
moon/flagpole acceptance sample:

```bash
python scripts/verify_live_demo.py --overlay outputs/live_demo_verified.png
```

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
