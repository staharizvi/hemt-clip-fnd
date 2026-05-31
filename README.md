# HEMT-CLIP — Multimodal Fake News Detection

**XAI for Multimodal Fake News Detection — Final Year Project, UMT Sialkot, 2022–2026**

Team: Syed Taha Faiz-ul-Hassan Rizvi, Ayesha Bukhari, Ali Mosa Raza
Advisor: Ma'am Hina Tufail

## Overview

HEMT-CLIP takes a social-media post (text + image) and predicts Real vs Fake with a confidence score, explained via cross-attention heatmaps and SHAP. The architecture combines:

- **RoBERTa-base** (text, last 2 layers trainable)
- **CLIP ViT-B/32** (vision tower, last 2 blocks trainable)
- **8-head cross-attention fusion** (text queries attend over image patches)
- **CLIP cosine similarity α** fed as an extra feature to the classifier
- **2-layer MLP classifier** → P(Real), P(Fake)

See [HEMT_CLIP_Implementation_Blueprint.md](HEMT_CLIP_Implementation_Blueprint.md) for the full specification.

## Setup

```bash
pip install -r requirements.txt
```

## Demo

Launch the live Streamlit demo (serves the headline α-gated HEMT-CLIP):

```bash
# public URL via ngrok (set a free token first)
NGROK_AUTHTOKEN=xxxxx python run_demo.py
# or local only
python run_demo.py --no-tunnel
```

`run_demo.py` discovers the latest `gated_fusion` checkpoint, launches `app/streamlit_app.py`,
and opens the tunnel. See `python run_demo.py --help` for flags (`--variant`, `--ckpt`, `--port`).

## Repository Layout

```
configs/         hyperparameters (base.yaml, debug.yaml)
data/            Fakeddit download, HDF5 packer, PyTorch Dataset, splits
models/          text/image encoders, fusion, classifier, full HEMT-CLIP
explainability/  cross-attention viz + SHAP/LIME for text
training/        training loop, evaluation, ablation runner
app/             Streamlit demo (streamlit_app.py)
run_demo.py      demo launcher (checkpoint discovery + streamlit + ngrok)
notebooks/       data exploration, smoke test, full training, eval, XAI
checkpoints/     trained weights (kept on Google Drive)
outputs/         plots and figures for the report
```

## Ablation Variants

| Variant | Description |
|---|---|
| A | Text-only baseline |
| B | Image-only baseline |
| C | Concat fusion `[text, image, α]` |
| D | HEMT-CLIP (cross-attention + α) |
