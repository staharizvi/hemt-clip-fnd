# HEMT-CLIP Implementation Blueprint
## XAI for Multimodal Fake News Detection — Final Year Project

**Team:** Syed Taha Faiz-ul-Hassan Rizvi, Ayesha Bukhari, Ali Mosa Raza
**Advisor:** Ma'am Hina Tufail
**Institution:** University of Management and Technology, Sialkot
**Session:** 2022–2026

---

## 1. Executive Summary

This blueprint implements a multimodal fake news detection system that:
1. Takes a social media post (text + image) as input
2. Predicts Real or Fake with a confidence score
3. Explains its prediction through attention heatmaps and SHAP analysis

The system uses **RoBERTa** for text understanding, **CLIP ViT-B/32** for image understanding, a **CLIP-guided cross-attention fusion module** to combine them, and a **2-layer MLP classifier** for final prediction. Explainability is provided through cross-attention visualization (intrinsic, real-time) and SHAP for text tokens (post-hoc, run on a sample subset).

**Target deliverables:**
- Trained HEMT-CLIP model on ~15K Fakeddit samples
- Four-variant ablation study (text-only, image-only, concat fusion, full HEMT-CLIP)
- Attention visualizations on ~15 examples
- SHAP text explanations on ~30 samples
- Streamlit demo
- Updated FYP report with Results & Discussion chapter

**Total estimated time:** 2.5–3 weeks of focused work.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                            │
│   Text (post title/caption)         Image (224×224 RGB)         │
└──────────────┬─────────────────────────────────┬────────────────┘
               │                                 │
               ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│   TEXT ENCODER           │      │   IMAGE ENCODER              │
│   RoBERTa-base (125M)    │      │   CLIP ViT-B/32              │
│   Last 2 layers trainable│      │   Last 2 blocks trainable    │
│   Output: 768-dim        │      │   Output: 512-dim            │
└──────────┬───────────────┘      └──────────────┬───────────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│   Projection: 768→512    │      │   Already 512-dim            │
└──────────┬───────────────┘      └──────────────┬───────────────┘
           │                                     │
           │       ┌─────────────────────────────┴──────┐
           │       │                                    │
           ▼       ▼                                    │
┌─────────────────────────────────────────────────┐    │
│   CROSS-ATTENTION FUSION (8 heads, 512-dim)     │    │
│   Q = text_features, K = V = image_features     │    │
│   Output: attended_features (512-dim)           │    │
└──────────────────────┬──────────────────────────┘    │
                       │                               │
                       │   ┌───────────────────────────┘
                       │   │
                       ▼   ▼
            ┌──────────────────────────┐
            │  CLIP Similarity α       │
            │  (cosine sim, scalar)    │
            └──────────────┬───────────┘
                           │
                           ▼
            ┌──────────────────────────────────┐
            │  CONCAT: [attended (512), α (1)] │
            │  Total: 513-dim                  │
            └──────────────┬───────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────┐
            │  CLASSIFIER MLP                  │
            │  513 → 256 → 2                   │
            │  ReLU, Dropout 0.3, Softmax      │
            └──────────────┬───────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  P(Real), P(Fake)      │
              └────────────────────────┘
```

**Key design decisions:**
- α (CLIP similarity) is fed to the classifier as an extra feature, not used as a fusion weight. This lets the model *learn* that low similarity often signals manipulation, rather than forcing a hand-coded fusion rule.
- Two-layer fine-tuning depth (instead of three) keeps training time manageable on mixed Colab GPUs.
- Cross-attention is text-to-image (text queries attend over image patches), which aligns with how humans verify: "given this claim, what in the image supports or contradicts it?"

---

## 3. Technology Stack

| Component | Library / Version |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.1+ with CUDA |
| Text model | `transformers` 4.40+ (HuggingFace) |
| Image model | `transformers` (CLIP via HuggingFace) — keeps everything in one library |
| Data | `pandas`, `numpy`, `Pillow`, `h5py` |
| ML utilities | `scikit-learn` (metrics, splits) |
| Explainability | `shap` 0.45+, custom attention extraction |
| Visualization | `matplotlib`, `seaborn` |
| Experiment tracking | `wandb` (free tier) |
| Demo interface | `streamlit` 1.32+ |
| Tunneling for demo | `pyngrok` (during viva) |
| Storage | Google Drive (mounted in Colab) |

**One-line install for Colab:**
```bash
!pip install -q transformers==4.40.0 shap==0.45.0 wandb==0.16.0 \
                streamlit==1.32.0 pyngrok h5py
```

---

## 4. Project Structure

```
hemt-clip-fnd/
│
├── configs/
│   ├── base.yaml                    # All hyperparameters in one place
│   └── debug.yaml                   # Tiny config for smoke tests
│
├── data/
│   ├── download_fakeddit.py         # Parallel image downloader
│   ├── build_hdf5.py                # Pack images into single HDF5 file
│   ├── dataset.py                   # PyTorch Dataset class
│   └── splits/
│       ├── train.csv                # Indices + labels for train split
│       ├── val.csv
│       └── test.csv
│
├── models/
│   ├── text_encoder.py              # RoBERTa wrapper with projection
│   ├── image_encoder.py             # CLIP ViT wrapper
│   ├── fusion.py                    # 8-head cross-attention module
│   ├── classifier.py                # 2-layer MLP head
│   └── hemt_clip.py                 # Full assembled model
│
├── explainability/
│   ├── attention_viz.py             # Heatmap generation from attention weights
│   └── shap_text.py                 # SHAP for text tokens via HF pipeline
│
├── training/
│   ├── train.py                     # Main training loop with resumption
│   ├── evaluate.py                  # Test set evaluation + metrics
│   └── ablation_runner.py           # Runs all 4 ablation variants
│
├── app/
│   └── streamlit_app.py             # Demo interface
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Look at Fakeddit, plan sampling
│   ├── 02_smoke_test.ipynb          # Overfit 500 samples (debug)
│   ├── 03_full_training.ipynb       # Production training notebook
│   ├── 04_evaluation.ipynb          # Metrics + plots
│   └── 05_explainability.ipynb      # Generate viz for report
│
├── checkpoints/                     # On Google Drive
│   ├── hemt_clip_best.pt
│   ├── text_only_best.pt
│   ├── image_only_best.pt
│   └── concat_fusion_best.pt
│
├── outputs/                         # Plots and figures for report
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── attention_examples/
│   └── shap_examples/
│
├── requirements.txt
└── README.md
```

---

## 5. Dataset Strategy

### 5.1 Source
**Fakeddit** dataset — multimodal Reddit posts with text + image + binary label.

### 5.2 Sampling Plan
| Stage | Sample Size | Purpose |
|---|---|---|
| Smoke test | 500 samples | Verify pipeline + overfit check |
| Pilot | 5,000 samples | First end-to-end run, debug |
| Main experiment | 15,000 samples | Target for final results |
| Stretch | 25,000 samples | Only if time + GPU allow |

Stratified sampling (balanced classes). Use Fakeddit's `2_way_label` column (0=real, 1=fake).

### 5.3 Splits
70% train / 15% validation / 15% test, stratified by label, fixed `random_state=42`.

### 5.4 Data Pipeline
1. Download Fakeddit metadata TSVs (small, instant)
2. Sample 18,000 rows (expect ~15K usable after broken URLs)
3. Parallel image download with 5s timeout, 1 retry, skip-on-fail
4. Resize all images to 224×224 RGB
5. Pack everything into a **single HDF5 file** stored on Drive
6. HDF5 contains: `images` (N×3×224×224 uint8), `texts` (N strings), `labels` (N int)

**Why HDF5?** Loading 15K loose JPEGs from Drive will dominate epoch time (Drive's small-file I/O is brutal). A single HDF5 file with sequential reads is roughly 10× faster.

### 5.5 Train/Val/Test CSV Format
```
hdf5_index, text, label, split
0, "Breaking: scientist discovers...", 1, train
1, "Local news: weather update...", 0, train
...
```

---

## 6. Model Specifications

### 6.1 Text Encoder
- **Base:** `roberta-base` from HuggingFace (125M params)
- **Max sequence length:** 128 tokens (Fakeddit titles rarely exceed this)
- **Frozen:** Embedding layer + first 10 transformer layers
- **Trainable:** Last 2 transformer layers + projection head
- **Output:** `[CLS]` token representation, projected from 768 → 512 dim
- **Projection head:** `Linear(768, 512) → LayerNorm → Dropout(0.1)`

### 6.2 Image Encoder
- **Base:** `openai/clip-vit-base-patch32` (vision tower only)
- **Input:** 224×224 RGB, normalized with CLIP's mean/std
- **Frozen:** Patch embedding + first 10 transformer blocks
- **Trainable:** Last 2 transformer blocks
- **Output:** pooled vision representation, 512-dim (matches projection)

### 6.3 Cross-Attention Fusion
- **Module:** `nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)`
- **Q = text_features**, **K = V = image_features**
- **Residual connection + LayerNorm** after attention
- **Feed-forward block:** `Linear(512, 2048) → GELU → Dropout → Linear(2048, 512) → LayerNorm + residual`
- **Returns:** fused 512-dim representation **and** attention weights (for explainability)

### 6.4 CLIP Similarity Module
- Use full CLIP model (both encoders) to compute cosine similarity between text and image embeddings
- **Crucial optimization:** Precompute α for the entire dataset once and cache to HDF5. CLIP forward pass is expensive; doing it every training step is wasteful.
- α is a scalar per sample, ranging roughly [-1, 1] but typically [0, 1] for normal pairs

### 6.5 Classifier Head
- **Input:** `[fused_features (512), α (1)]` concatenated → 513-dim
- **Architecture:** `Linear(513, 256) → ReLU → Dropout(0.3) → Linear(256, 2)`
- **Output:** logits for [Real, Fake], softmax applied at inference

### 6.6 Total Parameter Count
- RoBERTa (frozen + trainable): 125M (only ~28M trainable in last 2 layers)
- CLIP ViT (frozen + trainable): ~88M (only ~14M trainable in last 2 blocks)
- Projection: ~400K
- Cross-attention + FFN: ~2.1M
- Classifier: ~130K
- **Trainable total:** ~44M params (manageable on T4 with FP16)

---

## 7. Ablation Study Design

Four model variants, all trained on the same data splits:

| Variant | Inputs | Fusion | Purpose |
|---|---|---|---|
| **A: Text-only** | Text | None — direct classifier on text features | Baseline: how much can text alone achieve? |
| **B: Image-only** | Image | None — direct classifier on image features | Baseline: how much can image alone achieve? |
| **C: Concat fusion** | Text + Image | Simple concatenation `[text, image, α]` | Does multimodality help at all? |
| **D: HEMT-CLIP (full)** | Text + Image | 8-head cross-attention + α | Does cross-attention help over concatenation? |

**Story this tells:**
- A vs B: which modality is more informative for fake news?
- (A or B) vs C: does adding the other modality help?
- C vs D: is cross-attention worth the added complexity?

This is the **missing Chapter 6** of your current report, delivered as one structured experiment.

---

## 8. Training Strategy

### 8.1 Staged Fine-Tuning (2 stages)
**Stage 1: Head warmup (1 epoch)**
- Freeze both encoders entirely
- Train only: projection layer, cross-attention, classifier
- High learning rate: 1e-4
- Fast convergence, ~1 hour on T4

**Stage 2: Encoder fine-tuning (2-3 epochs)**
- Unfreeze last 2 RoBERTa layers + last 2 CLIP blocks
- Lower learning rate: 2e-5
- Early stopping on validation F1 with patience=2
- ~3-4 hours on T4

**Total training time:** 4-6 hours per ablation variant on T4. With 4 variants, plan for ~20 hours total (across multiple Colab sessions).

### 8.2 Hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | weight_decay=0.01 |
| Learning rate | Stage 1: 1e-4, Stage 2: 2e-5 | Linear warmup over 10% of steps |
| Batch size | 16 (T4) / 32 (L4) / 64 (A100) | Auto-detect GPU and adjust |
| Gradient accumulation | 2 steps if OOM | Effective batch stays at 32 |
| Mixed precision | FP16 via `torch.cuda.amp` | Mandatory — 1.5-2× speedup |
| Gradient checkpointing | Enabled on RoBERTa | Saves ~30% memory |
| Label smoothing | 0.1 | Helps generalization |
| Loss | CrossEntropyLoss | Standard for binary |
| Epochs | Stage 1: 1, Stage 2: up to 3 | Early stop on val F1 |
| Random seed | 42 | Set on Python, NumPy, PyTorch, CUDA |

### 8.3 Checkpoint Strategy
- Save checkpoint **after every epoch** (not just on improvement) — Colab sessions die unpredictably
- Save: model state dict, optimizer state, scheduler state, epoch number, best val F1 so far
- Resume logic: on script start, check if checkpoint exists, load and continue from saved epoch
- Best model (by val F1) saved separately as `*_best.pt`

### 8.4 Experiment Tracking
Use **wandb free tier**. Log:
- Train/val loss per step
- Train/val accuracy, F1 per epoch
- Learning rate schedule
- GPU memory usage
- Sample predictions every 500 steps

Why: when your advisor asks "why batch size 16?" or "what was the learning curve?", you have plots. Saves your final-week presentation prep.

---

## 9. Evaluation Plan

### 9.1 Metrics
- **Accuracy**
- **Precision** (macro and per-class)
- **Recall** (macro and per-class)
- **F1-score** (macro — most important for slight imbalance)
- **AUC-ROC**
- **Confusion matrix**

### 9.2 Comparison Table Format
```
Model              | Acc   | Prec  | Rec   | F1    | AUC   | Params (trainable)
-------------------|-------|-------|-------|-------|-------|-------------------
Text-only          | xx.x  | xx.x  | xx.x  | xx.x  | xx.x  | ~28M
Image-only         | xx.x  | xx.x  | xx.x  | xx.x  | xx.x  | ~14M
Concat fusion      | xx.x  | xx.x  | xx.x  | xx.x  | xx.x  | ~42M
HEMT-CLIP (full)   | xx.x  | xx.x  | xx.x  | xx.x  | xx.x  | ~44M
```

### 9.3 Required Plots
1. Training/validation loss curves (one plot per variant, or all four on one plot)
2. F1 progression over epochs
3. Confusion matrix heatmap for best model
4. ROC curve comparing all four variants
5. Per-class precision/recall bar chart

### 9.4 Comparison with Literature
Cite reported numbers from your existing literature review (FND-CLIP, BC-FND, FMC, etc.). **Do not reimplement them** — that's a separate research project. Just say "comparable to" or "within X% of" published baselines on similar dataset sizes.

---

## 10. Explainability Implementation

### 10.1 Attention Visualization (Primary Method)
- Extract attention weights from cross-attention layer during forward pass
- Weights shape: `(batch, num_heads, query_len, key_len)` — average across heads for visualization
- **Text-side visualization:** color-coded tokens showing attention intensity
- **Image-side visualization:** heatmap overlay on original image showing which patches were attended to
- Implementation: `matplotlib` + `seaborn` for plots, `PIL` for image overlays

### 10.2 SHAP for Text (Secondary Method)
- Use `shap.Explainer` with HuggingFace `pipeline()` wrapper — this is the *easy* path, avoid `GradientExplainer` on the fused model
- Run on a fixed subset of 30 test samples (mix of correct/incorrect predictions)
- Generate HTML output with `shap.plots.text()` — produces beautiful color-coded explanations
- **Skip multimodal SHAP** — too complex, marginal value for FYP

### 10.3 Confidence Score
- Always shown: `softmax(logits)` → probability for predicted class
- Calibration check: plot reliability diagram on test set (optional)

### 10.4 CLIP Similarity Display
- Show α value alongside prediction
- Interpret for users: "Text-image alignment: 0.74 (high)" or "0.21 (low — possible mismatch)"

### 10.5 Required Outputs for Report
- 10 attention visualization examples (5 correct + 5 misclassified — the latter are *more* interesting)
- 5 SHAP text explanations
- 1 figure showing the same example explained by both methods (do they agree?)

---

## 11. Streamlit Demo Specification

### 11.1 Layout
```
┌─────────────────────────────────────────────────────────────┐
│                  HEMT-CLIP Demo                              │
│         Multimodal Fake News Detection                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   [Text input area]                                          │
│   [Image upload button]                                      │
│                                                              │
│   [  Analyze  ]                                              │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│   Prediction: FAKE   Confidence: 87.3%                       │
│   ████████████████████░░░  87.3% Fake                        │
│   ██░░░░░░░░░░░░░░░░░░░░░  12.7% Real                        │
│                                                              │
│   Text-Image Alignment (α): 0.21 (low — possible mismatch)   │
├─────────────────────────────────────────────────────────────┤
│   [Tab: Attention Heatmap]                                   │
│   [Tab: SHAP Explanation]                                    │
│   [Tab: How it Works]                                        │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 Implementation Notes
- Load model checkpoint once on app startup (cached with `@st.cache_resource`)
- Run inference on CPU (Streamlit Cloud has no GPU) OR run backend on Colab and tunnel via ngrok during viva
- Generate attention visualization on the fly per prediction
- SHAP precomputed on a few canned examples (too slow for live)
- Include an "About" / "How it works" tab that explains the model briefly — useful for the panel

### 11.3 Deployment
- **For viva:** run locally in Colab with ngrok tunnel, share link with examiners
- **For long-term:** deploy on Streamlit Community Cloud (free) — only if time permits
- Don't waste time on deployment polish. Functional > pretty.

---

## 12. Week-by-Week Timeline

### Week 1: Data + Skeleton

**Day 1 — Environment + data exploration**
- Set up Colab Pro, mount Drive
- Install all dependencies, verify GPU
- Download Fakeddit metadata, explore class balance, examine sample posts
- Notebook: `01_data_exploration.ipynb`

**Day 2 — Data download + HDF5 build**
- Run `download_fakeddit.py` to pull ~18K images (will take 4-6 hours, run overnight)
- Implement `build_hdf5.py` while downloads run
- Once images are local, pack into HDF5 file on Drive
- Verify HDF5 loads correctly with random samples

**Day 3 — Model skeleton**
- Implement all model modules: `text_encoder.py`, `image_encoder.py`, `fusion.py`, `classifier.py`, `hemt_clip.py`
- Implement `dataset.py` reading from HDF5
- Write minimal `train.py` skeleton

**Day 4 — Smoke test + α precomputation**
- Notebook: `02_smoke_test.ipynb` — overfit 500 samples, verify loss decreases
- Run α precomputation on full dataset (one-time cost, ~30 min)
- Fix bugs found in smoke test

**End of Week 1 deliverable:** Working pipeline, all 18K images packed in HDF5, α cached, smoke test passes.

---

### Week 2: Training + Ablations

**Day 5 — Train HEMT-CLIP (variant D)**
- Stage 1 (heads only): 1 epoch, ~1 hour
- Stage 2 (with fine-tuning): up to 3 epochs with early stopping, ~4 hours
- Save best checkpoint to Drive
- Monitor on wandb

**Day 6 — Train Variant A (text-only)**
- Reuse text encoder, skip image branch, simple classifier
- Same staged training, ~3 hours total

**Day 7 — Train Variants B (image-only) and C (concat fusion)**
- Both can run in same day; image-only is ~2 hours, concat fusion ~4 hours
- May spill into Day 8 if GPU sessions disconnect

**Day 8 — Evaluation + plots**
- Notebook: `04_evaluation.ipynb`
- Compute all metrics for all 4 variants on test set
- Generate all required plots
- Build comparison table

**End of Week 2 deliverable:** 4 trained models, complete metrics table, 5+ plots ready for report.

---

### Week 3: Explainability + Demo + Report

**Day 9 — Attention visualization**
- Implement `attention_viz.py`
- Generate 10 examples (5 correct + 5 misclassified)
- Save high-quality PNGs to `outputs/attention_examples/`

**Day 10 — SHAP for text**
- Implement `shap_text.py` using HuggingFace pipeline wrapper
- Generate 30 sample explanations
- Save HTML outputs + selected PNGs for report
- **Time limit: if SHAP setup takes >4 hours, drop it and write up attention-only**

**Day 11 — Streamlit demo**
- Implement `streamlit_app.py`
- Test with 5 manual examples (3 fake, 2 real)
- Set up ngrok tunnel, verify external access works

**Days 12-13 — Report writing**
- Write Chapter 6: Results & Discussion (most important new content)
- Write Chapter 7: Conclusion & Future Work
- Fix existing issues in current report:
  - Duplicate acknowledgments paragraph
  - Scope inconsistency (text-only claim vs multimodal implementation)
  - Duplicated section 5.1.1
  - Grammar polish throughout
- Insert all generated figures with proper captions
- Update abstract with actual numbers

**Day 14 — Slides + dry-run + buffer**
- Build 12-15 slide presentation
- Two dry-runs of the Streamlit demo (Colab can be flaky)
- Prepare answers for the viva questions in your team document

**End of Week 3 deliverable:** Complete report, working demo, polished slides, ready for viva.

---

## 13. Risk Management

### Common Failure Modes and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Colab session disconnects mid-training | High | Checkpoint every epoch, resume logic in `train.py` |
| Get bumped to T4 when you need L4 | Medium | Auto-detect GPU, scale batch size, enable gradient checkpointing |
| Out of memory error | Medium | Fallback: batch size 8 + gradient accumulation 2 + FP16 |
| >15% images broken in Fakeddit URLs | Medium | Oversample (download 20K to get 17K usable) |
| SHAP library breaks on fused model | Medium | Use SHAP on text encoder only via HF pipeline; skip multimodal SHAP |
| Streamlit Cloud doesn't load model | Medium | Use ngrok tunnel during viva instead |
| Training overfits early | Low-Medium | Check for data leakage (same post in train+test), use post_id for split |
| Attention weights look uniform | Low | Verify gradients flow through fusion module, check `requires_grad` |
| Drive runs out of space | Low | HDF5 is ~10GB for 18K samples — verify Drive has room |
| Misclassified samples look correct to humans | Low (but possible) | Document as limitation in report — labels are noisy in Fakeddit |

### Time Contingencies

**If Week 1 slips:**
- Cut sample size to 10K (still defensible)
- Skip 2 of 4 ablation variants (drop concat fusion + image-only, keep text-only + full)

**If Week 2 slips:**
- Skip SHAP entirely, write up attention-only explainability
- Use fewer epochs in Stage 2 (1-2 instead of 3)

**If Week 3 slips:**
- Skip Streamlit, demo via Colab notebook instead
- Use placeholder figures, polish later

---

## 14. Defensive Engineering Defaults

These are baked into the code from Day 1, not added later:

1. **FP16 mixed precision always on** (`torch.cuda.amp.autocast`)
2. **Gradient checkpointing on RoBERTa** by default
3. **Checkpoint resume logic** in `train.py` from first commit
4. **Random seed fixed** (42) on Python, NumPy, PyTorch, CUDA
5. **wandb logging** from first training run
6. **Stratified splits** by label, fixed `random_state`
7. **Auto GPU detection** with batch size scaling
8. **Try-except around model forward pass** to catch OOM gracefully
9. **All hyperparameters in `config/base.yaml`** — no magic numbers in code
10. **Image preprocessing matches CLIP exactly** (mean/std from CLIP repo)

---

## 15. What Goes In The Final Report

### New chapters / sections to write:
- **Chapter 6: Results & Discussion** (10-15 pages)
  - 6.1 Experimental Setup (data, hardware, hyperparameters)
  - 6.2 Quantitative Results (metrics table, plots)
  - 6.3 Ablation Study (what each component contributes)
  - 6.4 Qualitative Analysis (attention visualizations, SHAP examples)
  - 6.5 Comparison with Literature
  - 6.6 Error Analysis (misclassified samples — what went wrong)
- **Chapter 7: Conclusion & Future Work** (3-5 pages)
  - 7.1 Summary of Contributions
  - 7.2 Limitations
  - 7.3 Future Directions

### Existing report fixes:
- Remove duplicate acknowledgments paragraph (Chapter starts)
- Update Section 1.4 (Scope) to acknowledge multimodal scope, not text-only
- Fix duplicated section numbering 5.1.1
- Update Section 4.3.1 to reflect what was actually implemented (2 stages, not 3)
- Add actual figures to replace placeholder references
- Update abstract with real metric numbers

---

## 16. Viva Preparation

### Defensible Talking Points
1. **"Why cross-attention over concatenation?"** → Show ablation: C vs D variants demonstrate the contribution quantitatively.
2. **"Why fine-tune only the last 2 layers?"** → Resource constraints + catastrophic forgetting prevention. Both well-established in NLP literature.
3. **"Why drop LIME?"** → Methodological redundancy with SHAP (both perturbation-based). Cleaner to defend two complementary methods (intrinsic + post-hoc) than three overlapping ones.
4. **"Why only 15K samples?"** → Realistic for Colab Pro session limits + Reddit URL reliability. Several papers in the literature review used similar or smaller subsets.
5. **"What's novel here?"** → CLIP-guided cross-attention with α as a learned feature (not a hand-coded fusion weight), combined with a unified two-method explainability framework. Both are described in your existing report's "Novel Contributions" section.

### Questions Your Team Document Already Handles Well
Your Q&A in the existing FYP_Simple_Implementation_Explanation document covers the basics well — keep using it.

### Additional Questions to Prepare For
- "How would your system handle adversarial examples (deepfakes)?" → Acknowledge as future work, mention deepfake-specific tools like ResNet forensics.
- "Why Fakeddit and not Twitter/Weibo?" → Multimodal + binary labels + public availability + scale. Twitter API restrictions make replication hard.
- "How does this differ from FND-CLIP / BC-FND?" → Architectural differences in fusion (cross-attention with α as feature vs their similarity-weighted fusion) + explainability focus.

---

## 17. Definition of Done

The project is complete when:

- [ ] 15K+ Fakeddit samples processed and stored in HDF5
- [ ] All 4 ablation variants trained, best checkpoints saved
- [ ] Test set metrics computed for all 4 variants
- [ ] 5+ plots generated (training curves, confusion matrix, ROC, ablation comparison)
- [ ] 10+ attention visualization examples saved
- [ ] 30+ SHAP text explanations generated (or attention-only if SHAP dropped)
- [ ] Streamlit demo runs end-to-end with example inputs
- [ ] Chapter 6 (Results & Discussion) written and inserted in report
- [ ] Chapter 7 (Conclusion) written
- [ ] Existing report issues fixed (duplicates, scope, numbering)
- [ ] Slide deck ready (12-15 slides)
- [ ] Demo dry-run completed twice without crashes
- [ ] Code pushed to GitHub with README and requirements.txt

---

## 18. First Steps (Right Now)

In order:

1. Create GitHub repo, clone to Colab
2. Mount Drive, create `hemt-clip-fnd/` folder structure
3. Sign up for wandb (free), create project "hemt-clip-fnd"
4. Download Fakeddit metadata, examine class balance
5. Run `01_data_exploration.ipynb` — confirm you can read metadata and sample a few images via URL
6. Implement and start `download_fakeddit.py` — let it run overnight
7. While it runs: implement `text_encoder.py` and `image_encoder.py`

That's Day 1 done. Don't try to do more on Day 1 — the download will take hours of wall time and you can't accelerate it. Use that time for code.

---

**End of Blueprint**
*Document version 1.0 — adjust as implementation progresses.*
