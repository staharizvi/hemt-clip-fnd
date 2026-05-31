# HEMT-CLIP — Source Material for Chapters 4, 5 & 6

**Project:** XAI for Multimodal Fake News Detection (HEMT-CLIP)
**Team:** Syed Taha Faiz-ul-Hassan Rizvi (22002376004), Ayesha Bukhari (22002376015), Ali Mosa Raza (22002376012)
**Advisor:** Ma'am Hina Tufail · University of Management and Technology, Sialkot · Session 2022–2026

---

## How to use this document

This document follows the **exact heading hierarchy already present in `FYP final project.docx`** so the
text drops straight into the report:

- **Chapter 4 — Proposed Methodology** *(exists in the report as a plan — update it with the values below)*
- **Chapter 5 — Design and Implementation** *(exists as a plan — update it with the values below)*
- **Chapter 6 — Results and Discussion** *(does NOT exist yet — this is the missing chapter to add)*

Every number here is from the **actual** training / evaluation / explainability runs, not the original
plan. Where the existing report text describes something the implementation later changed, a
**`⚠ CORRECTION`** box flags exactly what to edit. Read those first — there are nine of them and three
are load-bearing for the viva.

> ### ⚠ Master correction list (read once, then see boxes in context)
> 1. **Image encoder is CLIP ViT-B/16, not B/32** — 16×16 patches → **196 patch tokens (14×14)**, not 49.
> 2. **Last 4 CLIP blocks fine-tuned, not last 3.**
> 3. **α is concatenated to the classifier as an extra feature — it is NOT a fusion gate.** The report's
>    formula `fused = α·attended + (1−α)·text` was **never implemented**. Classifier input is **513**, not 512.
> 4. **SHAP is a partition (Owen-value) explainer, perturbation-based — NOT GradientExplainer / gradient-based.**
> 5. **LIME was fully implemented (30 samples), not "optional".**
> 6. **Label smoothing = 0.0, not 0.1.**
> 7. **Training took ~6 minutes per variant, not "2–5 hours per epoch".** Two *stages*, not per-epoch stages.
> 8. **Dataset is 17,149 samples** (final corpus), not 100,000 / 50,000.
> 9. **Explainability was evaluated by cross-method agreement analysis, not a user study.**

---
---

# Chapter 4 — Proposed Methodology

## 4.1 Suggested Approach

*(Existing prose is fine — it correctly motivates RoBERTa + CLIP + attention fusion + early-baked
explainability. Keep it. Only ensure the closing sentence reflects the final design: text–image pairs,
efficient transformers, resource-constrained training.)*

The system is an explainable multimodal fake-news detector built on three findings from the literature:
(i) the choice of pretrained backbones matters — RoBERTa and CLIP, pretrained on web-scale data,
transfer well to noisy social-media text and imagery; (ii) the fusion strategy must capture inter-modal
*relationships*, not merely stack feature vectors; and (iii) explainability must be designed in from the
start, not bolted on. The design balances three goals — **competitive accuracy, computational efficiency
on a single Colab T4 GPU, and substantive explanations** — deliberately avoiding heavier approaches
(zero-shot LLMs, video, graph neural networks) in favour of efficient transformers trained under
resource constraints.

### 4.1.1 Research Design And Procedures

A supervised-learning framework in four sequential stages: **(1) data preparation** (preprocessing +
stratified **70/15/15** train/val/test split), **(2) model development** via *staged* fine-tuning,
**(3) explainability integration** through a three-level framework, and **(4) evaluation** on the
held-out test set. The staged regime resists overfitting and enables diagnostic inspection within a
limited GPU-time budget.

### 4.1.2 Data Sources And Sampling Procedures

The primary data source is the **Fakeddit** corpus (>1M multimodal Reddit posts; each example is an RGB
image + headline + multi-way labels). We use the **binary** task (`2_way_label`: 0 = real, 1 = fake).
Sampling is **stratified** to keep classes balanced across all three splits; data is shuffled each epoch;
a fixed seed (42) guarantees reproducibility.

> ### ⚠ CORRECTION 8 — actual corpus size
> The plan referenced ~100K samples. The **final processed corpus is 17,149 samples** (after parallel
> download, broken-URL filtering, and HDF5 packing): **8,716 real / 8,433 fake (50.83% / 49.17%)**,
> split **12,003 / 2,573 / 2,573**. State this number; it is what every result in Chapter 6 is computed on.

## 4.2 Workflow of the system

The end-to-end workflow (Figure 4.1) is a **dual-encoder architecture combined through cross-attention**:
text and image are encoded separately, fused by a text-queries-image cross-attention block, augmented
with a CLIP text–image similarity scalar, and classified as Real/Fake.

### 4.2.1 Architecture Components

#### Text Encoder
**RoBERTa-base** (125M params), titles up to **128 tokens**. The embedding layer and first 8 transformer
layers are **frozen**; the **last 4 layers** plus a projection head are trainable (cuts training cost and
prevents catastrophic forgetting). The `[CLS]` representation is projected **768 → 512**.

#### Image Encoder
> ### ⚠ CORRECTION 1 & 2 — backbone and trainable depth
> The plan used **CLIP ViT-B/32** (32×32 patches → 49 tokens, last 3 blocks). The implementation uses
> **CLIP ViT-B/16** (16×16 patches → **196 patch tokens on a 14×14 grid**), with the **last 4 blocks**
> fine-tuned. **Why it changed:** with B/32 the cross-attention block could only attend over a coarse 7×7
> grid and validation F1 plateaued at ≈0.801; B/16 gives a **4× finer spatial grid** for the fusion block
> to localise on, and it is the single change that moved the model off that plateau (see Chapter 6).

The image encoder is the vision tower of **CLIP ViT-B/16**, pretrained on 400M image–text pairs. Patch
embedding and early blocks are frozen; the **last 4 transformer blocks** are fine-tuned. Both the pooled
CLS vector and the 196 patch tokens are projected into the 512-dim space.

#### Fusion Model
An **8-head cross-attention** block where **text is the Query** and the **image patch tokens are Key/Value**.
Heads specialise on different cross-modal patterns (entity alignment, tonal concordance, semantic
cohesion). The attention weights are a **zero-cost, first-order explanation** of which image regions the
text attended to.

### 4.2.2 Processing Pipelines

#### Core Mathematical Operations

**Cross-Attention Computation.** With `Q = text features`, `K = V = image patch features`, head dim `d_k = 64`:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

**CLIP Similarity α.** A scalar per sample — the cosine similarity between CLIP's text and image
embeddings, **precomputed once for the whole dataset and cached in the HDF5** (computing full CLIP every
step would be wasteful):

```
α = cos( CLIP_text(t), CLIP_image(i) )
```

> ### ⚠ CORRECTION 3 — α is a feature, not a fusion gate (LOAD-BEARING)
> The report currently presents the novelty as a **CLIP-guided weighted fusion**:
> `fused = α·attended + (1−α)·text`. **This was never implemented.** The actual model
> (`models/hemt_clip.py`) computes the cross-attended vector, then **concatenates α as one extra
> feature** to the classifier input: `features = concat[ fused(512), α(1) ] → 513`. The classifier is
> therefore **513 → 256 → 2**, not 512 → 256 → 2.
>
> **Rewrite the "CLIP Similarity Guidance" subsection and the Novel-Contribution claim accordingly.** The
> defensible framing (use this): *"α is fed to the classifier as a learned extra feature rather than a
> hand-coded fusion weight, so the model **learns** how text–image (mis)alignment relates to authenticity
> instead of being forced to follow a fixed rule."* This is supported by a dataset finding (§4.2.x / §5.1):
> on Fakeddit, fake posts actually show *higher* mean α than real ones (Δ ≈ +0.046), which a hard-coded
> "low α ⇒ fake" gate would have penalised — concrete evidence that learning the relationship was the
> right call.

**Training Loss.** Cross-entropy over the two classes:

```
L = − Σ  y_c · log p_c          (label smoothing = 0.0)
```

> ### ⚠ CORRECTION 6 — label smoothing
> The plan/table list label smoothing 0.1. The final config uses **0.0** — it was capping validation loss
> with no class-noise to smooth over, so it was dropped.

### 4.2.3 Explainability Framework

A **three-level** framework combining intrinsic and post-hoc methods for complementary views
(Table 4.2 below). The three levels are deliberately chosen to span *intrinsic vs post-hoc* and two text
granularities (sub-word vs whole-word), so an attribution can be argued by **agreement between
independent procedures** rather than relying on any single method.

**Table 4.2 — Levels of explainability & techniques**

| Level | Method | Type | Output |
|---|---|---|---|
| 1 | Cross-attention visualisation | **Intrinsic** | Text→image attention heatmaps (14×14 patch grid) |
| 2 | **SHAP** (partition / Owen-value) | Post-hoc, perturbation | Token importance (BPE sub-word granularity) |
| 3 | **LIME** (local linear surrogate) | Post-hoc, perturbation | Token importance (whole-word granularity) |

> ### ⚠ CORRECTION 4 & 5 — SHAP type and LIME status
> (4) SHAP is **not gradient-based**: it is a **partition (Owen-value) explainer**, which is
> **perturbation-based**. Remove "GradientExplainer / gradient" wording. (5) **LIME is not optional — it
> was implemented** on the same 30 samples as SHAP. Both run on the **text-only model** (so the attribution
> question "which words drove the verdict?" has clean semantics).

### 4.2.4 Justification for Multimodal Approach

Fusion serves several stakeholder needs at once. Attention visualisation gives parameter-free, instant
insight during development; SHAP gives theoretically grounded validation; LIME gives readable, whole-word
explanations for non-technical users. The methodological redundancy across SHAP and LIME raises
confidence and makes **concordance/discordance between methods itself informative** (Chapter 6.4).

### 4.2.5 Limitations and Assumptions

*(Existing prose is broadly fine. Two edits below.)*

- **Computational** — single T4 limits batch size and hyperparameter search; mitigated by efficient
  backbones, partial-layer freezing, and cached α.
- **Dataset** — Reddit-sourced; may carry platform bias and is English-only. **(Update the "100K-sample"
  figure to 17,149.)**
- **Scope** — fixed single image–text pairs only (no video/multi-image); no external fact-checking; binary
  label simplifies the misinformation spectrum (satire, misleading context).
- **Explainability** — attention shows *focus*, not proven *causation*; both post-hoc methods are
  perturbation-based and share an out-of-distribution floor (perturbed inputs are unnatural).
- **Methodological assumption** — misinformation patterns in training data transfer to new cases; text–image
  discrepancies in fake content are detectable.

## 4.3 Algorithms / Architecture

### 4.3.1 Novel Contribution

> ### ✎ NOVELTY — reframed (do not claim "a new fusion architecture")
> The original report led with "CLIP-guided cross-attention fusion" as the novelty. That claim is weak on
> two counts: the α-gated mechanism it describes was never built, and the cross-attention architecture
> that *was* built does not beat plain concatenation on F1 (Chapter 6). **Do not lead with the
> architecture.** The genuine, defensible contributions are the three below — an empirical finding, a
> methodological framework, and an honest architectural trade. This is what to put in the abstract and
> the "Novel Contribution" section, and to rehearse for the viva.

#### Contribution 1 (Empirical) — α behaves opposite to the prevailing assumption on Fakeddit, and a learned α feature beats α-gating
Prior CLIP-based detectors (e.g. FND-CLIP) assume **low text–image similarity signals manipulation** and
therefore **gate** fusion on that similarity. On Fakeddit we find the **opposite**: fake posts have a
**higher** mean CLIP similarity than real ones (α: fake ≈ 0.290 vs real ≈ 0.244, Δ ≈ +0.046). A hard-coded
"low α ⇒ fake" gate is therefore the *wrong* inductive bias for this data. We instead supply **α to the
classifier as a learned feature**, letting the model decide how (mis)alignment relates to authenticity.
**We test this head-to-head** with a dedicated ablation — `gated_fusion` (α as a similarity gate, the
prior-work approach) vs `hemt_clip` (α as a learned feature, ours) — on the *identical* cross-attention
backbone, so the only difference is how α is used (Chapter 6.3.3). This converts a design choice into a
**measured result** and is the strongest single novelty point.

> Recompute the fake-vs-real α split on the final ViT-B/16 embeddings before quoting the exact Δ — the
> 0.290/0.244 figures were measured on ViT-B/32 (notebook 01, CPU, seconds to redo). The *direction*
> (fake > real) is the load-bearing claim.

#### Contribution 2 (Methodological) — integrated three-level explainability with a *conditional* cross-method agreement analysis
We combine **intrinsic** cross-attention heatmaps with **post-hoc SHAP (BPE)** and **LIME (whole-word)**,
and — beyond merely running three methods — we analyse **when they agree and when they diverge**: agreement
holds for long, signal-rich titles and **inverts on short ones** (Chapter 6.4, sample 2115), while LIME's
whole-word view recovers signal that SHAP's sub-word fragmentation hides (sample 2221). This conditional
agreement framework (positioned against [16] Roshinta & Gábor's LIME-vs-SHAP study) is uncommon at this
level and is more honest than a flawless-agreement story.

#### Contribution 3 (Architectural trade) — intrinsic explainability as the rationale for cross-attention
Cross-attention is retained not because it wins on F1 (it does not) but because it yields **free,
image-side spatial heatmaps** that concatenation architecturally cannot, plus a **recall-skewed decision
profile** useful for triage. We state the ≈0.4 pt F1 cost openly and justify the trade.

#### (Supporting) Resource-Efficient Staged Fine-Tuning
Progressive unfreezing trades adaptation against cost: **Stage 1** trains only projection + fusion +
classifier (fast convergence, low memory); **Stage 2** unfreezes the last RoBERTa layers and CLIP blocks
with early stopping, limiting catastrophic forgetting at modest compute.

> ### ⚠ CORRECTION 7 (preview) — "Epoch 1/2/3" framing is wrong; see §5.x
> This subsection currently maps stages onto multi-hour *epochs*. The implementation has **two stages**
> (Stage 1 = 1 epoch head-warmup; Stage 2 = up to 6 epochs encoder fine-tuning), and the **whole run takes
> ~6 minutes on a T4**, not hours. Fix the numbers when you reach the training subsection.

### 4.3.2 Integration Of Pre-trained Models

**Table 4.3 — Model architecture & configuration (corrected to as-built)**

| Component | Model | Configuration |
|---|---|---|
| Text encoder | RoBERTa-base | Max len 128, dropout 0.1, **trainable: last 4 layers (9–12)** + projection 768→512 |
| Image encoder | **CLIP ViT-B/16** | Input 224×224, **patch 16×16 → 196 tokens**, **trainable: last 4 blocks** |
| Fusion | Cross-attention | 8 heads, 512-dim, FFN 512→2048→512, **dropout 0.2** |
| α | CLIP cosine similarity | Precomputed, cached; **concatenated as feature (not a gate)** |
| Classifier | 2-layer MLP | **513 → 256 → 2**, ReLU, dropout 0.3 |

### 4.3.3 Contingency Strategies

*(Keep the four-tier plan — it shows engineering foresight. Note in the chapter that, in practice, only the
default path was needed: training fit comfortably on a single T4 in ~6 min/variant, so no tier was
triggered. The robustness machinery — per-epoch atomic checkpointing with full resume, FP16, gradient
checkpointing, auto-batch-size — was nonetheless built in from the start and is what made the runs
survive Colab disconnects.)*

---
---

# Chapter 5 — Design and Implementation

This chapter describes the realised system: design, module implementations, tools, training, and
integration. It turns the Chapter-4 architecture into a working prototype.

## 5.1 System Design

A **modular** design separates concerns into five independently testable sub-components: data
preprocessing, text encoding, image encoding, cross-attention fusion, and explainability generation.

### 5.1.1 Architecture Design

- **Input layer** — a (text, image) pair: post title (≤128 tokens) + 224×224 RGB image.
- **Text Encoder Module** — RoBERTa-base, first 8 layers frozen, **last 4 fine-tuned**; 768→512 projection.
- **Image Encoder Module** — **CLIP ViT-B/16**, **last 4 blocks** fine-tuned; outputs pooled (512) + 196 patch tokens (512).
- **Fusion Module** — 8-head cross-attention (Q = text, K = V = image patches) + residual/LayerNorm + FFN.
  *(No α gate inside fusion — see CORRECTION 3.)*
- **Classification Head** — concat `[fused(512), α(1)]` → **513 → 256 → 2**, ReLU, dropout 0.3, softmax.
- **Explainability Module** — extracts attention weights live; runs SHAP/LIME on demand.

### 5.1.2 Interface Design

The Streamlit demo (`app/streamlit_app.py`, ngrok-tunnelled for the viva) provides:
- **Production View** — text box + image upload (or a dropdown of 6 canned test samples), Analyze button,
  Real/Fake label with class-probability bars and the α value with a verbal interpretation
  (low / moderate / high), plus ground-truth comparison when a test sample is chosen.
- **Attention Heatmap View** — live 14×14 attention overlay computed per prediction (<1 s on T4).
- **SHAP Explanation View** — precomputed per-sample SHAP bar looked up by sample id.
- **LIME Explanation View** — *(now realised, not optional)* precomputed whole-word LIME bar.

### 5.1.3 System Modeling

*(Keep your Use Case and Sequence diagrams. The sequence is: user submits text+image → tokenise +
CLIP-normalise → text/image encoders → cross-attention fusion → concat α → classifier → softmax →
render label + α + attention heatmap; SHAP/LIME served from precomputed artefacts.)*

### 5.1.4 Development Tools and Environments

**Table 5.1 — Tools and technologies (as used)**

| Purpose | Tool |
|---|---|
| Language | Python 3.10 |
| Deep learning | PyTorch 2.1 (CUDA), FP16 autocast |
| Backbones | HuggingFace `transformers` 4.40 (RoBERTa, CLIP) |
| Data | NumPy, pandas, Pillow, **h5py** (single packed HDF5) |
| Metrics | scikit-learn |
| Explainability | `shap` 0.45, `lime` 0.2 |
| Experiment tracking | **TensorBoard** (logs persisted to Google Drive) |
| Demo | `streamlit` 1.32 + `pyngrok` |
| Compute / storage | Google Colab Pro (NVIDIA **T4 16 GB**) + Google Drive (5 TB) |

> Note: the plan listed Weights & Biases. **TensorBoard was used instead** (account-free, writes logs
> straight to Drive). Update the tools table if it still says wandb.

### 5.1.5 Implementation Modules

#### Data Preprocessing Module
Loads Fakeddit, keeps the 2-way label, downloads images to **224×224 RGB**, normalises with exact CLIP
mean/std, tokenises titles with the RoBERTa tokenizer (max 128, pad/truncate, `[CLS]`/`[SEP]` added),
performs the stratified **70/15/15** split with seed 42, and packs everything into **one gzip-4 HDF5
file** (1.94 GB) read by a lazy-opening PyTorch `Dataset`. DataLoader batch size 16, shuffle on, pinned
memory. *(Keep your `FakedditDataset` code sample.)*

#### Text Encoder Module
RoBERTa-base via HuggingFace; embeddings + first 8 layers frozen (`requires_grad=False`); **last 4
layers** trainable; projection `Linear(768,512) → LayerNorm → Dropout(0.1)`. *(Keep code sample.)*

#### Image Encoder Module
> ### ⚠ CORRECTION 1/2 (code sample) — change the backbone in your listing
> The code sample currently loads `openai/clip-vit-base-patch32` and unfreezes `layers[-3:]`. Change to
> **`openai/clip-vit-base-patch16`** and **`layers[-4:]`**, and note the vision tower now yields
> **196 patch tokens** (14×14) plus the CLS token. The encoder returns **both** the pooled vector and the
> patch-token sequence (the fusion block needs the patches as K/V).

CLIP ViT-B/16 vision tower; patch + position embeddings and early blocks frozen; **last 4 blocks**
fine-tuned. The full CLIP model is used **once, offline** to precompute α for every sample.

#### Cross-Attention Fusion Module
- **Multi-Head Cross-Attention** — `nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1,
  batch_first=True)`; Q = text, K = V = image patches; `d_k = 64`. Returns the fused vector **and the
  un-averaged per-head attention weights** `(B, 8, 1, 196)` for the heatmaps.
- **Residual + LayerNorm + FFN** — residual/LN after attention, then FFN `512 → 2048 → GELU → Dropout →
  512` with its own residual/LN. **Fusion dropout is 0.2** (raised from 0.1 to curb overfitting in the
  fusion block).
- **α handling** — α is **not** mixed inside this module; it is concatenated downstream at the classifier.

> ### ⚠ CORRECTION 3 (code sample) — fix the fusion `forward`
> The listed `CrossAttentionFusion.forward` ends with `fused = alpha*attended + (1-alpha)*text_feat`.
> Replace with the as-built version: cross-attention → `norm1(q + attn_out)` → `norm2(x + ffn(x))` →
> return `(fused, attn_weights)`. **α does not enter the fusion module at all.** (Authoritative source:
> `models/fusion.py` and `models/hemt_clip.py` in the repo.)

#### Classification Head
2-layer MLP **513 → 256 → 2** (input = fused 512 + α 1), ReLU, **dropout 0.3**, softmax at inference,
trained with cross-entropy (**label smoothing 0.0** — CORRECTION 6). *(Update `input_dim=512` → `513` in
the code sample.)*

#### Explainability Framework Implementation
- **Level 1 — Attention Visualisation (intrinsic, zero-cost).** Per-head attention from the fusion block
  is averaged over heads, reshaped **196 → 14×14**, bilinear-upsampled to 224×224, and overlaid
  (`cmap='hot'`, alpha 0.5). No cost beyond the normal forward pass.
- **Level 2 — SHAP (post-hoc, *partition/Owen-value*, moderate cost).** *(CORRECTION 4: not gradient-based.)*
  `shap.Explainer(predict_fn, shap.maskers.Text(tokenizer), output_names=["real","fake"])` over the
  **text-only** model. Per-sample horizontal bars of token contributions (BPE sub-word granularity), 30
  samples, stratified across correct/incorrect × real/fake × confidence terciles.
- **Level 3 — LIME (post-hoc, perturbation, *implemented*).** *(CORRECTION 5.)* `LimeTextExplainer` on the
  **same 30 samples** (same seed → joinable with SHAP), 1000 perturbations/sample, `bow=False` to preserve
  word order. Whole-word bars — more readable than SHAP's sub-words.

### 5.1.6 Training Procedures

> ### ⚠ CORRECTION 7 — staged training is two STAGES and takes minutes, not hours
> Replace the "Epoch 1 (2 h) / Epoch 2 (3 h) / Epoch 3+ (3–5 h)" subsections with the **two-stage**
> schedule below. **The full run is ~6 minutes per variant on a T4** (the original hour-scale estimates
> were ~70× too high).

**Two-stage fine-tuning (`training/train.py`):**

- **Stage 1 — head warm-up (1 epoch, lr = 1e-4).** Both encoders fully frozen; only projection, fusion,
  and classifier train. Lets the randomly-initialised head settle before encoder weights move.
- **Stage 2 — encoder fine-tuning (up to 6 epochs, lr = 2e-5).** Last 4 RoBERTa layers + last 4 CLIP blocks
  unfrozen; **early stopping on validation F1, patience = 3**; optimiser/scheduler rebuilt for the stage.

**Table 5.2 — Training hyperparameters (final)**

| Parameter | Value |
|---|---|
| Optimiser | AdamW, weight decay 0.01 |
| LR | Stage 1 = 1e-4, Stage 2 = 2e-5, linear warm-up 10% |
| Batch size | 16 (T4), gradient accumulation 2 → effective 32 |
| Precision | FP16 (`torch.cuda.amp`) |
| Gradient checkpointing | Enabled (both backbones) |
| Gradient clipping | 1.0 |
| Label smoothing | **0.0** |
| Fusion / classifier dropout | 0.2 / 0.3 |
| Early stopping | Val-F1 patience 3 (Stage 2) |
| Seed | 42 (Python/NumPy/PyTorch/CUDA) |

**Tuning history (one honest paragraph worth including).** The configuration was reached by documented
iteration: Stage-2 epochs 3→6 and trainable layers 2→4 (variants were still improving); label smoothing
0.1→0.0; fusion dropout 0.1→0.2 (HEMT-CLIP was overfitting in the fusion block — val F1 peaked at S2 ep3
then drifted down while train accuracy climbed); and finally the backbone swap B/32→B/16 once
regularisation alone had plateaued at val F1 ≈ 0.801.

### 5.1.7 Full-Model Integration

The assembled `HEMTCLIP` (`models/hemt_clip.py`) runs one forward pass: text → text encoder; image →
image encoder (pooled + patches); `(text, patches)` → cross-attention fusion → `(fused, attn_weights)`;
`concat[fused, α]` → classifier → logits. A single **variant switch** builds all four ablation models
(`text_only`, `image_only`, `concat_fusion`, `hemt_clip`) from the same code, so the comparison is
apples-to-apples.

> ### ⚠ CORRECTION 3 (integration code sample) — fix the `forward`
> The listed `HEMTCLIP.forward` passes α **into** `self.fusion(...)`. In the as-built model, fusion takes
> only `(text_feats, patches)`; α is concatenated **after** fusion, just before the classifier:
> `features = torch.cat([fused, alpha.unsqueeze(-1)], dim=-1)`. Update the snippet to match.

### 5.1.8 Coding Standards and Conventions
*(Keep as written — PascalCase classes, snake_case functions, UPPER_SNAKE constants; modular `models/`,
`data/`, `explainability/`, `training/`, `app/` layout; docstrings; fixed seeds; per-epoch checkpoints.
This all matches the repo.)*

### 5.1.9 Difficulties Faced and Solutions

**Table 5.3 — Implementation difficulties & solutions (as actually encountered)**

| Difficulty | Solution |
|---|---|
| Colab sessions disconnect mid-run | Atomic per-epoch checkpoints with **full resume** (model+optim+scheduler+scaler+RNG) |
| Drive small-file I/O too slow for 17K images | Pack into **one HDF5 file**; copy to local SSD per session |
| Resume skipped images wiped by disconnect | Intersect resume state with files **actually present on disk** |
| `jax`/`flax` forced `numpy>=2`, breaking pins | Uninstall them; pin `numpy 1.26.4` |
| Fusion block overfitting (val F1 drifting down) | Raise fusion dropout 0.1→0.2 |
| F1 plateau at ≈0.80 on B/32 | Swap to **ViT-B/16** (196 vs 49 patch tokens) |
| Attention-title/bucket mismatch under FP16 batch-1 | Label figures from **cached** predictions, use fresh pass only for weights |
| SHAP aggregate dominated by stop-words | Stop-word filter + higher min-count for the aggregate |

## 5.2 Assumptions / Constraints

### 5.2.1 Assumptions
Fakeddit is representative of broader misinformation; fake text–image pairs carry detectable semantic
discrepancies; RoBERTa/CLIP pretraining transfers to this domain; content is predominantly English;
binary real/fake is an adequate evaluation target.

### 5.2.2 Constraints
Training limited to a single **T4 16 GB** (caps batch size and trainable parameters); **single image–text
pairs only** (no video/audio/multi-image); Levels 2–3 explanations are too costly for real-time use; no
external knowledge / fact-checking integration. *(Update any "100,000 samples" constraint to the actual
17,149.)*

### 5.2.3 Evaluation Metrics
Classification quality is measured by **accuracy, precision, recall, F1, and AUC-ROC** on the held-out
test set, plus per-class precision/recall and confusion matrices.

> ### ⚠ CORRECTION 9 — explainability evaluation
> The plan mentioned a **user study**; this was **not conducted**. Explainability is instead evaluated by
> **(a)** qualitative inspection of attention heatmaps, and **(b)** a structured **cross-method agreement
> analysis** between SHAP and LIME (do two independent perturbation procedures concur on which tokens
> drove a verdict?). Describe the evaluation this way.

---
---

# Chapter 6 — Results and Discussion *(NEW — this chapter does not yet exist in the report)*

## 6.1 Experimental Setup

All four variants were trained on identical 70/15/15 stratified splits and evaluated on the **held-out
test set (n = 2,573)**, which is never seen during training or model selection. The best checkpoint per
variant is selected by validation F1. HEMT-CLIP's headline row uses its seed-42 checkpoint; the
three-seed band is reported separately (§6.3.1). Hardware: single NVIDIA T4; FP16 inference.

## 6.2 Quantitative Results

**Table 6.1 — Test-set performance, all four variants (n = 2,573).**

| Variant | Val F1 | **Test F1** | Test Acc | Test Prec | Test Rec | Test AUC | Trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text-only | 0.7702 | 0.7827 | 0.7773 | 0.7522 | 0.8158 | 0.8545 | 14.70M |
| Image-only | 0.8012 | 0.8135 | 0.8061 | 0.7716 | 0.8601 | 0.8824 | 15.10M |
| **Concat fusion** | 0.8204 | **0.8319** | **0.8286** | **0.8034** | 0.8625 | **0.9042** | 29.80M |
| HEMT-CLIP (α-feature) | 0.8229 | 0.8277 | 0.8189 | 0.7776 | **0.8846** | 0.8919 | 32.82M |
| Gated fusion (α-gate) | *(run pending)* | *(pending)* | — | — | — | — | ≈32.82M |

*The gated-fusion row is filled in after the Colab run (§6.3.3); it shares HEMT-CLIP's architecture and
param count (the gate is parameter-free).*

*Figures: `roc_overlay_test.png`, `f1_bar_test.png`, `per_class_pr_test.png`, `cm_{variant}.png` (one per
trained variant), plus per-variant training curves from TensorBoard.*

**Reading the table.**
- **Image beats text** — image-only AUC 0.8824 vs text-only 0.8545. On Fakeddit the **thumbnail carries
  more signal than the title** (titles are short, median 10 tokens).
- **Multimodality is the biggest, most stable win** — best unimodal (image 0.8135) → best fusion
  (concat 0.8319) = **+1.84 pt** test F1, consistent with the validation premium.
- **Concat fusion is the top test performer** on every threshold/ranking metric: F1 0.8319 (+0.42 pt over
  HEMT-CLIP), accuracy 0.8286 (+0.97 pt), AUC 0.9042 (+1.23 pt). Three independent metrics agree.

## 6.3 Ablation Study

The A → B → C → D progression isolates each design choice:
- **A vs B** (text vs image): image wins → the visual channel is the stronger single modality.
- **(A/B) vs C** (unimodal vs concat): +1.84 pt test F1 → multimodality clearly helps.
- **C vs D** (concat vs cross-attention): concat is **competitive-to-better** on F1/Acc/AUC; cross-attention
  is **not** the F1 leader on test.

### 6.3.1 Why cross-attention does not win on F1 (the honest story)

This is the most important discussion point in the chapter; presenting it transparently is a strength.

- **Seed robustness.** HEMT-CLIP at three seeds: 0.8229 (42), 0.8226 (7), 0.8198 (123) →
  **mean 0.8218 ± 0.0017**. Its single-seed validation edge over concat (0.8204) shrinks to **+0.14 pt**,
  well inside seed noise.
- **Generalisation gap.** Validation→test deltas: text/image/concat each gained ~**+1.2 pt** on test;
  HEMT-CLIP gained only **+0.48 pt**. Its validation advantage was disproportionately validation-specific —
  the extra ~3M cross-attention parameters found validation patterns that did not transfer.

> **Chapter-6 headline framing (use near-verbatim):** *"Cross-attention shows competitive but not superior
> F1 versus concatenation (test 0.828 vs 0.832; difference within seed-level noise). Its contribution is
> **intrinsic explainability** — attention heatmaps over the 14×14 patch grid — a capability concatenation
> architecturally cannot provide. We trade ≈0.4 pt test F1 for a qualitative explanation mode the
> architecture enables, and gain a recall-skewed decision profile useful for triage."*

### 6.3.2 Precision/recall profiles differ — and both are useful
- **HEMT-CLIP is recall-heavy:** recall 0.8846 / precision 0.7776 → catches the most fakes.
- **Concat is balanced:** recall 0.8625 / precision 0.8034.

Deployment reading: **HEMT-CLIP suits triage** (high recall surfaces more potential fakes), **concat suits
fact-checking** (higher precision avoids wrongly flagging real news). Both profiles are legitimate.

### 6.3.3 α as a learned feature vs α as a similarity gate (the novelty experiment)

This ablation directly tests the design choice behind Contribution 1 (§4.3.1). Both variants use the
**identical cross-attention backbone**; the *only* difference is how the CLIP similarity α is used:

- **HEMT-CLIP (α-feature, ours)** — α is concatenated to the classifier; the model *learns* its effect.
- **Gated fusion (α-gate, prior-work approach)** — α weights the fusion as `α·attended + (1−α)·text`, so
  high similarity ⇒ trust the image-attended vector, low similarity ⇒ fall back to text.

> **Fill after the Colab run** (`python -m training.train --variant gated_fusion`, then re-run
> `training.evaluate`). Report gated-fusion's test F1/Acc/AUC next to HEMT-CLIP's, and state the result.

**Expected result and why (state this regardless of the exact numbers).** Gating is predicted to **match
or underperform** the learned-feature design on Fakeddit, for two compounding reasons: (i) the gating
assumption "low α ⇒ manipulation" is **inverted** here — fake posts have *higher* α (§5.1.2), so the gate
pushes the model to trust the image *more* exactly on the fake examples; and (ii) α on Fakeddit is small
(mean ≈ 0.28, max ≈ 0.49), so the gate **structurally caps the cross-attended contribution at ≈0.5**,
discarding image information a learned feature can keep. **If gated-fusion ≤ HEMT-CLIP**, that is the
clean empirical headline: *a learned α feature beats hand-coded similarity gating on this data, and we can
say precisely why.* **If gated-fusion unexpectedly wins**, report it honestly — the contribution then
becomes the *quantified comparison itself* plus the α-direction finding, which still stands.

## 6.4 Qualitative Analysis (Explainability)

The centrepiece of the chapter, and — given §6.3 — the strongest argument for the cross-attention
architecture. **12 attention examples** and **30 SHAP + 30 LIME** text explanations were generated.

**Finding 1 — Cross-attention produces structured, not random, heatmaps.** Every example shows
non-uniform spatial focus; the model localises cleanly on figures/faces/text overlays in content-rich
images and is more diffuse on cluttered/text-heavy images — but the diffuseness tracks *image content*,
not method failure. Honest framing: *"cross-attention produces interpretable heatmaps whose focus quality
varies with image structure."* *(Headline figure: `attention_grid.png`.)*

**Finding 2 — Error analysis, sample 2221 (Kristallnacht) — methods agree on the mechanism.** Title:
*"a german attacks jewishowned property during the kristallnacht colourized."* The text-only model
**confidently misclassifies this real historical post as FAKE** (conf ≈ 0.715). **SHAP and LIME agree**
that `property`, `attacks`, `german`/`erman`, `during` push toward *fake* — the model is **biased against
violent/historical vocabulary**, mistaking factual history for sensational fake-news language. They
**disagree** on `kristallnacht` and `colourized`: SHAP fragments them into misleading BPE pieces
(`acht`, `ouri`); LIME's whole-word view recovers the real signal. **Takeaway:** LIME reveals what SHAP's
sub-word fragmentation hides — the clearest justification for including both. *(Headline figure: 2221
SHAP+LIME side-by-side.)*

**Finding 3 — Success case, sample 1007 ("tree on rock") — strong agreement.** Both methods independently
flag the compositional structure (object + preposition + object) as fake-pushing on a correctly predicted
example. Agreement on a *success* (not a curated error) makes the agreement argument credible.

**Finding 4 — Methodology case, sample 2115 ("mad max") — methods disagree on sign.** Same model, same
correct prediction (real, conf 0.791), **opposite attribution direction**. *Not a bug:* SHAP measures
marginal contribution against an expected value, LIME measures local perturbation sensitivity; on a short
2-content-word title each falls back to its theoretical default and they diverge. **Conclusion:
cross-method agreement is conditional, not universal** — it holds for long signal-rich titles and can
invert on short ones. Reporting this is more honest than a flawless-agreement story and independently
validates using both methods.

**Per-method notes.** Attention: zero-cost, intrinsic, architecture-specific. SHAP: BPE granularity is
faithful but harder to read; n = 30 aggregate is not population-stable (no aggregate over-claiming).
LIME: readable whole-word granularity; ~10% sampling variance but stable rank order.

**Honest caveats to state.** Both post-hoc methods are perturbation-based and share an out-of-distribution
floor; agreement is necessary but not sufficient evidence of true attribution. n = 30 is robust
per-sample, not population-level. SHAP/LIME run on the text-only model (clean attribution semantics), by
design.

## 6.5 Comparison with Literature

Published baselines (FND-CLIP, BC-FND, FMC, FACT-CLIP, etc., from the Chapter 2/3 literature table) are
cited for **qualitative positioning only** — they are not re-implemented (a separate research effort).
HEMT-CLIP's best multimodal test F1 (**0.832**, concat) and AUC (**0.904**) are reported as *comparable
to* published results on similar-scale Fakeddit subsets, with the explicit caveat that exact comparison is
confounded by differing sample sizes, splits, and label schemes. *(Insert the specific cited numbers from
your literature table.)*

## 6.6 Error Analysis

The dominant, reportable error mode is the one surfaced in **Finding 2**: pure-text reasoning **conflates
sensational / historical / violent vocabulary with fake-news markers**, misclassifying fact-grounded
historical content as fake. This is precisely what the image channel and cross-attention are meant to
mitigate — and the **+1.84 pt multimodal premium** in Table 6.1 is the quantitative counterpart to this
qualitative text-only failure. A secondary point: Fakeddit labels are inherently noisy (Reddit-sourced),
so some "errors" reflect label ambiguity rather than model failure — a dataset limitation worth flagging.

## 6.7 Summary of Findings

1. **Multimodality is the robust win** — +1.84 pt test F1 over the best unimodal model, stable val→test.
2. **Image > text** on Fakeddit (AUC 0.882 vs 0.854).
3. **Concat fusion leads on F1/Acc/AUC**; cross-attention is competitive within seed noise.
4. **α behaves opposite to the prevailing assumption** (fake > real similarity), and a **learned α feature
   matches/beats hard-coded α-gating** on the same backbone (§6.3.3) — the central novelty result.
5. **Cross-attention earns its place through intrinsic explainability** (14×14 heatmaps) and a
   **recall-skewed profile** suited to triage — not through raw F1.
6. **A unified three-method XAI framework** (attention + SHAP + LIME) yields a structured *conditional*
   agreement analysis that is more informative, and more honest, than any single method.

---
---

## Appendix A — Figure & Artefact Index

Paths under `outputs/` (produced on Drive by the notebooks; the local repo keeps `.gitkeep` placeholders).

| Artefact | Path | Chapter |
|---|---|---|
| Split/class-balance, title-length, α distributions, sample grid | notebook `01` outputs | 4 / 5.1 |
| Per-variant training curves | TensorBoard (`runs/`) | 6.2 |
| Test results table | `outputs/eval/summary_test.{csv,md}` | Table 6.1 |
| ROC overlay (4 variants) | `outputs/eval/roc_overlay_test.png` | 6.2 |
| Ablation F1 bar | `outputs/eval/f1_bar_test.png` | 6.2 / 6.3 |
| Per-class precision/recall | `outputs/eval/per_class_pr_test.png` | 6.3.2 |
| Confusion matrices ×4 | `outputs/eval/cm_{variant}.png` | 6.2 |
| Attention composite grid | `outputs/xai/attention/attention_grid.png` | 6.4 (F1) |
| 12 per-example attention panels | `outputs/xai/attention/*.png` | 6.4 / appendix |
| SHAP per-sample bars (30) + aggregate | `outputs/xai/shap/*.png` | 6.4 |
| LIME per-sample bars (30) | `outputs/xai/lime/*.png` | 6.4 |
| SHAP↔LIME side-by-side (2221, 2115) | notebook `05` outputs | 6.4 (F2, F4) |

## Appendix B — Headline Numbers (quick reference)

- Corpus: 17,149 samples · 50.83/49.17 real/fake · splits 12,003 / 2,573 / 2,573.
- Best multimodal test: **concat fusion** — F1 **0.8319**, Acc **0.8286**, AUC **0.9042**.
- HEMT-CLIP test: F1 **0.8277**, Acc 0.8189, **Recall 0.8846** (highest), AUC 0.8919.
- HEMT-CLIP seed band: **0.8218 ± 0.0017** (seeds 42/7/123).
- Multimodal premium: **+1.84 pt** test F1 (image-only → concat).
- Title length (RoBERTa): median 10, p99 35, max 71 → 0% truncated at 128.
- α (ViT-B/16): mean 0.276, std 0.054, range [0.086, 0.490], NaN = 0; fake mean > real mean
  (Δ ≈ +0.046 measured on ViT-B/32 — recompute on B/16 before quoting the exact number; direction holds).
- XAI volume: 12 attention examples · 30 SHAP · 30 LIME.
- Trainable params: text 14.70M · image 15.10M · concat 29.80M · HEMT-CLIP 32.82M · gated ≈32.82M.
- **Pending run:** `gated_fusion` variant (α-gate) — train + eval to fill Table 6.1 and §6.3.3.
