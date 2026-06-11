# HEMT-CLIP — Project Defense Brief

**XAI for Multimodal Fake-News Detection — Final Year Project, UMT Sialkot (2022–2026)**
Team: Syed Taha Faiz-ul-Hassan Rizvi · Ayesha Bukhari · Ali Mosa Raza · Advisor: Ma'am Hina Tufail

> Purpose of this document: a single end-to-end narrative — *data → method → results → explainability → limitations → future work* — with the exact numbers, the **reason behind every decision**, and a ready answer to every hard question. Use it to establish the work and defend it. (A shorter, plain-language version lives in `Supervisor_Talk_Brief.md`.)

---

## 0. One-paragraph summary

HEMT-CLIP takes a social-media post (a **title + an image**) and classifies it as **Real or Fake**, while **explaining itself** — it shows which image regions and which words drove the decision. It combines a language model (RoBERTa) with a vision–language model (CLIP) through a **cross-attention fusion gated by the CLIP text–image similarity α**. On the held-out Fakeddit test set (2,573 unseen posts) the gated model reaches **F1 = 0.839 / AUC = 0.912**, the best of five ablation variants. The two defensible contributions are (1) a **controlled demonstration that *gating* the fusion with α beats both plain concatenation and using α as a raw feature**, and (2) a **unified explainability framework** that pairs an intrinsic image-side method (cross-attention heatmaps) with two post-hoc text-side methods (SHAP + LIME) and reasons about their agreement.

---

## 1. Problem & motivation

- **The problem.** Misinformation on social media is overwhelmingly **multimodal** — a misleading post is a caption *plus* a picture, and the deception often lives in the *mismatch* between them. Text-only or image-only detectors miss this by construction.
- **Why explainability (XAI).** A "fake/real" verdict with no justification is not trustworthy or auditable. For a moderation or fact-checking setting, *why* matters as much as *what*. We made explainability a first-class deliverable, not an afterthought.
- **Scope, honestly stated.** This is an undergraduate FYP: the goal is a **rigorous, reproducible, well-understood system** with a clear, tested design choice — not a new state-of-the-art leaderboard entry.

---

## 2. Data — selection and justification

### 2.1 Why Fakeddit
We needed a dataset that was **(a) genuinely multimodal** (paired text + image), **(b) real social-media content** (not synthetic or GAN-generated), **(c) large and publicly available**, and **(d) cleanly labelled**. **Fakeddit** (Yang et al., 2020) fits all four: ~1M Reddit posts with image + title and a binary `2_way_label` (0 = real, 1 = fake). Alternatives we considered were weaker on at least one axis — e.g. text-only or rumour-graph datasets (not multimodal), or small/curated sets (not representative of messy real posts).

**We used the binary (2-way) label** rather than the 3-/6-way taxonomy because the core scientific question — *does multimodal alignment help detect authenticity?* — is cleanest as a binary task, and the binary split keeps the classes balanced and the evaluation interpretable.

### 2.2 From raw metadata to a packed dataset
1. Sampled **18,000 rows stratified by label** from the Fakeddit metadata.
2. Ran a parallel, resumable downloader (`data/download_fakeddit.py`) that fetches each image, resizes to **224×224** in flight, and writes atomically. Posts whose image URL was dead/broken were dropped — **17,149 survived (~5% attrition)**, in line with Fakeddit's known URL freshness. (We deliberately over-sampled to absorb this loss.)
3. Packed the survivors into a single **gzip-compressed, chunked HDF5** (`data/build_hdf5.py`) — one row per post, with a **70/15/15 stratified split** baked in (`seed=42`).
4. Pre-computed the CLIP similarity **α** once per row and stored it in the file.

### 2.3 What the data looks like (and why it's fit for purpose)
- **Size & splits:** 17,149 posts → **12,003 train / 2,573 validation / 2,573 test**. Val and test are each large enough for stable F1 (±~1 pt at this scale).
- **Class balance:** **50.83% real / 49.17% fake**, preserved across all splits by stratification. *Consequence:* no class weighting needed — plain cross-entropy is appropriate, and **accuracy is a meaningful metric** (not inflated by imbalance).
- **Titles are short** (median 10 tokens, p99 35, max 71). We cap at `max_text_len = 128`, which truncates **0%** of titles — generous headroom, cost is only cheap pad tokens.
- **Images** are uniform 224×224 RGB.

### 2.4 The dataset observation that drives the whole design — the direction of α
For each post we compute **α = cosine similarity between the CLIP text embedding and the CLIP image embedding** — one number for "how well do the caption and picture match in CLIP's shared space?" On our data α ranges ≈ **0.09–0.49 (mean ≈ 0.28)**, and crucially:

> **Fake posts have a *higher* mean α than real ones.**

This is the **opposite** of the assumption baked into prior CLIP-based detectors ("low similarity ⇒ manipulation"). It makes sense for Fakeddit: satire and mislabeled posts typically pair a *dramatic caption with on-topic imagery* (high alignment), whereas real news often pairs a *generic stock photo with a specific headline* (lower alignment). **This finding is exactly why we don't hard-code a rule about α — we test, empirically, how it should be used.** That test became our headline result (§6).

---

## 3. Method — architecture and the reason for each part

**Design philosophy:** two strong, pre-trained "specialists" (one for text, one for vision), fused so the **text can interrogate the image**, with the fusion **modulated by how well the two modalities agree**.

```
Title  ─► RoBERTa-base ───────────────► text vector (512-d)
                                                 │
Image  ─► CLIP ViT-B/16 vision tower ─► 196 image-patch tokens (512-d each)
                                                 │
              text [CLS] "queries" patches ─► 8-head Cross-Attention Fusion
                                                 │  (residual + LayerNorm + FFN)
              α gates the blend  ─►  fused = α·attended + (1−α)·text
                                                 │
                                                 ▼
                          Classifier: Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→2)
                                                 │
                                                 ▼
                                          P(Real), P(Fake)
```

| Component | Choice | Why this choice |
|---|---|---|
| **Text encoder** | **RoBERTa-base**, last **4** layers fine-tuned, projected 768→512 | A robust, widely-used LM pre-trained on large web text — strong on informal social-media language. Fine-tuning only the top 4 layers preserves pre-trained knowledge and saves compute. |
| **Image encoder** | **CLIP ViT-B/16** vision tower, last **4** blocks fine-tuned, per-patch projection 768→512 | CLIP is pre-trained on 400M image–text pairs, so its image features are already **language-aligned** — ideal for a text+image task. B/16 yields a **14×14 = 196-patch grid**, fine enough to localize small regions in the attention heatmaps. |
| **Similarity α** | **Full CLIP**, cosine of the two embeddings, computed offline and cached | Injects CLIP's cross-modal judgement of "do these match?" as a single scalar — **no trainable parameters**. |
| **Fusion** | **8-head multi-head cross-attention** (text [CLS] = query; image patches = keys/values) + residual + LayerNorm + position-wise FFN | Lets the text *ask questions of the image* — the model can check whether the picture supports or contradicts the claim, rather than just concatenating two summaries. |
| **The gate (our contribution)** | `fused = α·attended + (1−α)·text` | A reliability knob on the image channel: high α lets the image-attended evidence in, low α falls back to text. **Parameter-free** — the gate value comes from CLIP, not a learned layer. |
| **Classifier** | 2-layer MLP (512→256→2), ReLU, Dropout 0.3 | A small head mapping the fused representation to Real/Fake. |

**What the gate actually blends.** After cross-attention with a residual connection, `attended ≈ text + image-evidence`. So `α·attended + (1−α)·text` is a weighted average between *"text + image"* and *"text alone"* — i.e. **α controls how strongly the image is allowed to influence the verdict**. The model never treats α as a fake-detector; the trained classifier learns what to do with the gated vector.

### 3.1 The five variants — and what each one isolates (ablation design)
The ablation is designed so each comparison changes exactly one thing:

| Variant | Modalities | Fusion mechanism | Isolates |
|---|---|---|---|
| `text_only` | text | — | text signal alone |
| `image_only` | image | — | image signal alone |
| `concat_fusion` | text + image | concatenate `[text, image, α]` | does fusion help at all? |
| `hemt_clip` | text + image | cross-attention; **α as a concatenated feature** | does cross-attention + α-as-feature beat concat? |
| **`gated_fusion`** (headline **HEMT-CLIP**) | text + image | cross-attention; **α as a gate** | **does *gating* with α beat using it as a feature?** |

`hemt_clip` vs `gated_fusion` is the key comparison: **same cross-attention backbone, same parameter count — the only difference is whether α gates the fusion or is appended as a feature.** That isolates the gating mechanism itself.

---

## 4. Training methodology

**Two-stage fine-tuning** (Blueprint §8.1):
- **Stage 1** (1 epoch, lr 1e-4): freeze both encoders, **warm up only the new heads** (projections + fusion + classifier, ~4.47 M params). This stabilises the randomly-initialised parts before touching the pre-trained weights.
- **Stage 2** (≤6 epochs, lr 2e-5): unfreeze the **last 4 layers/blocks** of each encoder (~61 M trainable), gently fine-tune, and **early-stop on validation F1 (patience = 3)**.

*Why two stages:* fine-tuning everything from step one would let large gradients from the untrained head corrupt the pre-trained encoders. Warm-up-then-fine-tune is standard practice and demonstrably avoids that.

**Other settings & why:** `label_smoothing = 0.0` (clean binary labels — smoothing only capped val-loss with no benefit); mixed-precision (fp16) + gradient checkpointing to fit a T4's 16 GB; batch size 16; Adam; `seed = 42`. All variants share splits, seed, and schedule, so comparisons are fair.

**Reproducibility & engineering.** Everything is config-driven (`configs/base.yaml`); runs write per-epoch checkpoints plus a best-on-val-F1 `best.pt`, with **full resume support** (optimiser/scheduler/scaler/RNG restored) — necessary because Colab sessions disconnect. TensorBoard logs every scalar and weight/grad histogram.

**The tuning was principled, not random.** The headline configuration came from a small, motivated search: CLIP **B/32 → B/16** (finer patch grid for the attention to use), encoder **last-2 → last-4** layers (more adaptation capacity), and Stage-2 **3 → 6** epochs (variants were still improving). Each change was made for a stated reason and validated on the validation set; the test set was touched **only once**, at the end.

---

## 5. Evaluation protocol

- **Held-out discipline:** all model selection (early stopping, the tuning above) used **validation** only; the **2,573-post test split was scored once**, at the end. This is the single most important guard against over-optimistic numbers.
- **Metrics & why each:**
  - **F1 (binary)** — primary; balances precision and recall on the "fake" class.
  - **AUC-ROC** — **threshold-independent** ranking quality; our most robust single number because it doesn't depend on the 0.5 cut-off.
  - **Accuracy** — meaningful here *because the classes are balanced*.
  - **Precision / recall** — expose the decision profile (false-alarm vs miss trade-off), which matters for deployment framing.

---

## 6. Results — what we found and what it establishes

### 6.1 Validation (model selection)
| Variant | Best val F1 |
|---|---:|
| `text_only` | 0.7702 |
| `image_only` | 0.8012 |
| `concat_fusion` | 0.8204 |
| `hemt_clip` (α feature) | 0.8229 |
| **`gated_fusion` (α gate)** | **0.8464** |

### 6.2 Held-out test (n = 2,573) — the numbers that count
| Variant | Test F1 | Test Acc | Test AUC | Test Recall | Test Precision |
|---|---:|---:|---:|---:|---:|
| `text_only` | 0.7827 | — | 0.8545 | — | — |
| `image_only` | 0.8135 | — | 0.8824 | — | — |
| `concat_fusion` | 0.8319 | 0.8286 | 0.9042 | 0.8625 | **0.8034** |
| `hemt_clip` (α feature) | 0.8277 | 0.8189 | 0.8919 | 0.8846 | 0.7776 |
| **`gated_fusion` (α gate) — HEMT-CLIP** | **0.8393** | **0.8313** | **0.9122** | **0.8957** | 0.7895 |

### 6.3 Seed robustness
The α-feature `hemt_clip` variant trained at three seeds (42, 7, 123) gives **0.8218 ± 0.0017** val F1 — i.e. the architecture is **stable across seeds (±0.002)**, so the differences between variants are not seed artefacts.

### 6.4 What the results establish (four claims, weakest-to-strongest evidence)
1. **Multimodal beats unimodal — large and consistent.** `image_only` 0.8135 → `concat_fusion` 0.8319 test F1 (**+1.84 pt**); both fusion variants beat both single-modality baselines on every metric, on both val and test. *This is the most solid result in the study.*
2. **Image carries more signal than the title on Fakeddit.** `image_only` AUC 0.882 > `text_only` 0.854 — short Reddit headlines are weak on their own.
3. **How α is used matters — gating wins.** On the same backbone, the α-**gate** beats the α-**feature** by **+1.2 pt F1 / +2.0 pt AUC**, and beats plain concatenation by **+0.7 pt F1 / +0.8 pt AUC**. (Defense and caveats in §7.)
4. **The validation ranking transfers to test.** Every variant's test F1 is within ~1 pt of its val F1, and `gated_fusion` keeps the lead (val 0.846 → test 0.839) — no sign of validation overfitting.

---

## 7. The headline finding — establish and defend

**Claim.** Using the CLIP similarity **as a gate** on a cross-attention fusion is the best of the five designs (test F1 0.839 / AUC 0.912), beating both plain concatenation and using α as a feature.

**Why it's interesting — the honest twist.** We initially expected gating might *hurt*, because on this data **fake posts have higher α** (§2.4), so a naïve "trust the image when α is high" gate would trust the image *more* on fakes. **It won anyway.** The reason: the model never uses α as a fake-detector. The gate is just a *soft reliability blend* between "text+image" and "text alone"; the trained classifier learns the rest. The smooth α-blend appears to **regularise** the fusion better than appending a raw α scalar — and it does so with **zero extra parameters**. The lesson is "**learn how to use α rather than hard-coding a rule about it**," and the ablation is the evidence.

**How to defend the margin (the question most likely to be pressed):**
- The gate wins on **four metrics at once** (F1, accuracy, AUC, recall) — not a single cherry-picked number.
- The **AUC margin (+0.8 pt over concat) is threshold-independent**, so it's the most reassuring single figure.
- We are transparent that `gated_fusion` is a **single-seed** result and that the F1 margins are small; we **do not** claim statistical significance. The seed-robustness study (§6.3) shows the architecture's seed variance is ~±0.002, which makes a ~0.7 pt gap *plausibly* real, but a multi-seed run on the gated variant would be needed to assert it formally — see future work.

---

## 8. Explainability — three lenses

We deliberately span two axes: **intrinsic ↔ post-hoc** and **image ↔ text**.

| Method | Type | Modality | Granularity | Available on |
|---|---|---|---|---|
| Cross-attention heatmaps | **Intrinsic** | Image | 14×14 patch grid → 224×224 | only the cross-attention model |
| SHAP (Owen partition) | Post-hoc | Text | BPE subword | any text classifier |
| LIME (local linear surrogate) | Post-hoc | Text | whole word | any text classifier |

**Why each:**
- **Cross-attention heatmaps** are *free and intrinsic* — they come straight from the fusion weights, showing **where on the image** the text-conditioned model looked. A concatenation model **physically cannot produce these**, so explainability is a genuine *capability* of our architecture, not a bolt-on.
- **SHAP and LIME on the *text-only* model.** We attribute on `text_only` (not the multimodal text branch) so the answer to "which words drove the verdict?" is unambiguous — the model's decision is a function of text alone. We run **both** methods so we can argue token attribution by **agreement between two independent procedures**, not by trusting one.

**Three findings worth presenting:**
1. **A defensible error (sample 2221, "Kristallnacht").** A *real* colorized historical photo predicted *fake* (conf 0.715). **SHAP and LIME independently agree** the drivers are violent/historical vocabulary (`property`, `attacks`, `german`, `during`). The model has a **vocabulary bias against sensational/historical language** — a concrete, defensible insight (and a fairness point: this is the kind of bias XAI exists to surface).
2. **A success (sample 1007, "the way this tree grew on top of a rock").** Correctly *fake*; both methods flag the whole compositional structure (object + on top of + object) rather than any single word — the model learned a Photoshop-battle *template*.
3. **A principled disagreement (sample 2115, "mad max").** SHAP and LIME assign **opposite signs** to the same words. This is **not a bug**: SHAP measures *marginal contribution* against a baseline, LIME measures *perturbation sensitivity*. The disagreement is informative and turns into a methodology point rather than an embarrassment — cross-method agreement is conditional, and we show exactly when it holds.

---

## 9. Limitations (state them before the panel does)

- **Single seed for the headline `gated_fusion` model.** The F1 margins between fusion variants are small; we report them honestly and lean on AUC and multi-metric agreement. *(The most fixable gap — see future work.)*
- **Subset of Fakeddit (17K of ~1M), binary label only.** Chosen for compute and clarity; it limits direct comparison to published Fakeddit numbers that use far more data and/or the 6-way label.
- **Both post-hoc methods are perturbation-based**, so they share a theoretical floor (deleting words creates inputs the model never trained on). A gradient-based method would be a genuinely independent third perspective.
- **n = 30 for the XAI sample.** Robust for per-sample analysis; we explicitly *do not* over-claim population-level token trends (the SHAP aggregate was unstable at n=30 and is dropped from the report).
- **Scope:** single image–text pairs, English, Reddit-specific. Not video, not multi-image, not cross-platform.

---

## 10. Future work

- **Multi-seed the headline model** (gated_fusion at seeds 7/123) to report mean ± std and settle significance — ~30 min of compute.
- **Scale the data** toward the full Fakeddit corpus and add the **6-way** label.
- **A gradient-based attribution** (integrated gradients / attention rollout) as an independent explainability axis.
- **Mitigate the vocabulary bias** surfaced by XAI (e.g. counterfactual augmentation on sensational-but-real posts).
- **Calibration & deployment study** — turn the recall-skewed profile into a tunable operating point for a triage vs fact-check setting.

---

## 11. Contributions (the recap line)

1. A **CLIP-guided, α-gated cross-attention** multimodal detector that is the **best of five variants on held-out test** (F1 0.839 / AUC 0.912).
2. A **controlled ablation** isolating the design choice — *gating* α beats both concatenation and α-as-feature on the same backbone — motivated by a real dataset observation (fake posts have *higher* text–image alignment).
3. A **unified explainability framework** (intrinsic attention + SHAP + LIME) with cross-method analysis, including a concrete, defensible model-bias finding.
4. A **reproducible, config-driven, resumable pipeline** with a live demo notebook.

---

## 12. Anticipated panel questions & answers

**"What is genuinely new here?"**
→ The **gated** use of CLIP similarity in a cross-attention fusion, *proven best by a same-backbone ablation*, plus a three-method explainability framework. We don't claim cross-attention or CLIP fusion are new; we claim our controlled comparison and XAI integration are our contribution.

**"Is the gate's advantage statistically significant?"**
→ We don't claim significance — honestly. It wins on four metrics simultaneously and on the threshold-independent AUC (+0.8 pt). Our seed study shows ~±0.002 architecture variance, which makes the ~0.7 pt gap plausibly real; a multi-seed run on the gated variant is the clean way to confirm it, and it's our top future-work item.

**"Why does the gate win if fake posts have *higher* α — isn't your gate backwards?"**
→ Because the model doesn't treat α as a fake signal. The gate is a soft reliability blend, and the classifier *learns* how to use the gated vector. The smooth blend regularises the fusion better than a raw α feature. That's precisely why we **learn** to use α rather than hard-code "low α = fake."

**"Why only ~17K posts and the binary label, not all of Fakeddit / 6-way?"**
→ Compute (a single free T4) and scientific clarity — the binary task most directly tests "does multimodal alignment help authenticity detection?" with balanced classes and interpretable metrics. Scaling up is explicit future work.

**"0.84 F1 isn't state-of-the-art — is it good enough?"**
→ For an undergraduate FYP, yes: our deliverable is a rigorous, explained, reproducible system and a tested design choice, not a leaderboard entry. The number is in a reasonable range for a Fakeddit subset, and our contribution is the *comparison and explainability*, which higher-F1 black boxes don't provide.

**"How do you know you didn't overfit the test set?"**
→ The test split was scored exactly once; all tuning used validation. The val→test gap is ~1 pt for every variant (§6.4), which is the signature of healthy generalisation, not overfitting.

**"The Kristallnacht example shows a real bias — isn't that a problem?"**
→ It is a real model bias against sensational/historical vocabulary, and **surfacing it is exactly the point of building in explainability.** We report it openly and propose a mitigation (counterfactual augmentation) in future work. A black-box detector would hide this.

**"Why two large models instead of one multimodal model?"**
→ Text and images need different specialists. RoBERTa is a strong language model; CLIP understands images *in a language-aligned space*. Using both, and letting text query the image, is the substance of multimodal detection.

**"What were the engineering challenges?"**
→ Fitting two transformers on a 16 GB T4 (mixed precision + gradient checkpointing), and Colab disconnects (we built atomic checkpointing with exact resume). The architecture itself was iterated with a small, motivated search (backbone, trainable depth, schedule).

**"Can you show a decision live?"**
→ Yes — `notebooks/06_demo.ipynb`, `analyze(title, image)`: prediction + confidence, the α value, and a live cross-attention heatmap, on a built-in test sample or a custom input.

---

## 13. Suggested presentation flow (~8–10 min)

1. **Pitch + problem** (§0–1) — multimodal misinformation; why text+image and why explainability. *(1 min)*
2. **Data & the α observation** (§2) — Fakeddit, the splits, and the *fake-posts-have-higher-α* insight that motivates the design. *(1.5 min)*
3. **Architecture** (§3 diagram) — two specialists, cross-attention, and the α-gate as the contribution. *(2 min)*
4. **Ablation & results** (§6 tables) — multimodal helps; the gate is best on held-out test. *(2 min)*
5. **Explainability** (§8) — the heatmap + the Kristallnacht bias finding (most memorable). *(1.5 min)*
6. **Limits + future work** (§9–10) — own the single-seed and scope caveats up front. *(1 min)*
7. **Live demo** (§12) — `analyze()` on a sample, then invite questions and lean on §12. *(remaining time)*

**Golden rule for the viva:** *state the modest margins and the single-seed caveat yourself, confidently.* Owning your limitations reads as mastery; getting caught by them reads as the opposite.
