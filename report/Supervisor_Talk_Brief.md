# HEMT-CLIP — Talk Brief for Supervisor Meeting

*A casual student–teacher walkthrough. Read it in your own words; the numbers are exact.*

---

## 1. The 30-second pitch

"We built a system that looks at a social-media post — **the title and the picture together** — and decides whether it's **real or fake** news. On top of the yes/no answer, it also **explains itself**: it shows which part of the image it focused on and which words in the title drove the decision. We trained and tested it on **Fakeddit**, a large multimodal dataset of Reddit posts."

---

## 2. The main architecture (in plain words)

Think of it as **two specialists and a referee**:

1. A **text specialist** reads the title.
2. An **image specialist** looks at the picture.
3. A **fusion step** lets the text "ask questions" of the image, and a **gate** decides how much to trust the image versus the text. The combined understanding goes to a small classifier that outputs **Real / Fake**.

The pipeline:

```
Title  ──► RoBERTa (text encoder)  ─────────────┐
                                                 │
Image  ──► CLIP ViT-B/16 (image encoder) ──► image patches
                                                 │
            text "queries" the image patches  ►  Cross-Attention Fusion
                                                 │
            CLIP similarity α gates the blend ►  fused = α·(image-attended) + (1−α)·text
                                                 │
                                                 ▼
                                   Classifier (512 → 256 → 2)  ──►  Real / Fake
```

---

## 3. The models used, and *why* each one

| Component | Model | Why this one |
|---|---|---|
| **Text** | **RoBERTa-base** | A strong, well-known language model pre-trained on huge web text — good at messy, informal social-media language. We only fine-tune its **last 4 layers** to save compute and avoid "forgetting" what it already knows. |
| **Image** | **CLIP ViT-B/16** | CLIP was trained on 400M image–text pairs, so its image understanding is already *aligned with language* — perfect for a text+image task. B/16 gives a fine **14×14 grid (196 patches)**, so the model can point to small regions. Last 4 blocks fine-tuned. |
| **Similarity α** | **Full CLIP** | We use CLIP a second way: to measure how well the title and image *match* (cosine similarity). This single number, **α**, tells the model how aligned the two modalities are. |
| **Fusion** | **8-head cross-attention** | The text acts as a "question" and attends over the image patches to find supporting/contradicting evidence — like how a human checks "does the picture back up this claim?" |
| **Classifier** | **2-layer MLP** | Small head that turns the fused understanding into Real/Fake probabilities. |

**The key idea / our contribution:** instead of just stapling text and image together, we **gate the fusion with α**:
`fused = α·(image-attended) + (1−α)·text`. When the image and text agree (high α), the model leans on the image evidence; when they don't, it falls back to the text. This **CLIP-guided gated cross-attention** is the heart of the system.

### 3.1 Deep-dive: how the α-gate works (in case they ask)

**What two things is it blending?** At the moment of gating, the fusion holds two vectors:
- **`text`** — the title on its own (RoBERTa's 512-dim output).
- **`attended`** — the cross-attention result: the text "queried" the image patches and pulled in the relevant image evidence. Because of a residual connection this vector is really **text + image-evidence**.

The gate is one line:

```
fused = α · attended  +  (1 − α) · text
```

So it is a **weighted average between "text + image" and "text alone."** The only thing that differs between those two is the image contribution — so **α controls how strongly the image evidence is let into the final decision.** *That* is what's being gated: the influence of the image channel.

**The extremes:**
- α = 1 → use the image evidence fully.
- α = 0 → ignore the image, decide on text alone.
- α = 0.3 (typical here) → `0.3·(text+image) + 0.7·(text)` → mostly text, image mixed in at 30% strength.

**What α is:** the **CLIP cosine similarity between the title and the image** — one number saying "do these two match in CLIP's shared space?" Computed **once per post, offline** (run full CLIP, take similarity of the two embeddings, cache it). Range on our data ≈ 0.09–0.49.

**Intuition — a reliability knob on the image:** *how much should I trust this image for this title?* If they align (high α) the image is probably relevant → let it in; if they don't (low α) the image might be misleading → lean on text. Like a student who uses a diagram only if it matches the question.

**How it differs from a "normal" gate** *(likely question)*: in an LSTM/GRU the gate value is produced by a **trained layer**. Here the gate value is **not learned** — it comes from **CLIP's external knowledge** (the similarity score). So it injects CLIP's cross-modal understanding for free, and it adds **zero trainable parameters** (just a weighted average). *What is learned* is everything around it — the attention and the classifier learn what to *do* with the gated vector; we hard-code no rule.

**The honest twist** *(your strongest point if pushed):* we expected gating might *hurt*, because on Fakeddit **fake posts have *higher* α** (satire pairs a dramatic caption with on-topic imagery), so the gate trusts the image *more* on fakes — seemingly backwards. **But it still won** (F1 0.839 vs 0.828 without the gate). Why? The model never treats α as a "fake detector"; it uses it as a reliability knob, and the trained classifier learns the rest. That's exactly why we **learn** to use α instead of hard-coding "low similarity = fake."

**One-sentence version:** *"α is the CLIP text–image similarity, used as a gate that controls how much image evidence enters the fusion — high similarity lets the image in, low falls back to text — and it's parameter-free because the gate value comes from CLIP, not a trained layer."*

---

## 4. Performance — and how to talk about it

We trained **5 versions** (an "ablation study" — turning parts on/off to see what each contributes) and tested them on the **held-out test set (2,573 posts the model never saw)**:

| Version | What it is | Test F1 | Test AUC |
|---|---|---:|---:|
| Text-only | title only | 0.783 | 0.854 |
| Image-only | picture only | 0.814 | 0.882 |
| Concat fusion | text + image stapled together | 0.832 | 0.904 |
| HEMT-CLIP (α as feature) | cross-attention, α just added on | 0.828 | 0.892 |
| **HEMT-CLIP (α-gate) — our model** | cross-attention + α-gate | **0.839** | **0.912** |

**What to say about it:**
- "**Combining text and image clearly helps** — both fusion models beat either single one."
- "**Image alone beats text alone** on this data — the thumbnail carries more signal than a short title."
- "**Our gated model is the best overall** — F1 0.839, AUC 0.912. It beats plain stapling, and importantly it beats the version that uses α as a plain feature. So it's specifically the **gate** that makes the difference."
- (F1 ≈ overall accuracy balanced across both classes; AUC ≈ how well it ranks real vs fake regardless of threshold. Both higher is better, max 1.0.)

---

## 5. The explainability part (your XAI angle — they'll like this)

The system gives **three kinds of explanation**:

1. **Attention heatmap (built-in, free):** overlays a "heat" map on the image showing *where the model looked*. This comes straight from the cross-attention — a plain stapling model **can't** do this. Live in under a second.
2. **SHAP (on the text):** shows which words pushed toward "fake" or "real," at the sub-word level.
3. **LIME (on the text):** same idea but at whole-word level, easier to read.

"We use SHAP and LIME together so we can **cross-check** the word explanations — when two independent methods agree, we trust the explanation more. One nice finding: on a real historical post about Kristallnacht, the model wrongly said *fake* because it's **biased against violent/historical vocabulary** — both SHAP and LIME agreed on that, which is exactly the kind of insight explainability is for."

---

## 6. Dataset & training (quick facts)

- **Dataset:** Fakeddit. We used **17,149 posts** (≈50/50 real/fake), split **70% train / 15% validation / 15% test**.
- **Hardware:** a single free-tier **T4 GPU** on Google Colab. Each model trains in **~6 minutes**.
- **Training trick:** **two stages** — first warm up the new layers with the big models frozen, then gently fine-tune the top layers. Keeps it fast and avoids overfitting.

---

## 7. Likely supervisor questions + short answers

**"What's actually new here?"**
→ "The CLIP-guided **gated** cross-attention fusion — using the text–image similarity to control the fusion — is our main contribution, and our ablation proves it's the best design. Plus a three-method explainability framework."

**"Why two big models instead of one?"**
→ "Text and images need different specialists. RoBERTa understands language; CLIP understands images *in a language-aligned way*. Combining them is the whole point of multimodal detection."

**"What is α exactly?"**
→ "A single number — how similar the title and image are in CLIP's space. We use it as a gate to decide how much to trust the image."

**"Interesting — does low similarity mean fake?"** *(good one to pre-empt)*
→ "Surprisingly, on this dataset **fake posts have *higher* similarity** than real ones — because mislabeled/satire posts pair a dramatic caption with on-topic imagery. So we *learn* how to use α rather than hard-coding 'low = fake.' That's why the learned gate works."

**"How good is it / is it usable?"**
→ "Best model: F1 0.839, AUC 0.912 — competitive with published work on similar Fakeddit subsets. Good enough to demo live."

**"Can it explain a decision?"**
→ "Yes — live attention heatmap on the image, plus SHAP/LIME word importance on the text. I can show it in the demo."

**"What were the challenges?"**
→ "Colab sessions disconnect, so we built checkpoint/resume. We also iterated a lot on the architecture — a bigger image backbone (B/16) and the α-gate were the changes that moved the numbers."

**"Limitations / future work?"**
→ "Single image–text pairs only (no video/multi-image), English-only, Reddit-specific, binary real/fake. Single random seed for the final model. Future: more seeds for robustness, gradient-based explanations, larger data."

**"Why Fakeddit?"**
→ "It's large, genuinely multimodal, real social-media content (not synthetic), and publicly available with clean binary labels."

---

## 8. Suggested flow for the talk (≈5 minutes)

1. **Pitch** (§1) — what it does, one line.
2. **Architecture** (§2 diagram) — two specialists + a gated fusion.
3. **Why these models** (§3) — RoBERTa + CLIP, and the α-gate as the novel bit.
4. **Results** (§4 table) — "fusion helps, our gated version is best."
5. **Explainability** (§5) — show or describe the heatmap + word importance.
6. Invite questions; lean on §7.

**Tip:** if you can, have the **live demo** (`python run_demo.py --no-tunnel`) open on a test sample — the attention heatmap is the most memorable thing in the whole project.
