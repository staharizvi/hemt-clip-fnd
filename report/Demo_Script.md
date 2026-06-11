# HEMT-CLIP — Defense Demo Script

A rehearsed, ~3-minute walkthrough of `notebooks/06_demo.ipynb`. Everything shown
comes from the **held-out Fakeddit test set** the model was scored on, so every
prediction is verified and reproducible. Run cells top-to-bottom before the panel
walks in (the model load takes ~1 min); then narrate.

---

## 0. One-line framing (say this first)

> "HEMT-CLIP takes a post — a **headline plus an image** — and predicts **REAL vs
> FAKE**, then explains the *same* decision two ways: a heatmap showing where on
> the image it looked, and a highlight showing which words moved the verdict."

**Define the label immediately** (pre-empts the most common confusion):

> "Important: *FAKE* here is the Fakeddit 2-way label. It's not 'this sentence is a
> lie' — it groups satire, manipulated images, misleading captions, and false
> text-image pairings into one FAKE class. We predict the dataset's label."

## 1. The numbers (cell-04 artifacts, optional slide)

> "On the test split, the gated-fusion model gets **F1 0.839, AUC 0.912** — and it
> beats text-only (0.78) and image-only (0.81), which is the whole point: fusion
> helps." Point at `outputs/confusion_matrices/` and `outputs/roc_pr_curves/`.

## 2. Run the curated examples (cell 12)

Three vetted rows; talk through each as it renders.

| Row | Say |
|---|---|
| **15662** REAL | "Liberation of Paris — Eiffel Tower and a WWII jeep. Authentic photo, model says REAL, ~93% confident. Watch the attention land on the tower/vehicle." |
| **15368** REAL | "Two polar bears on the ice. REAL, very confident. Clean example — the heatmap sits on the animals." |
| **16379** FAKE | **(the one to dwell on)** "The title says *'the almost full moon'* — but look at the image: that's a **lamp on a flagpole**, not the moon. The text and image disagree. That's a *false text-image connection*, and the model flags it FAKE at 99%. This is exactly what a multimodal detector buys you over text-only." |

For each, mention: **prediction + confidence**, the **attention heatmap**, and the
**highlighted words** (red = toward FAKE, green = toward REAL).

## 3. Prove the pipeline is honest (cell 14)

> "To show this isn't cherry-picking: here are six labelled-REAL test rows run cold
> — five come out REAL, and `live_a` matching `stored_a` confirms our alpha feature
> is computed correctly. The pipeline is sound."

## 4. Own the limitation (cells 17–18)

> "One honest caveat. If we feed an **arbitrary web image with a made-up title** —
> here, a stock lion photo — the model over-predicts FAKE. That's **distribution
> shift**: it's calibrated to Fakeddit, and this is off-distribution. It's a
> documented property, not a bug — the sanity check we just ran proves the code is
> correct on in-distribution data."

Leading with this *raises* credibility — it shows you understand the model's
operating envelope.

## 5. (Optional) Live moment

Uncomment `demo_custom_title(15368, "...")` (cell 18): keep the real polar-bear
image, type a new title live. Stays in-distribution, shows text sensitivity without
the OOD failure. Safer than pasting a random internet image.

---

## Anticipated Q&A

- **"Why is a normal-looking photo FAKE?"** → Fakeddit labels come from source
  subreddits and the FAKE class spans satire/manipulated/misleading — not just
  obvious fakes. We predict the dataset's label; label noise is a known dataset
  property (covered in the report).
- **"Why does it call everything fake on your own images?"** → Distribution shift.
  The model is fit to Fakeddit's distribution; arbitrary web content is OOD and
  skews toward FAKE (recall-skew). In-distribution behaviour is correct — see the
  sanity check.
- **"What is alpha?"** → A CLIP text-image cosine-similarity *feature* (how aligned
  the caption and image are), fed into the fusion gate. It is an input signal, not
  a hard gate/threshold.
- **"How do you know the explanation reflects the real decision?"** → SHAP and LIME
  are run on the *multimodal* model with the image held fixed, so the word
  attributions explain the actual verdict — not a separate text-only proxy. The
  attention heatmap is read straight from the model's cross-attention weights.
- **"Text-only already gets 0.78 — is multimodal worth it?"** → +0.06 F1 and a
  cleaner AUC, and qualitatively it catches false text-image pairings (the moon/
  flagpole case) that text alone cannot.
- **"Could you deploy this?"** → Within a Fakeddit-like distribution, yes; for open-
  world use it would need domain adaptation / recalibration — which is exactly the
  limitation we demonstrated.

## Pre-flight checklist

- [ ] Run all cells once; confirm the three curated rows render correct verdicts.
- [ ] `outputs/` populated (confusion matrices, ROC/PR, attention/SHAP/LIME panels).
- [ ] Internet available for the one OOD image (or pre-download it).
- [ ] Know the three curated stories cold; lead with #2 row 16379.
