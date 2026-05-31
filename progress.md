# HEMT-CLIP Progress Log

**Current phase:** **α-gate wins → `gated_fusion` is now the headline HEMT-CLIP; source doc fully reframed.** Test (n=2573): full HEMT-CLIP (α-gate) F1 **0.8393** / Acc **0.8313** / AUC **0.9122** — best of all variants (concat 0.8319/0.9042; α-feature ablation 0.8277/0.8919). User decided (2026-06-01): `gated_fusion` = headline HEMT-CLIP, α-feature variant demoted to ablation; single seed is fine (no seeds 7/123). `report/Chapters_4_5_6_Source.md` rewritten throughout: headline-result box, master-correction item 3 reversed, §4.2.1/§4.3.1 novelty (lead with gated architecture), Table 4.3, §5.1.1/5.1.5/5.1.7 (α-gate is the model, classifier 512), Chapter 6 (Table 6.1 5 rows, §6.3.1/§6.3.3 gate-wins, §6.7). Notebooks 03/04 cells updated with the result. **Next:** user integrates the reframed prose into the docx; Ch.7; slides; re-run notebooks 05/06 on Colab (now wired to gated_fusion) to regenerate attention figures + demo.
**Last updated:** 2026-06-01

---

### 2026-06-01 — Notebooks 05/06 + XAI/demo code reframed to gated_fusion (headline HEMT-CLIP)
Made the explainability + demo serve the α-gated headline model:
- **`explainability/attention_viz.py`** — added `--variant` flag (default `gated_fusion`); `load_model(variant=...)`; default `--preds-npz` → `preds_gated_fusion.npz`. Heatmaps are now the headline model's genuine cross-attention weights (gate is applied after attention).
- **`app/streamlit_app.py`** — new `HEMT_CLIP_VARIANT` env (default `gated_fusion`); `load_model`/`discover_ckpt` build+glob that variant (needed because gated classifier in_dim=512 vs α-feature 513); About-tab architecture text + headline table updated (gate eqn, 512→256→2, gated row 0.839/0.912 as best, last-4 layers).
- **Notebook 05** — discover/auto-recover/attn cells switched to `gated_fusion` + `preds_gated_fusion.npz`; intro flips the "concat beats cross-attn, XAI is consolation" framing to "gated cross-attn is best AND has intrinsic XAI"; take-aways method table / Finding 1 / per-method / "what's novel" reframed; note that attention picks change (re-inspect grid). SHAP/LIME findings (2221/2115/1007) unchanged (text_only).
- **Notebook 06** — ckpt-env cell sets `HEMT_CLIP_CKPT` + `HEMT_CLIP_VARIANT=gated_fusion`; intro/prereqs/viva points updated (α is the gate; gate-vs-feature Q&A; HEMT-CLIP is best model). launch-code already passes `env={**os.environ}` so the subprocess inherits the vars.

All notebooks parse; no leftover `preds_hemt_clip` refs. **Not yet committed/pushed** — Colab pulls from GitHub, so push before re-running 05/06.

---

## Done

### 2026-06-01 — `gated_fusion` result: α-gate wins on test (novelty inverts back to original architecture)
Trained `gated_fusion` (val F1 0.8464, reproduced twice) and evaluated all 5 variants on test (notebook 04, `hemt_gated_fusion_20260531-2116`):

| Variant | test F1 | test Acc | test AUC | test Rec | test Prec |
|---|---:|---:|---:|---:|---:|
| concat_fusion | 0.8319 | 0.8286 | 0.9042 | 0.8625 | **0.8034** |
| hemt_clip (α-feature) | 0.8277 | 0.8189 | 0.8919 | 0.8846 | 0.7776 |
| **gated_fusion (α-gate)** | **0.8393** | **0.8313** | **0.9122** | **0.8957** | 0.7895 |

**gated_fusion is the single best model** on F1/Acc/AUC/Rec. Beats α-feature by +1.16 F1 / +2.03 AUC; beats concat by +0.74 F1 / +0.80 AUC.

**This REVERSES the prior conclusion and CORRECTION 3.** The α-gated weighted fusion that the original report describes (and that I'd flagged as "never implemented / weak novelty") is now implemented *and* is the best detector. The project's originally-claimed novelty — CLIP-guided cross-attention with α as a similarity gate — is vindicated empirically. The α-direction finding (fake>real α) still holds as an observation, but the interpretation "⇒ gating is wrong" was falsified: the gate works as a learned soft-blend despite inverted α.

Caveats: gated val→test −0.7 pt (val optimistic, like hemt_clip, but stays #1 on test); single seed; concat retains best precision. Notebook 04 §6.3.3 cell updated with the result.

**Next:** (1) user decides whether gated_fusion becomes the headline "HEMT-CLIP"; (2) reframe source doc §4.3.1/Table 6.1/§6.3.3 + reverse CORRECTION 3; (3) optionally run gated seeds 7/123 for mean±std since it's now the headline.

> **NOTE (superseded):** the entry below framed the novelty as "α-as-feature beats gating." The 2026-06-01 test result above **reverses that** — the α-gate wins. Kept for history.

---

### 2026-05-31 — Novelty reframe + `gated_fusion` variant (code change)
**Decision (user-chosen):** novelty framing = **α-finding + XAI** (not "novel architecture"); and **yes, build the gated variant** to make the α design choice an empirical result.

**New ablation variant `gated_fusion`** — same cross-attention backbone as `hemt_clip`, but α is used as a CLIP-similarity **gate** inside fusion (`α·attended + (1−α)·text`) instead of being concatenated as a feature. Isolates *how α is used* (gate = prior-work approach e.g. FND-CLIP, vs feature = ours). Files touched:
- `models/fusion.py` — `forward()` takes optional `alpha`; when given, applies the parameter-free gate (cast to autocast dtype). Default path unchanged → existing `hemt_clip` checkpoints still load.
- `models/hemt_clip.py` — added `gated_fusion` to `VARIANTS`; `needs_fusion` includes it; classifier in_dim = proj_dim (α not concatenated); forward passes `alpha=batch["alpha"]` into fusion.
- `training/evaluate.py` — `discover_checkpoints` now **skips** missing variants with a warning instead of raising (so eval runs before the new variant is trained); added a color for it; docstrings de-hardcoded from "4 variants".
- `training/ablation_runner.py` — docstring lists variant E.
- `configs/base.yaml` — `gated_fusion` added to `ablation.variants` with rationale comment.

**Why this is the novelty fix:** the report's claimed novelty (α-gated "CLIP-guided fusion") was never implemented and the cross-attention architecture loses to concat on test F1. The α-inversion finding (fake α > real α on Fakeddit) means gating is the *wrong* inductive bias here; testing gate-vs-feature on the same backbone turns "we didn't build the gate" into "we tested it and here's why a learned feature is better." Source doc §4.3.1 + §6.3.3 rewritten accordingly.

**To run (Colab T4, ~6 min + ~4 min eval):**
- `!python -m training.train --variant gated_fusion` (writes `hemt_gated_fusion_*_best.pt` to Drive)
- `!python -m training.evaluate --split test` (regenerates the now-5-variant table + plots)
- Then recompute fake-vs-real α split on **B/16** in notebook 01 (CPU, seconds) before quoting the exact Δ.

**Notebooks wired (2026-05-31):** the run is now paste-free in the notebooks (no loose commands):
- **nb 03** — new `gated_fusion` section (3 cells: framing md / `!python -m training.train --variant gated_fusion` / val-results placeholder) inserted after the seed-robustness results, before the "After training" wrap-up.
- **nb 04** — `intro` updated to mention the 5th variant; `discover-code` and `figures-code` made tolerant of a not-yet-trained variant (guard `discovered.get()` / `png.exists()` — needed because `gated_fusion` is now in `VARIANTS`); new take-away cell with the α-gate-vs-α-feature comparison table (§6.3.3) to fill after eval.
- **nb 01** — `intro`/`alpha-md`/`takeaways-md` corrected from ViT-B/32 → **B/16** (α stats 0.09–0.49 / mean ≈ 0.28); the fake>real α finding now explicitly linked to the novelty + gated experiment. The `alpha-code` cell computes the by-label split dynamically, so re-running it on the B/16 HDF5 prints the current Δ.

### 2026-05-31 — Chapter 4/5/6 source document (`report/Chapters_4_5_6_Source.md`)
Extracted the existing report's full TOC/heading hierarchy from `FYP final project.docx` (it ends at Ch. 5; **Ch. 6 Results does not exist yet**). Wrote a single source doc organised by those exact headings so prose drops straight in. Chapters 4–5 in the docx are a pre-execution **plan** that diverged from what was built; the doc embeds **9 plan-vs-reality corrections**, 3 load-bearing:
- **α is a concatenated classifier feature (513→256→2), NOT a fusion gate.** The docx's novel-contribution formula `fused = α·attended + (1−α)·text` was never implemented (verified in `models/fusion.py` + `models/hemt_clip.py`). This touches the report's headline "novel contribution" claim.
- **Image encoder is CLIP ViT-B/16 (196 patch tokens, last 4 blocks), not B/32 (49 tokens, last 3).**
- **SHAP is partition/Owen-value (perturbation), not GradientExplainer; LIME implemented, not "optional".**
- Plus: label smoothing 0.0 not 0.1; training ~6 min/variant not 2–5 h/epoch (two *stages* not per-epoch stages); corpus 17,149 not 100K; explainability evaluated by cross-method agreement not a user study.
- Ch. 6 is fully drafted (Table 6.1, ablation, seed band, 4 XAI findings, error analysis) since it's net-new.

**Next:** user applies the corrections + prose into the docx, drafts Ch. 7 (Conclusion), slides, demo dry-runs.

---

### 2026-05-31 — Streamlit demo (`app/streamlit_app.py` + notebook 06)
**`app/streamlit_app.py` — new (~340 LOC, was a 13-line stub).** Per Blueprint §11.
- **Cached resources:** `load_model` (hemt_clip), `load_tokenizer` (RoBERTa), `load_clip` (ViT-B/16 for live α), `load_sample_examples` (6 test samples — 3 real + 3 fake — from HDF5, deterministic seed=0).
- **Layout:** sidebar with checkpoint info + device, two-column main panel (input on left: sample dropdown + image upload + text area + Analyze button; image preview on right).
- **Output on click:** prediction label (REAL / FAKE) with color coding, class-probability bars, α value with verbal interpretation (low / moderate / high), ground-truth comparison if a test sample was picked.
- **Three tabs:**
  - *Attention heatmap* — live extraction from `out["attention_weights"]` (B=1, H=8, Q=1, P=196) → mean heads → 14×14 grid → bilinear-upsample → overlay (`cmap='hot'`, alpha=0.5). Rendered as PNG via matplotlib `BytesIO`. < 1 s on T4.
  - *SHAP (text)* — lookup precomputed `outputs/xai/shap/shap_NN_*.png` from notebook 05's manifest, keyed by `hdf5_row` (matches `selected_ds_idx`). Graceful degrade if no match or no manifest (shows informative message).
  - *How it works* — architecture summary, test-set headline table (text_only 0.783 / image_only 0.814 / concat_fusion 0.832 / hemt_clip 0.828), explainability methods overview. Drops into a one-paragraph viva briefing.
- **Checkpoint discovery:** `HEMT_CLIP_CKPT` env var takes precedence; else walks `HEMT_CLIP_CKPT_DIR` (defaults to Drive checkpoints dir) for latest non-`_seed*` hemt_clip best.pt.

**`notebooks/06_demo.ipynb` — new (18 cells, was no stub).** Per Blueprint §11.3.
- Cells 0–4: idempotent bootstrap (Drive + repo + deps + HDF5 to local SSD), point cfg at local HDF5 — identical pattern to nb 02–05.
- Cells 5–6: **auto-recover SHAP artefacts** if missing in fresh Colab runtime. Runs `training.evaluate` (~3 min) + `explainability.shap_text` (~30 s) so the demo's SHAP tab is populated. Attention is live so no precompute needed.
- Cells 7–8: discover canonical `hemt_clip` ckpt via `training.evaluate.discover_checkpoints`, set `HEMT_CLIP_CKPT` env var for the streamlit subprocess to inherit.
- Cells 9–10: ngrok auth setup — reads `NGROK_AUTHTOKEN` env var first, falls back to `getpass.getpass()` so the token doesn't print to the notebook output. Tears down stale tunnels before reauthenticating.
- Cells 11–12: launches `streamlit run app/streamlit_app.py` as a `subprocess.Popen` with stdout to `/content/streamlit.log`, polls `curl http://127.0.0.1:8501/` up to 20 s for readiness, then opens HTTPS ngrok tunnel pointing at port 8501. Prints the public URL with horizontal-rule formatting for easy copy.
- Cells 13–14: optional `!tail -n 40 /content/streamlit.log` for debugging.
- Cells 15–16: shutdown cell — disconnects tunnels, kills ngrok, `pkill streamlit run`.
- Cell 17: viva talking points specific to the demo — what to highlight live, what to do if Colab disconnects mid-viva (recorded video backup + report figures + report prose), three anticipated examiner questions with crisp answers.

**Definition of Done movement (Blueprint §17):**
- ✅ Streamlit demo runs end-to-end with example inputs (test-sample dropdown gives 6 canned examples; custom upload also supported)
- ✅ Code pushed to GitHub with README and requirements.txt (already done; `requirements.txt` already had `streamlit==1.32.0` + `pyngrok>=7.0.0`, no new deps needed)
- ✅ All engineering deliverables now ticked

**Engineering scope is now complete.** Remaining items are pure writing/prep:
- Chapter 6 prose draft (using nb 04 + 05 outputs as figure/table inputs)
- Chapter 7 (Conclusion & Future Work)
- Existing-report fixes per Blueprint §15 (duplicate acknowledgments, scope inconsistency, etc.)
- Slide deck (12–15 slides)
- Two demo dry-runs before viva

**Next:** user owns the prose. I can draft outlines or specific sections on request, but Chapter 6 is best written in the user's own voice with the figures/numbers I've already produced as the scaffolding.

---

### 2026-05-31 — Chapter 6.4 take-aways written as near-final draft
**Notebook 05 cell `takeaways-md` rewritten from a "fill in after the run" template to a structured §6.4 draft** that drops into the report with light prose polish. Four findings, each backed by saved figures:

**Finding 1 — Cross-attention produces structured heatmaps.** Focus quality varies with image content density; clean localisation on figures/faces in `correct_hi` examples, diffuse on text-heavy images (honest limit, not claimed as a flaw). Composite grid `attention_grid.png` is the single §6.4 figure for the intrinsic method.

**Finding 2 — Sample 2221 (Kristallnacht) is the headline error-analysis figure.** SHAP and LIME *agree* on the failure mechanism: `property`, `attacks`, `german`/`erman`, `during` push fake → model is biased against violent/historical vocabulary in general, not against the named event. They *disagree* on `kristallnacht` and `colourized`: SHAP fragments both into BPE pieces (`acht`, `ouri`) that mislead; LIME catches the whole-word signal (`kristallnacht` modestly real-pushing, `colourized` real-pushing — Reddit cultural pattern). LIME's whole-word view reveals what SHAP's BPE fragmentation hid.

**Finding 3 — Sample 1007 ("tree on rock") is the headline success figure.** Both methods independently flag the compositional structure (object + preposition + object on top of object) as fake-pushing. Strong agreement on success tokens makes the cross-method agreement argument credible (not just cherry-picked errors).

**Finding 4 — Sample 2115 ("mad max") is the methodology figure: SHAP and LIME disagree on sign.** Same model, same correct prediction (real, conf=0.791), opposite attribution direction. SHAP says tokens push fake; LIME says tokens push real. **Not a bug** — SHAP measures marginal contribution against expected value, LIME measures perturbation sensitivity. On a 2-content-word title with low signal density, each method falls back to its respective theoretical default and they diverge. **Cross-method agreement is conditional, not universal** — holds for long signal-rich titles, can invert on short titles. This is *more* honest than a flawless-agreement story and validates including both methods.

**Per-method observations** documented for the §6.4 discussion paragraph:
- Cross-attention: zero-cost intrinsic, architecture-specific, focus varies with image structure
- SHAP: BPE granularity faithful to model internals but harder to read; aggregate at n=30 not stable (dropped from report — only "like" survived strict filter)
- LIME: whole-word granularity readable; ~10% sampling variance across runs, rank order stable

**Honest caveats** flagged for §6.4:
- Both post-hoc methods are perturbation-based; share a theoretical floor (deleted-word inputs are out-of-distribution). A gradient-based method would be genuinely independent — out of scope for FYP.
- n=30 is robust per-sample but not population-level — no over-claiming aggregate trends.
- SHAP+LIME run on `text_only` model, not `hemt_clip` text branch — deliberate methodological choice for clean attribution semantics.
- Multimodal SHAP intentionally skipped per Blueprint §10.2.

**§6.4 figure plan** (3 headline figures, ~1 page each in report):
1. Cross-attention composite grid (intrinsic method showcase).
2. Sample 2221 SHAP+LIME side-by-side (error analysis with cross-method agreement on mechanism).
3. Sample 2115 SHAP+LIME side-by-side (methodology figure: when methods disagree, what each measures).

**Viva talking points** (Blueprint §16, updated):
- "Why include LIME?" → not redundant with SHAP; answers different attribution question at different granularity; agreement and disagreement both informative.
- "What's novel?" → intrinsic explainability via cross-attention (unique to fusion architecture) + unified two-method post-hoc framework that *deliberately includes both LIME and SHAP* with structured agreement-vs-disagreement analysis.

**Definition of Done movement (blueprint §17):**
- ✅ 10+ attention visualizations (12 done)
- ✅ 30+ SHAP text explanations (30 done; LIME also 30 done)
- ✅ Chapter 6.4 qualitative-analysis subsection drafted (near-final, in notebook 05)

**Next:** notebook 06 (Streamlit demo) + Chapter 6 prose draft. The XAI deliverable is substantively done — figures are saved, findings are concrete, framing is honest, report integration plan is clear.

---

### 2026-05-30 — LIME added alongside SHAP (overrides blueprint §16's drop-LIME decision)
**`explainability/lime_text.py` — new (236 LOC).** Mirrors `shap_text.py` in structure for direct comparability:
- Same `pick_samples` logic and same default `--seed=42` → guarantees LIME and SHAP run on the *same* 30 test samples. Manifests are joinable on `ds_idx`.
- Same `build_predict_fn(model, tokenizer, device, max_len)` signature → same `text_only` predict_fn wired through both explainers.
- `LimeTextExplainer(class_names=["real","fake"], bow=False, random_state=42)` — `bow=False` preserves word order (matters on 10-token titles).
- `num_perturbations=1000` per sample → ~1–2 s/explanation × 30 ≈ 30–60 s on T4.
- Outputs: 30 per-sample bars (`lime_NN_*.png`), `lime_word_records.csv`, `lime_manifest.json` (includes the `word_weights` list per sample so downstream comparisons don't need re-running LIME).
- Per-sample plots use whole words on the y-axis (vs SHAP's BPE subwords) — much more readable; complementary granularity.

**`notebooks/05_explainability.ipynb` — three new cells (positions 15–17, between SHAP display and take-aways):**
- **Cell 15 (md):** LIME framing — perturbation-based vs SHAP's also-perturbation-based; explicit acknowledgement that the blueprint dropped LIME for methodological redundancy and a defence for including both anyway ("agreement between two independent perturbation procedures" as a stronger qualitative claim).
- **Cell 16 (code):** `!python -m explainability.lime_text --checkpoint "{TEXT_ONLY_CKPT}" --n-samples 30` (~30–60 s on T4).
- **Cell 17 (code):** Displays 6 confident-error LIME plots + a **SHAP vs LIME side-by-side block** for the top 3 confident errors. Loads both manifests, joins on `ds_idx`, renders sample text + SHAP bar + LIME bar in sequence. This is the headline Chapter-6.4 methodology figure.

**`requirements.txt` updated:** added `lime>=0.2.0.1`.

**Methodological framing for Chapter 6.4 / viva (revised from earlier):**
> "We use three complementary explainability methods: (1) cross-attention heatmaps (intrinsic, image-side, unique to hemt_clip's architecture); (2) SHAP via Owen-partition explainer (post-hoc, text-side, BPE-token granularity); (3) LIME via local linear surrogate (post-hoc, text-side, whole-word granularity). SHAP and LIME are both perturbation-based and could be redundant — we include both deliberately so token attribution can be argued by agreement between two independent procedures rather than relying on either alone."

**Caveats to flag in Chapter 6.4 honestly:**
- SHAP and LIME share the same theoretical floor (both perturbation-based) — agreement is necessary but not sufficient evidence of true attribution. Gradient-based attribution (e.g. Integrated Gradients) would be a genuinely independent perspective; out of scope here.
- LIME's `num_perturbations=1000` introduces sampling variance — same sample re-run may give slightly different weights. SHAP's Owen-partition is more deterministic in this regard.

**Next:** user runs notebook 05 on Colab (~50 s of new work — bootstrap/attention/SHAP cached, just LIME + side-by-side display to add). When figures land, review LIME outputs and the side-by-side comparison; populate take-aways with concrete findings; then notebook 06 (Streamlit demo) + Chapter 6 draft.

---

### 2026-05-30 — XAI **executed**: heatmaps work, SHAP aggregate fixed, error-analysis finding surfaced

---

## Done

### 2026-05-30 — XAI **first-pass executed**: heatmaps work, SHAP aggregate fixed, error-analysis finding surfaced
**Run summary (50 sec total wall-clock on T4):**
- Cell 8 (auto-recover): detected both `preds_*.npz` missing in fresh runtime, ran `training.evaluate` (~3 min), regenerated all eval artefacts. Self-recovery wire worked as designed.
- Cell 10 (attention_viz): 14 s. All 12 examples + composite grid saved.
- Cell 13 (shap_text): 35 s. **All 30 SHAP samples succeeded — no failures.** (Owen partition explainer on short titles is much faster than estimated; the 5–10 min estimate was for KernelExplainer-class methods.)

**Qualitative findings from inspection of the rendered figures:**

**Attention heatmaps (intrinsic, hemt_clip):**
- Cross-attention produces *structured* heatmaps, not noise — every example shows non-uniform spatial focus. Some examples localize cleanly on figures/faces/text overlays; others (e.g., a Reddit-thread screenshot) are diffuse because the image content itself is diffuse. **Honest framing for Chapter 6.4:** "cross-attention produces interpretable heatmaps with focus quality varying as a function of image structure" — defensible, not over-claiming.
- The composite grid figure (`outputs/xai/attention/attention_grid.png`) is single-figure-ready for Chapter 6.4.

**SHAP per-sample bars (post-hoc, text_only):**
- **Strong Chapter-6.4 finding from sample 2221:** title `"a german attacks jewishowned property during the kristallnacht colourized"` — text_only confidently misclassified as FAKE (conf=0.715). Top positive SHAP tokens: `property` (+0.18), `during` (+0.12), `erman` (+0.07), `attacks` (+0.06), `acht` (+0.05) — BPE pieces of "german" and "kristallnacht". **The model conflates sensational/historical vocabulary with fake-news markers, mistaking a real historical event description for a fake post.** Direct error-analysis material for Chapter 6.6 (Error Analysis), and a defensible viva talking point about why pure text models fail on fact-grounded historical content.
- BPE tokenization fragments multi-token words (`german` → `g` + `erman`, `kristallnacht` → `k` + `rist` + `all` + `n` + `acht`). Per-sample bars are still readable when you mentally join subwords; aggregate stats would benefit from BPE merging but that's a separate engineering pass.

**Bugs found and fixed in this turn:**

1. **`attention_viz.py`: cached-pred mismatch.** The script ran a fresh batch=1 forward pass to capture attention weights, but ALSO used that fresh pass's `argmax(probs)` to label `pred` and `confidence` in the figure titles. Under fp16 + batch=1 vs evaluate.py's batched fp16, borderline samples can flip — producing titles like `[correct_lo] ✗ pred=FAKE | true=REAL | wrong | conf=1.00` (bucket says correct + low-conf, fresh pass disagrees on both). **Fix:** use the cached `preds[ds_idx]` and `probs[ds_idx]` from the npz for labeling; only use the model's fresh forward for the attention weights, which is what it's there for. One-line change in `main()`.

2. **`shap_text.py`: aggregate dominated by stop words.** The original `plot_aggregate` had `min_count=2` and no stop-word filter, so the top-15 tokens for both "real" and "fake" rankings were `and`, `the`, `this`, `it`, `on`, `way`, `building`, ... — common English fillers with small SHAP values that average to spurious means across the 30 samples. **Fix:** bumped `min_count` 2→5 and added a `STOP_WORDS` set of ~80 English fillers + Reddit-platform words to exclude from the aggregate. Per-sample plots are untouched (stop words there still carry information about which words the model latched onto).

**What re-running will produce:** clean attention titles (consistent with the bucket they came from) + interpretable SHAP aggregate showing real content tokens. Then Chapter 6.4 is ready to draft.

**Definition of Done movement (blueprint §17):**
- ✅ 10+ attention visualization examples saved (12 done, with composite grid for the report)
- ✅ 30+ SHAP text explanations generated (30 done, zero failures)

**Next:** user re-runs cells 10 and 13 (attention + SHAP) — bootstrap and eval cells stay cached. ~50 s. Then notebook 06 (Streamlit demo) and Chapter 6 draft. The XAI artefacts now carry the Chapter 6 headline (since concat F1 > hemt_clip F1 on test), so getting these figures clean was load-bearing.

---

### 2026-05-30 — Explainability infrastructure (`attention_viz.py` + `shap_text.py` + notebook 05)
**`explainability/attention_viz.py` — new (was a 13-line stub).** Cross-attention heatmaps per Blueprint §10.1.
- Loads `hemt_clip` ckpt + `outputs/eval/preds_hemt_clip.npz` from notebook 04.
- Sample picking: 4 buckets × `--n-per-bucket` (default 3 = 12 total) = `{correct, wrong}` × `{hi-conf, lo-conf}`. Deterministic, seed=42.
- For each picked example: forward pass with `torch.autocast` fp16, capture per-head attention from `CrossAttentionFusion` (B=1, H=8, Q=1, P=196), average over heads, reshape to 14×14, bilinear-upsample to 224×224, overlay on original HDF5 image with `cmap='hot' alpha=0.5`. Plot is a 1×2 figure (original + overlay) with title showing pred / conf / true / status / α / text snippet.
- Composite grid figure (`attention_grid.png`) shows all 12 in a 4×3 grid with compact per-cell titles — single-figure for Chapter 6.4.
- Side artefacts: `attention_manifest.json` with `{bucket, ds_idx, hdf5_row, pred, label, confidence, alpha, file, text}` per pick — reproducible reference for the report's appendix.

**`explainability/shap_text.py` — new (was a 12-line stub).** Post-hoc text attribution per Blueprint §10.2.
- Loads `text_only` ckpt + `outputs/eval/preds_text_only.npz`. Uses **text_only** (not hemt_clip's text branch) deliberately: SHAP attributes a model's predictions, so the cleanest answer to "which words pushed the verdict?" comes from a model whose verdict is exactly a function of text alone.
- Builds a `predict_fn(texts)` that tokenizes + runs forward pass with zero-filled dummies for `pixel_values/alpha/label` (text_only's forward ignores them but the signature requires them).
- `shap.Explainer(predict_fn, shap.maskers.Text(tokenizer), output_names=["real","fake"])` — Owen-value partition explainer (faster than KernelExplainer, equivalent semantics for hierarchical text).
- Sample picking: stratified across `{correct, wrong}` × `{real, fake}` × 3 confidence quantiles (low/mid/high) per cell, tops up randomly to hit `--n-samples` (default 30 per Blueprint §10.5).
- Per-sample artefact: horizontal bar of token contributions to the predicted class (top 20 by |value|), positive bars red (→ fake), negative bars blue (→ real).
- Aggregate artefact (`shap_top_tokens.png`): two side-by-side panels showing the top-15 tokens by mean SHAP value toward each class (across all 30 samples, with `count ≥ 2` filter for stability).
- `shap_token_records.csv` long-form table + `shap_manifest.json` per-sample record. SHAP failures on individual samples log a warning and continue (rather than killing the whole run).

**`notebooks/05_explainability.ipynb` — new (was a 1-cell stub).** 14 cells (7 markdown + 7 code).
- Bootstrap cell verbatim from nb 02–04 (idempotent — Drive mount, repo pull, HDF5 → local SSD).
- Discovery cell — reuses `training.evaluate.discover_checkpoints` to pick the same canonical ckpts as nb 04 (hemt_clip v4 seed=42, text_only latest). Also asserts `outputs/eval/preds_{hemt_clip,text_only}.npz` exist (produced by nb 04).
- Attention section: framing markdown → `!python -m explainability.attention_viz ...` (~30s on T4) → display the composite grid + each of the 12 per-example panels via `IPython.display.Image`.
- SHAP section: framing markdown → `!python -m explainability.shap_text ...` (~5–10 min on T4) → display the aggregate top-tokens figure + curated subset (top 6 confident errors + top 4 confident successes) from the manifest.
- Take-aways markdown is templated: structure for what to claim under each method, plus a joint framing for Chapter 6.4 ("intrinsic on image, post-hoc on text — unified two-method framework"). Concrete observations get plugged in after the run.

**Why this matters now:** the test-set verdict from earlier today (concat F1 0.832 > hemt_clip F1 0.828, AUC +1.23 pt for concat) inverted Chapter 6's planned F1 headline. The intrinsic-XAI argument — attention heatmaps over a 14×14 patch grid, *unique to the cross-attention architecture* — is the contribution Chapter 6 will now lead with. Without notebook 05's figures, the report has no qualitative defence for the architectural choice. With them, it has a stronger story than F1 alone.

**Runtime budget on Colab:** ~30s (attention) + ~5–10 min (SHAP) + ~1 min (display + setup) = **~10–12 min total**. Cheap.

**Next:** user runs notebook 05 on Colab. When figures land, I review the attention heatmaps and SHAP token bars, populate the take-aways markdown with concrete observations, and update progress.md with the qualitative findings (do heatmaps localize on relevant image regions? does SHAP produce sensible token rankings?). Then notebook 06 (Streamlit demo) and Chapter 6 draft.

---

### 2026-05-30 — Test-set evaluation **executed**: concat beats hemt_clip on test, narrative pivots to XAI
**Full results (n=2,573 test split):**

| Variant | val F1 | test F1 | val→test Δ | test acc | test prec | test rec | test AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `text_only`     | 0.7702 | 0.7827 | +0.0125 | 0.7773 | 0.7522 | 0.8158 | 0.8545 |
| `image_only`    | 0.8012 | 0.8135 | +0.0123 | 0.8061 | 0.7716 | 0.8601 | 0.8824 |
| `concat_fusion` | 0.8204 | **0.8319** | +0.0115 | **0.8286** | **0.8034** | 0.8625 | **0.9042** |
| `hemt_clip`     | 0.8229 | 0.8277 | **+0.0048** | 0.8189 | 0.7776 | **0.8846** | 0.8919 |

**Key finding — concat_fusion beats hemt_clip on test:**
- test F1: concat 0.8319 > hemt_clip 0.8277 (concat ahead **+0.42 pt**).
- test accuracy: concat 0.8286 > hemt_clip 0.8189 (concat ahead **+0.97 pt**).
- test AUC: concat 0.9042 > hemt_clip 0.8919 (concat ahead **+1.23 pt**, threshold-agnostic).

Three different metrics agree — the val ordering did not generalize. The val→test delta tells the story: text_only/image_only/concat_fusion all gained ~+1.2 pt on test; hemt_clip gained only **+0.48 pt**. hemt_clip's val advantage was disproportionately val-specific, likely because the cross-attention block's extra ~3M trainable params over concat found val patterns that didn't transfer. `fusion.dropout=0.2` reduced but didn't eliminate this.

**This invalidates the "cross-attention beats concat on F1" headline** we briefly defended after v4 single-seed (later weakened by seeds to +0.14 pt mean, now negative on test).

**Findings that *do* hold robustly:**
- **Multimodal premium is large and stable** — best unimodal (image_only 0.8135) → best fusion (concat 0.8319) = **+1.84 pt on test F1**, consistent with the val premium. The case for multimodal fusion (of any kind) is the strongest result in the matrix.
- **Image > text on Fakeddit** — image_only test AUC 0.8824 vs text_only 0.8545. Thumbnails carry more signal than titles. Sentence-worth in Chapter 4 / 6.
- **Recall/precision profiles differ by variant** — hemt_clip is recall-heavy (rec 0.8846 / prec 0.7776), concat is balanced (rec 0.8625 / prec 0.8034). Has deployment implications: hemt_clip for triage (high recall catches more fakes); concat for fact-checking (high precision avoids flagging real news). Both have legitimate use cases.

**Chapter 6 narrative pivot — final framing:**
> "Cross-attention shows competitive but not superior F1 versus concatenation (test 0.828 vs 0.832; difference within seed-level noise). Its contribution is **intrinsic explainability** — attention heatmaps over the 14×14 patch grid — a capability concatenation architecturally cannot provide. We trade ≈0.4 pt test F1 for a qualitative XAI mode the architecture enables, and gain a recall-skewed decision profile useful for triage applications."

This is **stronger than overclaiming**. The intrinsic-XAI argument is genuinely unique to cross-attention; no honesty cost; examiner sees we understand our own numbers. Aligns with blueprint §16's defensible talking point #5 ("CLIP-guided cross-attention with α as a learned feature, combined with a unified two-method explainability framework").

**Optional methodological strengthening (not blocking):** running `concat_fusion` at seeds 7 and 123 would give mean ± std for both fusion variants — symmetric comparison. Conclusion almost certainly won't flip (three test metrics all agree), but closes the rigour gap if the examiner asks. ~30 min Colab + ~6 min eval.

**Artefacts now on disk (under `outputs/eval/` in repo):**
- 4 confusion-matrix PNGs
- 1 ROC overlay PNG
- 1 ablation F1 bar PNG  
- 1 per-class P/R PNG
- 4 metrics JSON files
- 4 predictions npz files (logits/probs/preds/labels — feeds notebook 05)
- `summary_test.{csv,md}` — Chapter 6 results table

**Definition of Done movement (blueprint §17):**
- ✅ Test-set metrics for all 4 variants
- ✅ 5+ plots (TB training curves + ROC + F1 bar + per-class P/R + 4 CMs = 8 figures)

**Next:** notebook 05 — attention heatmaps (~10 examples chosen from `preds_hemt_clip.npz`, mix of correct/incorrect, high/low confidence) + SHAP on text branch (~30 samples). Both produce qualitative figures for Chapter 6, with attention heatmaps now carrying the headline. The XAI work is the centrepiece of the report, not a side dish.

---

### 2026-05-30 — Test-set evaluation infrastructure (`training/evaluate.py` + notebook 04)
**`training/evaluate.py` — new (was an 11-line stub).**
- Auto-discovers `hemt_{variant}_*_best.pt` per variant from `cfg.checkpointing.dir`, preferring non-`_seed*` files (so `hemt_clip` picks the canonical v4 seed=42 ckpt, not seed=7/123). Override with `--checkpoints <json>`.
- Picks: `text_only` → v2 B/32 ckpt (latest text_only run; text branch is backbone-agnostic), `image_only`/`concat_fusion` → v4 B/16 (latest), `hemt_clip` → v4 seed=42 B/16 (latest non-seed).
- Per variant: build from cfg, load weights (`payload["model"]`), fp16 inference under `torch.no_grad()`, collect logits + labels, compute Blueprint §9.1 metrics (accuracy, F1 binary + macro, precision, recall, AUC-ROC, confusion matrix, per-class report). Releases GPU memory between variants (`del model` + `torch.cuda.empty_cache()`).
- Cross-variant outputs:
  - `summary_test.{csv,md}` — 4-row Chapter 6 results table (val F1 from ckpt + 5 test metrics + ckpt name).
  - `roc_overlay_test.png` — all 4 ROC curves on one axes with AUC labels.
  - `f1_bar_test.png` — ablation F1 bar with value labels.
  - `per_class_pr_test.png` — real/fake precision/recall grouped bars.
- Per-variant outputs:
  - `cm_{variant}.png` × 4 — confusion matrices with counts.
  - `metrics_{variant}.json` × 4 — full metric dump for the report.
  - `preds_{variant}.npz` × 4 — logits/probs/preds/labels, feeds notebook 05's XAI.
- CLI: `--config`, `--ckpt-dir`, `--checkpoints`, `--split` (default `test`), `--out-dir` (default `outputs/eval`), `--batch-size` (default 32), `--num-workers` (2), `--device`. Reusable on val split (`--split val`) for threshold calibration follow-ups.

**`notebooks/04_evaluation.ipynb` — new (was a 1-cell stub).**
- 14 cells (7 markdown + 7 code). Target audience: examiner reading Chapter 6.
- Bootstrap cell verbatim from nb 02/03 (idempotent — mounts Drive on Account B, pulls repo, copies HDF5 to local SSD).
- "Point evaluator at the local HDF5" cell — same yaml patch pattern as nb 03 (so cfg points at `/content/fakeddit.h5`, faster sequential reads than Drive FUSE).
- Discovery cell — calls `discover_checkpoints` and prints which `*_best.pt` was picked per variant + file size. Sanity check before evaluation.
- Run cell — single `!python -m training.evaluate --split test` invocation (~3–4 min).
- Results table cell — reads `summary_test.csv`, prints the table, computes val→test delta per variant (small delta = healthy generalization).
- Figures cell — renders the 7 PNGs inline via `IPython.display.Image` for the committed notebook to show them without re-running.
- Take-aways markdown (template): expected F1 ordering, val→test delta interpretation, AUC vs F1 comparison, per-class P/R interpretation, dominant error mode analysis — examiner-ready bullets pre-structured so they can be populated with concrete numbers after the run.

**No new requirements** — all deps (pandas, scikit-learn, matplotlib, pyyaml, h5py) already in `requirements.txt`.

**Definition of Done movement (blueprint §17):**
- ✓ Test-set metrics computed for all 4 variants ← *will tick after Colab run*
- ✓ 5+ plots generated (training curves from TB + ROC + F1 bar + per-class PR + 4 confusion matrices = 8 figures) ← *same*

**Next:** user runs notebook 04 on Colab. When the numbers come back, I fill in the take-aways markdown with concrete observations, update progress.md with the verdict, then we move to notebook 05 (XAI: attention heatmaps + SHAP).

---

### 2026-05-30 — Seed-robustness for hemt_clip (seeds 7, 123 added to seed=42)
**Results:**

| Run | val F1 | Best @ |
|---|---:|---|
| `hemt_clip` seed=42  | 0.8229 | S2 ep5 |
| `hemt_clip` seed=7   | 0.8226 | S2 ep6 |
| `hemt_clip` seed=123 | 0.8198 | S2 ep5 |
| **mean ± std (n=3)** | **0.8218 ± 0.0017** | — |

**Headline shift:** the single-seed v4 result (0.8229) was at the *high* end of the band. True performance is closer to **0.822 ± 0.002**. Vs concat_fusion 0.8204:
- Δ mean = **+0.14 pt** (was +0.25 pt on single seed — the gap halved).
- 1-std band on hemt_clip is [0.8201, 0.8235]; concat 0.8204 falls *inside* this band, 0.03 pt below `mean − std`.
- 2/3 hemt_clip seeds beat concat individually (seed=123 lost by 0.06 pt).

**What this means for Chapter 6:** the architectural F1 advantage is **real but small** — within typical seed noise. The headline contribution shifts from "cross-attention wins F1" to "**cross-attention provides intrinsic explainability that concat physically cannot, with no F1 cost**." Three-layer framing:
1. Single-seed F1: +0.25 pt (overstates).
2. Multi-seed F1: +0.14 pt (honest).
3. Intrinsic XAI: attention heatmaps over 14×14 patch grid → unique to hemt_clip.

Layer 3 leads; layers 1–2 are supporting evidence the architecture isn't *worse* on F1.

**Asymmetry — not addressing for now:** concat_fusion is still a single-seed result. To fairly compare mean ± std for both, we'd need ~30 min more Colab for concat at seeds 7 and 123. Skipping unless the user wants apples-to-apples — the conclusion won't change.

**Plumbing landed (`training/train.py` edits already on `main`):** `--seed N` CLI flag, auto-appends `_seed{N}` to run name. Backwards-compatible — runs without `--seed` use cfg.seed=42 as before. Three new checkpoints on Drive: `hemt_hemt_clip_20260530-0223_best.pt` (seed=42, v4), `hemt_hemt_clip_20260530-1332_seed7_best.pt`, `hemt_hemt_clip_20260530-1346_seed123_best.pt`.

**Next:** test-set evaluation. For headline test numbers, use seed=42's `best.pt` (strongest of the three) but report the seed band in Chapter 6 text.

---

### 2026-05-30 — v4 results: hemt_clip pulls ahead with ViT-B/16
**Ablation outcome (3 image-using variants re-run on B/16; text_only unchanged at 0.7702):**

| Variant | v3 best F1 | v4 best F1 | Δ | Best @ |
|---|---:|---:|---:|---|
| `image_only`    | 0.7823 | 0.8012 | +1.89 pt | S2 ep4 |
| `concat_fusion` | 0.8133 | 0.8204 | +0.71 pt | S2 ep5 |
| `hemt_clip`     | 0.8012 | **0.8229** | **+2.17 pt** | S2 ep5 |

**α re-precompute (CLIP ViT-B/16):** min=0.086, mean=0.276, max=0.490, std=0.054, NaN=0. Slight upward shift vs B/32 (mean 0.267 → 0.276) — consistent with B/16's tighter text-image alignment.

**Headline finding — hemt_clip beats concat_fusion (+0.25 pt).** First time across v1–v4 that the cross-attention variant leads. The Chapter-6 talking point "cross-attention is justified over concatenation" is now defensible by raw F1, not just by XAI.

**Mechanism — hemt_clip gained the *most* from the swap (+2.17 pt).** Cleanly explains the architecture's contribution: concat sees a pooled image vector (1 token), so spatial resolution barely helps it (+0.71 pt). hemt_clip's fusion attends over patch tokens as K/V — going 49 → 196 tokens gave it 4× finer localization. The architecture is *using* the new information, not just absorbing it. This is the cleanest possible defence in viva.

**Trajectory shape — clean.** v2/v3 hemt_clip peaked at S2 ep3 then declined (overfit). v4 hemt_clip peaks at S2 ep5 (0.8229), ep6 only dips 0.003 — no early stop, no train-acc runaway, no decline. `fusion.dropout=0.2` from v3 + the bigger backbone settled into a clean operating point.

**Wall-clock vs predicted:** 41.5 min for 3 variants + ~2 min α precompute = ~44 min total, well under the ~70 min estimate. α precompute was 8× faster than predicted (fp16 + batch=64 — the 25–30 min estimate was stale, predating those optimizations). Cells 18 + 20 in notebook 03 have full per-epoch logs.

**Honest caveats for the report:**
- +0.25 pt over concat is within typical seed-to-seed noise (±0.3–0.5 pt). Single-seed result. Worth running 2 additional hemt_clip seeds (7, 123) and reporting mean ± std for robustness — ~30 min on T4, optional.
- All variants share the same `fusion.dropout=0.2` (from v3) and 6-epoch stage 2 (from v2). Apples-to-apples comparison.
- B/16 means more compute per epoch (~75s for image_only vs B/32's ~62s, ~120s for hemt_clip vs ~90s). Still well within Colab Pro budget.

**Notebook 03 cell 21 (`v4 results` placeholder) filled in with the above table + decision-tree outcome.**

**Next (before test-set eval):** seed-robustness for `hemt_clip`. User chose to run 2 additional seeds (7 and 123) alongside the existing 42 to report **mean ± std** instead of a single point, since the +0.25 pt margin over concat_fusion is within typical seed noise.

**Plumbing (`training/train.py`, 3 small edits):**
- Added `--seed N` CLI flag. When set, overrides `cfg["seed"]` before `set_seed()` is called.
- When `--seed` is given without explicit `--run-name`, the auto-generated run name appends `_seed{N}` so seed=7 and seed=123 checkpoints don't collide with the seed=42 run (which keeps its existing name).
- No change to behaviour when `--seed` is not passed — backwards-compatible with all earlier ablation runs.

**Notebook 03 — added seed-robustness section (cells 22–26):**
- Cell 22 (md): rationale for the multi-seed run, what the seed actually controls in this pipeline (small randomly-init parts + shuffle order + dropout masks — backbones and α are deterministic).
- Cell 23 (code): `!python -m training.train --variant hemt_clip --seed 7` (~15 min on T4).
- Cell 24 (code): `!python -m training.train --variant hemt_clip --seed 123` (~15 min on T4).
- Cell 25 (md, placeholder): 3-seed comparison table with mean ± std, plus a reading guide (`mean − std > 0.8204` → strongest defence; mean above concat but `− std` below → still defensible with nuance; mean below concat → pivot to XAI headline).
- Cell 26 (md): existing "After training" wrap-up (unchanged).

After-test plan: load all four variants' best.pt (text_only v2, image_only v4, concat_fusion v4, and the seed-aggregated hemt_clip — likely use seed=42 as the single best.pt for headline numbers, with seeds 7+123 reported as ± std in Chapter 6 text), run on n=2573 test split, emit per-variant test_{acc, f1, prec, rec} + confusion matrix PNG + ROC PNG + 4-row comparison table.

---

### 2026-05-30 — Pivot: notebook implementation pass (01–03)
Paused v4 (B/16) work to bring the notebook deliverables up to report quality. Notebooks 02 and 03 are implemented and have committed outputs; notebook 01 was a one-cell stub. Implementing 01 first, then a review/tidy pass on 02 and 03 — they need narrative cleanup (e.g. 03 has v1 markdown headers next to v2/v3 outputs) but no rewrite.

**v4 state on `main` (commit 9812f31):**
- `configs/base.yaml` — `model.image.name: openai/clip-vit-base-patch16` (was patch32).
- `models/image_encoder.py` — docstring generalized for patch-count-agnostic interface.
- Not yet run: alpha re-precompute (B/16 embeddings differ from B/32), and ablation re-run for the three image-using variants.

**Notebook 01 — Dataset Characterization (implemented, run on Colab, evaluated):**
- 15 cells (7 markdown + 8 code). Target audience: examiner reading Chapter 4.
- Reads the packed HDF5 directly (no raw Fakeddit metadata needed — only the HDF5 survived).
- Sections: HDF5 schema dump, split sizes + class balance (table + bar chart), title token-length distribution (justifies `max_text_len=128`), α distribution overall + by label (frames α as a weak-but-useful extra feature), 8-panel sample grid (4 real / 4 fake from train, seed=42), and a Chapter-4 take-aways block.
- Bootstrap cell reused verbatim from notebook 02 — same Drive-mount + repo-pull pattern, idempotent.
- Reproducibility: `RNG = np.random.default_rng(42)` for the sample picks so the figure is stable across re-runs.

**Numbers from the Colab run (commit d35f37a):**
- Total samples: 17,149. Splits 12,003 / 2,573 / 2,573. Class balance 50.83 / 49.17 (real / fake), preserved across splits.
- Title token-lengths (RoBERTa tokenizer, incl. special): median 10, p90/p95/p99 = 19 / 22 / 35, max 71. Truncation at `max_text_len=128` is 0% — cap is ~2× the longest title.
- α (CLIP ViT-B/32 cosine sim): min 0.076, max 0.437, mean 0.267, std 0.051. NaN=0.
- File size: 1.94 GB gzip-4 compressed.

**Key Chapter-4 finding (new — not in the original take-aways):** α by label shows **fake mean=0.290 vs real mean=0.244** (Δ=+0.046, ~1 std apart). Fake posts have *higher* mean CLIP text-image agreement than real ones — counter-intuitive at first read, but consistent with Fakeddit's composition: mislabeled posts and satire typically pair on-topic imagery with dramatic captions, while real news often pairs generic stock photos with specific headlines. Heavy distribution overlap means α is still a weak predictor in isolation, but the consistent directional signal is what makes α non-trivial as an extra feature (and what justifies feeding it concatenated to the fused vector rather than as a fusion gate).

**Polish edits applied (notebook 01 cells `titles-code`, `alpha-code`, `takeaways-md`):**
- Take-aways markdown: updated median/p99 numbers to actuals (10 / 35, not the ~12 / ~30 I had drafted from memory), and added the fake>real α observation as a substantive bullet.
- `titles-code`: suppressed two HuggingFace warnings (`resume_download` FutureWarning, `HF_TOKEN` UserWarning) — cosmetic noise in the committed output.
- `alpha-code`: added an explicit Δ-mean print so the fake>real direction surfaces in the cell output, not just in the take-aways markdown.
- Pending: re-run cells 9 (`titles-code`) and 11 (`alpha-code`) on Colab to refresh their outputs against the new source. Both are CPU-only and complete in seconds.

**Pivot back to v4 (2026-05-30, same day):** user chose Fork A (F1-first) — resume v4 immediately. Notebook 02 polish deferred until v4 numbers are back. v4 cells now wired into `notebooks/03_full_training.ipynb` as a discrete `## Re-run — bigger backbone (v4, 2026-05-30)` section (cells 17–21), so the Colab run is paste-free:

- Cell 17 (md): v4 framing — why B/16, what changed, expected outcome, decision tree.
- Cell 18 (code, ~25–30 min): `data.precompute_alpha --model openai/clip-vit-base-patch16 --overwrite` against the Drive HDF5. Inline sanity-check guidance (NaN=0 hard requirement, mean roughly [0.20, 0.35]).
- Cell 19 (code, ~1 min): `shutil.copy` Drive → local SSD so the trainer reads the fresh α.
- Cell 20 (code, ~45 min): `training.ablation_runner --variants image_only concat_fusion hemt_clip --force`.
- Cell 21 (md, placeholder): v4 results table + decision tree, to be filled after the runner finishes.

Existing v1–v3 cells (0–16) untouched — they remain as the experiment history, with their outputs as the on-the-record trajectory.

---

### 2026-05-30 — v3 result + v4 backbone swap (ViT-B/32 → ViT-B/16)
**v3 hemt_clip run (only the cross-attention variant; baselines unchanged from v2):**

| Stage | Epoch | val F1 | train_acc | Note |
|---|---|---|---|---|
| S1 | 1 | 0.7447 | 0.675 | head warmup |
| S2 | 1 | 0.7487 | 0.759 | |
| S2 | 2 | 0.7963 | 0.784 | |
| S2 | 3 | 0.7992 | 0.800 | |
| S2 | 4 | 0.7986 | 0.812 | |
| S2 | 5 | 0.8008 | 0.819 | |
| S2 | 6 | **0.8012** | 0.821 | best — no early stop |

**Comparison hemt_clip across runs:**

| Run | fusion.dropout | Best F1 | Peak @ | Trajectory |
|---|---|---|---|---|
| v1 (3 epochs) | 0.1 | 0.8004 | S2 ep2 | peaks early, runs out of epochs |
| v2 (6 epochs) | 0.1 | 0.8069 | S2 ep3 | peaks ep3, declines ep4–6 (overfit) |
| v3 (6 epochs) | 0.2 | 0.8012 | S2 ep6 | monotonic climb, no overfit, lower plateau |

**Interpretation:** The dropout patch worked *structurally* — overfit shape gone, train/val gap narrower, val F1 climbed monotonically through ep6 with no early stop. But absolute F1 dropped 0.57 pt because we over-regularized: the model has less effective capacity now. Combined with v2 result, this tells us the model is **not generalization-limited** — it's **architecturally capped** on the input modality. Cutting capacity (more dropout) makes it worse; the next lever has to add *information*, not regularization.

**Hypothesis for v4:** The CLIP ViT-B/32 backbone gives 49 patch tokens (7×7 grid) — coarse for cross-attention to localize. ViT-B/16 gives 196 tokens (14×14 grid), 4× finer spatial resolution, which should help the fusion module pick up finer image cues that correlate with title misalignment (the signal Fakeddit's binary task actually rewards).

**Changes for v4 (configs/base.yaml + image_encoder.py docstring):**
- `model.image.name`: `openai/clip-vit-base-patch32` → **`openai/clip-vit-base-patch16`**
- `model.fusion.dropout`: kept at **0.2** (v3 trajectory was clean; revisit if v4 overfits)
- All other v2/v3 hyperparams unchanged.
- Image encoder docstring generalized — both backbones share hidden_dim=768, so the module is patch-count-agnostic; cross-attention K/V just sees a longer sequence.

**Re-precompute alpha (required):** existing `f['alpha']` was computed with B/32; B/16 produces different CLIP embeddings, so alpha values would be inconsistent. Re-run with `--model openai/clip-vit-base-patch16 --overwrite`. Expected ~25–30 min on T4 for 17K rows (B/16 is ~2× slower than B/32).

**Re-run variants:**
- `text_only` — **not re-run**; doesn't touch the image backbone. v2 number (0.7702) stands.
- `image_only`, `concat_fusion`, `hemt_clip` — re-run (~15 min × 3 = ~45 min on T4 with B/16's larger vision tower).
- Total Colab budget for v4: ~25 + 45 = **~70 min**.

**Realistic target:** hemt_clip val F1 0.82–0.84. If B/16 doesn't break 0.81, the F1 ceiling on Fakeddit titles+thumbnails is genuinely architectural and we accept it for the report — pivot Ch. 6 narrative to XAI as the headline contribution.

---

### 2026-05-30 — v2 ablation results in, fusion-dropout patch (v3)
**v2 vs v1 deltas (all 4 variants, 6-epoch stage 2, last-4 encoder layers, label_smoothing=0):**

| Variant | v1 best F1 | v2 best F1 | Δ | Best @ |
|---|---|---|---|---|
| text_only | 0.7654 | 0.7702 | +0.5 pt | S2 ep3 |
| image_only | 0.7775 | 0.7823 | +0.5 pt | S2 ep4 |
| concat_fusion | 0.7983 | **0.8133** | **+1.5 pt** | S2 ep5 |
| hemt_clip | 0.8004 | 0.8069 | +0.7 pt | S2 ep3 |

**Interpretation:**
- **v2 missed the 0.83–0.85 target on hemt_clip** — longer schedule + more trainable layers only bought +0.7 pt.
- **Concat now beats hemt_clip** by +0.64 pt. Narrative problem for the report: the headline architecture is no longer the F1 leader.
- **Overfit signal in hemt_clip:** val F1 peaks at S2 ep3 (0.8069) and *declines* through ep6 (0.8048) while train_acc keeps climbing (0.802→0.826). Same shape as v1, just one more epoch. Early-stop fired at ep6.
- **Concat is still learning at ep5** (0.8133) — extra epochs helped it cleanly, no overfit shape.
- text_only / image_only barely moved — at their unimodal ceilings on Fakeddit titles+thumbnails.

**Hypothesis:** Cross-attention adds ~3M trainable params on top of concat; the existing `fusion.dropout=0.1` isn't enough to regularize the extra capacity on 12K samples.

**Patch (configs/base.yaml):** `model.fusion.dropout 0.1 → 0.2`. Targets only `hemt_clip` (only variant using `CrossAttentionFusion`); other variants unchanged so v2 numbers remain valid for them. No other config changes (kept v2's 4 trainable layers, 6 stage-2 epochs, ls=0.0, patience=3).

**Expected:** hemt_clip val F1 0.81–0.83, with peak shifted to S2 ep4–5 instead of ep3 (regularization should delay overfit). If it works, hemt_clip closes the gap with concat or pulls ahead. If it doesn't move, the next lever is per-param-group LR (lower LR on fusion only) or seed ensemble.

**Re-run:** Just `hemt_clip` this time — `!python -m training.ablation_runner --variants hemt_clip --force` on Colab. ~12–15 min on T4. No need to re-run the three baselines; their numbers are stable.

---

### 2026-05-17 — Ablation matrix complete (3 remaining variants)
- Ran `training.ablation_runner` on T4 for `text_only`, `image_only`, `concat_fusion` — **16.6 min total** wall-clock (5–6 min each).
- Final val F1 ordering: `text_only` 0.7654 < `image_only` 0.7775 < `concat_fusion` 0.7983 < `hemt_clip` 0.8004.
- **Cross-attention beats concat by only +0.21 pt** — architectural complexity justified primarily by intrinsic XAI (attention heatmaps), not raw accuracy. Concat-only would be the pragmatic baseline; HEMT-CLIP earns its keep through explainability.
- **Multimodal premium**: +2.1 pt over best unimodal (`image_only` → `concat_fusion`).
- **Image > Text** on Fakeddit: 0.7775 vs 0.7654 — titles alone are weaker than thumbnails for binary fake-vs-real.
- No variant triggered early stopping; all reached the max stage-2 epoch budget.
- Per-variant logs: `runs/../ablation_logs/{variant}_20260517-*.log`. Summary: `outputs/ablation_summary_20260517-1355.{csv,md}` (3-row, baselines only — 4-row table built by hand in `notebooks/03_full_training.ipynb` for the report).
- Trainable-param counts (Stage 2, for the report's architecture comparison): text_only 14.70M, image_only 15.10M, concat_fusion 29.80M, hemt_clip 32.82M.

---

### 2026-05-12 — Project scaffolding
- Full repo structure per Blueprint §4 (configs, data, models, explainability, training, app, notebooks, outputs).
- `requirements.txt`, `.gitignore`, `README.md` in place.
- All Python modules stubbed with docstrings + `# TODO` markers.
- Public GitHub repo created (Account C) and pushed.

### 2026-05-12 — Compute setup decided
- Colab Pro on **Account A**, 5 TB Drive on **Account B**, public GitHub on **Account C**.
- Cross-account `drive.mount()` trick locked in: authorize as B in OAuth picker → all writes count against B's 5 TB.
- Experiment tracking: **TensorBoard**, not wandb (wandb signup steered into 30-day Pro trial; TensorBoard is account-free and writes logs to Drive).

### 2026-05-13 — Data pipeline implemented and run
- `data/download_fakeddit.py` — parallel downloader, resizes to 224×224 during download, atomic writes, resumable across session crashes (state file intersected with on-disk files so wiped JPEGs re-queue).
- `data/build_hdf5.py` — stratified 70/15/15 split, gzip-4 chunked HDF5, also writes per-split CSVs.
- `data/dataset.py` — `HEMTClipDataset` with lazy `h5py.File` open (survives DataLoader fork), exact CLIP normalisation, NaN-alpha warning at construction.
- **Bug fixed:** original resume logic trusted the Drive state file blindly; after a Colab disconnect (local `/content/` wiped) it would skip IDs whose JPEG no longer existed. Now intersects the state list with files actually present on disk.

### 2026-05-13 — HDF5 built and sanity-checked
- **Samples:** 17,149 (target was 15K, well over)
- **Class balance:** 8,716 real / 8,433 fake (50.8 / 49.2 — essentially perfect)
- **Splits:** train 12,003 / val 2,573 / test 2,573
- **Image tensor:** `(17149, 3, 224, 224) uint8`
- **File size:** ~1.9 GB on Drive (gzip-4 compressed; original 10 GB estimate was for uncompressed)
- **Alpha column:** all NaN (sentinel — fills in next step)
- **Location:** `/content/drive/MyDrive/hemt-clip-fnd/data/fakeddit.h5`

---

## Next (immediate)

### Re-run ablation with tuned config (v2)
- Patched `configs/base.yaml` on 2026-05-17: `trainable_layers 2→4`, `trainable_blocks 2→4`, `stage2.epochs 3→6`, `early_stopping_patience 2→3`, `label_smoothing 0.1→0.0`.
- Re-run via `!python -m training.ablation_runner --force` in `notebooks/03_full_training.ipynb` (cell added 2026-05-17). `--force` required because v1 best.pt files are within the 24h skip window.
- Expected ~45–55 min on T4. Target: `hemt_clip` val F1 0.83–0.85 (v1 was 0.8004).
- After re-run, refresh the v1 results table in the notebook with v2 numbers (mark v1 as historical) and re-stamp the `outputs/ablation_summary_*` files.

### Held-out test evaluation (`notebooks/04_evaluation.ipynb` → `training/evaluate.py`)
- Load each variant's `best.pt`, run on `test` split (n=2573), emit per-variant test_{acc, f1, prec, rec} + confusion matrix PNG + ROC curve PNG.
- 4-row test-set comparison table → drops into Chapter 6 alongside the val table.
- ~3–4 min total on T4 (inference only, no training).

### Then: XAI artefacts (Chapter 6 figures)
- `explainability/attention_viz.py` — pick ~10 test examples (mix of correct/incorrect, fake/real), extract `attn` from `CrossAttentionFusion`, reshape `(H, 1, P)` → 7×7 patch grid, overlay heatmap on the 224×224 image.
- `explainability/shap_text.py` — KernelExplainer over text-only branch on ~30 samples, save token-importance bar plots.

### Done 2026-05-16 — Full HEMT-CLIP training run
- `notebooks/03_full_training.ipynb` ran on T4, **6 min 24s total wall-clock** (overestimated 45 min by 7×).
- Stage 1 (1 ep, lr=1e-4, encoders frozen): val F1=0.7465.
- Stage 2 (3 ep, lr=2e-5, last-2-layers unfrozen): val F1 0.7529 → **0.8004** (best) → 0.7976.
- No early stop (patience=2, only 1 epoch w/o improvement).
- Tiny train/val gap (0.801 vs 0.794) → no overfitting on 12K samples; could push further but diminishing returns.
- Fine-tuning premium: +5.4 pt F1 over frozen-encoder head-only baseline (0.747 → 0.800) — useful for report's "why fine-tune" argument.
- Best ckpt: `hemt_hemt_clip_20260516-1428_best.pt` on Drive.
- Cosmetic warnings (no functional impact): `torch.cuda.amp.GradScaler` deprecation, gradient-checkpointing "no inputs require grad" warning from frozen early layers. Easy cleanups for later.

### Done 2026-05-16 — Ablation runner (`training/ablation_runner.py`)
- Subprocess-per-variant orchestration (`python -m training.train --variant X` × 4) — clean GPU memory between runs, isolated failure surface.
- Skip-if-recent default: looks for `*_{variant}_*_best.pt` mtime within 24h. Override with `--force`.
- Per-variant logfiles tee'd to `<runs>/../ablation_logs/{variant}_{ts}.log` alongside live console stream.
- Summary writer: reads `best_val_f1` from each `best.pt`'s embedded `TrainState`, emits `outputs/ablation_summary_{ts}.{csv,md}` with variant / F1 / stage / epoch / step / ckpt name.
- CLI: `--variants V1 V2 ...` (subset), `--force` (re-train all), `--skip-summary` (partial runs).

### Done 2026-05-16 — Training loop (`training/train.py`)
- Two-stage fine-tune: stage 1 freezes ALL encoder params (head warmup, lr=1e-4); stage 2 calls each encoder's `_freeze()` to restore last-N trainable (lr=2e-5). Optim + scheduler rebuilt fresh per stage.
- fp16 autocast + `torch.cuda.amp.GradScaler`, AdamW + `transformers.get_linear_schedule_with_warmup` (warmup_ratio from cfg), grad-clip(1.0), grad-accum from cfg.
- Gradient checkpointing toggled on both backbones if `cfg.training.gradient_checkpointing` is true.
- Auto batch-size detection (T4/L4/A100 via `cuda.get_device_name`).
- Atomic per-epoch ckpt (`.tmp` then `os.replace`), `cleanup_old_checkpoints` keeps last N. Separate `best.pt` on val-F1 improvement.
- Full resume: model + optim + scheduler + scaler + RNG (torch/cuda/numpy/random) + `TrainState(stage, epoch, global_step, best_val_f1, epochs_since_improve)`. Stage-aware: if resumed mid-stage, optim/scheduler reload; if resumed post-stage-1, stage 1 is skipped.
- TB logs: `train/{loss,lr}` per step (every `log_every_n_steps`), `train_epoch/{loss,acc}` and `val/{loss,acc,f1,prec,rec}` per epoch, weight+grad histograms per epoch.
- Early stop (stage 2 only): val F1 patience from cfg.

### Done 2026-05-16 — Smoke test
- `notebooks/02_smoke_test.ipynb` ran on T4, full output committed.
- 500-sample train / 100-sample val, `hemt_clip` variant, 5 epochs, **27s total**.
- Final: train_loss=0.228 / train_acc=98.6% (passed >85% gate).
- Val_acc held 67–80% (above chance — model learning real signal, not just memorising).
- Val_loss climbed from epoch 3 onwards → expect early-stopping to fire around stage-2 epoch 2–3 on the real run.
- Bootstrap cell added by user: env-var setup, idempotent clone/pull, `jax`/`flax` uninstall (they forced `numpy>=2` and broke pinned `numpy 1.26.4`), HDF5 copy from Drive to local SSD.

### Done 2026-05-16 — Alpha precomputation
- `data/precompute_alpha.py` implemented (CLIPModel + CLIPTokenizer, fp16 autocast, resumable in-place write to `f['alpha']`).
- Ran on T4: 17,149 rows, **NaN=0**, min=0.076 / mean=0.267 / max=0.437 — healthy band for ViT-B/32 on Fakeddit.

### Done 2026-05-16 — Model modules
- `models/text_encoder.py` — RoBERTa-base with embeddings + first 10 layers frozen, last 2 + projection (768→512, LayerNorm, Dropout) trainable. Outputs `[CLS]` projected to 512.
- `models/image_encoder.py` — CLIPVisionModel (ViT-B/32) with patch embedding + first 10 blocks frozen, last 2 + post_layernorm trainable. Local 768→512 projections for both pooled CLS and patch tokens (so cross-attention K/V are already in the 512 space). Returns `ImageEncoderOutput(pooled, patches)`.
- `models/fusion.py` — `CrossAttentionFusion`: 8-head MHA (text Q over patch K/V) + residual/LN + FFN(512→2048→512) + residual/LN. Returns `(fused (B,512), attn (B, H, 1, P))` un-averaged for XAI.
- `models/classifier.py` — `ClassifierHead(in_dim, 256, 2)`, in_dim wired by HEMTCLIP based on variant.
- `models/hemt_clip.py` — Assembler with variant switch over `{text_only, image_only, concat_fusion, hemt_clip}`. `use_alpha` auto-disabled for unimodal baselines. `build_from_config(cfg)` factory reads `configs/base.yaml`'s `model` block.

---

## Backlog (in order)

1. **Alpha precomputation** — `data/precompute_alpha.py` (next).
2. **Model modules** — implement `models/text_encoder.py`, `image_encoder.py`, `fusion.py`, `classifier.py`, `hemt_clip.py`. Variant switch in `hemt_clip.py` handles all four ablations.
3. **Smoke test** — overfit ~500 samples in `notebooks/02_smoke_test.ipynb` to verify pipeline before burning compute units on a real run.
4. **Training loop** — `training/train.py` with two-stage fine-tune, fp16, grad checkpointing, per-epoch Drive checkpoints, TensorBoard logging, resume-from-checkpoint.
5. **Ablation runner** — `training/ablation_runner.py` orchestrating all 4 variants.
6. **Evaluation** — `training/evaluate.py` for metrics + plots (confusion matrix, ROC, per-class P/R, ablation comparison table).
7. **Explainability** — `explainability/attention_viz.py` (10 examples) and `explainability/shap_text.py` (~30 samples).
8. **Streamlit demo** — `app/streamlit_app.py` with ngrok tunnel for viva.
9. **Report writing** — Chapter 6 (Results & Discussion), Chapter 7 (Conclusion), existing-report fixes.
10. **Slides + dry runs** — viva prep.

---

## Notes / gotchas (for future-me)

- **Drive FUSE is eventually consistent.** A file freshly `cp`'d to Drive may show in `ls` but fail on `h5py.File(...)` for a few seconds. Re-run the cell, or read the local copy.
- **`!command` is Colab notebook magic, not bash.** In the Colab terminal pane, drop the `!`.
- **Mount account matters more than Colab account.** Pick Account B (5 TB) in the OAuth popup — defaults to A and that's the trap.
- **Compute units are only consumed on accelerator runtimes.** Use CPU runtime for data work; switch to GPU only for alpha precompute and training.
- **Local `/content/` wipes on disconnect.** Don't trust anything there past the session. Drive is the only persistent layer.
- **HDF5 gzip-4 is tighter than I expected.** 17K samples = 1.9 GB, not 10 GB. Plenty of headroom on 5 TB.
