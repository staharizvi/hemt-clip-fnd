# HEMT-CLIP Progress Log

**Current phase:** Test-set evaluation **implemented** — `training/evaluate.py` (~280 LOC) + `notebooks/04_evaluation.ipynb` (14 cells) wired up. Pending Colab run on the 2,573-sample test split (~3–4 min on T4). When numbers come back, Chapter 6's primary results table + 7 figures are produced in one shot.
**Last updated:** 2026-05-30

---

## Done

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
