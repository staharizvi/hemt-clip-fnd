# HEMT-CLIP Progress Log

**Current phase:** v2 ablation done. `hemt_clip` plateaued at val F1=0.8069 (target was 0.83–0.85) and is now *behind* `concat_fusion` (0.8133). Patched `fusion.dropout 0.1→0.2` for v3 to fight overfit in the cross-attention block.
**Last updated:** 2026-05-30

---

## Done

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
