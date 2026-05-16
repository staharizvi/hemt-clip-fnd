# HEMT-CLIP Progress Log

**Current phase:** Model modules implemented; smoke test next.
**Last updated:** 2026-05-16

---

## Done

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

### Smoke test (`notebooks/02_smoke_test.ipynb`)
- Overfit ~500 samples on the `hemt_clip` variant to verify the full pipeline (dataset → model → loss → backward → optimiser) runs end-to-end before burning compute units on the real training run.
- Goal: training loss must drop near zero in a few epochs. If it doesn't, there's a wiring bug somewhere.

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
