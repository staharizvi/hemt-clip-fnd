"""Runs all four ablation variants end-to-end.

Variants (Blueprint §7):
    A: text_only       — RoBERTa + classifier
    B: image_only      — CLIP ViT + classifier
    C: concat_fusion   — [text, image, alpha] → classifier
    D: hemt_clip       — full cross-attention fusion + alpha → classifier

Trains each variant on the same data splits with the same seed, saves
best checkpoint per variant, then aggregates metrics into a single
comparison table for the report.
"""

# TODO: orchestrate train.py + evaluate.py across all variants
