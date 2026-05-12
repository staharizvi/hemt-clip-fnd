"""Intrinsic explainability — cross-attention heatmaps.

Extract attention weights from the fusion module during a forward pass:
    weights.shape == (batch, num_heads, query_len=1, key_len=num_patches)
Average across heads, reshape patch axis to a 7×7 grid (for ViT-B/32 at 224px),
upsample to image resolution, blend over the original PIL image with matplotlib.

Also colours text tokens by attention intensity (or by gradient * activation
for token-side attribution) for the side-by-side report figures.
"""

# TODO: implement attention extraction + overlay rendering
