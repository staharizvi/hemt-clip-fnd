"""CLIP ViT-B/32 image encoder (vision tower only).

Base: openai/clip-vit-base-patch32.
Input: 224×224 RGB, normalised with CLIP mean/std.
Frozen: patch embedding + first 10 transformer blocks.
Trainable: last 2 transformer blocks.
Output: 512-dim pooled vision representation (already matches text proj_dim).

Also exposes patch-token embeddings for cross-attention K/V and
patch-level attention visualisation.
"""

# TODO: implement ImageEncoder(nn.Module)
