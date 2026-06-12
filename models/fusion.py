"""Cross-attention fusion module.

Q = text features  (B, 1, 512)
K = V = image patch features  (B, P, 512)

nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)
    + residual + LayerNorm
Position-wise FFN: Linear(512, 2048) -> GELU -> Dropout -> Linear(2048, 512)
    + residual + LayerNorm

Returns:
    fused : (B, 512)
    attn  : (B, num_heads, 1, P) — per-head, kept un-averaged for the
            explainability viz; trainer can mean over heads for logging.

CLIP-similarity gating (optional, for the `gated_fusion` ablation):
    If `alpha` is passed to forward(), the attended representation is blended
    with the raw text query as `alpha * attended + (1 - alpha) * text`, the
    similarity-weighted fusion used by prior CLIP-based FND work (e.g. FND-CLIP).
    When `alpha` is None (the default, used by `hemt_clip`) the module is
    unchanged — the gate adds no parameters, so checkpoints are interchangeable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = embed_dim

    def forward(
        self,
        text_feats: torch.Tensor,        # (B, 512)
        image_patches: torch.Tensor,     # (B, P, 512)
        alpha: torch.Tensor | None = None,  # (B,) CLIP similarity — enables gating when given
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = text_feats.unsqueeze(1)              # (B, 1, 512)
        attn_out, attn_weights = self.attn(
            query=q,
            key=image_patches,
            value=image_patches,
            need_weights=True,
            average_attn_weights=False,          # keep per-head for viz: (B, H, 1, P)
        )
        attended = self.norm1(q + self.dropout(attn_out))   # (B, 1, 512)
        if alpha is not None:
            a = alpha.view(-1, 1, 1).to(attended.dtype)   # match autocast dtype
            attended = a * attended + (1.0 - a) * q
        x = self.norm2(attended + self.dropout(self.ffn(attended)))
        fused = x.squeeze(1)                          # (B, 512)
        return fused, attn_weights
