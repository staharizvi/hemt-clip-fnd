"""Cross-attention fusion module.

nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)
    Q = text_features, K = V = image_features (patch tokens).
Residual connection + LayerNorm after attention.
Position-wise FFN: Linear(512, 2048) → GELU → Dropout → Linear(2048, 512)
                   + residual + LayerNorm.

Returns the fused 512-dim representation AND the attention weights
(averaged across heads externally for explainability viz).
"""

# TODO: implement CrossAttentionFusion(nn.Module)
