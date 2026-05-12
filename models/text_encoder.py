"""RoBERTa text encoder with projection head.

Base: roberta-base (125M).
Frozen: embeddings + first 10 transformer layers.
Trainable: last 2 transformer layers + projection head.
Output: [CLS] representation, projected 768 → 512 dim.

Projection: Linear(768, 512) → LayerNorm → Dropout(0.1).
"""

# TODO: implement TextEncoder(nn.Module)
