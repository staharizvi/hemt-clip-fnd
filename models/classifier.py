"""2-layer MLP classifier head.

Input: [fused_features (512), alpha (1)] concatenated → 513-dim
       (or 512-dim when use_alpha=False — ablation flag).
Architecture: Linear(513, 256) → ReLU → Dropout(0.3) → Linear(256, 2).
Output: logits for [Real, Fake]; softmax applied at inference time.
"""

# TODO: implement ClassifierHead(nn.Module)
