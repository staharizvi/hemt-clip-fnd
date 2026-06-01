"""2-layer MLP classifier head.

Architecture: Linear(in_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, num_classes).
Output: logits for [Real, Fake]; softmax applied at inference time.

in_dim is variant-dependent (set by HEMTCLIP):
    text_only      : 512
    image_only     : 512
    concat_fusion  : 512 + 512 + (1 if use_alpha else 0)
    hemt_clip      : 512 + (1 if use_alpha else 0)
"""

from __future__ import annotations

import torch
import torch.nn as n


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
