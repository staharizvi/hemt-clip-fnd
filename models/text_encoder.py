"""RoBERTa text encoder with projection head.

Base: roberta-base (125M).
Frozen: embeddings + first 10 transformer layers.
Trainable: last 2 transformer layers + projection head.
Output: [CLS] representation, projected 768 -> 512 dim.

Projection: Linear(768, 512) -> LayerNorm -> Dropout(0.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import RobertaModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",
        proj_dim: int = 512,
        trainable_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_dim = self.backbone.config.hidden_size  # 768 for roberta-base

        self._freeze(trainable_layers)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.out_dim = proj_dim

    def _freeze(self, trainable_layers: int) -> None:
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        n_layers = len(self.backbone.encoder.layer)
        n_frozen = n_layers - trainable_layers
        for i, layer in enumerate(self.backbone.encoder.layer):
            for p in layer.parameters():
                p.requires_grad = i >= n_frozen

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.projection(cls)
