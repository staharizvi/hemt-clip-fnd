"""CLIP ViT image encoder (vision tower only).

Backbone is configurable via `model_name` — both ViT-B/32 (49 patches) and
ViT-B/16 (196 patches) share hidden_dim=768, so the module is agnostic to
patch count; cross-attention K/V just sees a longer sequence.
Input: 224x224 RGB, normalised with CLIP mean/std (done in the dataset).
Frozen: patch embedding + all but the last `trainable_blocks` transformer blocks.

Returns BOTH:
    pooled  : (B, proj_dim)        — CLS token, projected, for image_only / concat fusion.
    patches : (B, P, proj_dim)     — patch tokens, projected, used as K/V in cross-attention.
                                     P = (image_size / patch_size)^2.

The projections are local Linear(768 -> proj_dim) layers (not CLIP's
visual_projection) so the module is self-contained and the pooled and
patch streams can be tuned independently during fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


@dataclass
class ImageEncoderOutput:
    pooled: torch.Tensor   # (B, proj_dim)
    patches: torch.Tensor  # (B, num_patches, proj_dim)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        proj_dim: int = 512,
        trainable_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size  # 768 for both ViT-B/32 and ViT-B/16

        self._freeze(trainable_blocks)

        self.pooled_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.out_dim = proj_dim

    def _freeze(self, trainable_blocks: int) -> None:
        vm = self.backbone.vision_model
        for p in vm.embeddings.parameters():
            p.requires_grad = False
        for p in vm.pre_layrnorm.parameters():
            p.requires_grad = False
        n_layers = len(vm.encoder.layers)
        n_frozen = n_layers - trainable_blocks
        for i, layer in enumerate(vm.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = i >= n_frozen
        for p in vm.post_layernorm.parameters():
            p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> ImageEncoderOutput:
        out = self.backbone(pixel_values=pixel_values)
        hidden = out.last_hidden_state          # (B, 1 + P, 768): [CLS, patch_0, ..., patch_{P-1}]
        cls = hidden[:, 0]                      # (B, 768)
        patch_tokens = hidden[:, 1:]            # (B, P, 768)
        return ImageEncoderOutput(
            pooled=self.pooled_proj(cls),
            patches=self.patch_proj(patch_tokens),
        )
