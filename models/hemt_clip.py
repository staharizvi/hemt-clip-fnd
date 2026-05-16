"""Full assembled HEMT-CLIP model with ablation variants.

Pipeline (full variant):
    text  -> TextEncoder      -> text_feats   (B, 512)
    image -> ImageEncoder     -> pooled (B, 512), patches (B, P, 512)
    (text_feats, patches)     -> CrossAttentionFusion -> fused (B, 512), attn (B, H, 1, P)
    [fused, alpha]            -> ClassifierHead -> logits (B, 2)

Ablation variants (variant kwarg):
    "text_only"      : classifier on text_feats only
    "image_only"     : classifier on pooled image features only
    "concat_fusion"  : classifier on [text_feats, pooled_img, alpha]
    "hemt_clip"      : full cross-attention fusion + alpha

forward() accepts the dict produced by HEMTClipDataset:
    input_ids, attention_mask, pixel_values, alpha, label (label is unused).
Returns dict with `logits` and (for cross-attention variants) `attention_weights`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.classifier import ClassifierHead
from models.fusion import CrossAttentionFusion
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder

VARIANTS = ("text_only", "image_only", "concat_fusion", "hemt_clip")


class HEMTCLIP(nn.Module):
    def __init__(
        self,
        variant: str = "hemt_clip",
        text_cfg: dict | None = None,
        image_cfg: dict | None = None,
        fusion_cfg: dict | None = None,
        classifier_cfg: dict | None = None,
        use_alpha: bool = True,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

        text_cfg = text_cfg or {}
        image_cfg = image_cfg or {}
        fusion_cfg = fusion_cfg or {}
        classifier_cfg = classifier_cfg or {}

        self.variant = variant
        # alpha is a text-image agreement signal; ignore it for unimodal baselines.
        self.use_alpha = use_alpha and variant in {"concat_fusion", "hemt_clip"}

        needs_text = variant != "image_only"
        needs_image = variant != "text_only"
        needs_fusion = variant == "hemt_clip"

        self.text_encoder = TextEncoder(**text_cfg) if needs_text else None
        self.image_encoder = ImageEncoder(**image_cfg) if needs_image else None
        self.fusion = CrossAttentionFusion(**fusion_cfg) if needs_fusion else None

        proj_dim = (self.text_encoder.out_dim if self.text_encoder is not None
                    else self.image_encoder.out_dim)
        in_dim = self._classifier_in_dim(proj_dim)
        self.classifier = ClassifierHead(in_dim=in_dim, **classifier_cfg)

    def _classifier_in_dim(self, proj_dim: int) -> int:
        if self.variant in {"text_only", "image_only"}:
            return proj_dim
        if self.variant == "concat_fusion":
            return 2 * proj_dim + (1 if self.use_alpha else 0)
        if self.variant == "hemt_clip":
            return proj_dim + (1 if self.use_alpha else 0)
        raise AssertionError("unreachable")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        text_feats = None
        pooled_img = None
        patches = None

        if self.text_encoder is not None:
            text_feats = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        if self.image_encoder is not None:
            img_out = self.image_encoder(batch["pixel_values"])
            pooled_img = img_out.pooled
            patches = img_out.patches

        attn_weights: torch.Tensor | None = None

        if self.variant == "text_only":
            features = text_feats
        elif self.variant == "image_only":
            features = pooled_img
        elif self.variant == "concat_fusion":
            parts = [text_feats, pooled_img]
            if self.use_alpha:
                parts.append(batch["alpha"].unsqueeze(-1))
            features = torch.cat(parts, dim=-1)
        elif self.variant == "hemt_clip":
            fused, attn_weights = self.fusion(text_feats, patches)
            if self.use_alpha:
                features = torch.cat([fused, batch["alpha"].unsqueeze(-1)], dim=-1)
            else:
                features = fused
        else:
            raise AssertionError("unreachable")

        logits = self.classifier(features)
        out: dict[str, Any] = {"logits": logits}
        if attn_weights is not None:
            out["attention_weights"] = attn_weights
        return out

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_from_config(cfg: dict, variant: str | None = None) -> HEMTCLIP:
    """Construct a HEMTCLIP from the nested `model` block of base.yaml."""
    m = cfg["model"]
    return HEMTCLIP(
        variant=variant or "hemt_clip",
        text_cfg={
            "model_name": m["text"]["name"],
            "proj_dim": m["text"]["proj_dim"],
            "trainable_layers": m["text"]["trainable_layers"],
            "dropout": m["text"]["dropout"],
        },
        image_cfg={
            "model_name": m["image"]["name"],
            "proj_dim": m["image"]["embed_dim"],
            "trainable_blocks": m["image"]["trainable_blocks"],
        },
        fusion_cfg={
            "embed_dim": m["image"]["embed_dim"],
            "num_heads": m["fusion"]["num_heads"],
            "ffn_dim": m["fusion"]["ffn_dim"],
            "dropout": m["fusion"]["dropout"],
        },
        classifier_cfg={
            "hidden_dim": m["classifier"]["hidden_dim"],
            "dropout": m["classifier"]["dropout"],
            "num_classes": m["classifier"]["num_classes"],
        },
        use_alpha=m.get("use_alpha", True),
    )
