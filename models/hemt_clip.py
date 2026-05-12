"""Full assembled HEMT-CLIP model.

Pipeline:
    text  → TextEncoder      → text_feats   (B, 512)
    image → ImageEncoder     → img_feats    (B, num_patches, 512)
    (text_feats, img_feats)  → CrossAttentionFusion → fused (B, 512), attn (B, H, 1, P)
    [fused, alpha]           → ClassifierHead → logits (B, 2)

Ablation variants (variant kwarg):
    "text_only"      → classifier on text_feats only
    "image_only"     → classifier on pooled img_feats only
    "concat_fusion"  → classifier on [text_feats, pooled_img, alpha]
    "hemt_clip"      → full cross-attention fusion + alpha
"""

# TODO: implement HEMTCLIP(nn.Module) with variant switch
