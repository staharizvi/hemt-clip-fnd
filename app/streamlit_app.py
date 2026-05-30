"""Streamlit demo for HEMT-CLIP — multimodal fake-news detection (Blueprint §11).

Layout:
    - Text + image input (custom upload OR pick a test sample)
    - Prediction (REAL/FAKE) with confidence bars and ground truth (if sample picked)
    - CLIP text-image alignment α with verbal interpretation
    - Tabs: Attention heatmap | SHAP (text) | How it works

Models are loaded once via @st.cache_resource. Attention extraction is live (one
forward pass, <1 s on GPU); SHAP attribution is looked up from precomputed
outputs/xai/shap/ produced by explainability/shap_text.py — running SHAP live
would block the UI for ~1 s per perturbation × 1000 perturbations per call.

Configuration via environment variables (set before `streamlit run`):
    HEMT_CLIP_CKPT      : explicit checkpoint path. If unset, auto-discovers
                          the latest non-`_seed*` hemt_clip best.pt in
                          HEMT_CLIP_CKPT_DIR (defaults to the Drive path).
    HEMT_CLIP_CKPT_DIR  : checkpoint discovery directory.
"""

from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Repo path setup so this file is runnable as `streamlit run app/streamlit_app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import CLIP_MEAN, CLIP_STD  # noqa: E402
from models.hemt_clip import build_from_config  # noqa: E402

# Streamlit page config — must be first Streamlit call.
st.set_page_config(page_title="HEMT-CLIP Demo", page_icon="🔍", layout="wide")


# ────────────────────────────────────────────────────────────────────────────
# Model + tokenizer loading (cached for the lifetime of the streamlit process)
# ────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading HEMT-CLIP model…")
def load_model(ckpt_path: str):
    with open(ROOT / "configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_from_config(cfg, variant="hemt_clip").to(device).eval()
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    return model, device, cfg


@st.cache_resource(show_spinner="Loading RoBERTa tokenizer…")
def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


@st.cache_resource(show_spinner="Loading CLIP for α…")
def load_clip(model_name: str):
    from transformers import CLIPModel, CLIPTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_name).to(device).eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return clip_model, clip_tokenizer, device


@st.cache_data(show_spinner="Loading test samples…")
def load_sample_examples(hdf5_path: str, n: int = 6, seed: int = 0):
    """Load a small set of test examples for the dropdown (3 real + 3 fake)."""
    examples: list[dict] = []
    if not Path(hdf5_path).exists():
        return examples
    with h5py.File(hdf5_path, "r") as f:
        splits = f["splits"][:].astype(str)
        labels = f["labels"][:]
        test_mask = splits == "test"
        rng = np.random.default_rng(seed)
        per_class = n // 2
        for label_val, name in [(0, "REAL"), (1, "FAKE")]:
            pool = np.where(test_mask & (labels == label_val))[0]
            if len(pool) == 0:
                continue
            for idx in rng.choice(pool, size=min(per_class, len(pool)), replace=False):
                img = f["images"][idx].transpose(1, 2, 0).astype(np.uint8)
                text = f["texts"][idx]
                text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
                examples.append({
                    "idx": int(idx),
                    "image": Image.fromarray(img),
                    "text": text,
                    "label": int(label_val),
                    "label_name": name,
                })
    return examples


# ────────────────────────────────────────────────────────────────────────────
# Inference + α + attention helpers
# ────────────────────────────────────────────────────────────────────────────


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return (arr - CLIP_MEAN[:, None, None]) / CLIP_STD[:, None, None]


@torch.no_grad()
def compute_alpha(clip_model, clip_tokenizer, device, pil_img: Image.Image, text: str) -> float:
    pixel_values = torch.from_numpy(preprocess_image(pil_img)).unsqueeze(0).to(device)
    enc = clip_tokenizer([text], padding="max_length", truncation=True,
                          max_length=77, return_tensors="pt").to(device)
    img_emb = F.normalize(clip_model.get_image_features(pixel_values=pixel_values).float(), dim=-1)
    txt_emb = F.normalize(clip_model.get_text_features(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).float(), dim=-1)
    return float((img_emb * txt_emb).sum(dim=-1).item())


@torch.no_grad()
def predict(model, tokenizer, device, pil_img: Image.Image, text: str,
            alpha: float, max_text_len: int = 128):
    """Returns (probs[2], attn_grid[P,P] or None)."""
    pixel_values = torch.from_numpy(preprocess_image(pil_img)).unsqueeze(0).to(device)
    enc = tokenizer([text], padding="max_length", truncation=True,
                    max_length=max_text_len, return_tensors="pt").to(device)
    batch = {
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "pixel_values":   pixel_values,
        "alpha":          torch.tensor([alpha], dtype=torch.float32, device=device),
        "label":          torch.tensor([0], dtype=torch.long, device=device),
    }
    out = model(batch)
    logits = out["logits"].float().cpu().numpy()[0]
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    attn_grid = None
    if "attention_weights" in out:
        a = out["attention_weights"].float().cpu().numpy()[0].mean(axis=0).squeeze(0)
        p = int(round(np.sqrt(len(a))))
        attn_grid = a.reshape(p, p)
    return probs, attn_grid


def render_attention_overlay(pil_img: Image.Image, attn_grid: np.ndarray) -> Image.Image:
    img_arr = np.array(pil_img.convert("RGB").resize((224, 224)))
    t = torch.from_numpy(attn_grid).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(224, 224), mode="bilinear",
                       align_corners=False).squeeze().numpy()
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_arr)
    ax.imshow(up, cmap="hot", alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def interpret_alpha(alpha: float) -> str:
    if alpha < 0.15:
        return f"**low** ({alpha:.3f}) — text and image semantics weakly aligned in CLIP space."
    if alpha < 0.30:
        return f"**moderate** ({alpha:.3f}) — typical band for Fakeddit headline + thumbnail pairs."
    return f"**high** ({alpha:.3f}) — strong CLIP text-image alignment."


def find_shap_for_sample(ds_idx: int, shap_dir: Path) -> Path | None:
    manifest_path = shap_dir / "shap_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in manifest:
        if entry.get("hdf5_row") == ds_idx:
            png = shap_dir / entry["file"]
            return png if png.exists() else None
    return None


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ────────────────────────────────────────────────────────────────────────────


def discover_ckpt() -> str | None:
    env_path = os.environ.get("HEMT_CLIP_CKPT")
    if env_path and Path(env_path).exists():
        return env_path
    ckpt_dir = Path(os.environ.get(
        "HEMT_CLIP_CKPT_DIR",
        "/content/drive/MyDrive/hemt-clip-fnd/checkpoints",
    ))
    if not ckpt_dir.exists():
        return None
    candidates = [c for c in ckpt_dir.glob("hemt_hemt_clip_*_best.pt")
                  if "_seed" not in c.name]
    if not candidates:
        return None
    return str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])


# ────────────────────────────────────────────────────────────────────────────
# Sidebar — config + about
# ────────────────────────────────────────────────────────────────────────────

st.sidebar.title("HEMT-CLIP")
st.sidebar.markdown("Multimodal fake-news detection on Fakeddit.")
st.sidebar.markdown("---")

ckpt_path = discover_ckpt()
if ckpt_path is None:
    st.error(
        "No `hemt_clip` checkpoint found. Set `HEMT_CLIP_CKPT` to an explicit "
        "`*_best.pt` path, or `HEMT_CLIP_CKPT_DIR` to a directory containing one."
    )
    st.stop()

model, device, cfg = load_model(ckpt_path)
tokenizer = load_tokenizer(cfg["model"]["text"]["name"])
clip_model, clip_tokenizer, _ = load_clip(cfg["model"]["image"]["name"])
samples = load_sample_examples(cfg["data"]["hdf5_path"])

st.sidebar.markdown(f"**Checkpoint:** `{Path(ckpt_path).name}`")
st.sidebar.markdown(f"**Device:** `{device.type}`")
st.sidebar.markdown(f"**Test samples loaded:** {len(samples)}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Live: prediction, α, attention heatmap. "
    "Precomputed: SHAP attribution (test samples only — live SHAP is too slow for an interactive demo)."
)

# ────────────────────────────────────────────────────────────────────────────
# Main panel
# ────────────────────────────────────────────────────────────────────────────

st.title("🔍 HEMT-CLIP Demo")
st.markdown("Paste a headline, upload an image (or pick a test sample), get a prediction with explanations.")

col_input, col_image = st.columns([1, 1])

uploaded_image: Image.Image | None = None
default_text = ""
truth_label: str | None = None
selected_ds_idx: int | None = None

with col_input:
    st.subheader("Input")

    sample_choice = 0
    if samples:
        sample_options = ["(none — custom input)"] + [
            f"[{s['label_name']}] sample {s['idx']}: {s['text'][:60]}…"
            if len(s['text']) > 60 else f"[{s['label_name']}] sample {s['idx']}: {s['text']}"
            for s in samples
        ]
        sample_choice = st.selectbox(
            "Or pick a test sample:",
            options=range(len(sample_options)),
            format_func=lambda i: sample_options[i],
        )

    if sample_choice > 0:
        selected = samples[sample_choice - 1]
        default_text = selected["text"]
        uploaded_image = selected["image"]
        truth_label = selected["label_name"]
        selected_ds_idx = selected["idx"]
    else:
        upload = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if upload is not None:
            uploaded_image = Image.open(upload)

    title_text = st.text_area("Title / headline:", value=default_text, height=80)
    analyze = st.button("🔬 Analyze", type="primary", use_container_width=True)

with col_image:
    if uploaded_image is not None:
        st.subheader("Image")
        st.image(uploaded_image, use_container_width=True)
    elif sample_choice == 0:
        st.info("Upload an image or pick a test sample to begin.")

# ────────────────────────────────────────────────────────────────────────────
# Output panel — fires on Analyze click
# ────────────────────────────────────────────────────────────────────────────

if analyze:
    if uploaded_image is None:
        st.error("No image provided.")
        st.stop()
    if not title_text.strip():
        st.error("No title provided.")
        st.stop()

    with st.spinner("Running HEMT-CLIP inference + cross-attention extraction…"):
        alpha = compute_alpha(clip_model, clip_tokenizer, device, uploaded_image, title_text)
        probs, attn_grid = predict(model, tokenizer, device, uploaded_image, title_text, alpha)

    pred = int(probs.argmax())
    pred_name = "FAKE" if pred == 1 else "REAL"
    pred_color = "red" if pred == 1 else "green"
    conf = float(probs[pred])

    st.markdown(f"### Prediction: :{pred_color}[**{pred_name}**] — confidence {conf:.1%}")
    if truth_label is not None:
        match = "✓ correct" if truth_label == pred_name else "✗ wrong"
        st.markdown(f"**Ground truth:** `{truth_label}` ({match})")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**Class probabilities**")
        st.progress(float(probs[1]), text=f"FAKE: {probs[1]:.1%}")
        st.progress(float(probs[0]), text=f"REAL: {probs[0]:.1%}")
    with pcol2:
        st.markdown("**CLIP text-image alignment (α)**")
        st.markdown(interpret_alpha(alpha))

    st.markdown("---")
    tab_attn, tab_shap, tab_about = st.tabs(["Attention heatmap", "SHAP (text)", "How it works"])

    with tab_attn:
        st.markdown(
            "Cross-attention from the text-side [CLS] query over CLIP ViT-B/16 image patch tokens "
            "(8-head MHA, averaged over heads). Reshaped from the 14×14 patch grid and bilinear-"
            "upsampled to 224×224."
        )
        if attn_grid is not None:
            overlay = render_attention_overlay(uploaded_image, attn_grid)
            st.image(overlay, caption="Cross-attention overlay (hotter = higher attention weight)")
        else:
            st.info("No attention weights returned (not a `hemt_clip` model?).")

    with tab_shap:
        st.markdown(
            "SHAP attribution on the **text-only** model branch (Blueprint §10.2 — methodologically "
            "cleaner than running SHAP on the multimodal branch). Live SHAP would block the UI for "
            "~1 s per perturbation × 1000 perturbations; pre-computation makes the demo interactive."
        )
        shap_png = (find_shap_for_sample(selected_ds_idx, ROOT / "outputs/xai/shap")
                    if selected_ds_idx is not None else None)
        if shap_png is not None:
            st.image(str(shap_png),
                     caption=f"Pre-computed SHAP attribution — test sample {selected_ds_idx}")
        elif selected_ds_idx is not None:
            st.info(
                f"Sample {selected_ds_idx} wasn't included in the 30-sample SHAP pre-computation. "
                "Pick a different test sample, or see the report's §6.4 for cross-method "
                "(SHAP + LIME) examples."
            )
        else:
            st.info(
                "SHAP is only shown for test samples (not custom inputs). "
                "Pick a test sample from the dropdown to see its pre-computed attribution."
            )

    with tab_about:
        st.markdown("""
### HEMT-CLIP — architecture

A binary classifier (Real vs Fake) on the Fakeddit multimodal corpus, combining:

- **Text encoder:** RoBERTa-base (last 2 transformer layers fine-tuned), projected 768 → 512.
- **Image encoder:** CLIP ViT-B/16 vision tower (last 2 transformer blocks fine-tuned), projected per patch 768 → 512.
- **Cross-attention fusion:** 8-head MHA with text [CLS] as query, image patch tokens as keys/values; residual + LayerNorm + position-wise FFN.
- **α scalar:** CLIP text-image cosine similarity (computed externally with the full CLIP model), concatenated to the fused 512-dim representation as a 513-dim feature vector.
- **Classifier head:** Linear(513 → 256) → ReLU → Dropout(0.3) → Linear(256 → 2).

### Test-set headline (n = 2,573 held-out samples)

| Variant | test F1 | test AUC |
|---|---:|---:|
| `text_only` | 0.783 | 0.854 |
| `image_only` | 0.814 | 0.882 |
| `concat_fusion` | **0.832** | **0.904** |
| `hemt_clip` | 0.828 | 0.892 |

Cross-attention does **not** beat simple concatenation on raw F1 (concat wins by +0.4 pt). Its
contribution is **intrinsic explainability** — the attention heatmap shown above is a capability
concatenation architecturally cannot provide. See report Chapter 6 for the full discussion.

### Explainability

- **Cross-attention heatmap** (live, this tab) — intrinsic to the fusion architecture.
- **SHAP** (precomputed, text tab) — Owen partition explainer over BPE subword tokens.
- **LIME** (offline, see report §6.4) — local linear surrogate over word-deletion perturbations.

SHAP and LIME are run on the `text_only` model so attribution is unambiguous; multimodal SHAP is
intentionally skipped per Blueprint §10.2.
""")
