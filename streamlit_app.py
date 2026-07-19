"""HEMT-CLIP project showcase and interactive model demo.

Run from the repository root with ``streamlit run streamlit_app.py``.
"""

from __future__ import annotations

import html
import hashlib
import importlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import inference as inference_module  # noqa: E402

# Streamlit keeps imported modules alive across script hot-reloads. During local
# development an already-running server can therefore retain an older
# ``app.inference`` that predates a newly added helper. Reload only in that stale
# case; normal runs preserve the cached model class and resource identity.
if not hasattr(inference_module, "missing_runtime_packages"):
    inference_module = importlib.reload(inference_module)

CHECKPOINT_DIR = inference_module.CHECKPOINT_DIR
HEMTPredictor = inference_module.HEMTPredictor
PredictionResult = inference_module.PredictionResult
discover_checkpoint = inference_module.discover_checkpoint
load_hdf5_sample = inference_module.load_hdf5_sample
missing_runtime_packages = inference_module.missing_runtime_packages
resolve_hdf5_path = inference_module.resolve_hdf5_path


st.set_page_config(
    page_title="HEMT-CLIP | Multimodal Classification Study",
    page_icon="α",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUMMARY_PATH = ROOT / "outputs" / "roc_pr_curves" / "summary_test.csv"
OUTPUTS = ROOT / "outputs"

MODEL_NAMES = {
    "text_only": "Text-only baseline",
    "image_only": "Image-only baseline",
    "concat_fusion": "Feature concatenation",
    "hemt_clip": "Cross-attention + α feature",
    "gated_fusion": "α-gated cross-attention",
}

CURATED = {
    "False connection · lamp vs moon": {
        "row": 16379,
        "story": "FAKE-labelled test instance: the title–image relation provides an observable multimodal inconsistency.",
    },
    "Liberation of Paris · real": {
        "row": 15662,
        "story": "REAL-labelled test instance depicting a Liberation of Paris scene.",
    },
    "Polar bears · real": {
        "row": 15368,
        "story": "REAL-labelled wildlife instance used to inspect patch-level cross-attention.",
    },
    "Graduation claim · fake": {
        "row": 15657,
        "story": "FAKE-labelled instance in which the image and textual claim provide different evidential signals.",
    },
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f5f3ee; --paper-deep: #ece9e1; --ink: #171717;
            --muted: #6f6b63; --line: #d4d0c7; --red: #b42318;
            --blue: #2457d6; --green: #13795b; --amber: #9a6700;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stSidebar"] {
            background: #ebe8e0; border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #5d5952; }
        [data-testid="stSidebar"] .stRadio label {
            padding: .5rem 0; border-radius: 0; border-bottom: 1px solid transparent;
            font-size: .88rem;
        }
        [data-testid="stSidebar"] .stRadio label:hover { border-bottom-color: #aaa59a; }
        [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .15rem; }
        .block-container { max-width: 1160px; padding-top: 2.7rem; padding-bottom: 4rem; }
        h1, h2, h3 { letter-spacing: -.035em; color: var(--ink); }
        .brand-lockup { margin: .2rem 0 2.8rem; }
        .brand-mark {
            display: block; padding-bottom: .55rem; border-bottom: 3px solid var(--red);
            font-family: Georgia, 'Times New Roman', serif; font-size: 1.35rem;
            font-weight: 700; color: var(--ink); letter-spacing: -.04em;
        }
        .brand-sub { color: #777269; font-size: .66rem; letter-spacing: .12em; margin: .65rem 0 0; }
        .hero {
            border-top: 5px solid var(--ink); border-bottom: 1px solid var(--line);
            padding: 2.4rem 0 2.7rem; background: transparent;
        }
        .hero-grid {
            display: grid; grid-template-columns: minmax(0,1.75fr) minmax(15rem,.75fr);
            gap: 4rem; align-items: end;
        }
        .eyebrow {
            color: var(--red); font-size: .65rem; font-weight: 750;
            letter-spacing: .14em; text-transform: uppercase; margin-bottom: .8rem;
        }
        .hero h1 {
            color: var(--ink); font-family: Georgia, 'Times New Roman', serif;
            font-weight: 500; font-size: clamp(2.6rem,5.3vw,4.8rem);
            line-height: .98; margin: 0; max-width: 46rem;
        }
        .hero-copy { color: #5f5b54; font-size: 1rem; line-height: 1.65; max-width: 42rem; margin: 1.35rem 0 1.25rem; }
        .pill {
            display: inline-block; padding: .22rem 0; margin: .15rem 1rem .15rem 0;
            border-bottom: 1px solid #aaa59a; color: #514d46;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: .68rem; font-weight: 600;
        }
        .hero-result { border-left: 1px solid var(--line); padding-left: 1.7rem; }
        .hero-result-label { color: var(--muted); font-size: .65rem; letter-spacing: .12em; text-transform: uppercase; }
        .hero-score {
            display: block; color: var(--red); font-family: Georgia, 'Times New Roman', serif;
            font-size: 4.2rem; line-height: 1; margin: .5rem 0 .2rem;
        }
        .hero-unit { font-size: .82rem; font-weight: 700; }
        .hero-meta { margin-top: 1.2rem; font-size: .72rem; color: var(--muted); line-height: 1.7; }
        .hero-meta b { color: var(--ink); font-weight: 650; }
        .section-intro { max-width: 47rem; margin: 3.8rem 0 1.4rem; }
        .section-intro.first { margin-top: 0; border-top: 5px solid var(--ink); padding-top: 1.8rem; }
        .section-intro h2 {
            margin: .15rem 0 .55rem; font-family: Georgia, 'Times New Roman', serif;
            font-weight: 500; font-size: 2rem; color: var(--ink);
        }
        .section-intro p { color: var(--muted); line-height: 1.68; margin: 0; }
        .metric-card, .info-card, .verdict-card {
            border: 0; border-top: 1px solid var(--ink); border-radius: 0; background: transparent;
        }
        .metric-card { padding: 1rem .15rem .75rem; min-height: 6.8rem; }
        .metric-value { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; font-size: 2.2rem; font-weight: 500; letter-spacing: -.04em; }
        .metric-label { color: #46423c; font-size: .75rem; font-weight: 700; margin-top: .3rem; }
        .metric-note { color: #858078; font-size: .67rem; margin-top: .3rem; }
        .metric-accent { display: none; }
        .info-card { padding: 1rem .2rem 1.1rem; min-height: 8.5rem; }
        .card-index { color: var(--red); font-size: .62rem; letter-spacing: .12em; font-weight: 750; }
        .info-card h3 { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; font-weight: 500; font-size: 1.15rem; margin: .6rem 0 .4rem; }
        .info-card p { color: #6b665e; line-height: 1.58; font-size: .82rem; margin: 0; }
        .pipeline { display: grid; grid-template-columns: repeat(4,1fr); gap: 0; margin: 1rem 0 1.5rem; border-top: 1px solid var(--ink); }
        .pipe-step { border: 0; border-right: 1px solid var(--line); background: transparent; border-radius: 0; padding: 1rem 1rem 1rem 0; margin-right: 1rem; }
        .pipe-step:last-child { border-right: 0; }
        .pipe-step b { color: var(--ink); display: block; margin-bottom: .4rem; font-size: .8rem; }
        .pipe-step small { color: #777269; line-height: 1.45; }
        .status-row { margin-bottom: 1rem; }
        .status-ok, .status-warn {
            display: inline-block; padding: .3rem 0; margin: .15rem 1.1rem .15rem 0;
            font-size: .68rem; font-weight: 700; border-bottom: 1px solid currentColor;
        }
        .status-ok { color: var(--green); } .status-ok::before { content: '● '; font-size: .55rem; }
        .status-warn { color: var(--amber); } .status-warn::before { content: '● '; font-size: .55rem; }
        .verdict-card { padding: 1.1rem 0 1.3rem; margin: .4rem 0 1.5rem; border-bottom: 1px solid var(--line); }
        .verdict-card.real { border-top: 5px solid var(--green); }
        .verdict-card.fake { border-top: 5px solid var(--red); }
        .verdict-overline { color: #777269; font-size: .65rem; letter-spacing: .13em; font-weight: 750; }
        .verdict { font-family: Georgia, 'Times New Roman', serif; font-size: 3rem; font-weight: 500; margin: .1rem 0; }
        .verdict.real { color: var(--green); } .verdict.fake { color: var(--red); }
        .verdict-copy { color: #68635c; font-size: .8rem; }
        .prob-row { margin: .8rem 0 1rem; }
        .prob-label { display: flex; justify-content: space-between; color: #514d46; font-size: .75rem; margin-bottom: .35rem; }
        .prob-track { height: .38rem; background: #dedad1; overflow: hidden; }
        .prob-fill { height: 100%; }
        .evidence-line { font-size: 1.25rem; line-height: 2.25; padding: .8rem 0; }
        .evidence-word { padding: .22rem .38rem; margin: .08rem; border-radius: 0; color: var(--ink); }
        .score-row { margin: 1rem 0 1.25rem; }
        .score-label { display: flex; justify-content: space-between; color: #49453f; font-size: .8rem; margin-bottom: .32rem; }
        .score-track { height: .5rem; background: #dedad1; overflow: hidden; }
        .score-fill { height: 100%; background: #7d7971; }
        .score-fill.best { background: var(--red); }
        .callout {
            border-left: 3px solid var(--amber); padding: .8rem 1rem; color: #625d55;
            background: #eee9dc; border-radius: 0;
            font-size: .83rem; line-height: 1.6;
        }
        .footer { color: #777269; font-size: .68rem; border-top: 1px solid var(--ink); margin-top: 4rem; padding-top: 1rem; }
        [data-testid="stFileUploader"] { border: 1px dashed #aaa59a; border-radius: 0; padding: .25rem; }
        .stButton > button {
            border: 1px solid var(--ink); border-radius: 0; font-weight: 700;
            background: var(--ink); color: white; box-shadow: none;
        }
        .stButton > button:hover { border-color: var(--red); background: var(--red); color: white; }
        [data-baseweb="tab-list"] { gap: 1.3rem; border-bottom: 1px solid var(--line); }
        [data-baseweb="tab"] { background: transparent; padding-left: 0; padding-right: 0; }
        [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div { border-radius: 0 !important; }
        code { border-radius: 0 !important; }
        @media (max-width: 800px) {
            .pipeline, .hero-grid { grid-template-columns: 1fr 1fr; gap: 1.2rem; }
            .hero-result { border-left: 0; border-top: 1px solid var(--line); padding: 1.2rem 0 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_PATH)


@st.cache_resource(show_spinner=False)
def load_predictor() -> HEMTPredictor:
    return HEMTPredictor()


def section(kicker: str, title: str, copy: str, first: bool = False) -> None:
    css_class = "section-intro first" if first else "section-intro"
    st.markdown(
        f'<div class="{css_class}"><div class="eyebrow">{html.escape(kicker)}</div>'
        f'<h2>{html.escape(title)}</h2><p>{html.escape(copy)}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(value: str, label: str, note: str) -> None:
    st.markdown(
        '<div class="metric-card"><div class="metric-accent"></div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def info_card(index: str, title: str, copy: str) -> None:
    st.markdown(
        '<div class="info-card">'
        f'<div class="card-index">{html.escape(index)}</div>'
        f'<h3>{html.escape(title)}</h3><p>{html.escape(copy)}</p></div>',
        unsafe_allow_html=True,
    )


def score_bar(label: str, score: float, best: bool = False) -> None:
    best_class = " best" if best else ""
    st.markdown(
        '<div class="score-row"><div class="score-label">'
        f'<span>{html.escape(label)}</span><b>{score:.4f}</b></div>'
        f'<div class="score-track"><div class="score-fill{best_class}" style="width:{score * 100:.2f}%"></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def probability_bar(label: str, value: float, color: str) -> None:
    st.markdown(
        '<div class="prob-row"><div class="prob-label">'
        f'<span>{label}</span><b>{value:.1%}</b></div>'
        f'<div class="prob-track"><div class="prob-fill" style="width:{value * 100:.2f}%;background:{color}"></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def input_signature(title: str, image: Image.Image | None) -> str | None:
    """Small fingerprint used to avoid showing a result for changed inputs."""

    if image is None or not title.strip():
        return None
    digest = hashlib.sha256(title.strip().encode("utf-8"))
    digest.update(image.convert("RGB").resize((32, 32)).tobytes())
    return digest.hexdigest()


def show_artifact(path: Path, caption: str) -> None:
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Artifact not found: {path.relative_to(ROOT)}")


def sidebar() -> str:
    st.sidebar.markdown(
        '<div class="brand-lockup"><span class="brand-mark">HEMT—CLIP</span>'
        '<div class="brand-sub">RESEARCH INTERFACE / 2026</div></div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Sections", ["Study overview", "Inference", "Results", "Interpretability", "Methodology"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Held-out evaluation · Fakeddit binary task · n = 2,573")
    st.sidebar.markdown('<span class="status-ok">Selected model · α-gated fusion</span>', unsafe_allow_html=True)
    st.sidebar.caption("Final Year Project · UMT Sialkot · 2022–2026")
    return page


def overview_page(summary: pd.DataFrame) -> None:
    best = summary.loc[summary["variant"] == "gated_fusion"].iloc[0]
    st.markdown(
        """
        <div class="hero">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">Final year project · UMT Sialkot</div>
              <h1>Explainable multimodal classification of social-media posts.</h1>
              <div class="hero-copy">HEMT-CLIP is evaluated on the binary Fakeddit classification task using
              paired headline and image inputs. This interface reports the predicted dataset class, softmax
              score, CLIP cosine similarity, and patch-level cross-attention diagnostics.</div>
              <span class="pill">RoBERTa-base</span><span class="pill">CLIP ViT-B/16</span>
              <span class="pill">8-head cross-attention</span><span class="pill">α-gated fusion</span>
            </div>
            <div class="hero-result">
              <div class="hero-result-label">Held-out test estimate</div>
              <span class="hero-score">0.839</span><span class="hero-unit">F1 · FAKE class</span>
              <div class="hero-meta"><b>ROC AUC</b> 0.9122<br><b>Test n</b> 2,573<br><b>Selection</b> Validation F1 only</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Evaluation protocol", "Held-out test performance", "The α-gated fusion checkpoint was selected using validation F1 and evaluated once on 2,573 test instances excluded from training and model selection.")
    cols = st.columns(4)
    values = [
        (f"{best['test_f1']:.3f}", "F1 · FAKE class", "FAKE is the positive class"),
        (f"{best['test_auc']:.3f}", "ROC AUC", "Highest observed among five variants"),
        (f"{best['test_acc']:.3f}", "Accuracy", "2,139 / 2,573 correctly classified"),
        (f"{best['test_rec']:.3f}", "Recall · FAKE class", "1,133 / 1,265 FAKE-labelled instances"),
    ]
    for col, card in zip(cols, values):
        with col:
            metric_card(*card)

    section("Fusion formulation", "Multimodal representation and similarity gating", "The CLIP cosine similarity α is used as a continuous, parameter-free fusion coefficient; it is neither a class label nor a decision threshold.")
    st.markdown(
        """
        <div class="pipeline">
          <div class="pipe-step"><b>01 · Text encoding</b><small>RoBERTa projects the normalized title representation to 512 dimensions.</small></div>
          <div class="pipe-step"><b>02 · Image encoding</b><small>CLIP ViT-B/16 yields a 14 × 14 sequence of projected visual patch tokens.</small></div>
          <div class="pipe-step"><b>03 · Cross-attention</b><small>The text representation is the query; image patches provide keys and values.</small></div>
          <div class="pipe-step"><b>04 · Gating and classification</b><small>α combines attended image and text representations before the MLP classifier.</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("fused = α × attended_image + (1 − α) × text", language="text")

    section("Controlled ablation", "Empirical observations across model variants", "All five variants use identical data partitions, random seed, optimization procedure, and checkpoint-selection criterion; the principal manipulated factor is the modality or fusion pathway.")
    cols = st.columns(3)
    cards = [
        ("OBSERVATION 01", "Multimodal performance difference", "The gated variant exceeds the text-only and image-only baselines by 5.66 and 2.58 F1 points, respectively, on this test split."),
        ("OBSERVATION 02", "Similarity-gate formulation", "Using α as a fusion coefficient yields higher test F1 than appending α as a scalar feature, without introducing additional trainable parameters."),
        ("ANALYSIS PROTOCOL", "Complementary model diagnostics", "Cross-attention summarizes image-patch weighting; SHAP and LIME estimate text-token attribution with the paired image held fixed."),
    ]
    for col, card in zip(cols, cards):
        with col:
            info_card(*card)


def render_word_evidence(result: PredictionResult) -> None:
    if not result.word_evidence:
        return
    by_position = {int(item["position"]): float(item["score"]) for item in result.word_evidence}
    max_abs = max((abs(value) for value in by_position.values()), default=1.0) or 1.0
    spans = []
    for position, word in enumerate(result.cleaned_title.split()):
        score = by_position.get(position, 0.0)
        opacity = min(abs(score) / max_abs, 1.0) * 0.66
        background = f"rgba(180,35,24,{opacity:.2f})" if score > 0 else f"rgba(36,87,214,{opacity:.2f})" if score < 0 else "transparent"
        spans.append(f'<span class="evidence-word" style="background:{background}" title="ΔP(fake)={score:+.3f}">{html.escape(word)}</span>')
    st.markdown('<div class="evidence-line">' + " ".join(spans) + "</div>", unsafe_allow_html=True)
    st.caption("Red indicates a positive ΔP(FAKE); blue indicates a negative ΔP(FAKE). Intensity is normalized within this instance.")


def render_prediction(result: PredictionResult, source_image: Image.Image, truth: str | None) -> None:
    verdict_class = result.label.lower()
    reference_comparison = ""
    if truth:
        agreement = "agreement" if truth == result.label else "disagreement"
        reference_comparison = f" · Reference label: {truth} · {agreement}"
    st.markdown(
        f'<div class="verdict-card {verdict_class}"><div class="verdict-overline">PREDICTED DATASET CLASS</div>'
        f'<div class="verdict {verdict_class}">{result.label}</div>'
        f'<div class="verdict-copy">Maximum softmax score: {result.confidence:.1%}{html.escape(reference_comparison)}</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.image(source_image, caption="Input image observation", width="stretch")
    with right:
        if result.attention_overlay is not None:
            st.image(result.attention_overlay, caption="Mean cross-attention across eight heads", width="stretch")
        else:
            st.info("Cross-attention weights were not returned for this inference pass.")
    probability_col, alignment_col = st.columns([1.15, .85])
    with probability_col:
        st.subheader("Softmax class scores")
        probability_bar("REAL", result.probabilities["REAL"], "#13795b")
        probability_bar("FAKE", result.probabilities["FAKE"], "#b42318")
        st.caption("These scores have not been independently calibrated as empirical probabilities.")
    with alignment_col:
        st.subheader("Text–image similarity")
        st.metric("CLIP cosine similarity α", f"{result.alpha:.3f}")
        st.caption(result.alpha_band)
        st.caption("α parameterizes fusion and is not interpreted as evidence for either class in isolation.")
    if result.word_evidence:
        st.subheader("Local token-removal sensitivity")
        render_word_evidence(result)
        with st.expander("Sensitivity-analysis procedure"):
            st.write("Each content token is removed in turn while the paired image and α remain fixed. The displayed quantity is the resulting change in the FAKE-class softmax score. This local perturbation diagnostic is descriptive and does not constitute a causal attribution.")
    st.caption(f"Normalized title: “{result.cleaned_title}” · Compute device: {result.device} · Checkpoint: {result.checkpoint}")


def demo_page() -> None:
    section("Inference demonstration", "Apply the trained model to a paired observation", "Select a held-out Fakeddit instance or provide a title–image pair. The application executes the fixed α-gated checkpoint and reports class scores and model-behaviour diagnostics.", first=True)
    try:
        checkpoint = discover_checkpoint(CHECKPOINT_DIR)
    except FileNotFoundError:
        checkpoint = None
    hdf5_path = resolve_hdf5_path()
    missing_runtime = missing_runtime_packages()
    checkpoint_status = f'<span class="status-ok">Checkpoint · {html.escape(checkpoint.name)}</span>' if checkpoint else '<span class="status-warn">Checkpoint missing</span>'
    dataset_status = (
        '<span class="status-ok">Curated dataset available</span>'
        if hdf5_path and "HDF5" not in missing_runtime else
        '<span class="status-warn">Curated dataset unavailable</span>'
    )
    runtime_status = (
        '<span class="status-ok">Inference runtime ready</span>'
        if not missing_runtime else
        f'<span class="status-warn">Runtime missing · {html.escape(", ".join(missing_runtime))}</span>'
    )
    st.markdown(
        f'<div class="status-row">{checkpoint_status}{dataset_status}{runtime_status}</div>',
        unsafe_allow_html=True,
    )

    modes = ["User-supplied pair"]
    if hdf5_path and "HDF5" not in missing_runtime:
        modes.insert(0, "Held-out Fakeddit instance")
    mode = st.radio("Observation source", modes, horizontal=True)
    source_image: Image.Image | None = None
    title = ""
    truth: str | None = None

    if mode == "Held-out Fakeddit instance":
        pick = st.selectbox("Select a held-out instance", list(CURATED))
        meta = CURATED[pick]
        sample = load_hdf5_sample(int(meta["row"]), hdf5_path)
        st.caption(meta["story"])
        left, right = st.columns([.85, 1.15])
        with left:
            source_image = sample["image"]
            st.image(source_image, caption=f"Fakeddit row {sample['row']} · reference label {sample['truth']}", width="stretch")
        with right:
            title = st.text_area("Headline", value=sample["title"], height=120, key=f"curated-title-{sample['row']}")
            truth = sample["truth"]
            if sample["stored_alpha"] is not None:
                st.caption(f"Precomputed dataset α: {sample['stored_alpha']:.3f}")
    else:
        left, right = st.columns([.85, 1.15])
        with left:
            uploaded = st.file_uploader("Image observation", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                try:
                    source_image = Image.open(uploaded).convert("RGB")
                    st.image(source_image, caption="User-supplied image observation", width="stretch")
                except (UnidentifiedImageError, OSError):
                    st.error("The uploaded file could not be decoded as an image.")
        with right:
            title = st.text_area("Headline", placeholder="Example: the almost full moon atop a flagpole", height=120, key="upload-title")
            st.markdown('<div class="callout"><b>External-validity constraint.</b> The model was developed on Fakeddit. User-supplied observations may be out of distribution; the resulting class score is not an independent assessment of factual accuracy.</div>', unsafe_allow_html=True)

    explain_words = st.checkbox("Compute local token-removal sensitivity", value=True)
    current_signature = input_signature(title, source_image)
    analyze = st.button(
        "Run multimodal inference", type="primary", width="stretch",
        disabled=(
            source_image is None or not title.strip() or checkpoint is None or
            bool(missing_runtime)
        ),
    )
    if analyze and source_image is not None:
        try:
            with st.spinner("Loading the fixed checkpoint and computing model outputs…"):
                result = load_predictor().predict(source_image, title, explain_words=explain_words)
            st.session_state["demo_result"] = result
            st.session_state["demo_image"] = source_image.copy()
            st.session_state["demo_truth"] = truth
            st.session_state["demo_signature"] = current_signature
        except FileNotFoundError as exc:
            st.error(str(exc))
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"Inference did not complete: {exc}")
            st.info("A first execution may require retrieval of the configured RoBERTa and CLIP files. Verify network access and available memory before repeating the procedure.")
        except Exception as exc:  # Keep the Streamlit session alive on backend/model errors.
            st.error("The inference procedure terminated before producing an output.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")

    result = st.session_state.get("demo_result")
    result_image = st.session_state.get("demo_image")
    result_is_current = st.session_state.get("demo_signature") == current_signature
    if (
        result_is_current and isinstance(result, PredictionResult) and
        isinstance(result_image, Image.Image)
    ):
        section("Inference output", "Predicted class and diagnostic quantities", "The visual overlay is obtained by averaging the model's cross-attention weights across eight heads and interpolating the 14 × 14 patch grid.")
        render_prediction(result, result_image, st.session_state.get("demo_truth"))
    elif source_image is None:
        st.caption("Provide an image observation to enable inference. Model parameters are loaded only after the execution control is selected.")


def findings_page(summary: pd.DataFrame) -> None:
    section("Controlled ablation study", "Comparative test-set evaluation", "All variants use the same stratified 70/15/15 partition, random seed, optimizer, and two-stage fine-tuning schedule. The comparison varies the active modality or fusion formulation.", first=True)
    gated = summary.loc[summary["variant"] == "gated_fusion"].iloc[0]
    text_only = summary.loc[summary["variant"] == "text_only"].iloc[0]
    image_only = summary.loc[summary["variant"] == "image_only"].iloc[0]
    concat = summary.loc[summary["variant"] == "concat_fusion"].iloc[0]
    cols = st.columns(4)
    highlights = [
        (f"+{gated['test_f1'] - text_only['test_f1']:.4f}", "ΔF1 vs text-only", "Observed test-set difference"),
        (f"+{gated['test_f1'] - image_only['test_f1']:.4f}", "ΔF1 vs image-only", "Observed test-set difference"),
        (f"+{gated['test_auc'] - concat['test_auc']:.4f}", "ΔAUC vs concatenation", "Observed test-set difference"),
        (f"{gated['test_rec']:.1%}", "Recall · FAKE class", "Highest observed among five variants"),
    ]
    for col, item in zip(cols, highlights):
        with col:
            metric_card(*item)

    section("Variant-level estimates", "Performance under a common evaluation protocol", "F1 is reported with FAKE as the positive class. ROC AUC summarizes discrimination across classification thresholds.")
    f1_tab, auc_tab, table_tab = st.tabs(["F1 · FAKE class", "ROC AUC", "Complete estimates"])
    with f1_tab:
        for _, row in summary.iterrows():
            score_bar(MODEL_NAMES[row["variant"]], float(row["test_f1"]), row["variant"] == "gated_fusion")
    with auc_tab:
        for _, row in summary.iterrows():
            score_bar(MODEL_NAMES[row["variant"]], float(row["test_auc"]), row["variant"] == "gated_fusion")
    with table_tab:
        display = summary[["variant", "test_acc", "test_f1", "test_prec", "test_rec", "test_auc"]].copy()
        display["variant"] = display["variant"].map(MODEL_NAMES)
        display.columns = ["Model", "Accuracy", "F1", "Precision", "Recall", "AUC"]
        st.dataframe(
            display.style.format({column: "{:.4f}" for column in display.columns[1:]}).highlight_max(subset=["Accuracy", "F1", "Recall", "AUC"], color="#174d42"),
            hide_index=True, width="stretch",
        )

    section("Class-conditional errors", "Confusion-matrix analysis", "The selected model correctly classifies 1,133 of 1,265 FAKE-labelled instances and assigns the FAKE class to 302 of 1,308 REAL-labelled instances.")
    chart_col, note_col = st.columns([1.15, .85])
    with chart_col:
        show_artifact(OUTPUTS / "confusion_matrices" / "cm_gated_fusion.png", "α-gated cross-attention · held-out test set")
    with note_col:
        info_card("TRUE NEGATIVES", "1,006 REAL-labelled instances", "Of 1,308 REAL-labelled test instances, 76.9% are assigned the REAL class.")
        st.write("")
        info_card("TRUE POSITIVES", "1,133 FAKE-labelled instances", "Of 1,265 FAKE-labelled test instances, 89.6% are assigned the FAKE class.")
        st.write("")
        st.markdown('<div class="callout"><b>Metric trade-off.</b> Feature concatenation yields higher FAKE-class precision (0.8034 versus 0.7895), whereas α-gated cross-attention yields higher F1, recall, accuracy, and ROC AUC on this test split.</div>', unsafe_allow_html=True)
    section("Threshold analysis", "Discrimination across operating thresholds", "The selected variant attains ROC AUC 0.9122. The overlaid curves permit comparison without fixing a single classification threshold.")
    show_artifact(OUTPUTS / "roc_pr_curves" / "roc_overlay_test.png", "ROC curves for the five controlled variants")


def explainability_page() -> None:
    section("Interpretability analysis", "Complementary model-behaviour diagnostics", "Image-side cross-attention and post-hoc text attribution are evaluated for the same multimodal checkpoint. These quantities characterize model behaviour and are not treated as causal explanations.", first=True)
    cols = st.columns(3)
    cards = [
        ("IMAGE-SIDE", "Cross-attention weights", "The projected text representation attends to 196 CLIP image patches. The heatmap is derived from the trained model without an auxiliary explainer."),
        ("TEXT-SIDE", "SHAP and LIME attribution", "Both procedures perturb the normalized title while holding its paired image and α fixed, thereby retaining the multimodal inference context."),
        ("FUSION", "CLIP similarity α", "Cosine similarity parameterizes the fusion operation. It does not, independently, estimate authenticity or factual accuracy."),
    ]
    for col, card in zip(cols, cards):
        with col:
            info_card(*card)
    section("Image-side diagnostics", "Stratified cross-attention examples", "The sample includes agreement and disagreement cases at higher and lower maximum softmax scores, reducing selection bias in the qualitative analysis.")
    show_artifact(OUTPUTS / "attention_examples" / "attention_grid.png", "Stratified cross-attention diagnostics from the held-out test set")

    section("Case-level analysis", "Inspect individual attention distributions", "Select instances by label agreement and maximum softmax score. Artifact filenames encode the predicted class and reference label.")
    attention_files = [path for path in sorted((OUTPUTS / "attention_examples").glob("*.png")) if path.name != "attention_grid.png"]
    selected_attention = st.selectbox("Cross-attention instance", attention_files, format_func=lambda path: path.stem.replace("_", " ").replace("pred-", "predicted ").replace("true-", "reference "))
    if selected_attention:
        show_artifact(selected_attention, selected_attention.stem.replace("_", " "))

    section("Text-side diagnostics", "Post-hoc token attribution", "SHAP estimates additive token contributions, whereas LIME fits a sparse local surrogate around the selected title. Agreement between methods does not validate a causal interpretation or the predicted class.")
    shap_files = sorted((OUTPUTS / "shap_examples").glob("shap_[0-9]*.png"))
    lime_files = sorted((OUTPUTS / "lime_examples").glob("*.png"))
    left, right = st.columns(2)
    with left:
        shap_pick = st.selectbox("SHAP example", shap_files, format_func=lambda path: path.stem.replace("_", " "))
        if shap_pick:
            show_artifact(shap_pick, "SHAP · subword-level attribution")
    with right:
        lime_pick = st.selectbox("LIME example", lime_files, format_func=lambda path: path.stem.replace("_", " "))
        if lime_pick:
            show_artifact(lime_pick, "LIME · word-level local attribution")
    st.markdown('<div class="callout"><b>Interpretive limitation.</b> Cross-attention and token attribution summarize associations within the fitted model. They do not establish causal relevance, factual correctness, or source credibility.</div>', unsafe_allow_html=True)


def project_page() -> None:
    section("Methodology", "Experimental protocol and reproducibility", "HEMT-CLIP comprises a configuration-controlled pipeline for Fakeddit preparation, staged fine-tuning, validation-based checkpoint selection, held-out evaluation, and multimodal interpretability analysis.", first=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Model specification")
        st.markdown("""
        - **Text:** RoBERTa-base, last four layers fine-tuned
        - **Vision:** CLIP ViT-B/16, last four blocks fine-tuned
        - **Fusion:** eight-head text-to-image cross-attention
        - **Gate:** CLIP text–image cosine similarity α
        - **Head:** two-layer MLP, REAL / FAKE softmax
        """)
        st.subheader("Optimization and model selection")
        st.markdown("""
        - 70/15/15 stratified split, fixed seed 42
        - One encoder-frozen head-warm-up epoch
        - Up to six encoder fine-tuning epochs with validation-based early stopping
        - Mixed precision, gradient checkpointing, full resume state
        - Canonical checkpoint selected exclusively by validation F1
        """)
    with right:
        st.subheader("Operational definition of the outcome")
        st.write("The dependent variable is Fakeddit's binary label. The FAKE category aggregates source-defined satire, manipulated media, misleading captions, and false text–image associations. Consequently, the predicted class is a dataset-label estimate rather than an independent factual adjudication.")
        st.subheader("Threats to validity")
        st.markdown("""
        - **External validity:** arbitrary web content may differ materially from the Fakeddit distribution.
        - **Construct validity:** the binary target combines heterogeneous forms of misleading content.
        - **Dataset bias:** source, topic, and annotation regularities may be learned by the model.
        - **Calibration:** reported softmax scores have not been independently calibrated.
        - **Interpretability:** attention and attribution characterize model behaviour, not causal relevance.
        - **Deployment:** open-world use would require domain adaptation, recalibration, and prospective evaluation.
        """)

    section("Research team", "Final Year Project · UMT Sialkot", "Undergraduate research project on explainable multimodal classification, completed within the 2022–2026 academic programme.")
    cols = st.columns(3)
    team = [
        ("STUDENT RESEARCHER", "Syed Taha Faiz-ul-Hassan Rizvi", "Model development, software implementation, and experimental analysis"),
        ("STUDENT RESEARCHER", "Ayesha Bukhari", "Research design, analysis, and project development"),
        ("STUDENT RESEARCHER", "Ali Mosa Raza", "Research design, analysis, and project development"),
    ]
    for col, person in zip(cols, team):
        with col:
            info_card(*person)
    st.caption("Project supervisor · Ma'am Hina Tufail")
    with st.expander("Reproduction environment"):
        st.code("pip install -r requirements.txt\nrun_app.cmd", language="bash")
        st.write("The results and interpretability sections read the stored evaluation artifacts. Live inference additionally requires the canonical α-gated checkpoint and the configured Hugging Face backbones. Curated-instance inference requires `data/fakeddit.h5` or an explicit `HEMT_HDF5` path.")


inject_css()
page = sidebar()
summary_frame = load_summary()

if page == "Study overview":
    overview_page(summary_frame)
elif page == "Inference":
    demo_page()
elif page == "Results":
    findings_page(summary_frame)
elif page == "Interpretability":
    explainability_page()
else:
    project_page()

st.markdown(
    '<div class="footer">HEMT-CLIP · Academic research prototype · Outputs are predictions of Fakeddit labels and do not constitute independent factual verification.</div>',
    unsafe_allow_html=True,
)
