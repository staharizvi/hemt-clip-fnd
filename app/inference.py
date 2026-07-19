"""Lazy, notebook-equivalent inference helpers for the Streamlit frontend.

The module deliberately does not import PyTorch or Transformers at import time.
The findings dashboard therefore starts quickly; the heavyweight model stack is
only loaded after a visitor asks for a prediction.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from importlib.util import find_spec
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"
CHECKPOINT_DIR = ROOT / "checkpoints"

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "is", "it", "its",
    "was", "were", "be", "by", "for", "and", "or", "as", "this", "that",
    "with", "from", "has", "have", "had", "will", "would", "can", "could",
    "i", "you", "he", "she", "they", "we", "his", "her", "their", "our",
    "my", "me", "so", "but", "if", "no", "not", "do", "does", "did",
}


def missing_runtime_packages() -> list[str]:
    """Return packages required only when the visitor runs live inference."""

    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "h5py": "HDF5",
        "matplotlib": "Matplotlib",
    }
    return [display_name for module, display_name in required.items() if find_spec(module) is None]


def resolve_model_source(model_name: str) -> str:
    """Use a complete local Hugging Face snapshot when one is cached.

    Passing the snapshot directory (instead of the repository id) also prevents
    Transformers 4.40 from making remote safetensors probes during an offline
    defense demo. A cold machine simply receives the original id and downloads
    it through the normal ``from_pretrained`` path.
    """

    try:
        from huggingface_hub import try_to_load_from_cache

        cached_config = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached_config, str):
            snapshot = Path(cached_config).parent
            weight_names = ("model.safetensors", "pytorch_model.bin")
            if any((snapshot / name).exists() for name in weight_names):
                return str(snapshot)
    except (ImportError, OSError):
        pass
    return model_name


@dataclass
class PredictionResult:
    """Serializable result returned by :class:`HEMTPredictor`."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    alpha: float
    alpha_band: str
    cleaned_title: str
    attention_overlay: Image.Image | None = None
    word_evidence: list[dict[str, float | str]] = field(default_factory=list)
    device: str = "cpu"
    checkpoint: str = ""


def clean_text(text: str) -> str:
    """Match the lowercase ``clean_title`` representation used in training."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Return the exact CLIP-normalized CHW array used by the dataset."""

    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (array - CLIP_MEAN[:, None, None]) / CLIP_STD[:, None, None]


def interpret_alpha(alpha: float) -> tuple[str, str]:
    """Return the display-only similarity band used by the interface."""

    if alpha < 0.20:
        return "Low-similarity band", "Display criterion: α < 0.20."
    if alpha < 0.33:
        return "Intermediate-similarity band", "Display criterion: 0.20 ≤ α < 0.33."
    return "High-similarity band", "Display criterion: α ≥ 0.33."


def discover_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    """Resolve the latest canonical alpha-gated checkpoint."""

    candidates = list(checkpoint_dir.glob("hemt_gated_fusion_*_best.pt"))
    canonical = [path for path in candidates if "_seed" not in path.name]
    pool = canonical or candidates
    if not pool:
        raise FileNotFoundError(
            f"No gated-fusion checkpoint found in {checkpoint_dir}. "
            "Expected hemt_gated_fusion_*_best.pt."
        )
    return max(pool, key=lambda path: path.stat().st_mtime)


def resolve_hdf5_path(config_path: Path = DEFAULT_CONFIG) -> Path | None:
    """Find the optional packed dataset used by the curated demo mode."""

    candidates: list[Path] = []
    if os.environ.get("HEMT_HDF5"):
        candidates.append(Path(os.environ["HEMT_HDF5"]).expanduser())
    candidates.append(ROOT / "data" / "fakeddit.h5")
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        candidates.append(Path(cfg["data"]["hdf5_path"]).expanduser())
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        pass
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_hdf5_sample(row: int, hdf5_path: Path | None = None) -> dict[str, Any]:
    """Load one absolute HDF5 row for the curated test-set demo."""

    path = hdf5_path or resolve_hdf5_path()
    if path is None:
        raise FileNotFoundError(
            "fakeddit.h5 was not found. Put it at data/fakeddit.h5 or set HEMT_HDF5."
        )
    import h5py

    with h5py.File(path, "r") as handle:
        raw_image = handle["images"][row]
        if raw_image.shape[0] == 3:
            raw_image = raw_image.transpose(1, 2, 0)
        image = Image.fromarray(raw_image.astype("uint8"), mode="RGB")
        raw_title = handle["texts"][row]
        title = raw_title.decode("utf-8") if isinstance(raw_title, bytes) else str(raw_title)
        label = "FAKE" if int(handle["labels"][row]) == 1 else "REAL"
        stored_alpha = float(handle["alpha"][row]) if "alpha" in handle else None
    return {
        "row": row,
        "image": image,
        "title": title,
        "truth": label,
        "stored_alpha": stored_alpha,
    }


def attention_overlay(image: Image.Image, grid: np.ndarray) -> Image.Image:
    """Blend a perceptually clear heat map over the original image."""

    from matplotlib import colormaps

    values = np.asarray(grid, dtype=np.float32)
    values = (values - values.min()) / (np.ptp(values) + 1e-8)
    heat = Image.fromarray(np.uint8(values * 255), mode="L").resize(
        image.size, Image.Resampling.BILINEAR
    )
    colored = colormaps["inferno"](np.asarray(heat, dtype=np.float32) / 255.0)[..., :3]
    colored_image = Image.fromarray(np.uint8(colored * 255), mode="RGB")
    return Image.blend(image.convert("RGB"), colored_image, alpha=0.48)


class HEMTPredictor:
    """The trained alpha-gated HEMT-CLIP model and its exact preprocessing."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        checkpoint_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        # Environment switches must precede the transformers import.
        os.environ.setdefault("USE_FLAX", "FALSE")
        os.environ.setdefault("USE_TF", "FALSE")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        missing = missing_runtime_packages()
        if missing:
            raise ImportError(
                "Live inference dependencies are missing: " + ", ".join(missing) +
                ". Install requirements.txt in the environment running Streamlit."
            )

        import torch
        import torch.nn.functional as functional
        from transformers import AutoTokenizer, CLIPModel, CLIPTokenizer

        from models.hemt_clip import build_from_config

        self.torch = torch
        self.functional = functional
        self.config_path = Path(config_path)
        self.checkpoint_path = checkpoint_path or discover_checkpoint()
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.max_len = int(self.cfg["data"]["max_text_len"])

        text_name = self.cfg["model"]["text"]["name"]
        clip_name = self.cfg["model"]["image"]["name"]
        text_source = resolve_model_source(text_name)
        clip_source = resolve_model_source(clip_name)

        runtime_cfg = deepcopy(self.cfg)
        runtime_cfg["model"]["text"]["name"] = text_source
        runtime_cfg["model"]["image"]["name"] = clip_source
        self.model = build_from_config(
            runtime_cfg, variant="gated_fusion"
        ).to(self.device).eval()
        payload = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.model.load_state_dict(state, strict=True)

        self.tokenizer = AutoTokenizer.from_pretrained(text_source)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_source)
        self.clip_model = CLIPModel.from_pretrained(clip_source).to(self.device).eval()

    def _pixel_values(self, image: Image.Image):
        return self.torch.from_numpy(preprocess_image(image)).unsqueeze(0).to(self.device)

    def _compute_alpha(self, pixel_values, title: str) -> float:
        encoded = self.clip_tokenizer(
            [title], padding="max_length", truncation=True, max_length=77,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            image_embedding = self.functional.normalize(
                self.clip_model.get_image_features(pixel_values=pixel_values).float(), dim=-1
            )
            text_embedding = self.functional.normalize(
                self.clip_model.get_text_features(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                ).float(),
                dim=-1,
            )
        return float((image_embedding * text_embedding).sum(dim=-1).item())

    def _predict_batch(self, titles: list[str], pixel_values, alpha: float):
        encoded = self.tokenizer(
            titles, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        ).to(self.device)
        batch_size = len(titles)
        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": pixel_values.expand(batch_size, -1, -1, -1),
            "alpha": self.torch.full(
                (batch_size,), alpha, dtype=self.torch.float32, device=self.device
            ),
        }
        with self.torch.inference_mode():
            output = self.model(batch)
            probabilities = self.torch.softmax(output["logits"].float(), dim=-1)
        return probabilities.cpu().numpy(), output.get("attention_weights")

    def _word_ablation(
        self,
        title: str,
        pixel_values,
        alpha: float,
        base_fake: float,
    ) -> list[dict[str, Any]]:
        """Fast local word evidence with the image and alpha held fixed.

        Each score is the change in fake-class probability after removing one
        content word. Positive scores support FAKE; negative scores support REAL.
        """

        words = title.split()
        positions = [
            index for index, word in enumerate(words)
            if word not in STOPWORDS and len(word) > 1
        ][:18]
        if not positions:
            return []
        variants = []
        for position in positions:
            variants.append(" ".join(word for index, word in enumerate(words) if index != position))

        # Encode the image once. Only the much cheaper text → fusion → head path
        # is replayed for the word perturbations (the exact strategy used by the
        # offline SHAP/LIME helpers in explainability/mm_common.py).
        with self.torch.inference_mode():
            patches = self.model.image_encoder(pixel_values).patches
        perturbed_fake: list[float] = []
        for start in range(0, len(variants), 16):
            chunk = variants[start:start + 16]
            encoded = self.tokenizer(
                chunk, padding="max_length", truncation=True,
                max_length=self.max_len, return_tensors="pt",
            ).to(self.device)
            batch_size = len(chunk)
            with self.torch.inference_mode():
                text_features = self.model.text_encoder(
                    encoded["input_ids"], encoded["attention_mask"]
                )
                fused, _ = self.model.fusion(
                    text_features,
                    patches.expand(batch_size, -1, -1),
                    alpha=self.torch.full(
                        (batch_size,), alpha, dtype=self.torch.float32,
                        device=self.device,
                    ),
                )
                logits = self.model.classifier(fused)
                probabilities = self.torch.softmax(logits.float(), dim=-1)
            perturbed_fake.extend(probabilities[:, 1].cpu().tolist())

        evidence = []
        for position, masked_fake in zip(positions, perturbed_fake):
            evidence.append({
                "word": words[position],
                "score": base_fake - masked_fake,
                "position": position,
            })
        return evidence

    def predict(
        self,
        image: Image.Image,
        title: str,
        explain_words: bool = True,
    ) -> PredictionResult:
        cleaned = clean_text(title)
        if not cleaned:
            raise ValueError("Enter a headline containing at least one letter or number.")

        source_image = image.convert("RGB")
        pixel_values = self._pixel_values(source_image)
        alpha = self._compute_alpha(pixel_values, cleaned)
        probabilities, attention = self._predict_batch([cleaned], pixel_values, alpha)
        real_probability, fake_probability = map(float, probabilities[0])
        predicted_index = int(np.argmax(probabilities[0]))
        label = "FAKE" if predicted_index == 1 else "REAL"

        overlay = None
        if attention is not None:
            weights = attention.float().cpu().numpy()[0].mean(axis=0).squeeze(0)
            side = int(round(np.sqrt(len(weights))))
            if side * side == len(weights):
                overlay = attention_overlay(source_image, weights.reshape(side, side))

        band, explanation = interpret_alpha(alpha)
        evidence = (
            self._word_ablation(cleaned, pixel_values, alpha, fake_probability)
            if explain_words else []
        )
        return PredictionResult(
            label=label,
            confidence=max(real_probability, fake_probability),
            probabilities={"REAL": real_probability, "FAKE": fake_probability},
            alpha=alpha,
            alpha_band=f"{band}. {explanation}",
            cleaned_title=cleaned,
            attention_overlay=overlay,
            word_evidence=evidence,
            device=str(self.device),
            checkpoint=self.checkpoint_path.name,
        )
