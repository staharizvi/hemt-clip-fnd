"""PyTorch Dataset reading from the packed HDF5 file.

Returns dicts with:
    input_ids, attention_mask  (RoBERTa tokenizer, max_len from config)
    pixel_values               (CLIP-normalised float32 image tensor)
    alpha                      (precomputed CLIP cosine similarity, scalar)
    label                      (0=real, 1=fake)

The HDF5 handle is opened lazily inside __getitem__: PyTorch DataLoader
workers fork after Dataset construction, and h5py.File handles do not
survive fork. Opening per-worker on first access is the only stable pattern.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LOG = logging.getLogger(__name__)

# CLIP normalisation constants — must match openai/clip-vit-base-patch32 exactly.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class HEMTClipDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str | Path,
        split: str,
        tokenizer_name: str = "roberta-base",
        max_text_len: int = 128,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.hdf5_path = str(hdf5_path)
        self.split = split
        self.max_text_len = max_text_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        with h5py.File(self.hdf5_path, "r") as f:
            splits = f["splits"][:].astype(str)
            self.indices = np.nonzero(splits == split)[0]
            alpha = f["alpha"][self.indices]

        n_nan = int(np.isnan(alpha).sum())
        if n_nan > 0:
            warnings.warn(
                f"HEMTClipDataset[{split}]: {n_nan}/{len(self.indices)} alpha values are NaN. "
                "Run data/precompute_alpha.py before training, or set model.use_alpha=False.",
                stacklevel=2,
            )

        self._h5: Optional[h5py.File] = None

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_open()
        assert self._h5 is not None
        row = int(self.indices[idx])

        img_u8 = self._h5["images"][row]  # (3, H, W) uint8
        img = img_u8.astype(np.float32) / 255.0
        img = (img - CLIP_MEAN[:, None, None]) / CLIP_STD[:, None, None]
        pixel_values = torch.from_numpy(img)

        raw_text = self._h5["texts"][row]
        text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        alpha_val = float(self._h5["alpha"][row])
        if np.isnan(alpha_val):
            alpha_val = 0.0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "alpha": torch.tensor(alpha_val, dtype=torch.float32),
            "label": torch.tensor(int(self._h5["labels"][row]), dtype=torch.long),
        }

    def __del__(self) -> None:
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
