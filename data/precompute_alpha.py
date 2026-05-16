"""Precompute CLIP text-image cosine similarity (alpha) for each HDF5 row.

Runs the full CLIP model (text + vision encoders) over every sample in
batches on GPU, computes cosine similarity between the L2-normalised
embeddings, and writes the scalar back into f['alpha'] in place.

Why precompute: CLIP forward pass is expensive (~hundreds of ms per batch).
Doing it every training step would 2-3x epoch time. Once-off precompute on
a T4 takes ~20-30 min for ~17K samples, and the value never changes (CLIP
weights are frozen anyway).

Resumable: by default skips rows whose alpha is already non-NaN, so re-runs
after a Colab disconnect pick up where they stopped. Pass --overwrite to
recompute from scratch.

Example (in Colab, on a GPU runtime, after Drive is mounted):
    !python -m data.precompute_alpha \\
        --hdf5 /content/drive/MyDrive/hemt-clip-fnd/data/fakeddit.h5 \\
        --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from data.dataset import CLIP_MEAN, CLIP_STD

LOG = logging.getLogger("precompute-alpha")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hdf5", required=True, type=Path,
                   help="Packed HDF5 from build_hdf5.py. Updated in place.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model", default=CLIP_MODEL_NAME)
    p.add_argument("--max-text-len", type=int, default=77,
                   help="CLIP tokenizer max length (77 is the native ceiling).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                   help="Run encoders in fp16 (default on; GPU only).")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute all rows, ignoring existing non-NaN alpha values.")
    p.add_argument("--flush-every", type=int, default=10,
                   help="Flush HDF5 to disk every N batches (durability vs throughput).")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@torch.no_grad()
def compute_alpha_batch(
    model: CLIPModel,
    images_u8: np.ndarray,
    texts: list[str],
    tokenizer: CLIPTokenizer,
    max_text_len: int,
    device: str,
    use_fp16: bool,
) -> np.ndarray:
    """Returns (B,) float32 cosine similarities for one batch."""
    imgs = images_u8.astype(np.float32) / 255.0
    imgs = (imgs - CLIP_MEAN[:, None, None]) / CLIP_STD[:, None, None]
    pixel_values = torch.from_numpy(imgs).to(device, non_blocking=True)

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_fp16):
        img_emb = model.get_image_features(pixel_values=pixel_values)
        txt_emb = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    img_emb = F.normalize(img_emb.float(), dim=-1)
    txt_emb = F.normalize(txt_emb.float(), dim=-1)
    sim = (img_emb * txt_emb).sum(dim=-1)
    return sim.detach().cpu().numpy().astype(np.float32)


def main() -> int:
    setup_logging()
    args = parse_args()

    if not args.hdf5.exists():
        LOG.error("HDF5 not found: %s", args.hdf5)
        return 1

    use_fp16 = args.fp16 and args.device.startswith("cuda")
    if args.fp16 and not use_fp16:
        LOG.warning("fp16 requested but device=%s; running in fp32.", args.device)

    LOG.info("device=%s | fp16=%s | batch=%d", args.device, use_fp16, args.batch_size)
    LOG.info("loading CLIP model: %s", args.model)
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(args.device).eval()

    with h5py.File(args.hdf5, "r+") as f:
        n = f["alpha"].shape[0]
        existing = f["alpha"][:]
        if args.overwrite:
            todo = np.arange(n)
            LOG.info("--overwrite: recomputing all %d rows", n)
        else:
            todo = np.nonzero(np.isnan(existing))[0]
            LOG.info("resuming: %d/%d already computed, %d remaining",
                     n - len(todo), n, len(todo))

        if len(todo) == 0:
            LOG.info("nothing to do.")
            return 0

        bar = tqdm(range(0, len(todo), args.batch_size), desc="alpha", unit="batch")
        for bi, start in enumerate(bar):
            idx = todo[start:start + args.batch_size]
            idx_list = idx.tolist()

            imgs = f["images"][idx_list]
            raw_texts = f["texts"][idx_list]
            texts = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in raw_texts]

            sims = compute_alpha_batch(
                model, imgs, texts, tokenizer,
                args.max_text_len, args.device, use_fp16,
            )
            f["alpha"][idx_list] = sims

            if (bi + 1) % args.flush_every == 0:
                f.flush()

        f.flush()
        final = f["alpha"][:]
        n_nan = int(np.isnan(final).sum())
        LOG.info("done. alpha stats: min=%.3f mean=%.3f max=%.3f std=%.3f | NaN=%d/%d",
                 float(np.nanmin(final)), float(np.nanmean(final)),
                 float(np.nanmax(final)), float(np.nanstd(final)), n_nan, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
