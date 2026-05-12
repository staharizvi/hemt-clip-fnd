"""Pack staged Fakeddit images + texts + labels into a single HDF5 file.

Reads the sample CSV produced by download_fakeddit.py, filters to rows whose
JPEG actually made it into the staging dir, makes a 70/15/15 stratified split,
and writes everything into one HDF5 file.

Why HDF5: loading 15K loose JPEGs from Drive is dominated by small-file I/O
overhead. A single HDF5 file with sequential chunked reads is ~10x faster
end-to-end per epoch.

HDF5 schema:
    images : uint8   (N, 3, 224, 224)   per-sample chunks, gzip-compressed
    texts  : str     (N,)               variable-length UTF-8
    ids    : str     (N,)               Fakeddit post ids
    labels : int8    (N,)               0=real, 1=fake
    splits : str     (N,)               "train" | "val" | "test"
    alpha  : float32 (N,)               CLIP text-image cosine similarity
                                        (NaN until filled by precompute_alpha.py)

Also writes train.csv / val.csv / test.csv into splits-dir, one row per
sample, with the HDF5 row index for direct lookup.

Example:
    !python -m data.build_hdf5 \\
        --sample-csv /content/drive/MyDrive/hemt-clip-fnd/data/sample.csv \\
        --images-dir /content/images_staging \\
        --hdf5-out /content/fakeddit.h5 \\
        --splits-dir /content/drive/MyDrive/hemt-clip-fnd/data/splits

    # then move the HDF5 to Drive in one big sequential copy:
    !cp /content/fakeddit.h5 /content/drive/MyDrive/hemt-clip-fnd/data/fakeddit.h5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

LOG = logging.getLogger("build-hdf5")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sample-csv", required=True, type=Path,
                   help="CSV from download_fakeddit.py (columns: id,image_url,label,text).")
    p.add_argument("--images-dir", required=True, type=Path,
                   help="Local staging dir holding <id>.jpg files.")
    p.add_argument("--hdf5-out", required=True, type=Path,
                   help="Output HDF5 path (build locally on /content/, copy to Drive after).")
    p.add_argument("--splits-dir", required=True, type=Path,
                   help="Directory to write train.csv / val.csv / test.csv (Drive is fine; small files).")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--compression-level", type=int, default=4,
                   help="gzip level 0-9. 4 is a good speed/size tradeoff.")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def filter_to_downloaded(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    available = {p.stem for p in images_dir.glob("*.jpg")}
    before = len(df)
    df = df[df["id"].astype(str).isin(available)].reset_index(drop=True)
    LOG.info("filter_to_downloaded: %d -> %d (dropped %d missing JPEGs)",
             before, len(df), before - len(df))
    return df


def make_splits(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    """Stratified 70/15/15 split. test_frac and val_frac are absolute fractions of the full dataset."""
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed,
    )
    val_relative = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val["label"], random_state=seed,
    )
    out = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ]).reset_index(drop=True)
    LOG.info("splits: train=%d val=%d test=%d",
             (out["split"] == "train").sum(),
             (out["split"] == "val").sum(),
             (out["split"] == "test").sum())
    return out


def main() -> int:
    setup_logging()
    args = parse_args()

    args.hdf5_out.parent.mkdir(parents=True, exist_ok=True)
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("reading sample CSV: %s", args.sample_csv)
    df = pd.read_csv(args.sample_csv)
    df = df.dropna(subset=["id", "text", "label"]).reset_index(drop=True)
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    df = filter_to_downloaded(df, args.images_dir)
    if len(df) == 0:
        LOG.error("no usable rows after filtering; aborting")
        return 1

    df = make_splits(df, args.val_frac, args.test_frac, args.seed)

    N = len(df)
    LOG.info("packing %d samples into %s", N, args.hdf5_out)

    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(args.hdf5_out, "w") as f:
        images_ds = f.create_dataset(
            "images",
            shape=(N, 3, args.image_size, args.image_size),
            dtype=np.uint8,
            chunks=(1, 3, args.image_size, args.image_size),
            compression="gzip",
            compression_opts=args.compression_level,
        )
        texts_ds = f.create_dataset("texts", shape=(N,), dtype=str_dt)
        ids_ds = f.create_dataset("ids", shape=(N,), dtype=str_dt)
        labels_ds = f.create_dataset("labels", shape=(N,), dtype=np.int8)
        splits_ds = f.create_dataset("splits", shape=(N,), dtype=str_dt)
        alpha_ds = f.create_dataset("alpha", shape=(N,), dtype=np.float32)
        alpha_ds[:] = np.nan  # sentinel: not computed yet

        f.attrs["image_size"] = args.image_size
        f.attrs["seed"] = args.seed
        f.attrs["n_samples"] = N
        f.attrs["created_by"] = "build_hdf5.py"

        for i, row in enumerate(tqdm(df.itertuples(index=False), total=N,
                                      desc="packing", unit="img")):
            img_path = args.images_dir / f"{row.id}.jpg"
            img = Image.open(img_path).convert("RGB")
            if img.size != (args.image_size, args.image_size):
                img = img.resize((args.image_size, args.image_size), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)  # HWC -> CHW
            images_ds[i] = arr
            texts_ds[i] = row.text
            ids_ds[i] = row.id
            labels_ds[i] = int(row.label)
            splits_ds[i] = row.split

    size_gb = args.hdf5_out.stat().st_size / 1e9
    LOG.info("HDF5 written: %s (%.2f GB)", args.hdf5_out, size_gb)

    for split_name in ("train", "val", "test"):
        sub = (df[df["split"] == split_name]
                 .reset_index()
                 .rename(columns={"index": "hdf5_index"}))
        out = args.splits_dir / f"{split_name}.csv"
        sub[["hdf5_index", "id", "text", "label"]].to_csv(out, index=False)
        LOG.info("wrote %s (%d rows)", out, len(sub))

    return 0


if __name__ == "__main__":
    sys.exit(main())
