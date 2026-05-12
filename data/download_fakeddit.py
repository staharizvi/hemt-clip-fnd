"""Parallel Fakeddit image downloader.

Reads a Fakeddit metadata TSV/CSV, stratified-samples N rows (default 18K so we
end up with ~15K usable after broken URLs), downloads images concurrently with
a short timeout + 1 retry, resizes to 224x224 RGB JPEGs, and writes them to a
local staging directory. Broken URLs are skipped.

Resumable: maintains a "downloaded_ids.txt" state file on Drive — on restart,
already-fetched IDs are skipped. A disconnect costs only the few in-flight
requests, not the whole batch.

The output sample CSV uses normalised column names (`id`, `image_url`,
`label`, `text`) so build_hdf5.py is agnostic to the source schema.

Example (in Colab, after Drive is mounted):
    !python -m data.download_fakeddit \\
        --metadata /content/multimodal_train.tsv \\
        --out-dir /content/images_staging \\
        --state-file /content/drive/MyDrive/hemt-clip-fnd/data/downloaded_ids.txt \\
        --sample-csv /content/drive/MyDrive/hemt-clip-fnd/data/sample.csv \\
        --sample-size 18000 --workers 16
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOG = logging.getLogger("fakeddit-download")

DEFAULT_UA = "Mozilla/5.0 (compatible; HEMTCLIPBot/1.0; FYP-UMT-2026)"
DEFAULT_TIMEOUT = (5, 5)  # (connect, read)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--metadata", required=True, type=Path,
                   help="Fakeddit metadata TSV/CSV (needs id, image_url, 2_way_label, clean_title/title).")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Local staging dir for resized JPEGs (use /content/... in Colab, NOT Drive).")
    p.add_argument("--state-file", required=True, type=Path,
                   help="Drive-resident text file logging completed IDs (for resume).")
    p.add_argument("--sample-csv", required=True, type=Path,
                   help="Where to write the sampled subset CSV (columns: id,image_url,label,text).")
    p.add_argument("--sample-size", type=int, default=18000)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--text-col", default="clean_title",
                   help="Source column to use for post text. Falls back to 'title' if missing.")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_metadata(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    required = {"id", "image_url", "2_way_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    if "hasImage" in df.columns:
        df = df[df["hasImage"] == True]  # noqa: E712 — Fakeddit stores literal True/False
    df = df.dropna(subset=["image_url", "2_way_label"])
    return df.reset_index(drop=True)


def normalise_columns(df: pd.DataFrame, source_text_col: str) -> pd.DataFrame:
    text_col = source_text_col if source_text_col in df.columns else ("title" if "title" in df.columns else None)
    if text_col is None:
        raise ValueError(f"No usable text column ('{source_text_col}' or 'title') in metadata.")
    df = df.rename(columns={text_col: "text", "2_way_label": "label"})
    df["text"] = df["text"].fillna("").astype(str)
    df["id"] = df["id"].astype(str)
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    per_class = n // 2
    parts = []
    for _, group in df.groupby("label"):
        parts.append(group.sample(n=min(per_class, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_state(path: Path, post_id: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(post_id + "\n")


def download_one(session: requests.Session, post_id: str, url: str, out_dir: Path,
                 image_size: int, jpeg_quality: int, max_retries: int) -> str | None:
    target = out_dir / f"{post_id}.jpg"
    if target.exists():
        return post_id
    attempt = 0
    while attempt <= max_retries:
        try:
            r = session.get(url, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            if not r.content:
                raise ValueError("empty body")
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img = img.resize((image_size, image_size), Image.BICUBIC)
            tmp = target.with_suffix(".jpg.part")
            img.save(tmp, "JPEG", quality=jpeg_quality)
            tmp.replace(target)
            return post_id
        except Exception:
            attempt += 1
    return None


def main() -> int:
    setup_logging()
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    args.sample_csv.parent.mkdir(parents=True, exist_ok=True)

    LOG.info("loading metadata: %s", args.metadata)
    df = load_metadata(args.metadata)
    df = normalise_columns(df, args.text_col)
    LOG.info("metadata rows after filter: %d", len(df))

    sample = stratified_sample(df, args.sample_size, args.seed)
    LOG.info("sampled %d rows | class balance: %s",
             len(sample), dict(sample["label"].value_counts()))

    sample[["id", "image_url", "label", "text"]].to_csv(args.sample_csv, index=False)
    LOG.info("wrote sample CSV: %s", args.sample_csv)

    done = load_state(args.state_file)
    LOG.info("resume state: %d IDs already downloaded", len(done))

    work = sample[~sample["id"].isin(done)]
    LOG.info("queueing %d new downloads (%d workers)", len(work), args.workers)
    if len(work) == 0:
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = DEFAULT_UA

    success = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, session, row.id, row.image_url, args.out_dir,
                        args.image_size, args.jpeg_quality, args.max_retries): row.id
            for row in work.itertuples(index=False)
        }
        with tqdm(total=len(futures), desc="downloading", unit="img") as bar:
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    result = fut.result()
                except Exception:
                    result = None
                if result is not None:
                    append_state(args.state_file, result)
                    success += 1
                else:
                    fail += 1
                bar.update(1)
                bar.set_postfix(ok=success, fail=fail)

    LOG.info("done. success=%d fail=%d total=%d", success, fail, success + fail)
    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
