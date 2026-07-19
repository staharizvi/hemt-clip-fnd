"""Run one real HEMT-CLIP prediction outside Streamlit.

This is a small acceptance test for the frontend's live inference path. It loads
the same checkpoint, preprocessing, alpha model, and attention renderer used by
the app, then checks the prediction for a known curated Fakeddit row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import HEMTPredictor, load_hdf5_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--row", type=int, default=16379)
    parser.add_argument("--expected", choices=("REAL", "FAKE"), default="FAKE")
    parser.add_argument("--explain-words", action="store_true")
    parser.add_argument("--overlay", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = load_hdf5_sample(args.row)
    predictor = HEMTPredictor()
    result = predictor.predict(
        sample["image"], sample["title"], explain_words=args.explain_words
    )

    if result.attention_overlay is None:
        raise AssertionError("The gated-fusion model returned no attention overlay.")
    if result.label != args.expected:
        raise AssertionError(
            f"Expected {args.expected} for row {args.row}, got {result.label}."
        )
    if sample["truth"] != args.expected:
        raise AssertionError(
            f"Dataset truth for row {args.row} is {sample['truth']}, not {args.expected}."
        )

    if args.overlay is not None:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        result.attention_overlay.save(args.overlay)

    print(json.dumps({
        "row": args.row,
        "title": sample["title"],
        "truth": sample["truth"],
        "prediction": result.label,
        "confidence": round(result.confidence, 6),
        "alpha_live": round(result.alpha, 6),
        "alpha_stored": round(sample["stored_alpha"], 6),
        "attention_overlay": result.attention_overlay.size,
        "checkpoint": result.checkpoint,
        "device": result.device,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
