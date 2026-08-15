"""Classify a MODI / Devanagari character image.

Usage::

    python -m src.predict feature_data/k/k1.jpg
    python -m src.predict feature_data/k/k1.jpg --model 2 --topk 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .classes import classes_for, strokes_for
from .model import load, load_image


def predict(image_path: str | Path, model_id: int = 1, topk: int = 5):
    """Return ``[(character, probability), ...]`` for the top ``topk`` classes."""
    names = classes_for(model_id)
    probs = load(model_id).predict(load_image(image_path), verbose=0)[0]
    order = np.argsort(probs)[::-1][:topk]
    return [(names[i], float(probs[i])) for i in order]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="path to a character image")
    parser.add_argument(
        "--model", type=int, default=1, choices=(1, 2),
        help="1 = 47-class MODI (default), 2 = 80-class MODI + Devanagari",
    )
    parser.add_argument("--topk", type=int, default=5, help="how many predictions to show")
    args = parser.parse_args(argv)

    results = predict(args.image, args.model, args.topk)

    label = "47-class MODI" if args.model == 1 else "80-class MODI + Devanagari"
    print(f"{args.image}  (model {args.model}, {label})\n")
    width = max(len(name) for name, _ in results)
    for rank, (name, prob) in enumerate(results, start=1):
        bar = "#" * round(prob * 40)
        print(f"  {rank}. {name:<{width}}  {prob:6.2%}  {bar}")

    top = results[0][0]
    strokes = strokes_for(top, args.model)
    if strokes:
        print(f"\n  stroke primitives of '{top}': {' '.join(strokes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
