"""Render the convolutional feature maps a trained model extracts from an image.

This is the repo's namesake: a forward pass is tapped at a chosen conv layer and
every channel of that layer's activation is drawn as a small grayscale tile.

Usage::

    python -m src.featuremaps feature_data/k/k1.jpg
    python -m src.featuremaps feature_data/k/k1.jpg --model 2 --layer all -o out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .model import load, load_image


def conv_layer_names(model) -> list[str]:
    """Names of the model's Conv2D layers, in forward order."""
    return [l.name for l in model.layers if l.__class__.__name__ == "Conv2D"]


def activations(image_path: str | Path, layer_name: str, model_id: int = 1) -> np.ndarray:
    """Activations at ``layer_name`` for one image, shaped ``(h, w, channels)``."""
    import tensorflow as tf

    model = load(model_id)
    tap = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return tap.predict(load_image(image_path), verbose=0)[0]


def render(image_path: str | Path, layer_name: str, out_path: Path,
           model_id: int = 1, max_channels: int = 64) -> Path:
    """Draw the activation grid for one layer and save it to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = activations(image_path, layer_name, model_id)
    count = min(maps.shape[-1], max_channels)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for i, ax in enumerate(np.atleast_1d(axes).ravel()[:count]):
        ax.imshow(maps[:, :, i], cmap="gray")
    fig.suptitle(f"{layer_name} — {count} of {maps.shape[-1]} channels")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="path to a character image")
    parser.add_argument("--model", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--layer", default=None,
        help="conv layer name, or 'all' for every conv layer "
             "(default: the second conv layer, as used in the notebooks)",
    )
    parser.add_argument("-o", "--out", default="out", help="output directory")
    parser.add_argument("--list-layers", action="store_true", help="print conv layer names and exit")
    args = parser.parse_args(argv)

    model = load(args.model)
    convs = conv_layer_names(model)

    if args.list_layers:
        print("\n".join(convs))
        return 0

    if args.layer == "all":
        targets = convs
    elif args.layer:
        if args.layer not in convs:
            parser.error(f"unknown conv layer {args.layer!r}; choose from: {', '.join(convs)}")
        targets = [args.layer]
    else:
        targets = [convs[1]]

    out_dir = Path(args.out)
    stem = Path(args.image).stem
    for name in targets:
        path = render(args.image, name, out_dir / f"{stem}_model{args.model}_{name}.png", args.model)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
