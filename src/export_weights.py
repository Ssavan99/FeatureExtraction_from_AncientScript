"""Export a trained model's weights for the browser demo in ``docs/``.

The demo runs a hand-written forward pass in JavaScript rather than pulling in
TensorFlow.js: the network is only ~175k parameters, so a plain implementation
of conv2d / batch-norm / max-pool / global-average-pool / dense is small, has no
dependencies, and cannot version-rot. ``tests/test_js_parity.py`` asserts the
exported numbers reproduce Keras logits.

Three files are written:

``model2.json``    layer plan, tensor shapes and byte offsets, class names
``model2.bin``     every weight tensor concatenated as little-endian float32
``samples.bin``    the committed ``feature_data/`` glyphs, already preprocessed
                   to 32x32x3 uint8, so the page shows the model exactly what
                   the Python pipeline would.

Usage::

    python -m src.export_weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .classes import REPO_ROOT, classes_for
from .model import INPUT_SIZE, load, load_image

# Layer classes the JavaScript runtime knows how to execute. Anything absent
# here raises at export time, where it is diagnosable, rather than producing a
# plan the browser cannot run. Keep this in sync with docs/cnn.js.
_SUPPORTED = {
    "Conv2D", "BatchNormalization", "MaxPooling2D", "Dropout",
    "GlobalAveragePooling2D", "Dense", "InputLayer",
}

# Activations docs/cnn.js implements.
_SUPPORTED_ACTIVATIONS = {"relu", "softmax", "linear"}


def _activation_name(layer) -> str:
    """Activation of ``layer``, rejecting anything the JS runtime lacks."""
    name = layer.activation.__name__
    if name not in _SUPPORTED_ACTIVATIONS:
        raise NotImplementedError(
            f"layer {layer.name!r} uses activation {name!r}, which docs/cnn.js "
            f"does not implement (supported: {sorted(_SUPPORTED_ACTIVATIONS)})"
        )
    return name


def build_plan(model) -> tuple[list[dict], list[np.ndarray]]:
    """Describe the model as a flat op list plus the tensors those ops need."""
    plan: list[dict] = []
    tensors: list[np.ndarray] = []

    def add(tensor: np.ndarray) -> int:
        tensors.append(np.ascontiguousarray(tensor, dtype=np.float32))
        return len(tensors) - 1

    for layer in model.layers:
        kind = layer.__class__.__name__
        if kind not in _SUPPORTED:
            raise NotImplementedError(
                f"layer {layer.name!r} of type {kind} has no JavaScript equivalent"
            )

        if kind in ("InputLayer", "Dropout"):
            # Dropout is inference-time identity; InputLayer carries no maths.
            continue

        step: dict = {"type": kind, "name": layer.name}

        if kind == "Conv2D":
            kernel, bias = layer.get_weights()
            step.update(
                kernel=add(kernel), bias=add(bias),
                kernelShape=list(kernel.shape),
                activation=_activation_name(layer),
            )
            if layer.padding != "valid" or tuple(layer.strides) != (1, 1):
                raise NotImplementedError(
                    f"{layer.name}: JS conv2d only implements valid padding, stride 1"
                )
        elif kind == "BatchNormalization":
            gamma, beta, mean, var = layer.get_weights()
            step.update(
                gamma=add(gamma), beta=add(beta), mean=add(mean),
                variance=add(var), epsilon=float(layer.epsilon),
                channels=int(gamma.shape[0]),
            )
        elif kind == "MaxPooling2D":
            step.update(pool=list(layer.pool_size), strides=list(layer.strides))
        elif kind == "Dense":
            kernel, bias = layer.get_weights()
            step.update(
                kernel=add(kernel), bias=add(bias),
                kernelShape=list(kernel.shape),
                activation=_activation_name(layer),
            )
        plan.append(step)

    return plan, tensors


def export_samples(out_dir: Path) -> list[str]:
    """Preprocess every committed sample glyph to uint8 and write ``samples.bin``."""
    sample_dir = REPO_ROOT / "feature_data"
    names, arrays = [], []
    for folder in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
        images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if not images:
            continue
        # load_image applies exactly the notebook preprocessing; scale back to
        # uint8 purely to keep the payload small.
        array = load_image(images[0])[0]
        arrays.append(np.round(array * 255).astype(np.uint8))
        names.append(folder.name)

    stacked = np.stack(arrays) if arrays else np.zeros((0, *INPUT_SIZE, 3), np.uint8)
    (out_dir / "samples.bin").write_bytes(stacked.tobytes())
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=int, default=2, choices=(1, 2))
    parser.add_argument("-o", "--out", default=str(REPO_ROOT / "docs" / "weights"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load(args.model)
    plan, tensors = build_plan(model)

    offset, entries, blobs = 0, [], []
    for tensor in tensors:
        entries.append({"offset": offset, "shape": list(tensor.shape), "size": int(tensor.size)})
        blobs.append(tensor.astype("<f4").tobytes())
        offset += tensor.size * 4

    stem = f"model{args.model}"
    (out_dir / f"{stem}.bin").write_bytes(b"".join(blobs))

    manifest = {
        "model": args.model,
        "inputShape": [*INPUT_SIZE, 3],
        "classes": classes_for(args.model),
        "tensors": entries,
        "layers": plan,
        "samples": export_samples(out_dir),
        "sampleShape": [*INPUT_SIZE, 3],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(manifest), encoding="utf-8")

    print(f"wrote {out_dir / (stem + '.bin')}  ({offset / 1024:.0f} KB, "
          f"{sum(t.size for t in tensors):,} params)")
    print(f"wrote {out_dir / (stem + '.json')}  ({len(plan)} layers, "
          f"{len(manifest['classes'])} classes)")
    print(f"wrote {out_dir / 'samples.bin'}  ({len(manifest['samples'])} glyphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
