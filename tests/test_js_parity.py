"""Check the weights exported for the browser demo still reproduce Keras.

``src/export_weights.py`` flattens the model into an op list plus a float32
blob, which ``docs/cnn.js`` then executes by hand. This module re-implements
that same op list in NumPy and asserts it matches Keras. If the export layout
and the JavaScript runtime ever drift apart, this fails first — and it does so
without needing a browser or a JS toolchain in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.model import REPO_ROOT, load, load_image

WEIGHTS_DIR = REPO_ROOT / "docs" / "weights"
MANIFEST = WEIGHTS_DIR / "model2.json"
BLOB = WEIGHTS_DIR / "model2.bin"

pytestmark = pytest.mark.skipif(
    not (MANIFEST.is_file() and BLOB.is_file()),
    reason="exported weights missing; run `python -m src.export_weights`",
)


def load_export():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw = np.frombuffer(BLOB.read_bytes(), dtype="<f4")
    tensors = [
        raw[t["offset"] // 4 : t["offset"] // 4 + t["size"]].reshape(t["shape"])
        for t in manifest["tensors"]
    ]
    return manifest, tensors


def forward(manifest, tensors, x: np.ndarray) -> np.ndarray:
    """Run the exported op list. Mirrors what docs/cnn.js does, in NumPy."""
    for step in manifest["layers"]:
        kind = step["type"]

        if kind == "Conv2D":
            kernel, bias = tensors[step["kernel"]], tensors[step["bias"]]
            kh, kw, _, out_c = kernel.shape
            h, w, _ = x.shape
            out = np.empty((h - kh + 1, w - kw + 1, out_c), dtype=np.float32)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    patch = x[i : i + kh, j : j + kw, :]
                    out[i, j] = np.tensordot(patch, kernel, axes=([0, 1, 2], [0, 1, 2])) + bias
            x = out
        elif kind == "BatchNormalization":
            gamma, beta = tensors[step["gamma"]], tensors[step["beta"]]
            mean, var = tensors[step["mean"]], tensors[step["variance"]]
            x = gamma * (x - mean) / np.sqrt(var + step["epsilon"]) + beta
            continue  # BN carries no activation
        elif kind == "MaxPooling2D":
            ph, pw = step["pool"]
            sh, sw = step["strides"]
            h, w, c = x.shape
            out = np.empty(((h - ph) // sh + 1, (w - pw) // sw + 1, c), dtype=np.float32)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    out[i, j] = x[i * sh : i * sh + ph, j * sw : j * sw + pw].max(axis=(0, 1))
            x = out
            continue
        elif kind == "GlobalAveragePooling2D":
            x = x.mean(axis=(0, 1))
            continue
        elif kind == "Dense":
            x = x @ tensors[step["kernel"]] + tensors[step["bias"]]
        else:
            raise AssertionError(f"unexported layer type {kind}")

        activation = step.get("activation", "linear")
        if activation == "relu":
            x = np.maximum(x, 0.0)
        elif activation == "softmax":
            shifted = x - x.max()
            x = np.exp(shifted) / np.exp(shifted).sum()
        elif activation != "linear":
            raise AssertionError(f"unhandled activation {activation}")
    return x


def test_manifest_describes_every_weight():
    manifest, tensors = load_export()
    assert manifest["classes"] == sorted(manifest["classes"])
    assert len(manifest["classes"]) == 80
    assert manifest["inputShape"] == [32, 32, 3]
    assert sum(t.size for t in tensors) == 177_040  # model 2 parameter count


def test_samples_blob_matches_manifest():
    manifest, _ = load_export()
    raw = (WEIGHTS_DIR / "samples.bin").read_bytes()
    assert len(raw) == len(manifest["samples"]) * 32 * 32 * 3
    assert set(manifest["samples"]) <= set(manifest["classes"])


@pytest.mark.parametrize("character", ["k", "a", "m", "ph"])
def test_exported_weights_match_keras(character):
    manifest, tensors = load_export()
    image = load_image(REPO_ROOT / "feature_data" / character / f"{character}1.jpg")

    expected = load(2).predict(image, verbose=0)[0]
    actual = forward(manifest, tensors, image[0].astype(np.float32))

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-4)
    assert manifest["classes"][int(np.argmax(actual))] == character
