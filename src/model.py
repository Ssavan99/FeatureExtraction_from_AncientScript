"""Loading trained models and preparing images for them."""

from __future__ import annotations

import os

# TensorFlow and matplotlib each ship their own OpenMP runtime in the Anaconda
# builds this project was developed against; importing both otherwise aborts
# with "OMP: Error #15: Initializing libiomp5md.dll".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Quieten TF's C++ info logging; warnings and errors still surface.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import functools
from pathlib import Path

import numpy as np
from PIL import Image

from .classes import REPO_ROOT

#: Both models take 32x32 RGB input normalised to [0, 1].
INPUT_SIZE = (32, 32)

MODEL_DIRS = {
    1: REPO_ROOT / "custom_cnn_model_1",
    2: REPO_ROOT / "custom_cnn_model_2",
}


@functools.lru_cache(maxsize=None)
def load(model_id: int = 1):
    """Load a trained SavedModel. Cached, so repeated calls are free."""
    import tensorflow as tf

    try:
        path = MODEL_DIRS[model_id]
    except KeyError:
        raise ValueError(f"model_id must be 1 or 2, got {model_id!r}") from None
    if not path.is_dir():
        raise FileNotFoundError(f"No SavedModel at {path}")
    return tf.keras.models.load_model(str(path), compile=False)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file into the ``(1, 32, 32, 3)`` float32 batch the models expect.

    Matches the preprocessing in ``data_preprocess.ipynb``: convert to RGB,
    resize to 32x32, scale to [0, 1].
    """
    with Image.open(path) as img:
        img = img.convert("RGB").resize(INPUT_SIZE)
        array = np.asarray(img, dtype=np.float32) / 255.0
    return array[None, ...]
