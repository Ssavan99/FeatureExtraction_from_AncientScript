"""End-to-end checks that run from a clone with no dataset download.

Everything here relies only on committed artifacts: the two SavedModel
directories, the two CSVs, and the 47 sample glyphs in ``feature_data/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.classes import STROKE_FEATURES, classes_for, strokes_for
from src.model import REPO_ROOT, load, load_image
from src.predict import predict

SAMPLE_DIR = REPO_ROOT / "feature_data"

#: Measured at 46/47 = 97.9%. The margin absorbs nondeterminism without
#: letting a genuine regression through.
MIN_TOP1_ACCURACY = 0.90


def sample_images() -> list[tuple[str, Path]]:
    pairs = []
    for folder in sorted(p for p in SAMPLE_DIR.iterdir() if p.is_dir()):
        images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if images:
            pairs.append((folder.name, images[0]))
    return pairs


def test_class_lists_have_expected_sizes():
    assert len(classes_for(1)) == 47
    assert len(classes_for(2)) == 80


def test_class_lists_are_sorted():
    # The mapping is only valid because LabelBinarizer sorts alphabetically.
    for model_id in (1, 2):
        names = classes_for(model_id)
        assert names == sorted(names)
        assert len(names) == len(set(names))


def test_modi_classes_are_a_subset_of_combined():
    assert set(classes_for(1)) <= set(classes_for(2))


def test_model_output_widths_match_class_lists():
    for model_id in (1, 2):
        assert load(model_id).output_shape[-1] == len(classes_for(model_id))


def test_models_accept_the_documented_input_shape():
    for model_id in (1, 2):
        assert load(model_id).input_shape[1:] == (32, 32, 3)


def test_sample_glyphs_are_present():
    assert len(sample_images()) == 47


def test_load_image_is_normalised():
    _, path = sample_images()[0]
    batch = load_image(path)
    assert batch.shape == (1, 32, 32, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_model1_recognises_the_sample_glyphs():
    """The headline check: reconstructed class map + committed weights work."""
    samples = sample_images()
    misses = []
    for true_name, path in samples:
        top1 = predict(path, model_id=1, topk=1)[0][0]
        if top1 != true_name:
            misses.append((true_name, top1))

    accuracy = 1 - len(misses) / len(samples)
    assert accuracy >= MIN_TOP1_ACCURACY, (
        f"top-1 accuracy {accuracy:.1%} below {MIN_TOP1_ACCURACY:.0%}; misses: {misses}"
    )


def test_model2_predicts_modi_glyphs_within_its_larger_class_space():
    # Model 2 chooses between 80 classes rather than 47, so it is held to a
    # looser bar than model 1 on the same images.
    samples = sample_images()
    hits = sum(predict(path, model_id=2, topk=1)[0][0] == name for name, path in samples)
    assert hits / len(samples) >= 0.70


def test_predictions_are_a_probability_distribution():
    _, path = sample_images()[0]
    results = predict(path, model_id=1, topk=47)
    assert len(results) == 47
    assert sum(p for _, p in results) == pytest.approx(1.0, abs=1e-4)
    assert results == sorted(results, key=lambda r: -r[1])


def test_stroke_lookup():
    assert strokes_for("dha", 1) == ["(", "o", "\\"]
    assert set(strokes_for("k", 1)) <= set(STROKE_FEATURES)
    assert strokes_for("not-a-character", 1) == []


def test_featuremaps_render(tmp_path):
    from src.featuremaps import conv_layer_names, render

    _, path = sample_images()[0]
    layers = conv_layer_names(load(1))
    assert len(layers) == 4

    out = render(path, layers[0], tmp_path / "maps.png", model_id=1)
    assert out.is_file() and out.stat().st_size > 0


def test_class_lists_are_not_shared_between_callers():
    # The list order is the model's output order, so a caller mutating it must
    # not be able to corrupt anyone else's copy.
    first = classes_for(1)
    first.append("BOGUS")
    assert "BOGUS" not in classes_for(1)
    assert len(classes_for(1)) == 47
