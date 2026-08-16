"""Class lists and stroke-feature lookup for the two trained models.

The label -> index mapping was never saved alongside the models. It is
recoverable because training used ``sklearn.preprocessing.LabelBinarizer``,
which sorts its classes alphabetically. So the index of a character in the
model's softmax output is simply its position in the sorted list of class
names.

Both lists are derived from the committed CSVs rather than hardcoded:

* Model 1 (47 classes, MODI only)      -> ``feature_list.csv``
* Model 2 (80 classes, MODI + Devanagari) -> ``feature_list_combined.csv``

``feature_list_combined.csv`` holds 83 rows (47 MODI + 36 Devanagari) but only
80 unique names: ``dha``, ``ja`` and ``tha`` appear in both scripts and collapse
to a shared class, which is why model 2 has 80 outputs and not 83.

Verified in ``tests/test_smoke.py``: with this mapping, model 1 classifies 46 of
the 47 sample glyphs in ``feature_data/`` correctly.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_LIST_CSV = REPO_ROOT / "feature_list.csv"
FEATURE_LIST_COMBINED_CSV = REPO_ROOT / "feature_list_combined.csv"

#: The ten stroke primitives a character is decomposed into.
STROKE_FEATURES = ["(", "o", "\\", "||", "X", "^", "v", ">", "|", "/\\"]


@functools.lru_cache(maxsize=None)
def _read_classes(csv: Path) -> tuple[str, ...]:
    # Cached as a tuple so a caller cannot mutate the shared copy. The order of
    # this sequence *is* the model's output order, so corrupting it would
    # silently mislabel every subsequent prediction.
    return tuple(sorted(set(pd.read_csv(csv)["character"].astype(str))))


def modi_classes() -> list[str]:
    """The 47 MODI character names, in model-1 output order."""
    return list(_read_classes(FEATURE_LIST_CSV))


def combined_classes() -> list[str]:
    """The 80 MODI + Devanagari character names, in model-2 output order."""
    return list(_read_classes(FEATURE_LIST_COMBINED_CSV))


def classes_for(model_id: int) -> list[str]:
    """Class list for model ``1`` (47 MODI) or ``2`` (80 combined).

    Returns a fresh list each call; mutating it does not affect other callers.
    """
    if model_id == 1:
        return modi_classes()
    if model_id == 2:
        return combined_classes()
    raise ValueError(f"model_id must be 1 or 2, got {model_id!r}")


@functools.lru_cache(maxsize=None)
def _stroke_table(model_id: int) -> dict[str, tuple[int, ...]]:
    csv = FEATURE_LIST_CSV if model_id == 1 else FEATURE_LIST_COMBINED_CSV
    df = pd.read_csv(csv)
    table: dict[str, tuple[int, ...]] = {}
    for _, row in df.iterrows():
        # Characters shared between the two scripts collapse to one class, so
        # keep the first row seen rather than letting the later one win.
        table.setdefault(str(row["character"]), tuple(int(row[f]) for f in STROKE_FEATURES))
    return table


def strokes_for(character: str, model_id: int = 1) -> list[str]:
    """Stroke primitives that make up ``character``, per the hand-built CSV.

    Returns an empty list if the character is not in the table.
    """
    vector = _stroke_table(model_id).get(character)
    if vector is None:
        return []
    return [name for name, present in zip(STROKE_FEATURES, vector) if present]
