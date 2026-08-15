# Data

The training data is **not in this repository** — it is roughly 3.7 GB of image
archives. Everything needed to *run* the trained models is committed, so you can
clone, install, predict and run the test suite without downloading anything. You
only need the datasets below if you want to re-run `data_preprocess.ipynb` and
retrain.

## Sources

Both datasets are publicly available and free.

| Dataset | Contents | Used for |
|---|---|---|
| **MODI-HChar** — Historical MODI Script Handwritten Character Dataset | 47 MODI character classes (vowels, consonants, digits) | Model 1, and the MODI half of Model 2 |
| **Devanagari Handwritten Character Dataset** (UCI Machine Learning Repository) | 46 Devanagari classes, of which 36 characters are used here | The Devanagari half of Model 2 |

MODI is a historical script used to write Marathi from roughly the 13th century
until the mid-20th, when Devanagari replaced it. Large volumes of administrative
and historical records survive in MODI and comparatively few people can read it,
which is what makes automatic recognition worth doing.

Check the licence terms on each dataset before redistributing either one. They
are deliberately not vendored here.

## Expected layout

The notebooks read two folders of class subdirectories, each subdirectory named
after the character it contains:

```
data/                 # MODI only — 47 class folders
  a/  aa/  ah/  ...  y/
      a1.jpg  a2.jpg  ...

data2/                # MODI + Devanagari combined — 80 class folders
  a/  aa/  adna/  ...  yna/
```

`data2/` has 80 folders rather than 83 because `dha`, `ja` and `tha` exist in
both scripts and share a class.

Both folders are listed in `.gitignore`. Keep them out of the repo.

## Regenerating the preprocessed bundles

`data_preprocess.ipynb` turns those folders into three compressed NumPy bundles.
Each image is converted to RGB, resized to 32×32 and scaled to `[0, 1]`.

| Output | Built from | Contents | Size |
|---|---|---|---|
| `character_images_dataset.npz` | `data/` + `feature_list.csv` | `images`, `features`, `paths`, `names` | ~362 MB |
| `images_label_dataset.npz` | `data/` | `images`, `labels` | ~361 MB |
| `images_label_dataset_combined.npz` | `data2/` | `images`, `labels` | ~422 MB |

Roughly 475,000 images for the MODI-only bundles and 547,000 for the combined
one. These are also gitignored.

## The stroke-feature CSVs (committed)

`feature_list.csv` and `feature_list_combined.csv` are hand-built and small
enough to live in the repo. Each row maps a character to a ten-dimensional
binary vector over the stroke primitives:

```
(    o    \    ||    X    ^    v    >    |    /\
```

So `dha,1,1,1,0,0,0,0,0,0,0` says the glyph `dha` is composed of an open curve,
a loop and a downward diagonal. `feature_list_combined.csv` adds a leading
`Script` column (`MODI` or `Devanagri`) and covers both scripts.

These drive the multi-label experiment in
`character_feature_classification.ipynb` and the stroke lookup printed by
`python -m src.predict`.

## Sample glyphs (committed)

`feature_data/` holds one representative image for each of the 47 MODI
characters — 47 files, a few kilobytes each. These are what the test suite
scores against and what the browser demo's gallery displays, and they are the
reason the project is runnable from a bare clone.

## A caveat about the local dataset copy

The `data/` folder used during development is missing the `dha`, `ja` and `tha`
folders that were present when Model 1 was trained. Model 1 therefore has 47
outputs while a fresh preprocessing run over that particular copy would yield
44 classes. If you rebuild from scratch, verify your class count matches your
model's output width — `tests/test_smoke.py` checks exactly this.
