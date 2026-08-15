# SYNC.md — local vs origin

_Last updated: 2026-08-15 (Phase 0 audit)_

Remote: `git@github.com:Ssavan99/FeatureExtraction_from_AncientScript.git` (origin)

## Summary

| Check | Result |
|---|---|
| Local branch | `main` @ `1a3cad5` |
| Tracking | `origin/main` @ `1a3cad5` — **in sync** |
| Commits ahead of origin | 0 |
| Commits behind origin | 0 |
| Other local branches | none |
| Branches only on GitHub | none |
| Stashes | none |
| Tags | none (local or remote) |
| Untracked files | none |
| `git fsck --lost-found` | clean — no dangling/unreachable objects |
| Uncommitted changes | **1 file** — `character_label_classification.ipynb` (+127 / −69) |
| `.git` size | 7.7 MB |

An extra read-only remote `ghcheck` also points at the same commit; it is a local convenience remote, not a divergent copy.

## What is pushed

Everything in history through `1a3cad5 model 2`:

```
1a3cad5  model 2          <- HEAD = origin/main
dc45906  file changes
67412e1  model-2
9f887b2  model
c9d89ce  model data
979641e  Initial commit
```

Tracked content on origin:
- `README.md` (one line), `.gitignore`
- 3 notebooks: `data_preprocess.ipynb`, `character_feature_classification.ipynb`, `character_label_classification.ipynb`
- `feature_list.csv`, `feature_list_combined.csv`
- `Images/` — 9 result PNGs (training curves, predictions, feature maps, model plot)
- `feature_data/` — 47 folders, one sample JPG each (one per MODI character)
- `custom_cnn_model_1/`, `custom_cnn_model_2/` — Keras SavedModel dirs (~2.1 MB of weights each)
- `custom_cnn_model_2.zip` — 1.9 MB

## What is NOT pushed (local only)

1. **`character_label_classification.ipynb` working-tree edits** — swaps the `pydot`/`graphviz` installs for `!pip install --upgrade tf-keras-vis`, re-runs several cells (execution counts change), and adds an Activation-Maximization cell that **fails** with `UnimplementedError: Cast string to float is not supported` because `seed_input` is given a layer *name* instead of an image tensor. Real but unfinished work. Decision: keep the tf-keras-vis direction, fix the broken cell, then commit.

2. **The datasets** — deliberately outside the repo, in the sibling folder `C:\Users\savan\source\repos\FeatureExtraction-data\` (~3.7 GB). Contains `MODI_HChar.rar` (1.87 GB), the MODI-HChar zip (1.84 GB), the Devanagari dataset zip (80 MB), and `github repo data/` holding `data/` (44 class folders), `data2/` (80 class folders) and the three generated `.npz` bundles (362 MB / 361 MB / 422 MB). **Correctly kept out of git** — must stay out.

## What exists only on GitHub

Nothing. Origin has no branch, tag, or commit absent locally.

## Flags

**Untracked things that should be committed:** none — the working tree has no untracked files.

**Committed things that arguably should not be:**

- `custom_cnn_model_2.zip` (1.9 MB) is a byte-redundant zipped copy of the `custom_cnn_model_2/` directory that sits next to it. Dead weight; slated for removal.
- The two SavedModel directories (~2.1 MB weights each, ~5 MB total) are large-ish binaries, but they are **kept on purpose**: they are what makes the repo runnable without the 3.7 GB dataset, and they power the smoke test and the browser demo. Well under any GitHub limit.

**Secrets / keys:** none found. No `.env`, no credentials, no tokens in tracked files or history.

**Leaked local paths in committed notebook outputs:** `character_label_classification.ipynb` embeds tracebacks and pip logs exposing `c:\Program Files\Comp. sci\Anaconda\envs\myenv\...` and the HPC hostname `c3618.swan.hcc.unl.edu` (UNL Holland Computing Center). Cosmetic, not a credential leak — worth tidying but not urgent.

**Reproducibility gap:** the notebooks read `data/`, `data2/` and `*.npz` which are **not in the repo and not reachable from it**. A clone cannot re-run any notebook end to end. The committed models plus `feature_data/` do allow inference and evaluation, which is the gap the plan closes.

## Class-map recovery (audit finding)

The label→index mapping was never persisted — `LabelBinarizer` sorts alphabetically, so it is reconstructible:

- **Model 1 (47 classes):** `sorted(feature_list.csv['character'])`. Note local `data/` has only 44 folders — `dha`, `ja`, `tha` are absent from the local dataset copy but *were* present at training time, which is why the model has 47 outputs.
- **Model 2 (80 classes):** `sorted(os.listdir('data2/'))`, exactly 80 folders.

Verified: running `custom_cnn_model_1` over the 47 committed `feature_data/` samples with the reconstructed mapping gives **46/47 (97.9 %)** top-1, with the single miss being `aa → tha`. The mapping is correct.
