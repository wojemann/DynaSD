<img width="862" alt="logo" src="https://github.com/user-attachments/assets/75f0e87c-8209-4e9b-af48-72e6dc6c1120" />

# DynaSD

**Dyna**mic **S**eizure **D**etection models and utilities for iEEG analysis,
based on work described in the preprint [*Unsupervised seizure annotation
and detection with neural dynamic divergence*][preprint].

[preprint]: https://www.medrxiv.org/content/10.64898/2026.02.15.26346325v1

This package provides multiple detector implementations under a shared
`fit` → `forward` → `get_onset_and_spread` API, including neural dynamic
divergence (NDD) approaches and complementary baseline models for seizure
onset detection and annotation workflows.

## Project status

This repository is being hardened from research code to a publishable
package. API and packaging are stabilizing, with emphasis on:

- timing-focused unit tests for smoothing and windowing behavior,
- optional dependencies by model,
- contributor-friendly base class documentation.

## Installation

> **Pre-release status:** DynaSD is not yet on PyPI. Until the first
> tagged release, install directly from GitHub using the snippets in
> [Pre-release install (from GitHub)](#pre-release-install-from-github).
> The `pip install dynasd` form below will start working once v0.1.0
> is published.

### Core install

```bash
pip install dynasd
```

### Pre-release install (from GitHub)

```bash
# Latest commit on main (recommended for collaborators).
pip install "git+https://github.com/wojemann/DynaSD.git"

# Or pin to a specific commit / tag for reproducibility.
pip install "git+https://github.com/wojemann/DynaSD.git@<commit-or-tag>"

# With extras (note the quotes — your shell will otherwise eat the brackets).
pip install "dynasd[torch] @ git+https://github.com/wojemann/DynaSD.git"
pip install "dynasd[all] @ git+https://github.com/wojemann/DynaSD.git"
```

To upgrade to the newest commit on main, re-run the same command — pip
re-clones the branch tip every time when the spec uses a branch name.

### Optional model extras

Install only the dependencies needed for the model family you use:

```bash
pip install "dynasd[torch]"        # PyTorch-backed detectors (NDD, GIN, LiNDDA, ONCET)
pip install "dynasd[tensorflow]"   # WaveNet/TensorFlow-backed detector (WVNT)
pip install "dynasd[viz]"          # matplotlib for plotting
```

### Development install

```bash
pip install -e ".[test,docs]"
```

`ieeg` is intentionally treated as test/dev tooling for remote data
workflows, not a required runtime dependency.

## Quick start

End-to-end walkthrough on a bundled synthetic seizure recording lives in
[`examples/quickstart.ipynb`](examples/quickstart.ipynb). The four-step
pipeline:

```python
from dynasd import ABSSLP, load_example_seizure

example = load_example_seizure()           # 60s synthetic iEEG, 8ch, fs=256
X, fs = example.signal, example.fs

model = ABSSLP(fs=fs, w_size=1.0, w_stride=0.5)
model.fit(X.iloc[: int((example.seizure_start_sec - 10) * fs)])   # 20s of baseline

sz_prob = model(X)                 # window-by-channel detector scores
onsets = model.get_onset_and_spread(
    sz_prob, threshold=150.0,
    filter_w=10.0, rwin_size=5.0, rwin_req=4.0,
)                                          # per-channel onset times in seconds
```

Every detector class follows the same interface; swap `ABSSLP` for any of
the available models below. See the spec at
[`docs/spec_windowing_smoothing.md`](docs/spec_windowing_smoothing.md)
for the windowing/smoothing/spread math.

## Pretrained-checkpoint detectors

`ONCET` and `WVNT` use pretrained weights that are not shipped in the
wheel. On first instantiation without an explicit `checkpoint_path` /
`model_path`, the package downloads the file from the project's GitHub
Release into a per-user cache directory and verifies its SHA-256.
Subsequent calls load directly from cache.

- **Cache location**: `~/Library/Caches/dynasd/` (macOS),
  `~/.cache/dynasd/` (Linux), `%LOCALAPPDATA%\dynasd\Cache` (Windows).
  Override with `DYNASD_CACHE_DIR=/path/to/dir`.
- **Offline use**: drop the file at the cache path manually; the SHA-256
  check will validate it.
- **Override**: pass an explicit path to skip the download path entirely:

```python
ONCET(checkpoint_path="/abs/path/to/best_model.pth",
      config_path="/abs/path/to/final_training_config.json", ...)
```

## Available models

Exported from `dynasd`:

| Class | Method | Extra |
|---|---|---|
| `ABSSLP` | Mean abs first difference | core |
| `HFER` | High frequency energy ratio | core |
| `IMPRINT` | Mahalanobis-distance MAD score | core |
| `NDD` | Neural Dynamic Divergence (multi-step LSTM) | `[torch]` |
| `GIN` | NDD with residual GRU connections | `[torch]` |
| `LiNDDA` | Linear NDD Approximation | `[torch]` |
| `ONCET` | Pretrained dilated CNN classifier | `[torch]` |
| `WVNT` | Pretrained WaveNet classifier | `[tensorflow]` |
| `DynaSDBase`, `NDDBase` | Base classes for new detectors | core |
| `load_example_seizure` | Bundled synthetic recording | core |

## Testing

```bash
pytest
```

See [`docs/testing_strategy.md`](docs/testing_strategy.md) for the
release-focused plan.

## Contributing

If you want to add a new detector, start with:

- [`dynasd/base.py`](dynasd/base.py)
- [`dynasd/NDDBase.py`](dynasd/NDDBase.py)
- [`docs/extending_models.md`](docs/extending_models.md)

## Citation / contact

If DynaSD is useful in your work, please cite the project and reach out
for collaboration: `wojemann@seas.upenn.edu`.

> Ojemann WKS, Xu Z, Shi H, Walsh K, Pattnaik A, Sinha N, et al.
> Unsupervised seizure annotation and detection with neural dynamic
> divergence. *medRxiv*. Posted online February 17, 2026.
> doi:10.64898/2026.02.15.26346325

BibTeX:

```bibtex
@article{ojemann2026dynasd,
  title   = {Unsupervised seizure annotation and detection with neural dynamic divergence},
  author  = {Ojemann, William K. S. and Xu, Zhongchuan and Shi, Haoer
             and Walsh, Katie and Pattnaik, Akash and Sinha, Nishant
             and Lavelle, Sarah and Aguila, Carlos and Gallagher, Ryan
             and Revell, Andrew and LaRocque, Joshua J. and Korzun, Jacob
             and Kulick-Soper, Catherine V. and Zhou, Daniel J.
             and Galer, Peter D. and Sinha, Saurabh R.
             and Shinohara, Russell T. and Davis, Kathryn A.
             and Litt, Brian and Conrad, Erin C.},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.02.15.26346325},
  url     = {https://www.medrxiv.org/content/10.64898/2026.02.15.26346325v1},
  note    = {Preprint, posted 2026-02-17}
}
```