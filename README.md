<img width="862" alt="logo" src="https://github.com/user-attachments/assets/75f0e87c-8209-4e9b-af48-72e6dc6c1120" />

# DynaSD

Dynamic seizure detection models and utilities for iEEG analysis.

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

### Core install

```bash
pip install DynaSD
```

### Optional model extras

Install only the dependencies needed for the model family you use:

```bash
pip install "DynaSD[torch]"        # PyTorch-backed detectors (NDD, GIN, LiNDDA, ONCET)
pip install "DynaSD[tensorflow]"   # WaveNet/TensorFlow-backed detector (WVNT)
pip install "DynaSD[viz]"          # matplotlib for plotting
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
from DynaSD import HFER, load_example_seizure

example = load_example_seizure()           # 60s synthetic iEEG, 8ch, fs=256
X, fs = example.signal, example.fs

model = HFER(fs=fs, w_size=1.0, w_stride=0.5)
model.fit(X.iloc[: int(example.seizure_start_sec * fs)])   # baseline only

sz_prob = model.forward(X)                 # window-by-channel detector scores
onsets = model.get_onset_and_spread(
    sz_prob, threshold=8.0,
    filter_w=10.0, rwin_size=5.0, rwin_req=4.0,
)                                          # per-channel onset times in seconds
```

Every detector class follows the same interface; swap `HFER` for any of
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

Exported from `DynaSD`:

| Class | Method | Extra |
|---|---|---|
| `ABSSLP` | Mean abs first difference | core |
| `HFER` | High-to-low band-power ratio | core |
| `IMPRINT` | Mahalanobis-distance MAD score | core |
| `NDD` | Neural Dynamic Divergence (LSTM) | `[torch]` |
| `GIN` | GRU NDD | `[torch]` |
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
