<img width="862" alt="logo" src="https://github.com/user-attachments/assets/75f0e87c-8209-4e9b-af48-72e6dc6c1120" />

# DynaSD

Dynamic seizure detection models and utilities for iEEG analysis.

This package provides multiple detector implementations under a shared API, including
neural dynamic divergence (NDD) approaches and complementary baseline models for
seizure onset detection and annotation workflows.

## Project status

This repository is currently being hardened from research code to a publishable package.
API and packaging are stabilizing, with emphasis on:
- timing-focused unit tests for smoothing and windowing behavior,
- optional dependencies by model,
- contributor-friendly base class documentation.

## Installation

### Core install

Install the lightweight core package:

```bash
pip install DynaSD
```

### Optional model extras

Install only the dependencies needed for the model family you use:

```bash
pip install "DynaSD[torch]"        # PyTorch-backed detectors
pip install "DynaSD[tensorflow]"   # WaveNet/TensorFlow-backed detectors
```

### Development install

```bash
pip install -e ".[test,docs]"
```

`ieeg` is intentionally treated as test/dev tooling for remote data workflows, not a
required runtime dependency for package functionality.

## Quick start

```python
from DynaSD import ABSSLP

# X is expected to be a pandas DataFrame (samples x channels)
model = ABSSLP(fs=128, w_size=1.0, w_stride=0.5)
model.fit(X_train)
features = model.forward(X_eval)
```

Different model classes may require additional extras and model-specific initialization.

## Available models

Current exported classes include:
- `DynaSDBase`, `NDDBase`
- `ABSSLP`
- `NDD`, `GIN`, `ONCET`
- `LiNDDA`, `MINDD`, `LiRNDDA`
- `IMPRINT`, `HFER`
- `WVNT` (WaveNet wrapper)

## Testing

Run tests with:

```bash
pytest
```

See `docs/testing_strategy.md` for the release-focused plan, with emphasis on
timing-sensitive smoothing/windowing functionality and dependency isolation.

## Contributing

If you want to add a new model, start with:
- `DynaSD/base.py`
- `DynaSD/NDDBase.py`
- `docs/extending_models.md`

## Citation / contact

If DynaSD is useful in your work, please cite the project and reach out for collaboration:
`wojemann@seas.upenn.edu`.
