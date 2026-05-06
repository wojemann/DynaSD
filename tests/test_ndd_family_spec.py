"""Mathematical property tests for NDD-family detectors (NDD, GIN, LiNDDA)."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))

from dynasd import GIN, LiNDDA, NDD


FS = 256
W_SIZE = 1.0
W_STRIDE = 0.5
N_SECONDS = 30
N_CHANNELS = 4
TORCH_SEED = 1234


def _make_signal(amplitude=1.0, seed=0):
    rng = np.random.RandomState(seed)
    n_samples = FS * N_SECONDS
    data = rng.normal(0.0, amplitude, size=(n_samples, N_CHANNELS))
    cols = [f"ch{i}" for i in range(N_CHANNELS)]
    return pd.DataFrame(data, columns=cols)


# Lightweight shared training config (small model, few epochs).
_NN_BASE = dict(
    fs=FS,
    w_size=W_SIZE,
    w_stride=W_STRIDE,
    sequence_length=12,
    forecast_length=1,
    num_epochs=5,
    batch_size="full",
    lr=0.01,
    use_cuda=False,
    verbose=False,
)


def _build_ndd():
    return NDD(hidden_size=8, num_layers=1, **_NN_BASE)


def _build_gin():
    return GIN(hidden_size=8, num_layers=1, num_stacks=1, **_NN_BASE)


def _build_lindda():
    # LiNDDA's _boundary table only covers sequence_length ∈ {1..7}.
    return LiNDDA(**{**_NN_BASE, "sequence_length": 4, "forecast_length": 4})


MODELS = [
    pytest.param(_build_ndd, id="NDD"),
    pytest.param(_build_gin, id="GIN"),
    pytest.param(_build_lindda, id="LiNDDA"),
]


def _fit(build, x_fit):
    """Build under a fixed torch seed for reproducible weight init."""
    torch.manual_seed(TORCH_SEED)
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_fit)
    return model


@pytest.mark.parametrize("build", MODELS)
def test_features_are_non_negative(build):
    """forward output is >= 0 and finite."""
    x = _make_signal()
    model = _fit(build, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.forward(x)
    assert np.all(out.values >= 0.0)
    assert np.all(np.isfinite(out.values))


@pytest.mark.parametrize("build", MODELS)
def test_repeated_forward_is_deterministic(build):
    """forward(X) is deterministic across repeated calls."""
    x = _make_signal()
    model = _fit(build, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = model.forward(x)
        b = model.forward(x)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("build", MODELS)
def test_anomalous_amplitude_increases_features(build):
    """A louder copy of the fit signal yields strictly larger per-channel mean MSE."""
    x_baseline = _make_signal(amplitude=1.0, seed=0)
    x_loud = _make_signal(amplitude=5.0, seed=0)  # same seed → same shape

    model = _fit(build, x_baseline)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat_baseline = model.forward(x_baseline)
        feat_loud = model.forward(x_loud)

    assert np.all(feat_loud.mean(axis=0) > feat_baseline.mean(axis=0))
