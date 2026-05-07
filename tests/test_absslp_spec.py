"""Mathematical property tests for the ABSSLP detector."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from dynasd import ABSSLP


FS = 128
W_SIZE = 1.0
W_STRIDE = 0.5
N_SECONDS = 30
N_CHANNELS = 4


def _make_signal(amplitude=1.0, seed=0):
    rng = np.random.RandomState(seed)
    n_samples = FS * N_SECONDS
    data = rng.normal(0.0, amplitude, size=(n_samples, N_CHANNELS))
    cols = [f"ch{i}" for i in range(N_CHANNELS)]
    return pd.DataFrame(data, columns=cols)


def _fit_and_forward(model, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        return model.forward(x)


def test_features_are_non_negative():
    """ABSSLP output must be >= 0 everywhere — it's a scaled mean of |diff|."""
    model = ABSSLP(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_signal()
    out = _fit_and_forward(model, x)
    assert np.all(out.values >= 0.0)


def test_constant_signal_per_window_has_zero_features():
    """A signal constant within each (non-overlapping) window yields zero features."""
    n_samples = FS * N_SECONDS
    levels = np.repeat(np.arange(N_SECONDS), FS).astype(float)
    data = np.tile(levels[:, None], (1, N_CHANNELS))
    cols = [f"ch{i}" for i in range(N_CHANNELS)]
    x = pd.DataFrame(data, columns=cols)

    model = ABSSLP(fs=FS, w_size=W_SIZE, w_stride=W_SIZE)  # non-overlapping
    out = _fit_and_forward(model, x)
    assert np.allclose(out.values, 0.0)


def test_higher_amplitude_increases_features():
    """A louder copy of the fit signal yields strictly larger per-channel mean features."""
    x_baseline = _make_signal(amplitude=1.0, seed=0)
    x_loud = _make_signal(amplitude=3.0, seed=0)  # same seed → same shape

    model = ABSSLP(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_baseline)
        feat_baseline = model.forward(x_baseline)
        feat_loud = model.forward(x_loud)

    # Mean across windows for each channel; loud > baseline strictly.
    assert np.all(feat_loud.mean(axis=0) > feat_baseline.mean(axis=0))


def test_repeated_forward_is_deterministic():
    """forward(X) is a pure function: same fit + same X → identical output."""
    model = ABSSLP(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_signal()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        a = model.forward(x)
        b = model.forward(x)
    pd.testing.assert_frame_equal(a, b)
