"""
Mathematical property tests for the ABSSLP detector.

ABSSLP scores each window by the mean absolute first difference of the
robust-scaled signal, divided by the per-channel std of the fit data and
multiplied by ``fs``. These tests pin down the basic invariants:

- output is non-negative (the per-window quantity is a scaled mean of
  absolute values),
- a louder copy of the fit signal produces strictly larger feature values
  on every channel,
- a signal with no per-sample change (constant within each window) yields
  zero feature values.

Detection-quality / planted-seizure tests live in the deferred end-to-end
suite (``docs/testing_strategy.md`` § 6).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from DynaSD import ABSSLP


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
    """When w_stride == w_size (non-overlapping windows) and the signal is a
    step function that changes only at window boundaries, every window sees
    a single level, so np.diff within each window is zero and ABSSLP
    features are exactly zero.

    Uses w_stride=w_size for this test specifically so no window straddles
    a level boundary.
    """
    n_samples = FS * N_SECONDS
    levels = np.repeat(np.arange(N_SECONDS), FS).astype(float)
    data = np.tile(levels[:, None], (1, N_CHANNELS))
    cols = [f"ch{i}" for i in range(N_CHANNELS)]
    x = pd.DataFrame(data, columns=cols)

    model = ABSSLP(fs=FS, w_size=W_SIZE, w_stride=W_SIZE)  # non-overlapping
    out = _fit_and_forward(model, x)
    assert np.allclose(out.values, 0.0)


def test_higher_amplitude_increases_features():
    """Scoring a louder copy of the fit signal yields strictly larger
    features on every channel mean."""
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
