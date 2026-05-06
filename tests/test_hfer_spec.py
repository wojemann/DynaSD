"""Mathematical property tests for the HFER detector."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from dynasd import HFER


FS = 256
W_SIZE = 1.0
W_STRIDE = 0.5
N_SECONDS = 30
N_CHANNELS = 4


def _make_noise(amplitude=1.0, seed=0):
    rng = np.random.RandomState(seed)
    n_samples = FS * N_SECONDS
    data = rng.normal(0.0, amplitude, size=(n_samples, N_CHANNELS))
    cols = [f"ch{i}" for i in range(N_CHANNELS)]
    return pd.DataFrame(data, columns=cols)


def _make_sine(freq_hz, seed=0):
    """Sine at ``freq_hz`` plus tiny noise floor (avoids divide-by-zero in HFER ratio)."""
    rng = np.random.RandomState(seed)
    n_samples = FS * N_SECONDS
    t = np.arange(n_samples) / FS
    base = np.sin(2 * np.pi * freq_hz * t)
    noise = rng.normal(0.0, 1e-3, size=(n_samples, N_CHANNELS))
    data = np.tile(base[:, None], (1, N_CHANNELS)) + noise
    cols = [f"ch{i}" for i in range(N_CHANNELS)]
    return pd.DataFrame(data, columns=cols)


def _fit_and_forward(model, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        return model.forward(x)


def test_features_are_non_negative_and_finite():
    """HFER output is >= 0 and finite."""
    model = HFER(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_noise()
    out = _fit_and_forward(model, x)
    assert np.all(out.values >= 0.0)
    assert np.all(np.isfinite(out.values))


def test_amplitude_scale_invariance():
    """``forward(alpha * X) == forward(X)`` for ``alpha > 0`` (band-power ratio is scale-invariant)."""
    model = HFER(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_noise()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        baseline = model.forward(x)
        scaled = model.forward(x * 7.5)
    np.testing.assert_allclose(scaled.values, baseline.values, rtol=1e-10, atol=1e-12)


def test_high_band_signal_has_higher_hfer_than_low_band():
    """A 50 Hz (gamma) sine produces strictly larger HFER than a 5 Hz (theta) sine."""
    x_fit = _make_noise()
    x_low = _make_sine(freq_hz=5.0, seed=1)    # theta-band content
    x_high = _make_sine(freq_hz=50.0, seed=1)  # gamma-band content

    model = HFER(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_fit)
        feat_low = model.forward(x_low)
        feat_high = model.forward(x_high)

    assert np.all(feat_high.mean(axis=0) > feat_low.mean(axis=0))


def test_repeated_forward_is_deterministic():
    """forward(X) is a pure function: same fit + same X → identical output."""
    model = HFER(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_noise()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        a = model.forward(x)
        b = model.forward(x)
    pd.testing.assert_frame_equal(a, b)
