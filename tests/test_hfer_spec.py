"""
Mathematical property tests for the HFER detector.

HFER scores each window by the high-to-low frequency band-power ratio
``(P_beta + P_gamma) / (P_theta + P_alpha)`` computed from a Welch PSD
of the raw signal. ``forward`` is a pure function of ``X`` — the fitted
``RobustScaler`` is stored but not consumed during inference — which
makes a few invariants easy to pin:

- output is non-negative (every band power is a non-negative integral
  of a PSD),
- multiplying ``X`` by a positive scalar leaves the ratio unchanged
  (PSD scales as ``alpha**2`` in both numerator and denominator, so
  the scalar cancels),
- a signal whose spectral mass lives in the high band has strictly
  higher HFER than one whose mass lives in the low band,
- ``forward(X)`` is deterministic across repeated calls.

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

from DynaSD import HFER


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
    """Sine at ``freq_hz`` plus a tiny noise floor so the low-band term
    in the HFER ratio is never identically zero (avoids divide-by-zero
    on pure-tone inputs)."""
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
    """HFER output must be >= 0 and finite — both numerator and
    denominator are sums of band powers (non-negative integrals of a
    PSD), and on real-valued noise neither vanishes."""
    model = HFER(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_noise()
    out = _fit_and_forward(model, x)
    assert np.all(out.values >= 0.0)
    assert np.all(np.isfinite(out.values))


def test_amplitude_scale_invariance():
    """``forward(alpha * X) == forward(X)`` for ``alpha > 0``: PSD
    scales as ``alpha**2`` in both numerator and denominator, so the
    ratio cancels exactly (modulo floating-point rounding)."""
    model = HFER(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_noise()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        baseline = model.forward(x)
        scaled = model.forward(x * 7.5)
    np.testing.assert_allclose(scaled.values, baseline.values, rtol=1e-10, atol=1e-12)


def test_high_band_signal_has_higher_hfer_than_low_band():
    """A 50 Hz sine (well within the gamma band 24-97 Hz) must produce
    strictly larger HFER values than a 5 Hz sine (within the theta band
    3.5-7.4 Hz) on every channel mean."""
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
