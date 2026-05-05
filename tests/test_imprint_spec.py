"""
Mathematical property tests for the IMPRINT detector.

IMPRINT scores each window by a Mahalanobis-distance MAD score. Per
window per channel, it builds a feature vector
``log([line_length, energy, P_delta, P_theta, P_alpha, P_beta,
P_low_gamma, P_high_gamma])``, computes the squared Mahalanobis
distance against the cleaned preictal reference distribution for that
channel, then converts to a robust z-score by subtracting the
reference median and dividing by the reference scaled-MAD. Properties
worth pinning:

- on typical seeded noise the output is finite (no NaN/Inf), since the
  reference scaled-MAD is non-degenerate;
- a louder copy of the fit signal lands further from the reference
  distribution and produces strictly larger MAD scores on average;
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

from DynaSD import IMPRINT


FS = 256          # ≥ 240 Hz needed for IMPRINT's high_gamma band (70-120 Hz).
W_SIZE = 1.0
W_STRIDE = 0.5
N_SECONDS = 30   # ≥ IMPRINT.min_sz_dur (9s) so check_inclusion passes.
N_CHANNELS = 4


def _make_signal(amplitude=1.0, seed=0):
    rng = np.random.RandomState(seed)
    n_samples = FS * N_SECONDS
    data = rng.normal(0.0, amplitude, size=(n_samples, N_CHANNELS))
    cols = [f"ch{i}" for i in range(N_CHANNELS)]
    return pd.DataFrame(data, columns=cols)


def _fit_and_forward(model, x_fit, x_score=None):
    if x_score is None:
        x_score = x_fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_fit)
        return model.forward(x_score)


def test_features_are_finite_on_seeded_noise():
    """For seeded N(0, 1) input the cleaned-reference scaled-MAD is
    non-degenerate, so every per-window MAD score must be finite. The
    NaN-fallback at IMPRINT.py:145-147 only triggers when the
    reference scaled-MAD vanishes — never on this fixture."""
    model = IMPRINT(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_signal()
    out = _fit_and_forward(model, x)
    assert np.all(np.isfinite(out.values))


def test_anomalous_amplitude_increases_features():
    """Scoring a louder copy of the fit signal yields strictly larger
    mean MAD scores on every channel. The MAD score grows monotonically
    with how far the test window's feature vector lands from the
    cleaned preictal reference, and 5x amplitude shifts the
    log-energy / log-line-length / log-band-power features uniformly
    upward."""
    x_baseline = _make_signal(amplitude=1.0, seed=0)
    x_loud = _make_signal(amplitude=5.0, seed=0)  # same seed → same shape

    model = IMPRINT(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_baseline)
        feat_baseline = model.forward(x_baseline)
        feat_loud = model.forward(x_loud)

    assert np.all(feat_loud.mean(axis=0) > feat_baseline.mean(axis=0))


def test_repeated_forward_is_deterministic():
    """forward(X) is a pure function once fit: Mahalanobis distance,
    median, and MAD are all closed-form numpy operations with no
    sources of randomness."""
    model = IMPRINT(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_signal()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        a = model.forward(x)
        b = model.forward(x)
    pd.testing.assert_frame_equal(a, b)
