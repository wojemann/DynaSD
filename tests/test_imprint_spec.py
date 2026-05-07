"""Mathematical property tests for the IMPRINT detector."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from dynasd import IMPRINT


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
    """Every MAD score is finite on seeded N(0, 1) input."""
    model = IMPRINT(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_signal()
    out = _fit_and_forward(model, x)
    assert np.all(np.isfinite(out.values))


def test_anomalous_amplitude_increases_features():
    """A louder copy of the fit signal yields strictly larger per-channel mean MAD scores."""
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
    """forward(X) is deterministic across repeated calls."""
    model = IMPRINT(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)
    x = _make_signal()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        a = model.forward(x)
        b = model.forward(x)
    pd.testing.assert_frame_equal(a, b)
