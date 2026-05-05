"""
Mathematical property tests for the NN-forecaster detectors
(``NDD``, ``GIN``, ``LiNDDA``).

All three share :class:`DynaSD.NDDBase` and produce per-window features
via ``sqrt(mean(per_sequence_MSE))`` aggregated by
``_aggregate_sequences_to_windows_mse`` (see
``tests/test_nddbase_internals.py`` for the closed-form aggregation
contract). The math properties pinned here are weaker than those for
the closed-form detectors because the inner forecaster is learned —
true determinism requires seeding ``torch`` before construction, and
"correctness" reduces to monotone behavior under amplitude shifts:

- forward output is non-negative everywhere (a square root of a mean
  of squared errors),
- forward(X) is deterministic across repeated calls once a model is
  fit (the inner ``nn.Module`` runs in ``eval``-equivalent mode for
  inference: no dropout layers in NDD / GIN / LiNDDA, and no
  randomness in ``_get_features``),
- a louder copy of the fit signal lands further from the
  RobustScaler-normalized regime the model was trained in and
  produces strictly larger mean MSE on every channel.

Detection-quality / planted-seizure tests live in the deferred end-to-end
suite (``docs/testing_strategy.md`` § 6).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))

from DynaSD import GIN, LiNDDA, NDD


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


# Lightweight training config shared across all three forecasters. Tiny
# hidden sizes plus a handful of epochs keep wall-time low while still
# letting the model learn something — a totally untrained model would
# produce noisy MSE that doesn't separate the louder anomaly cleanly.
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
    # LiNDDA's pretrained _boundary table only covers
    # sequence_length ∈ {1..7}; carry a different sequence length.
    return LiNDDA(**{**_NN_BASE, "sequence_length": 4, "forecast_length": 4})


MODELS = [
    pytest.param(_build_ndd, id="NDD"),
    pytest.param(_build_gin, id="GIN"),
    pytest.param(_build_lindda, id="LiNDDA"),
]


def _fit(build, x_fit):
    """Build the model under a fixed torch seed so weight init is
    reproducible across test invocations."""
    torch.manual_seed(TORCH_SEED)
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_fit)
    return model


@pytest.mark.parametrize("build", MODELS)
def test_features_are_non_negative(build):
    """Output is ``sqrt(mean(seq_mse))`` per window — a square root of
    an average of squared errors — so every entry must be >= 0."""
    x = _make_signal()
    model = _fit(build, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.forward(x)
    assert np.all(out.values >= 0.0)
    assert np.all(np.isfinite(out.values))


@pytest.mark.parametrize("build", MODELS)
def test_repeated_forward_is_deterministic(build):
    """Once fit, ``forward(X)`` involves no randomness: the inner
    ``nn.Module`` has no dropout, and ``_get_features`` runs the model
    deterministically over the input. Two consecutive calls must
    produce identical DataFrames."""
    x = _make_signal()
    model = _fit(build, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = model.forward(x)
        b = model.forward(x)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("build", MODELS)
def test_anomalous_amplitude_increases_features(build):
    """A 5x-amplitude copy of the fit signal lands far outside the
    RobustScaler-normalized regime the inner forecaster was trained
    in, so its per-window MSE must be strictly larger on every
    channel mean than scoring the fit signal itself."""
    x_baseline = _make_signal(amplitude=1.0, seed=0)
    x_loud = _make_signal(amplitude=5.0, seed=0)  # same seed → same shape

    model = _fit(build, x_baseline)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat_baseline = model.forward(x_baseline)
        feat_loud = model.forward(x_loud)

    assert np.all(feat_loud.mean(axis=0) > feat_baseline.mean(axis=0))
