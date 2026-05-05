"""
Per-model API contract tests.

Every detector class exported from ``DynaSD.__all__`` must satisfy the
unified inference contract documented on
:class:`DynaSD.base.DynaSDBase.forward`. These tests exercise that contract
end-to-end on small seeded synthetic data:

- constructor accepts ``fs``, ``w_size``, ``w_stride``;
- ``forward(X)`` before ``fit(X)`` raises (loud, not silent);
- ``fit(X)`` sets ``is_fitted = True``;
- ``forward(X)`` returns a ``DataFrame`` of shape
  ``(num_wins(len(X), fs, w_size, w_stride), n_channels)`` with channel-name
  columns;
- ``model(X)`` and ``model.forward(X)`` produce identical output.

Detection-quality / numerical-correctness tests on simulated seizure
fixtures are out of scope here; see ``docs/testing_strategy.md`` § 6 for
the deferred end-to-end suite.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from DynaSD import ABSSLP, HFER, NDD
from DynaSD.utils import num_wins


# ----------------------------------------------------------------------
# Fixture: small deterministic synthetic input
# ----------------------------------------------------------------------

FS = 128
W_SIZE = 1.0
W_STRIDE = 0.5
N_SECONDS = 30
N_CHANNELS = 4


@pytest.fixture
def synthetic_signal():
    """30 seconds of seeded random iEEG-like data, 4 channels."""
    rng = np.random.RandomState(0)
    n_samples = FS * N_SECONDS
    data = rng.normal(0.0, 1.0, size=(n_samples, N_CHANNELS))
    cols = [f"ch{i}" for i in range(N_CHANNELS)]
    return pd.DataFrame(data, columns=cols)


# ----------------------------------------------------------------------
# Per-model build helpers
# ----------------------------------------------------------------------

def _build_absslp():
    return ABSSLP(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)


def _build_hfer():
    return HFER(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)


def _build_ndd():
    # Deliberately tiny config so fit completes quickly under tests.
    return NDD(
        fs=FS,
        w_size=W_SIZE,
        w_stride=W_STRIDE,
        hidden_size=4,
        num_layers=1,
        sequence_length=12,
        forecast_length=1,
        num_epochs=2,
        batch_size="full",
        lr=0.01,
        use_cuda=False,
        verbose=False,
    )


MODELS = [
    pytest.param(_build_absslp, id="ABSSLP"),
    pytest.param(_build_hfer, id="HFER"),
    pytest.param(_build_ndd, id="NDD"),
]


# ----------------------------------------------------------------------
# Contract tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("build", MODELS)
def test_constructor_does_not_raise(build):
    """Every model accepts (fs, w_size, w_stride) at construction."""
    model = build()
    assert model.fs == FS
    assert model.w_size == W_SIZE
    assert model.w_stride == W_STRIDE


@pytest.mark.parametrize("build", MODELS)
def test_forward_before_fit_raises(build, synthetic_signal):
    """Calling forward() before fit() must raise loudly, not return junk."""
    model = build()
    with pytest.raises((AssertionError, ValueError, AttributeError, NotImplementedError)):
        model.forward(synthetic_signal)


@pytest.mark.parametrize("build", MODELS)
def test_is_fitted_set_after_fit(build, synthetic_signal):
    """fit(X) must set is_fitted=True."""
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(synthetic_signal)
    assert model.is_fitted is True


@pytest.mark.parametrize("build", MODELS)
def test_forward_returns_dataframe(build, synthetic_signal):
    """forward(X) returns a pandas DataFrame."""
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(synthetic_signal)
        out = model.forward(synthetic_signal)
    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize("build", MODELS)
def test_forward_shape_matches_num_wins(build, synthetic_signal):
    """Output shape is (num_wins(len(X), fs, w_size, w_stride), n_channels)."""
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(synthetic_signal)
        out = model.forward(synthetic_signal)
    expected_n_windows = num_wins(len(synthetic_signal), FS, W_SIZE, W_STRIDE)
    assert out.shape == (expected_n_windows, N_CHANNELS)


@pytest.mark.parametrize("build", MODELS)
def test_forward_columns_match_input(build, synthetic_signal):
    """forward output columns are the input channel names, in order."""
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(synthetic_signal)
        out = model.forward(synthetic_signal)
    assert list(out.columns) == list(synthetic_signal.columns)


@pytest.mark.parametrize("build", MODELS)
def test_call_equals_forward(build, synthetic_signal):
    """model(X) is exactly model.forward(X)."""
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(synthetic_signal)
        out_forward = model.forward(synthetic_signal)
        out_call = model(synthetic_signal)
    pd.testing.assert_frame_equal(out_forward, out_call)
