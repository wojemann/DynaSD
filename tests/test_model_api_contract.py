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

from DynaSD import ABSSLP, GIN, HFER, IMPRINT, LiNDDA, NDD
from DynaSD.NDDBase import NDDBase
from DynaSD.utils import num_wins


# ----------------------------------------------------------------------
# Fixture: small deterministic synthetic input
# ----------------------------------------------------------------------

FS = 256          # ≥ 240 Hz needed for IMPRINT's high_gamma band (70-120 Hz).
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


def _build_imprint():
    return IMPRINT(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)


# Shared lightweight config for the NN forecasters. Tiny hidden sizes and 2
# epochs keep test wall-time small. ``sequence_length`` is the longest
# parameter that affects fit-time; defaults across these models are 12-16.
_NN_BASE = dict(
    fs=FS,
    w_size=W_SIZE,
    w_stride=W_STRIDE,
    sequence_length=12,
    forecast_length=1,
    num_epochs=2,
    batch_size="full",
    lr=0.01,
    use_cuda=False,
    verbose=False,
)


def _build_ndd():
    return NDD(hidden_size=4, num_layers=1, **_NN_BASE)


def _build_gin():
    return GIN(hidden_size=4, num_layers=1, num_stacks=1, **_NN_BASE)


def _build_lindda():
    # LiNDDA's _boundary table only covers sequence_length ∈ {1..7}; using a
    # different value raises KeyError in __init__. Override the shared
    # _NN_BASE config for this model.
    return LiNDDA(**{**_NN_BASE, "sequence_length": 4, "forecast_length": 4})


MODELS = [
    pytest.param(_build_absslp, id="ABSSLP"),
    pytest.param(_build_hfer, id="HFER"),
    pytest.param(_build_imprint, id="IMPRINT"),
    pytest.param(_build_ndd, id="NDD"),
    pytest.param(_build_gin, id="GIN"),
    pytest.param(_build_lindda, id="LiNDDA"),
]

# ONCET and WVNT are intentionally excluded: both require a ``model_path``
# pointing to a pretrained checkpoint, which is not available in CI. The
# unified forward(X) → DataFrame contract is documented to apply equally
# to those two; the per-model integration suite (deferred until simulated-
# signal fixtures land — see docs/testing_strategy.md § 6) will exercise
# them with real checkpoints.


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
def test_forward_index_is_realized_window_times(build, synthetic_signal):
    """forward(X)'s row index is the realized window-start times in seconds
    (spec section 5, Phase F). Equal to ``model.get_win_index(len(X))``."""
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(synthetic_signal)
        out = model.forward(synthetic_signal)
    expected_index = model.get_win_index(len(synthetic_signal))
    assert out.index.name == "t_sec"
    np.testing.assert_allclose(out.index.values, expected_index.values, rtol=0, atol=1e-12)


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


# ----------------------------------------------------------------------
# NDDBase-specific tests (kwarg validation, training-param absorption)
# ----------------------------------------------------------------------

def test_nddbase_rejects_unknown_kwargs():
    """An unexpected kwarg must raise a clear TypeError naming the offender,
    not silently propagate to DynaSDBase.

    Regression test for the pre-Phase-B bug where typos like ``train_win``
    or ``num_chunnels=...`` would surface as a bewildering
    ``DynaSDBase.__init__() got an unexpected keyword argument`` error.
    """
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        NDD(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE, totally_made_up_kwarg=42)


def test_nddbase_error_message_names_offender():
    """The TypeError mentions the offending kwarg name verbatim."""
    try:
        NDD(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE, my_typoed_param=1)
    except TypeError as e:
        assert "my_typoed_param" in str(e)
    else:
        pytest.fail("Expected TypeError naming the offender, none raised.")


def test_nddbase_error_message_names_actual_class():
    """The TypeError attributes the failure to the user-facing class
    (NDD, GIN, ...), not to NDDBase or DynaSDBase. This is what makes the
    error actionable for users."""
    try:
        GIN(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE, hidden_size=4,
            num_layers=1, num_stacks=1, foo=42)
    except TypeError as e:
        assert "GIN" in str(e), f"Error should attribute failure to GIN, got: {e}"
    else:
        pytest.fail("Expected TypeError, none raised.")


def test_nddbase_absorbs_known_training_params():
    """Training-loop parameters (early_stopping, val_split, etc.) are
    accepted as **kwargs and stored on the instance."""
    model = NDD(
        fs=FS, w_size=W_SIZE, w_stride=W_STRIDE,
        hidden_size=4, num_layers=1,
        sequence_length=12, forecast_length=1,
        num_epochs=2, batch_size="full", lr=0.01,
        use_cuda=False, verbose=False,
        early_stopping=True, val_split=0.3, patience=7, tolerance=1e-3,
    )
    assert model.early_stopping is True
    assert model.val_split == 0.3
    assert model.patience == 7
    assert model.tolerance == 1e-3


def test_nddbase_passes_scaler_kwargs_to_base():
    """Base-class kwargs (scaler_class, scaler_kwargs) thread through
    NDDBase to DynaSDBase without being mistaken for training params."""
    from sklearn.preprocessing import StandardScaler

    model = NDD(
        fs=FS, w_size=W_SIZE, w_stride=W_STRIDE,
        hidden_size=4, num_layers=1,
        sequence_length=12, forecast_length=1,
        num_epochs=2, batch_size="full", lr=0.01,
        use_cuda=False, verbose=False,
        scaler_class=StandardScaler,
    )
    assert model.scaler_class is StandardScaler


def test_gin_unsupported_sequence_length_raises():
    """GIN's pretrained _boundary table is sparse; unsupported
    sequence_length values must raise ValueError naming the supported
    set, not KeyError mid-construction."""
    with pytest.raises(ValueError, match="sequence_length"):
        GIN(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE,
            hidden_size=4, num_layers=1, num_stacks=1,
            sequence_length=99, forecast_length=1,
            num_epochs=1, use_cuda=False, verbose=False)


def test_lindda_unsupported_sequence_length_raises():
    """LiNDDA's pretrained _boundary table covers only sequence_length
    ∈ {1..7}; unsupported values must raise ValueError, not KeyError."""
    with pytest.raises(ValueError, match="sequence_length"):
        LiNDDA(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE,
               sequence_length=99, forecast_length=1,
               num_epochs=1, use_cuda=False, verbose=False)


def test_nddbase_is_abstract_for_forward():
    """Instantiating NDDBase directly and calling forward() without a
    fitted subclass-specific model must not silently return junk.
    NDDBase requires sequence_length and a torch model to forward; without
    those it should error rather than return garbage."""
    base = NDDBase(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE, use_cuda=False)
    assert base.is_fitted is False
    rng = np.random.RandomState(0)
    x = pd.DataFrame(
        rng.normal(size=(FS * 5, 2)), columns=["ch0", "ch1"]
    )
    with pytest.raises((AssertionError, AttributeError, TypeError)):
        base.forward(x)
