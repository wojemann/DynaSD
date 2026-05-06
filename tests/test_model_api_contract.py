"""Per-model API contract tests for every detector in ``DynaSD.__all__``."""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from dynasd import ABSSLP, GIN, HFER, IMPRINT, LiNDDA, NDD, ONCET, WVNT
from dynasd.NDDBase import NDDBase
from dynasd.utils import num_wins


# ----------------------------------------------------------------------
# Fixture: small deterministic synthetic input
# ----------------------------------------------------------------------

# Default fs ≥ 240 Hz so IMPRINT's high_gamma band (70-120 Hz) fits.
# Per-model fs overrides (e.g. WVNT @ 128 Hz) are read from model.fs.
FS = 256
W_SIZE = 1.0
W_STRIDE = 0.5
N_SECONDS = 30
N_CHANNELS = 4


def _make_signal(fs, n_seconds=N_SECONDS, n_channels=N_CHANNELS, seed=0):
    """Seeded random iEEG-like data at the given sampling rate."""
    rng = np.random.RandomState(seed)
    n_samples = fs * n_seconds
    data = rng.normal(0.0, 1.0, size=(n_samples, n_channels))
    cols = [f"ch{i}" for i in range(n_channels)]
    return pd.DataFrame(data, columns=cols)


def _signal_for(model):
    """Build a synthetic signal matched to ``model.fs``."""
    return _make_signal(fs=model.fs)


# ----------------------------------------------------------------------
# Per-model build helpers
# ----------------------------------------------------------------------

def _build_absslp():
    return ABSSLP(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)


def _build_hfer():
    return HFER(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)


def _build_imprint():
    return IMPRINT(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE)


# Lightweight NDD-family config (tiny model, 2 epochs) for fast tests.
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
    # LiNDDA's _boundary table only covers sequence_length ∈ {1..7}.
    return LiNDDA(**{**_NN_BASE, "sequence_length": 4, "forecast_length": 4})


# ----------------------------------------------------------------------
# Pretrained-checkpoint detectors (ONCET, WVNT). Skip if checkpoint missing;
# override paths via DYNASD_ONCET_CHECKPOINT / DYNASD_WVNT_CHECKPOINT etc.
# ----------------------------------------------------------------------

_ONCET_CHECKPOINT = os.environ.get(
    "DYNASD_ONCET_CHECKPOINT",
    "/Users/wojemann/local_data/dynasd_data/PROCESSED_DATA/CHECKPOINTS/ONCET/balanced_xl/best_model.pth",
)
_ONCET_CONFIG = os.environ.get(
    "DYNASD_ONCET_CONFIG",
    "/Users/wojemann/local_data/dynasd_data/PROCESSED_DATA/CHECKPOINTS/ONCET/balanced_xl/final_training_config.json",
)
_WVNT_CHECKPOINT = os.environ.get(
    "DYNASD_WVNT_CHECKPOINT",
    "/Users/wojemann/local_data/dynasd_data/PROCESSED_DATA/CHECKPOINTS/WaveNet/v111.hdf5",
)


def _build_oncet():
    if not (os.path.exists(_ONCET_CHECKPOINT) and os.path.exists(_ONCET_CONFIG)):
        pytest.skip(
            f"ONCET checkpoint not available at {_ONCET_CHECKPOINT} / "
            f"{_ONCET_CONFIG}. Set DYNASD_ONCET_CHECKPOINT / "
            f"DYNASD_ONCET_CONFIG to override."
        )
    # ONCET trained at 256 Hz / 1s windows; force CPU for determinism.
    return ONCET(
        fs=256, w_size=1.0, w_stride=0.5,
        checkpoint_path=_ONCET_CHECKPOINT,
        config_path=_ONCET_CONFIG,
        device="cpu",
        verbose=False,
    )


def _build_wvnt():
    if not _WVNT_CHECKPOINT or not os.path.exists(_WVNT_CHECKPOINT):
        pytest.skip(
            f"WVNT checkpoint not available at {_WVNT_CHECKPOINT}. "
            f"Set DYNASD_WVNT_CHECKPOINT to override."
        )
    pytest.importorskip("tensorflow")
    # WVNT trained at 128 Hz / 1s windows; per-test signal sized from model.fs.
    return WVNT(
        fs=128, w_size=1.0, w_stride=0.5,
        model_path=_WVNT_CHECKPOINT,
        verbose=False,
    )


MODELS = [
    pytest.param(_build_absslp, id="ABSSLP"),
    pytest.param(_build_hfer, id="HFER"),
    pytest.param(_build_imprint, id="IMPRINT"),
    pytest.param(_build_ndd, id="NDD"),
    pytest.param(_build_gin, id="GIN"),
    pytest.param(_build_lindda, id="LiNDDA"),
    pytest.param(_build_oncet, id="ONCET"),
    pytest.param(_build_wvnt, id="WVNT"),
]


# ----------------------------------------------------------------------
# Contract tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("build", MODELS)
def test_constructor_does_not_raise(build):
    """Every model accepts (fs, w_size, w_stride) at construction."""
    model = build()
    assert isinstance(model.fs, (int, float))
    assert isinstance(model.w_size, (int, float))
    assert isinstance(model.w_stride, (int, float))


@pytest.mark.parametrize("build", MODELS)
def test_forward_before_fit_raises(build):
    """Calling forward() before fit() must raise loudly, not return junk."""
    model = build()
    signal = _signal_for(model)
    with pytest.raises((AssertionError, ValueError, AttributeError, NotImplementedError)):
        model.forward(signal)


@pytest.mark.parametrize("build", MODELS)
def test_is_fitted_set_after_fit(build):
    """fit(X) must set is_fitted=True."""
    model = build()
    signal = _signal_for(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(signal)
    assert model.is_fitted is True


@pytest.mark.parametrize("build", MODELS)
def test_forward_returns_dataframe(build):
    """forward(X) returns a pandas DataFrame."""
    model = build()
    signal = _signal_for(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(signal)
        out = model.forward(signal)
    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize("build", MODELS)
def test_forward_shape_matches_num_wins(build):
    """Output shape is (num_wins(len(X), fs, w_size, w_stride), n_channels)."""
    model = build()
    signal = _signal_for(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(signal)
        out = model.forward(signal)
    expected_n_windows = num_wins(len(signal), model.fs, model.w_size, model.w_stride)
    assert out.shape == (expected_n_windows, signal.shape[1])


@pytest.mark.parametrize("build", MODELS)
def test_forward_columns_match_input(build):
    """forward output columns are the input channel names, in order."""
    model = build()
    signal = _signal_for(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(signal)
        out = model.forward(signal)
    assert list(out.columns) == list(signal.columns)


@pytest.mark.parametrize("build", MODELS)
def test_forward_index_is_realized_window_times(build):
    """forward(X)'s row index is realized window-start times in seconds."""
    model = build()
    signal = _signal_for(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(signal)
        out = model.forward(signal)
    expected_index = model.get_win_index(len(signal))
    assert out.index.name == "t_sec"
    np.testing.assert_allclose(out.index.values, expected_index.values, rtol=0, atol=1e-12)


@pytest.mark.parametrize("build", MODELS)
def test_call_equals_forward(build):
    """model(X) is exactly model.forward(X)."""
    model = build()
    signal = _signal_for(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(signal)
        out_forward = model.forward(signal)
        out_call = model(signal)
    pd.testing.assert_frame_equal(out_forward, out_call)


# ----------------------------------------------------------------------
# NDDBase-specific tests (kwarg validation, training-param absorption)
# ----------------------------------------------------------------------

def test_nddbase_rejects_unknown_kwargs():
    """An unexpected kwarg raises TypeError naming the offender."""
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
    """The TypeError names the user-facing class (e.g. NDD, GIN), not the base."""
    try:
        GIN(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE, hidden_size=4,
            num_layers=1, num_stacks=1, foo=42)
    except TypeError as e:
        assert "GIN" in str(e), f"Error should attribute failure to GIN, got: {e}"
    else:
        pytest.fail("Expected TypeError, none raised.")


def test_nddbase_absorbs_known_training_params():
    """Training-loop kwargs (early_stopping, val_split, ...) are stored on the instance."""
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
    """Base-class kwargs (scaler_class, scaler_kwargs) thread through to DynaSDBase."""
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
    """Unsupported sequence_length raises ValueError, not KeyError."""
    with pytest.raises(ValueError, match="sequence_length"):
        GIN(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE,
            hidden_size=4, num_layers=1, num_stacks=1,
            sequence_length=99, forecast_length=1,
            num_epochs=1, use_cuda=False, verbose=False)


def test_lindda_unsupported_sequence_length_raises():
    """Unsupported sequence_length (outside {1..7}) raises ValueError, not KeyError."""
    with pytest.raises(ValueError, match="sequence_length"):
        LiNDDA(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE,
               sequence_length=99, forecast_length=1,
               num_epochs=1, use_cuda=False, verbose=False)


@pytest.mark.parametrize("n_samples", [
    7680,   # 30.0000s — exact window boundary
    7807,   # 30.4961s — between window boundaries
    7808,   # 30.5000s — exact window boundary
    7935,   # 30.9961s — between window boundaries
])
def test_nddbase_forward_length_matches_num_wins(n_samples):
    """NDDBase.forward output has exactly num_wins rows for any input length."""
    fs = 256
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        rng.normal(size=(n_samples, 2)),
        columns=["ch0", "ch1"],
    )
    model = NDD(
        fs=fs, w_size=1.0, w_stride=0.5,
        hidden_size=4, num_layers=1,
        sequence_length=12, forecast_length=1,
        num_epochs=1, batch_size="full", lr=0.01,
        use_cuda=False, verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
        out = model.forward(X)
    expected_n_wins = num_wins(n_samples, fs, 1.0, 0.5)
    assert out.shape[0] == expected_n_wins, (
        f"forward returned {out.shape[0]} rows; expected num_wins={expected_n_wins}"
    )
    assert out.index.name == "t_sec"
    np.testing.assert_allclose(
        out.index.values, model.get_win_index(n_samples).values, rtol=0, atol=1e-12
    )


def test_nddbase_is_abstract_for_forward():
    """Bare NDDBase.forward() (no fitted subclass model) raises rather than returning junk."""
    base = NDDBase(fs=FS, w_size=W_SIZE, w_stride=W_STRIDE, use_cuda=False)
    assert base.is_fitted is False
    rng = np.random.RandomState(0)
    x = pd.DataFrame(
        rng.normal(size=(FS * 5, 2)), columns=["ch0", "ch1"]
    )
    with pytest.raises((AssertionError, AttributeError, TypeError)):
        base.forward(x)
