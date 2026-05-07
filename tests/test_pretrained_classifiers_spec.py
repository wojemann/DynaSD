"""Mathematical property tests for pretrained-classifier detectors (ONCET, WVNT).

Tests skip if checkpoints are unavailable; override paths via
DYNASD_ONCET_CHECKPOINT / DYNASD_ONCET_CONFIG / DYNASD_WVNT_CHECKPOINT.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from dynasd import ONCET, WVNT


N_SECONDS = 30
N_CHANNELS = 4


def _make_signal(fs, n_seconds=N_SECONDS, n_channels=N_CHANNELS, seed=0):
    rng = np.random.RandomState(seed)
    n_samples = fs * n_seconds
    data = rng.normal(0.0, 1.0, size=(n_samples, n_channels))
    cols = [f"ch{i}" for i in range(n_channels)]
    return pd.DataFrame(data, columns=cols)


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
    # ONCET trained at 256 Hz / 1s windows; pin CPU for reproducibility.
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
    # WVNT trained at 128 Hz on 1-second windows.
    return WVNT(
        fs=128, w_size=1.0, w_stride=0.5,
        model_path=_WVNT_CHECKPOINT,
        verbose=False,
    )


MODELS = [
    pytest.param(_build_oncet, id="ONCET"),
    pytest.param(_build_wvnt, id="WVNT"),
]


def _fit_and_forward(model, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        return model.forward(x)


@pytest.mark.parametrize("build", MODELS)
def test_output_is_bounded_to_unit_interval(build):
    """Per-window class-1 probability lies in [0, 1]."""
    model = build()
    x = _make_signal(fs=model.fs)
    out = _fit_and_forward(model, x)
    assert np.all(out.values >= 0.0)
    assert np.all(out.values <= 1.0)
    assert np.all(np.isfinite(out.values))


@pytest.mark.parametrize("build", MODELS)
def test_repeated_forward_is_deterministic(build):
    """forward(X) is deterministic across repeated calls."""
    model = build()
    x = _make_signal(fs=model.fs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
        a = model.forward(x)
        b = model.forward(x)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("build", MODELS)
def test_output_is_not_constant(build):
    """Output has non-trivial spread across windows on random input."""
    model = build()
    x = _make_signal(fs=model.fs)
    out = _fit_and_forward(model, x)
    spread = out.values.std()
    assert spread > 1e-3, (
        f"Output std {spread:.6g} is suspiciously small; classifier may "
        "be returning a near-constant probability."
    )
