"""
Tests for the windowing & smoothing spec.

Reference: docs/spec_windowing_smoothing.md

Each test corresponds to a bullet in section 11 of the spec. Tests are
written against the locked contract; failures indicate either a code-spec
divergence or a test bug. During the code-consolidation phase, tests for
not-yet-implemented behavior (validation, UserWarnings, sample-count-first
math) are expected to fail until the corresponding code lands.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from DynaSD.utils import num_wins, moving_win_clips
from DynaSD.base import DynaSDBase


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_base(fs=512, w_size=1.0, w_stride=1.0):
    return DynaSDBase(fs=fs, w_size=w_size, w_stride=w_stride)


def _expected_win_samples(w_size, fs):
    return int(round(w_size * fs))


def _expected_step_samples(w_stride, fs):
    return int(round(w_stride * fs))


def _seconds_to_idx(D, w_size, w_stride):
    """Spec section 6 closed-form: floor((D - w_size)/w_stride) + 1."""
    return int(np.floor((D - w_size) / w_stride)) + 1


def _make_sz_prob(n_windows, n_channels=3, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        rng.uniform(0, 1, size=(n_windows, n_channels)),
        columns=[f"ch{i}" for i in range(n_channels)],
    )


# ----------------------------------------------------------------------
# Section 2: canonical sample counts
# ----------------------------------------------------------------------

@pytest.mark.parametrize("fs,w_size,w_stride,exp_win,exp_step", [
    (512, 1.0, 1.0, 512, 512),
    (256, 2.0, 0.5, 512, 128),
    (1024, 0.5, 0.25, 512, 256),
    (500, 2.0, 1.0, 1000, 500),
])
def test_s2_integer_products_exact(fs, w_size, w_stride, exp_win, exp_step):
    """Integer products produce exact win_samples and step_samples (verified
    indirectly via moving_win_clips shape and get_win_times stride)."""
    x = np.arange(fs * 10, dtype=float)
    arr = moving_win_clips(x, fs, w_size, w_stride)
    assert arr.shape[1] == exp_win, "win_samples mismatch in moving_win_clips output width"

    base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
    t = base.get_win_times(len(x))
    if len(t) > 1:
        diffs = np.diff(t)
        np.testing.assert_allclose(diffs, exp_step / fs, rtol=0, atol=1e-12)


def test_s2_non_integer_product_emits_userwarning():
    """Non-integer w_stride*fs emits at least one UserWarning."""
    fs = 512
    w_stride = 1.0 / 3  # 170.666... samples
    x = np.arange(fs * 10, dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        moving_win_clips(x, fs, 1.0, w_stride)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) >= 1, (
        "Expected UserWarning for non-integer w_stride*fs"
    )
    msg = str(user_warnings[0].message).lower()
    assert "stride" in msg or "step" in msg, (
        "Warning message must name the affected parameter"
    )


def test_s2_integer_product_no_warning():
    """Integer products emit no UserWarning."""
    fs = 512
    x = np.arange(fs * 10, dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        moving_win_clips(x, fs, 1.0, 1.0)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0, (
        f"Expected no UserWarning for integer products, got: "
        f"{[str(w.message) for w in user_warnings]}"
    )


def test_s2_warning_upgradable_to_error():
    """warnings.filterwarnings('error') correctly upgrades UserWarning to exception."""
    fs = 512
    x = np.arange(fs * 10, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning):
            moving_win_clips(x, fs, 1.0, 1.0 / 3)


# ----------------------------------------------------------------------
# Section 3: window count
# ----------------------------------------------------------------------

@pytest.mark.parametrize("n_samples,fs,w_size,w_stride,expected", [
    (5120, 512, 1.0, 1.0, 10),     # 10 seconds, non-overlapping 1s windows
    (5120, 512, 1.0, 0.5, 19),     # finer stride, overlapping
    (5120, 512, 2.0, 1.0, 9),      # 2s window, overlapping
    (512,  512, 1.0, 1.0, 1),      # exact one-window
    (5121, 512, 1.0, 1.0, 10),     # one extra sample → partial discarded
])
def test_s3_window_count_closed_form(n_samples, fs, w_size, w_stride, expected):
    assert num_wins(n_samples, fs, w_size, w_stride) == expected


def test_s3_exact_one_window():
    """n_samples == win_samples → exactly 1 window, t == [0.0]."""
    fs, w_size, w_stride = 512, 1.0, 1.0
    n_samples = 512
    base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
    t = base.get_win_times(n_samples)
    assert len(t) == 1
    assert t[0] == 0.0
    arr = moving_win_clips(np.arange(n_samples, dtype=float), fs, w_size, w_stride)
    assert arr.shape == (1, 512)


def test_s3_partial_final_window_discarded():
    """Extra samples beyond an exact-window boundary do not produce an extra window."""
    fs, w_size, w_stride = 512, 1.0, 1.0
    base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
    # 10s + 0.2s extra
    assert len(base.get_win_times(fs * 10 + 100)) == 10
    assert num_wins(fs * 10 + 100, fs, w_size, w_stride) == 10


# ----------------------------------------------------------------------
# Section 4: window indexing (sample bounds)
# ----------------------------------------------------------------------

def test_s4_movingwinclips_row_matches_slice_integer_products():
    """Row i of moving_win_clips equals x[i*step:i*step+win] exactly."""
    fs, w_size, w_stride = 512, 1.0, 1.0
    x = np.arange(fs * 10, dtype=float)
    arr = moving_win_clips(x, fs, w_size, w_stride)
    step = _expected_step_samples(w_stride, fs)
    win = _expected_win_samples(w_size, fs)
    for i in range(arr.shape[0]):
        np.testing.assert_array_equal(arr[i], x[i * step : i * step + win])


def test_s4_window_length_constant_for_overlapping():
    """All rows of moving_win_clips output have length win_samples exactly."""
    fs = 512
    x = np.arange(fs * 10, dtype=float)
    arr = moving_win_clips(x, fs, 1.0, 0.5)  # overlapping
    assert all(len(row) == 512 for row in arr)


def test_s4_stride_between_starts_constant():
    """Stride between consecutive window start indices equals step_samples exactly."""
    fs, w_size, w_stride = 512, 1.0, 0.5
    x = np.arange(fs * 10, dtype=float)
    arr = moving_win_clips(x, fs, w_size, w_stride)
    step = _expected_step_samples(w_stride, fs)
    starts = arr[:, 0].astype(int)  # x is arange so first sample of row i is the start index
    np.testing.assert_array_equal(np.diff(starts), step)


def test_s4_movingwinclips_row_matches_slice_non_integer():
    """Even for non-integer products, rows are exact slices of the input
    using the rounded step_samples (no float drift across windows)."""
    fs, w_size, w_stride = 512, 1.0, 1.0 / 3
    x = np.arange(fs * 30, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arr = moving_win_clips(x, fs, w_size, w_stride)
    step = _expected_step_samples(w_stride, fs)
    win = _expected_win_samples(w_size, fs)
    for i in range(arr.shape[0]):
        np.testing.assert_array_equal(arr[i], x[i * step : i * step + win])


# ----------------------------------------------------------------------
# Section 5: window timestamp
# ----------------------------------------------------------------------

def test_s5_integer_product_timestamps_exact():
    """For integer products, t[i] == i * w_stride exactly."""
    fs, w_size, w_stride = 512, 1.0, 1.0
    base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
    t = base.get_win_times(fs * 10)
    np.testing.assert_array_equal(t, np.arange(10) * w_stride)


def test_s5_non_integer_per_step_drift_bound():
    """Per-window stride deviation from the requested stride is bounded by
    0.5/fs (spec section 5 — bound is per-step, not cumulative)."""
    fs = 512
    w_stride = 1.0 / 3
    base = _make_base(fs=fs, w_size=1.0, w_stride=w_stride)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t = base.get_win_times(fs * 30)
    realized_step = np.diff(t)
    assert np.all(np.abs(realized_step - w_stride) <= 0.5 / fs + 1e-12)


def test_s5_constant_diff():
    """np.diff(t) is constant."""
    fs = 512
    base = _make_base(fs=fs, w_size=1.0, w_stride=0.5)
    t = base.get_win_times(fs * 10)
    diffs = np.diff(t)
    np.testing.assert_allclose(diffs, diffs[0], rtol=0, atol=1e-12)


# ----------------------------------------------------------------------
# Section 6: seconds → window-count conversion
# ----------------------------------------------------------------------

@pytest.mark.parametrize("D,w_size,w_stride,expected", [
    (10.0, 1.0, 1.0, 10),
    (5.0,  1.0, 1.0, 5),
    (5.0,  2.0, 1.0, 4),    # rwin_size with overlapping windows
    (1.0,  1.0, 1.0, 1),    # D == w_size lower bound
    (10.0, 1.0, 0.5, 19),   # finer stride
    (4.0,  2.0, 1.0, 3),    # rwin_req with overlapping windows
])
def test_s6_seconds_to_idx_closed_form(D, w_size, w_stride, expected):
    assert _seconds_to_idx(D, w_size, w_stride) == expected


def test_s6_D_equals_w_size_gives_one():
    """D == w_size → idx == 1 (smallest valid value)."""
    assert _seconds_to_idx(1.0, 1.0, 1.0) == 1
    assert _seconds_to_idx(2.0, 2.0, 1.0) == 1


# ----------------------------------------------------------------------
# Section 7: smoothing pipeline
# ----------------------------------------------------------------------

def test_s7_smoothed_output_length_equals_input():
    """S2: uniform_filter1d preserves input length."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    sz_prob = _make_sz_prob(50)
    _, sz_clf_ff = base.get_onset_and_spread(
        sz_prob, threshold=0.5, ret_smooth_mat=True,
        filter_w=10.0, rwin_size=5.0, rwin_req=4.0,
    )
    assert len(sz_clf_ff) == len(sz_prob)


def test_s7_padded_rows_equal_last_valid():
    """S5: trailing padded rows equal the last valid convolution row."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 50
    data = np.zeros((n, 1))
    data[10:30, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    _, sz_clf_ff = base.get_onset_and_spread(
        sz_prob, threshold=0.5, ret_smooth_mat=True,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    rwin_size_idx = _seconds_to_idx(5.0, 1.0, 1.0)
    last_valid = sz_clf_ff.iloc[-rwin_size_idx]
    for offset in range(1, rwin_size_idx):
        np.testing.assert_array_equal(
            sz_clf_ff.iloc[-offset].values, last_valid.values
        )


def test_s7_idxmax_gives_first_true_row():
    """S6: onset is the first True row of sz_clf_ff (idxmax)."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 100
    data = np.zeros((n, 1))
    data[40:60, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs, sz_clf_ff = base.get_onset_and_spread(
        sz_prob, threshold=0.5, ret_smooth_mat=True,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    onset = sz_idxs["ch0"].iloc[0]
    expected = sz_clf_ff["ch0"].idxmax()
    assert onset == expected


def test_s7_non_seizing_channel_returns_nan_and_zero_column():
    """Non-seizing channels: NaN onset and zeroed sz_clf_ff column."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 100
    data = np.zeros((n, 2))
    data[40:60, 0] = 1.0  # only ch0 seizes
    sz_prob = pd.DataFrame(data, columns=["ch0", "ch1"])
    sz_idxs, sz_clf_ff = base.get_onset_and_spread(
        sz_prob, threshold=0.5, ret_smooth_mat=True,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    assert pd.isna(sz_idxs["ch1"].iloc[0])
    assert (sz_clf_ff["ch1"] == 0).all()


def test_s7_onset_seconds_equal_index_times_step_over_fs():
    """S7: when integer products, onset_time = onset_idx * w_stride exactly.
    (Spec only specifies the window-index return; absolute-second conversion
    is the consumer's responsibility, but we verify the relationship.)"""
    fs = 1
    base = _make_base(fs=fs, w_size=1.0, w_stride=1.0)
    n = 100
    data = np.zeros((n, 1))
    data[40:60, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    onset_idx = sz_idxs["ch0"].iloc[0]
    step_samples = _expected_step_samples(1.0, fs)
    assert onset_idx * step_samples / fs == onset_idx * 1.0


def test_s7_time_indexed_sz_prob_returns_onsets_in_seconds():
    """When ``sz_prob`` is fed in time-indexed (the standard Phase-F path
    produced by ``model.forward(X)``), ``idxmax`` returns onset times in
    seconds directly — no manual window-index → seconds conversion needed.

    Uses ``w_stride=2.0`` so that the time labels and positional indices
    are numerically distinct (positional 19 vs. time 38.0); without that
    distinction the test could be satisfied trivially by either contract.
    """
    base = _make_base(fs=1, w_size=1.0, w_stride=2.0)
    n_rows = 51
    data = np.zeros((n_rows, 1))
    k = 20  # positional row of sustained activity onset
    data[k : k + 10, 0] = 1.0

    # Choose n_samples so that num_wins(n_samples, fs, w_size, w_stride)
    # equals n_rows, then attach the matching time index.
    n_samples = (n_rows - 1) * 2 + 1  # = 101
    time_index = base.get_win_index(n_samples)
    sz_prob = pd.DataFrame(data, columns=["ch0"], index=time_index)

    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0,    # filter_w_idx == 1 (no smoothing)
        rwin_size=5.0,   # rwin_size_idx == 3
        rwin_req=4.0,    # rwin_req_idx == 2
    )
    # Spread shift is rwin_size_idx - rwin_req_idx = 1 window (= 2.0s).
    # Expected onset position: k - 1 = 19; in seconds: 19 * 2 = 38.0.
    onset = sz_idxs["ch0"].iloc[0]
    assert onset == 38.0


def test_s7_positional_sz_prob_returns_window_indices():
    """When ``sz_prob`` has the default :class:`RangeIndex` (legacy /
    manually constructed input), ``idxmax`` returns onset window indices
    as integers — backward-compatible with pre-Phase-F callers."""
    base = _make_base(fs=1, w_size=1.0, w_stride=2.0)
    n_rows = 51
    data = np.zeros((n_rows, 1))
    k = 20
    data[k : k + 10, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])  # default RangeIndex(0..50)

    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    onset = sz_idxs["ch0"].iloc[0]
    # Same fixture as the time-indexed test, but onset is reported in
    # window-index units instead: k - 1 = 19. The numerical distinction
    # from the time-indexed test (38.0 vs 19) is what makes both tests
    # meaningful — preserving the input-index convention is the contract.
    assert onset == 19


# ----------------------------------------------------------------------
# Section 7.4: bias from forward-looking convolution
# ----------------------------------------------------------------------

def test_s74_spread_shift_matches_documented_formula():
    """Contiguous activity at row k → reported onset ≈ k - (rwin_size_idx - rwin_req_idx).
    Smoothing disabled (filter_w == w_size) to isolate spread shift."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 100
    k = 50
    data = np.zeros((n, 1))
    data[k : k + 20, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0,    # filter_w_idx = 1, no smoothing
        rwin_size=5.0,
        rwin_req=4.0,
    )
    onset = sz_idxs["ch0"].iloc[0]
    expected = k - (_seconds_to_idx(5.0, 1.0, 1.0) - _seconds_to_idx(4.0, 1.0, 1.0))
    assert onset == expected, f"expected onset {expected}, got {onset}"


def test_s74_seizure_at_start_does_not_overflow():
    """A seizure starting at row 0 must not produce a negative onset index;
    the smoothing edge mode and end-padding must preserve correct semantics."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 100
    data = np.zeros((n, 1))
    data[0:30, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    onset = sz_idxs["ch0"].iloc[0]
    assert onset is not None and not pd.isna(onset)
    assert onset >= 0


def test_s74_late_seizure_outside_lookahead_not_flagged():
    """Activity confined to the last rwin_size_idx-1 windows cannot be
    flagged because no full lookahead window is available."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 50
    data = np.zeros((n, 1))
    data[n - 3 : n, 0] = 1.0  # only 3 windows of activity at the end
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    # rwin_req_idx = 4 but only 3 True rows exist → should not be flagged
    assert pd.isna(sz_idxs["ch0"].iloc[0])


# Bias formula under non-trivial smoothing. A centered ``uniform_filter1d``
# applied to a step from 0 to 1 does NOT shift the threshold-``0.5``
# crossing — by symmetry the smoothed value crosses ``0.5`` at the
# planted step itself for odd ``filter_w_idx``, and exactly one row
# later for even ``filter_w_idx`` (the strict ``>`` comparison breaks
# the half-and-half tie on the right side). The full bias for
# threshold-``0.5`` step inputs is:
#     total_shift_windows = filter_offset - (rwin_size_idx - rwin_req_idx)
#     where filter_offset = +1 if filter_w_idx even else 0
# These tests pin that formula across filter parities and a couple of
# spread configurations. Spec § 7.4 / R3 documents the same.

def _filter_offset(filter_w_idx):
    """Smoothing shift in windows for a threshold-0.5 step input.

    +1 for even ``filter_w_idx`` because the strict ``>`` comparison
    rules out the exact-half tie on the right side of the centered
    window; 0 otherwise.
    """
    return 1 if filter_w_idx % 2 == 0 else 0


@pytest.mark.parametrize("filter_w,rwin_size,rwin_req,planted_idx", [
    # filter_w_idx odd (no parity correction)
    (11.0, 5.0, 4.0, 50),
    (11.0, 5.0, 5.0, 50),  # rwin_req == rwin_size → no spread shift
    (11.0, 7.0, 4.0, 50),  # larger spread shift
    # filter_w_idx even (+1 parity correction)
    (10.0, 5.0, 4.0, 50),
    (10.0, 5.0, 5.0, 50),
    # filter_w_idx == 1 (no smoothing) — sanity baseline
    (1.0, 5.0, 4.0, 40),
])
def test_s74_combined_bias_step_threshold_0_5(filter_w, rwin_size, rwin_req, planted_idx):
    """For a clean step at row ``planted_idx`` and threshold = 0.5, the
    detected onset matches the corrected R3 formula exactly:

        detected = planted + filter_offset - (rwin_size_idx - rwin_req_idx)
    """
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 100
    data = np.zeros((n, 1))
    data[planted_idx:, 0] = 1.0  # step from 0 to 1 at planted_idx
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=filter_w, rwin_size=rwin_size, rwin_req=rwin_req,
    )
    filter_w_idx = _seconds_to_idx(filter_w, 1.0, 1.0)
    rwin_size_idx = _seconds_to_idx(rwin_size, 1.0, 1.0)
    rwin_req_idx = _seconds_to_idx(rwin_req, 1.0, 1.0)

    expected = planted_idx + _filter_offset(filter_w_idx) - (rwin_size_idx - rwin_req_idx)
    onset = sz_idxs["ch0"].iloc[0]
    assert onset == expected, (
        f"filter_w={filter_w} (idx={filter_w_idx}, parity_offset="
        f"{_filter_offset(filter_w_idx)}), rwin_size={rwin_size} "
        f"(idx={rwin_size_idx}), rwin_req={rwin_req} "
        f"(idx={rwin_req_idx}): expected onset {expected}, got {onset}"
    )


def test_s74_smoothing_step_at_midpoint_threshold_default_params_zero_net_shift():
    """Worked example from spec § 7.4: with default parameters
    (``filter_w=10s``, ``rwin_size=5s``, ``rwin_req=4s``) at threshold
    ``0.5``, a step at row ``k=50`` produces detected onset at row 50
    exactly — the +1 even-parity smoothing offset is canceled by the
    -1 spread shift."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 100
    k = 50
    data = np.zeros((n, 1))
    data[k:, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=10.0, rwin_size=5.0, rwin_req=4.0,
    )
    assert sz_idxs["ch0"].iloc[0] == k


# ----------------------------------------------------------------------
# Section 8: validation
# ----------------------------------------------------------------------

def test_s8_short_input_num_wins_raises():
    with pytest.raises(ValueError, match="(?i)shorter|window"):
        num_wins(3 * 512, 512, 10.0, 1.0)


def test_s8_short_input_movingwinclips_raises():
    with pytest.raises(ValueError, match="(?i)shorter|window"):
        moving_win_clips(np.arange(3 * 512, dtype=float), 512, 10.0, 1.0)


def test_s8_short_input_get_win_times_raises():
    base = _make_base(fs=512, w_size=10.0, w_stride=1.0)
    with pytest.raises(ValueError, match="(?i)shorter|window"):
        base.get_win_times(3 * 512)


@pytest.mark.parametrize("filter_w,rwin_size,rwin_req,bad_param", [
    (0.5, 5.0, 4.0, "filter_w"),     # filter_w < w_size
    (10.0, 0.5, 0.5, "rwin_size"),   # rwin_size < w_size
    (10.0, 5.0, 0.5, "rwin_req"),    # rwin_req < w_size
    (10.0, 4.0, 5.0, "rwin_req"),    # rwin_req > rwin_size
])
def test_s8_get_onset_validation_raises(filter_w, rwin_size, rwin_req, bad_param):
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    sz_prob = _make_sz_prob(50)
    with pytest.raises(ValueError, match=bad_param):
        base.get_onset_and_spread(
            sz_prob, threshold=0.5,
            filter_w=filter_w, rwin_size=rwin_size, rwin_req=rwin_req,
        )


# ----------------------------------------------------------------------
# Section 9: cross-function invariants
# ----------------------------------------------------------------------

@pytest.mark.parametrize("fs,w_size,w_stride", [
    (512, 1.0, 1.0),
    (256, 2.0, 1.0),       # overlapping
    (1024, 0.5, 0.25),
    (500, 1.0, 0.5),
])
def test_s9_num_wins_eq_get_win_times_length(fs, w_size, w_stride):
    n_samples = fs * 30
    base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
    assert num_wins(n_samples, fs, w_size, w_stride) == len(base.get_win_times(n_samples))


@pytest.mark.parametrize("fs,w_size,w_stride", [
    (512, 1.0, 1.0),
    (256, 2.0, 1.0),
    (1024, 0.5, 0.25),
])
def test_s9_movingwinclips_shape_matches_num_wins(fs, w_size, w_stride):
    n_samples = fs * 30
    x = np.arange(n_samples, dtype=float)
    expected_n_windows = num_wins(n_samples, fs, w_size, w_stride)
    expected_win_samples = _expected_win_samples(w_size, fs)
    arr = moving_win_clips(x, fs, w_size, w_stride)
    assert arr.shape == (expected_n_windows, expected_win_samples)


def test_s9_invariants_under_non_integer_products():
    """All invariants hold for non-integer w_stride*fs (post-warning)."""
    fs, w_size, w_stride = 512, 1.0, 1.0 / 3
    n_samples = fs * 30
    x = np.arange(n_samples, dtype=float)
    base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nw = num_wins(n_samples, fs, w_size, w_stride)
        arr = moving_win_clips(x, fs, w_size, w_stride)
        t = base.get_win_times(n_samples)
    assert nw == len(t)
    assert arr.shape == (nw, _expected_win_samples(w_size, fs))
    diffs = np.diff(t)
    np.testing.assert_allclose(diffs, diffs[0], rtol=0, atol=1e-12)


# ----------------------------------------------------------------------
# Section 10: edge cases
# ----------------------------------------------------------------------

def test_s10_filter_w_eq_w_size_is_no_op_smoothing():
    """filter_w == w_size → filter_w_idx == 1, smoothing is a no-op."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 100
    data = np.zeros((n, 1))
    data[20:40, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    # No smoothing → spread shift is rwin_size_idx - rwin_req_idx = 5-4 = 1
    expected = 20 - 1
    assert sz_idxs["ch0"].iloc[0] == expected


def test_s10_all_channels_non_seizing():
    """Quiet input → all NaN onsets, all-zero sz_clf_ff."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 50
    sz_prob = pd.DataFrame(np.zeros((n, 3)), columns=["ch0", "ch1", "ch2"])
    sz_idxs, sz_clf_ff = base.get_onset_and_spread(
        sz_prob, threshold=0.5, ret_smooth_mat=True,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    assert sz_idxs.iloc[0].isna().all()
    assert (sz_clf_ff.values == 0).all()


def test_s10_rwin_req_eq_rwin_size_strict_criterion():
    """rwin_req == rwin_size → every counted window must fire."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 100
    data = np.zeros((n, 1))
    # Exactly 5 contiguous fired windows (= rwin_size_idx); criterion satisfied
    data[30:35, 0] = 1.0
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0, rwin_size=5.0, rwin_req=5.0,
    )
    # rwin_req_idx == rwin_size_idx == 5; spread shift = 0; onset == 30
    assert sz_idxs["ch0"].iloc[0] == 30


def test_s10_isolated_above_threshold_not_flagged():
    """A single isolated above-threshold window does not flag a channel."""
    base = _make_base(fs=1, w_size=1.0, w_stride=1.0)
    n = 50
    data = np.zeros((n, 1))
    data[25, 0] = 1.0  # one window only
    sz_prob = pd.DataFrame(data, columns=["ch0"])
    sz_idxs = base.get_onset_and_spread(
        sz_prob, threshold=0.5,
        filter_w=1.0, rwin_size=5.0, rwin_req=4.0,
    )
    assert pd.isna(sz_idxs["ch0"].iloc[0])
