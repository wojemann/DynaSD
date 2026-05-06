"""Internals tests for ``DynaSD.NDDBase`` sequence-prep, aggregation, and cache."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))

from dynasd.NDDBase import NDDBase


def _make_base(fs=256, w_size=1.0, w_stride=0.5):
    """NDDBase with caching disabled and quiet output."""
    return NDDBase(fs=fs, w_size=w_size, w_stride=w_stride,
                   use_cuda=False, verbose=False)


def _ramp_data(n_samples, n_channels=2):
    """Per-channel monotonic ramp; sub-channel offset makes channel mixups visible."""
    base = np.arange(n_samples, dtype=np.float64)[:, None]
    offsets = np.arange(n_channels, dtype=np.float64)[None, :] * 1e-3
    cols = [f"ch{i}" for i in range(n_channels)]
    return pd.DataFrame(base + offsets, columns=cols)


# ----------------------------------------------------------------------
# _prepare_multistep_sequences_vectorized — shape & content
# ----------------------------------------------------------------------

class TestPrepareMultistepSequences:
    """Shape and content correctness for the sliding-window splitter."""

    def test_default_stride_is_forecast_length(self):
        """Omitting ``stride`` defaults it to ``forecast_length``."""
        base = _make_base()
        data = _ramp_data(n_samples=20, n_channels=2)
        S, F = 4, 2

        inp, tgt = base._prepare_multistep_sequences_vectorized(
            data, sequence_length=S, forecast_length=F
        )
        # n_sequences = (20 - 6) // 2 + 1 = 8
        assert inp.shape == (8, S, 2)
        assert tgt.shape == (8, F, 2)

    def test_custom_stride_changes_n_sequences(self):
        """``n_sequences = (N - (S+F)) // stride + 1`` for explicit stride."""
        base = _make_base()
        data = _ramp_data(n_samples=20, n_channels=2)
        S, F, stride = 4, 2, 3

        inp, tgt = base._prepare_multistep_sequences_vectorized(
            data, sequence_length=S, forecast_length=F, stride=stride
        )
        # n_sequences = (20 - 6) // 3 + 1 = 14 // 3 + 1 = 5
        assert inp.shape == (5, S, 2)
        assert tgt.shape == (5, F, 2)

    def test_exact_boundary_yields_single_sequence(self):
        """``N == sequence_length + forecast_length`` produces exactly one sequence."""
        base = _make_base()
        S, F = 4, 2
        data = _ramp_data(n_samples=S + F, n_channels=2)

        inp, tgt = base._prepare_multistep_sequences_vectorized(
            data, sequence_length=S, forecast_length=F
        )
        assert inp.shape == (1, S, 2)
        assert tgt.shape == (1, F, 2)

    def test_too_short_input_raises(self):
        """``N < sequence_length + forecast_length`` raises."""
        base = _make_base()
        S, F = 4, 2
        data = _ramp_data(n_samples=S + F - 1, n_channels=2)

        with pytest.raises(ValueError, match="Not enough data"):
            base._prepare_multistep_sequences_vectorized(
                data, sequence_length=S, forecast_length=F
            )

    def test_input_target_content_matches_slicing(self):
        """``input_data[i]`` and ``target_data[i]`` are exact slices of the input (float32)."""
        base = _make_base()
        N, S, F = 20, 4, 2
        data = _ramp_data(n_samples=N, n_channels=2)
        data_np = data.to_numpy().astype(np.float32)
        stride = F  # default

        inp, tgt = base._prepare_multistep_sequences_vectorized(
            data, sequence_length=S, forecast_length=F
        )
        assert inp.dtype == torch.float32
        assert tgt.dtype == torch.float32
        n_sequences = inp.shape[0]
        inp_np = inp.numpy()
        tgt_np = tgt.numpy()
        for i in range(n_sequences):
            seq_start = i * stride
            np.testing.assert_array_equal(inp_np[i], data_np[seq_start:seq_start + S])
            np.testing.assert_array_equal(tgt_np[i], data_np[seq_start + S:seq_start + S + F])

    def test_seq_positions_indices_match_stride_arithmetic(self):
        """``seq_positions[i]`` records correct sample indices and times in seconds."""
        fs = 256
        base = _make_base(fs=fs)
        N, S, F, stride = 20, 4, 2, 2
        data = _ramp_data(n_samples=N, n_channels=2)

        inp, tgt, positions = base._prepare_multistep_sequences_vectorized(
            data, sequence_length=S, forecast_length=F, ret_positions=True, stride=stride
        )
        assert len(positions) == inp.shape[0]
        for i, pos in enumerate(positions):
            seq_start = i * stride
            assert pos["seq_idx"] == i
            assert pos["input_start"] == seq_start
            assert pos["input_end"] == seq_start + S
            assert pos["target_start"] == seq_start + S
            assert pos["target_end"] == seq_start + S + F
            assert pos["seq_time_start"] == pytest.approx(seq_start / fs)
            assert pos["seq_time_end"] == pytest.approx((seq_start + S) / fs)
            assert pos["input_time_start"] == pytest.approx(seq_start / fs)
            assert pos["target_time_start"] == pytest.approx((seq_start + S) / fs)


# ----------------------------------------------------------------------
# _aggregate_sequences_to_windows_mse — closed-form aggregation
# ----------------------------------------------------------------------

class TestAggregateSequencesToWindowsMse:
    """Per-window aggregation: inclusive-start/strict-end overlap rule and sqrt(mean(mse))."""

    @staticmethod
    def _seq(start_t, end_t, mse_vec):
        """Build the minimal seq_results dict consumed by the aggregator."""
        return {
            "seq_start_time": float(start_t),
            "target_end_time": float(end_t),
            "mse": np.asarray(mse_vec, dtype=np.float64),
        }

    @staticmethod
    def _make_X(n_samples, n_channels):
        cols = [f"ch{i}" for i in range(n_channels)]
        return pd.DataFrame(np.zeros((n_samples, n_channels)), columns=cols)

    def test_window_grid_matches_get_win_times(self):
        """Aggregator's window grid is exactly ``get_win_times(N)``."""
        fs, w_size, w_stride = 10, 1.0, 0.5
        base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
        N = 30
        X = self._make_X(N, 2)

        # One sequence covering window 0 fully — irrelevant content.
        seq_results = [self._seq(0.0, 0.5, [1.0, 1.0])]
        out = base._aggregate_sequences_to_windows_mse(seq_results, X)

        expected_starts = base.get_win_times(N)
        assert out.shape == (len(expected_starts), 2)
        np.testing.assert_allclose(base.window_start_times, expected_starts)

    def test_overlap_rule_inclusive_start_strict_end(self):
        """Overlap rule: ``start >= window_start`` (inclusive), ``end < window_end`` (strict)."""
        fs, w_size, w_stride = 10, 1.0, 0.5
        base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
        X = self._make_X(30, 1)

        # Window 0 spans [0.0, 1.0). 'borderline' has start=0.0 (inclusive),
        # end=1.0 (excluded by strict <). It must contribute to NO window:
        # window 0 rejects on end, window 1 rejects on start (0.0 < 0.5).
        seq_results = [self._seq(0.0, 1.0, [99.0])]
        out = base._aggregate_sequences_to_windows_mse(seq_results, X)
        # All windows should be NaN — no sequence contributed anywhere.
        assert np.all(np.isnan(out.values))

        # In contrast, ``end=0.999`` is < 1.0 so this sequence DOES land
        # in window 0 (start 0.0 >= 0.0).
        seq_results = [self._seq(0.0, 0.999, [4.0])]
        out = base._aggregate_sequences_to_windows_mse(seq_results, X)
        assert out.iloc[0, 0] == pytest.approx(2.0)  # sqrt(mean([4.0]))

    def test_window_mse_is_sqrt_of_mean(self):
        """Per-window aggregate is ``sqrt(mean(seq_mse))`` over overlapping sequences."""
        fs, w_size, w_stride = 10, 1.0, 0.5
        base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
        X = self._make_X(30, 2)

        # Two sequences both inside window 0 only.
        seq_results = [
            self._seq(0.0, 0.4, [1.0, 9.0]),
            self._seq(0.1, 0.6, [3.0, 16.0]),
        ]
        # Window 0: [0.0, 1.0). Both seqs satisfy start >= 0 and end < 1.0.
        # Window 1: [0.5, 1.5). seq0 start 0.0 < 0.5 → out. seq1 start 0.1 < 0.5 → out.
        out = base._aggregate_sequences_to_windows_mse(seq_results, X)

        expected_w0 = np.sqrt(np.mean([[1.0, 9.0], [3.0, 16.0]], axis=0))
        np.testing.assert_allclose(out.iloc[0].values, expected_w0)
        # All other windows: no sequences → NaN.
        assert np.all(np.isnan(out.iloc[1:].values))

    def test_multi_window_assignment_with_known_layout(self):
        """Aggregation across multiple windows with hand-picked overlap layouts."""
        fs, w_size, w_stride = 10, 1.0, 0.5
        base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
        X = self._make_X(30, 2)
        # Window starts: [0.0, 0.5, 1.0, 1.5, 2.0]
        # Window ends:   [1.0, 1.5, 2.0, 2.5, 3.0]

        seq_results = [
            self._seq(0.0, 0.5, [1.0, 1.0]),   # win 0 only
            self._seq(0.6, 0.9, [2.0, 4.0]),   # win 0 and win 1
            self._seq(1.5, 1.9, [9.0, 1.0]),   # win 2 and win 3
            self._seq(2.5, 2.9, [16.0, 1.0]),  # win 4 only
        ]
        out = base._aggregate_sequences_to_windows_mse(seq_results, X)

        np.testing.assert_allclose(
            out.iloc[0].values, np.sqrt(np.mean([[1.0, 1.0], [2.0, 4.0]], axis=0))
        )
        np.testing.assert_allclose(out.iloc[1].values, np.sqrt([2.0, 4.0]))
        np.testing.assert_allclose(out.iloc[2].values, np.sqrt([9.0, 1.0]))
        np.testing.assert_allclose(out.iloc[3].values, np.sqrt([9.0, 1.0]))
        np.testing.assert_allclose(out.iloc[4].values, np.sqrt([16.0, 1.0]))

    def test_unassigned_window_is_nan(self):
        """Windows with no overlapping sequence emit NaN (not 0 or other sentinel)."""
        fs, w_size, w_stride = 10, 1.0, 0.5
        base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
        X = self._make_X(30, 1)

        seq_results = [self._seq(0.0, 0.5, [1.0])]  # only win 0
        out = base._aggregate_sequences_to_windows_mse(seq_results, X)
        assert out.iloc[0, 0] == pytest.approx(1.0)
        assert np.all(np.isnan(out.iloc[1:].values))

    def test_empty_seq_results_raises(self):
        """Empty seq_results raises."""
        base = _make_base()
        X = self._make_X(30, 1)
        with pytest.raises(ValueError, match="No sequences"):
            base._aggregate_sequences_to_windows_mse([], X)


# ----------------------------------------------------------------------
# Sequence cache
# ----------------------------------------------------------------------

class TestSequenceCache:
    """Content-addressed cache for prepared sequences (opt-in)."""

    def test_cache_disabled_by_default(self):
        """Caching is off by default."""
        base = _make_base()
        assert base._cache_enabled is False
        assert base._sequence_cache == {}

    def test_first_call_populates_cache_when_enabled(self):
        base = _make_base()
        base.enable_sequence_cache()
        data = _ramp_data(n_samples=20)

        base._prepare_multistep_sequences_vectorized(data, sequence_length=4, forecast_length=2)
        assert len(base._sequence_cache) == 1

    def test_repeat_call_returns_cached_object(self):
        """Cache hit returns the cached tensors verbatim (no recompute)."""
        base = _make_base()
        base.enable_sequence_cache()
        data = _ramp_data(n_samples=20)
        S, F = 4, 2

        inp1, tgt1 = base._prepare_multistep_sequences_vectorized(
            data, sequence_length=S, forecast_length=F
        )
        cache_key = next(iter(base._sequence_cache))
        sentinel_inp = torch.full_like(inp1, 99.0)
        sentinel_tgt = torch.full_like(tgt1, 88.0)
        base._sequence_cache[cache_key] = (sentinel_inp, sentinel_tgt, [])

        inp2, tgt2 = base._prepare_multistep_sequences_vectorized(
            data, sequence_length=S, forecast_length=F
        )
        torch.testing.assert_close(inp2, sentinel_inp)
        torch.testing.assert_close(tgt2, sentinel_tgt)

    def test_different_sequence_length_is_cache_miss(self):
        """``sequence_length`` is part of the cache key."""
        base = _make_base()
        base.enable_sequence_cache()
        data = _ramp_data(n_samples=20)

        base._prepare_multistep_sequences_vectorized(data, sequence_length=4, forecast_length=2)
        base._prepare_multistep_sequences_vectorized(data, sequence_length=6, forecast_length=2)
        assert len(base._sequence_cache) == 2

    def test_different_forecast_length_is_cache_miss(self):
        """``forecast_length`` is part of the cache key."""
        base = _make_base()
        base.enable_sequence_cache()
        data = _ramp_data(n_samples=20)

        base._prepare_multistep_sequences_vectorized(data, sequence_length=4, forecast_length=2)
        base._prepare_multistep_sequences_vectorized(data, sequence_length=4, forecast_length=3)
        assert len(base._sequence_cache) == 2

    def test_different_stride_is_cache_miss(self):
        """``stride`` is part of the cache key."""
        base = _make_base()
        base.enable_sequence_cache()
        data = _ramp_data(n_samples=20)

        base._prepare_multistep_sequences_vectorized(
            data, sequence_length=4, forecast_length=2, stride=2
        )
        base._prepare_multistep_sequences_vectorized(
            data, sequence_length=4, forecast_length=2, stride=3
        )
        assert len(base._sequence_cache) == 2

    def test_mutated_input_data_is_cache_miss(self):
        """Cache key includes a content fingerprint; new data is a miss."""
        base = _make_base()
        base.enable_sequence_cache()
        data1 = _ramp_data(n_samples=20)
        data2 = data1 + 1000.0

        base._prepare_multistep_sequences_vectorized(data1, sequence_length=4, forecast_length=2)
        base._prepare_multistep_sequences_vectorized(data2, sequence_length=4, forecast_length=2)
        assert len(base._sequence_cache) == 2

    def test_clear_sequence_cache_drops_entries(self):
        """``clear_sequence_cache`` empties entries but leaves caching enabled."""
        base = _make_base()
        base.enable_sequence_cache()
        data = _ramp_data(n_samples=20)
        base._prepare_multistep_sequences_vectorized(data, sequence_length=4, forecast_length=2)
        assert len(base._sequence_cache) == 1

        base.clear_sequence_cache()
        assert base._sequence_cache == {}
        assert base._cache_enabled is True

    def test_disable_sequence_cache_clears_and_disables(self):
        base = _make_base()
        base.enable_sequence_cache()
        data = _ramp_data(n_samples=20)
        base._prepare_multistep_sequences_vectorized(data, sequence_length=4, forecast_length=2)

        base.disable_sequence_cache()
        assert base._cache_enabled is False
        assert base._sequence_cache == {}
