"""
Internals tests for ``DynaSD.NDDBase``.

These pin down the implementation contract of the helpers that produce
per-window forecast-error features for every NDD-family detector
(``NDD``, ``GIN``, ``LiNDDA``):

- :meth:`NDDBase._prepare_multistep_sequences_vectorized` — sliding-window
  splitter that produces ``(input, target, positions)`` from a
  ``DataFrame``;
- :meth:`NDDBase._aggregate_sequences_to_windows_mse` — vectorized
  aggregator that maps per-sequence MSE arrays into the canonical
  ``(num_wins, n_channels)`` grid;
- the content-hash sequence cache that backs the prepare-sequences
  helper.

Detection-quality tests live in the deferred end-to-end suite
(``docs/testing_strategy.md`` § 6); the per-model API contract lives in
``tests/test_model_api_contract.py``.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))

from DynaSD.NDDBase import NDDBase


def _make_base(fs=256, w_size=1.0, w_stride=0.5):
    """NDDBase with caching disabled and quiet output. The class is
    abstract for ``fit`` / ``forward`` but its sequence-prep and
    aggregation helpers are concrete and can be exercised directly."""
    return NDDBase(fs=fs, w_size=w_size, w_stride=w_stride,
                   use_cuda=False, verbose=False)


def _ramp_data(n_samples, n_channels=2):
    """Per-channel monotonic ramp ``data[i, c] = i * 1.0 + c * 0.001``.
    The sub-channel offset makes channel mixups detectable."""
    base = np.arange(n_samples, dtype=np.float64)[:, None]
    offsets = np.arange(n_channels, dtype=np.float64)[None, :] * 1e-3
    cols = [f"ch{i}" for i in range(n_channels)]
    return pd.DataFrame(base + offsets, columns=cols)


# ----------------------------------------------------------------------
# _prepare_multistep_sequences_vectorized — shape & content
# ----------------------------------------------------------------------

class TestPrepareMultistepSequences:
    """Shape arithmetic and content correctness for the sliding-window
    splitter. Cache-disabled for these tests so we exercise the canonical
    code path."""

    def test_default_stride_is_forecast_length(self):
        """When ``stride`` is omitted it defaults to ``forecast_length``."""
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
        """``N == sequence_length + forecast_length`` is the smallest valid
        case and produces exactly one sequence."""
        base = _make_base()
        S, F = 4, 2
        data = _ramp_data(n_samples=S + F, n_channels=2)

        inp, tgt = base._prepare_multistep_sequences_vectorized(
            data, sequence_length=S, forecast_length=F
        )
        assert inp.shape == (1, S, 2)
        assert tgt.shape == (1, F, 2)

    def test_too_short_input_raises(self):
        """``N < sequence_length + forecast_length`` produces zero
        sequences, which the helper must reject loudly."""
        base = _make_base()
        S, F = 4, 2
        data = _ramp_data(n_samples=S + F - 1, n_channels=2)

        with pytest.raises(ValueError, match="Not enough data"):
            base._prepare_multistep_sequences_vectorized(
                data, sequence_length=S, forecast_length=F
            )

    def test_input_target_content_matches_slicing(self):
        """For ramp data, ``input_data[i]`` must equal
        ``data[i*stride : i*stride+S, :]`` and ``target_data[i]`` must
        equal ``data[i*stride+S : i*stride+S+F, :]``. The helper casts
        to ``torch.FloatTensor`` (float32) on the way out, so the
        reference must also be in float32 for an exact comparison —
        this pins both the content and the dtype convention."""
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
        """``seq_positions[i]`` records ``input_start = i*stride``,
        ``input_end = target_start = i*stride + S``,
        ``target_end = i*stride + S + F``, and the corresponding times
        in seconds (``index / fs``)."""
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
    """Hand-computed expected outputs on synthetic ``seq_results`` dicts.
    Pins the ``seq_starts >= window_starts`` and strict
    ``seq_ends < window_ends`` overlap rule at NDDBase.py:676-677, and
    the ``sqrt(mean(seq_mse))`` aggregation at NDDBase.py:685."""

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
        """The aggregator's window grid is exactly ``get_win_times(N)`` —
        not a recomputed grid (Phase F.1 guarantee). Output row count
        matches ``num_wins`` regardless of where sequences land."""
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
        """A sequence with ``start == window_start`` is included; a
        sequence with ``end == window_end`` is NOT (strict ``<``).
        Pins the asymmetric overlap criterion."""
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
        """Per window: aggregate is ``sqrt(mean(seq_mse))`` over all
        sequences whose interval falls inside the window."""
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
        """End-to-end aggregation across multiple windows. Hand-picked
        sequences whose target intervals overlap explicit window subsets
        let us pin the (window_idx → sequence_indices) mapping."""
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
        """Windows that no sequence overlaps emit NaN, not 0 or any
        sentinel (preserves "no evidence" semantics for downstream)."""
        fs, w_size, w_stride = 10, 1.0, 0.5
        base = _make_base(fs=fs, w_size=w_size, w_stride=w_stride)
        X = self._make_X(30, 1)

        seq_results = [self._seq(0.0, 0.5, [1.0])]  # only win 0
        out = base._aggregate_sequences_to_windows_mse(seq_results, X)
        assert out.iloc[0, 0] == pytest.approx(1.0)
        assert np.all(np.isnan(out.iloc[1:].values))

    def test_empty_seq_results_raises(self):
        """No sequences at all is not a meaningful aggregation request and
        must raise."""
        base = _make_base()
        X = self._make_X(30, 1)
        with pytest.raises(ValueError, match="No sequences"):
            base._aggregate_sequences_to_windows_mse([], X)


# ----------------------------------------------------------------------
# Sequence cache
# ----------------------------------------------------------------------

class TestSequenceCache:
    """Content-addressed cache backing the prepare-sequences helper. The
    cache is opt-in (``enable_sequence_cache``); shape-bearing parameters
    (``sequence_length``, ``forecast_length``, ``stride``) and a
    statistical fingerprint of the data are folded into the cache key."""

    def test_cache_disabled_by_default(self):
        """Constructor must leave caching off; otherwise repeated training
        sessions could silently reuse stale features."""
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
        """Hitting the cache must return the cached tensors verbatim,
        not recompute. Verified by mutating the cached entry to a
        sentinel and watching the next call return the sentinel."""
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
        """``sequence_length`` is part of the cache key; changing it
        forces recomputation under a separate entry."""
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
        """Explicit ``stride`` is part of the cache key (default
        ``stride=forecast_length`` is folded in too)."""
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
        """The cache key incorporates a content fingerprint of the data;
        meaningfully changing the data forces recomputation."""
        base = _make_base()
        base.enable_sequence_cache()
        data1 = _ramp_data(n_samples=20)
        # Different content with same shape: large constant offset
        # changes the statistical fingerprint reliably.
        data2 = data1 + 1000.0

        base._prepare_multistep_sequences_vectorized(data1, sequence_length=4, forecast_length=2)
        base._prepare_multistep_sequences_vectorized(data2, sequence_length=4, forecast_length=2)
        assert len(base._sequence_cache) == 2

    def test_clear_sequence_cache_drops_entries(self):
        """``clear_sequence_cache`` empties the cache but leaves caching
        enabled (vs ``disable_sequence_cache``)."""
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
