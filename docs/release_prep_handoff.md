# Release Prep Handoff

Date: 2026-05-06 (updated after Phases G/H, full spec coverage, Phase I.1-I.3 spread/onset spec tests)
Active branch: `release-prep/cleanup-packaging`
Original wo_dev tip preserved at git tag: `submission` (commit `897c3d5`)

## How to resume on a new machine

```bash
git fetch --all --tags
git checkout release-prep/cleanup-packaging
git log --oneline submission..HEAD
git status
```

If branch is missing locally:
```bash
git checkout -b release-prep/cleanup-packaging origin/release-prep/cleanup-packaging
```

Test environment: `stim-env` conda env (has pytest, neurodsp, torch,
matplotlib, seaborn). The package's own dev install (`pip install -e ".[test]"`
in `dynasd_env`) was set up but not used end-to-end.

```bash
conda run -n stim-env python -m pytest
# Expected: 204 passed
```

The `stim-env` conda environment also needs `neurodsp>=2.2` (lives in
the package's `[test]` extras but `stim-env` predates the dev install,
so install manually if missing: `conda run -n stim-env pip install
"neurodsp>=2.2"`).

ONCET and WVNT contract / spec tests need pretrained checkpoints; the
defaults point at the developer's local layout
(`/Users/wojemann/local_data/dynasd_data/PROCESSED_DATA/CHECKPOINTS/...`)
and tests skip cleanly when missing. Override with environment variables
`DYNASD_ONCET_CHECKPOINT`, `DYNASD_ONCET_CONFIG`, `DYNASD_WVNT_CHECKPOINT`.

## What's been done since the `submission` tag

17 commits on `release-prep/cleanup-packaging`. From oldest to newest:

| Hash | Phase | Summary |
|---|---|---|
| `87bba10` | (prior) | Original release-prep scaffolding (archive directory, draft pyproject) |
| `9aa2e30` | (prior) | Initial release-prep docs (testing strategy, extending models, this doc) |
| `892f514` | C | Lock down windowing/smoothing spec; refactor `DynaSD/utils.py` into the new `DynaSD/tools/` subpackage; introduce `DynaSDBase.get_onset_and_spread` validation; rename `MovingWinClips → moving_win_clips`; add 55-test windowing/smoothing spec suite |
| `e7d79bf` | A | Move 7 dead model files to `archive/legacy-models/` (~2400 LOC archived) |
| `d3b1361` | B | NDDBase kwarg validation; refresh `tests/test_ndd.py` to current API |
| `b58b1d4` | D | Document unified `forward(X)` contract on `DynaSDBase`; fix NDD test |
| `67544e6` | E | Per-model API contract test + ABSSLP math-property tests; HFER `is_fitted` consistency fix |
| `f42b26b` | E+ | Extend contract coverage to all NN models in `__all__`; standardize IMPRINT `is_fitted`; document Phase F decision in `docs/phase_f_time_index_api.md`; add neurodsp/matplotlib/seaborn to `[test]` extras |
| `8cd0f69` | E++ | GIN and LiNDDA reject unsupported `sequence_length` with ValueError |
| `ba1efea` | A+ | Archive LiRNDDA and MINDD (not in published paper); drop from `__all__` |
| `806189e` | F | Time-indexed DataFrame outputs across all detectors; `forward(X)` row index is realized window-start times in seconds named `t_sec` |
| `a2dc277` | F.1 | Align NDDBase aggregation grid with `get_win_times`; remove silent length-mismatch guards |
| `6e41b8a` | F.3 | Pin time-vs-positional index contract for `get_onset_and_spread` |
| `ed5338c` | F.4 | Remove IMPRINT input cropping; align IMPRINT's frame of reference with the rest of the package |
| `a3f16cf` | G | NDDBase internal-helper tests (21): `_prepare_multistep_sequences_vectorized` shape/content/positions, `_aggregate_sequences_to_windows_mse` overlap rule and aggregation, sequence cache behavior. Pure test additions. |
| `69857d0` | H | Extend contract suite to ONCET and WVNT (now covers all 8 detectors in `__all__`). Caught and fixed three production bugs: missing `is_fitted=True` in both `fit()` methods; float-vs-int `win_len_idx` blowing up `np.zeros` when `w_size=1.0`; ONCET running with active dropout because `model.eval()` was never called. |
| `578ae1d` | (spec) | Spec tests for HFER, IMPRINT, NN forecasters (NDD/GIN/LiNDDA), pretrained classifiers (ONCET/WVNT). Replaces stale `test_gin.py` and `test_wavenet.py` relics. 22 new tests; spec coverage now exists for every detector in `__all__`. |
| `dbee9a5` | (Tier 1 cleanup) | IMPRINT.fit() returns None like every other model; dedupe `plot_iEEG_data` (delete `examples/example_utils.py`, update notebooks to import `DynaSD.tools.viz.plot_ieeg_data`) |
| `a9eecbb` | I.1 | Pin `get_onset_and_spread` bias formula on synthetic step inputs (7 new parametrized cases). Empirical probing revealed spec § 7.4 / R3 had the wrong formula — claimed smoothing shift ≈ `filter_w_idx/2`, but a centered moving average around a step doesn't move the threshold-midpoint crossing. Corrected formula: `total_shift_windows = filter_offset - (rwin_size_idx - rwin_req_idx)` where `filter_offset = +1 if filter_w_idx even else 0`. Spec § 7.4 and R3 worked example rewritten to match. |
| `ce7bdc2` | I.2 | Multi-channel `get_onset_and_spread` test (7 parametrized cases, including all-seizing and all-quiet edge cases) — pins per-channel independence with planted vs unplanted channels. Production fix: `IMPRINT.get_onset_and_spread` had an unconditional threshold-override at line 276 on the default `legacy=False` path that silently replaced any caller-supplied threshold with the hardcoded pretrained value (7.82 / 13.80). Wrapped in `if threshold is None` so explicit thresholds are now respected. **Implication for prior work**: any `IMPRINT(...).get_onset_and_spread(sz_prob, threshold=X, legacy=False)` call with an explicit `X` was silently using the pretrained value instead — threshold-sweep / per-subject-tuned-threshold experiments through this path collapsed to the pretrained constant. |
| `019a626` | I.3 | `get_onset_and_spread` on realistic post-thresholded `sz_clf` patterns (4 tests): sparse drop-outs that still meet `rwin_req`, dense drop-outs that fail `rwin_req`, brief pre-onset flicker that doesn't trigger early detection, cascading onset times across channels. Tests with `filter_w = w_size` to isolate the spread step's `rwin_req` sliding-sum logic from smoothing. |

## Current package surface

`DynaSD/__init__.py` exports:
```python
__all__ = ["NDD", "DynaSDBase", "NDDBase", "ABSSLP", "WVNT", "GIN",
           "LiNDDA", "IMPRINT", "HFER", "ONCET"]
```

Layout:
```
DynaSD/
  __init__.py
  base.py                        # DynaSDBase: shared windowing, smoothing, onset, threshold logic
  utils.py                       # core windowing helpers (canonical sample counts, num_wins, moving_win_clips)
  NDDBase.py                     # NN-detector shared training/inference base
  ABSSLP.py, HFER.py, IMPRINT.py # classical detectors (DynaSDBase subclasses)
  ONCET.py, WAVENET.py           # torch-classical hybrids (need pretrained checkpoints)
  NDD.py, GIN.py, LiNDDA.py      # NN forecasters (NDDBase subclasses)
  tools/                         # researcher tooling (separate from runtime utils)
    io.py                        # iEEG.org loading, config, label cleaning
    preprocessing.py             # bad-channel detect, montage, filter, AR(1), pipeline
    viz.py                       # multi-channel plotting (matplotlib/seaborn)
    stats.py                     # cohens_d
archive/
  legacy-models/                 # 9 archived files: GIN_old, NDD_old, NDD_fixed, LiRNDDA_backup, LiRNDDA, MINDD, ONDD, absolute_slope, models.py
  legacy-tests/                  # earlier exploratory tests
  notes/                         # earlier optimization notes
docs/
  spec_windowing_smoothing.md    # locked spec; 11 sections + R1-R6 appendix (R3 corrected in Phase I.1)
  phase_f_time_index_api.md      # design doc for the time-indexed switch (now implemented)
  testing_strategy.md            # broader testing direction; § 6 was the deferred e2e suite (largely retired by Phase I)
  extending_models.md
  release_prep_handoff.md        # this file
tests/
  test_*.py                      # 10 test files, 204 tests passing
```

## Locked contracts

`docs/spec_windowing_smoothing.md` is the source of truth for windowing,
smoothing, onset detection, and the time-indexed inference DataFrame
contract. It records six resolved design decisions:

- **R1.** Spread convolution pads at the END with the last valid row.
  Reported onset is shifted earlier than first raw threshold crossing
  by `(rwin_size_idx - rwin_req_idx)` windows; this is intentional
  forward-looking semantics.
- **R2.** Window timestamps are window START times. Acausal but
  matches the package's "earliest-detection" intent.
- **R3.** Full window-index → seconds chain. Bias of detected onset
  vs the first raw threshold crossing for a step input at threshold
  `0.5`:
  ```
  total_shift_windows = filter_offset - (rwin_size_idx - rwin_req_idx)
                        filter_offset = +1 if filter_w_idx even else 0
  total_shift_sec     = total_shift_windows * w_stride
  ```
  A centered moving average does not move the threshold-midpoint
  crossing; the parity offset is +1 for even `filter_w_idx` due to
  the strict `>` test (Phase I.1 corrected this; the prior
  `filter_w_idx/2` claim was incorrect).
- **R4.** Input shorter than one window raises `ValueError` (loud
  failure across all three windowing entry points).
- **R5.** Sample-count-first canonical math; non-integer `w_size*fs` /
  `w_stride*fs` products emit `UserWarning` once per affected parameter
  per call site.
- **R6.** Seconds → window-count uses start-and-end containment:
  `floor((D - w_size) / w_stride) + 1`. Validated with
  `D >= w_size` and `rwin_req <= rwin_size`.

Every detector in `__all__` satisfies the unified inference contract:
- `(fs, w_size, w_stride)` accepted at construction
- `forward(X)` raises if not fit; `fit(X)` sets `is_fitted = True` and returns `None`
- `forward(X) → pd.DataFrame` of shape `(num_wins(len(X), fs, w_size, w_stride), n_channels)`
- columns are `X.columns`; index is `get_win_index(len(X))` named `"t_sec"`
- `model(X)` is exactly `model.forward(X)` via `DynaSDBase.__call__`

## Test suite

204 deterministic tests passing across 10 files:

| File | Tests | Coverage |
|---|---|---|
| `test_model_api_contract.py` | 76 | 8 contract checks × 8 detectors + NDDBase kwarg validation + sequence_length validation + 4 length-agreement regression tests |
| `test_windowing_smoothing_spec.py` | 75 | Spec § 2-9 + § 7.4 onset/spread (21 tests, including bias formula, multi-channel independence, and realistic post-thresholded patterns) |
| `test_nddbase_internals.py` | 21 | Phase G: `_prepare_multistep_sequences_vectorized` shape/content/positions; `_aggregate_sequences_to_windows_mse` overlap and aggregation; sequence cache |
| `test_nn_forecasters_spec.py` | 9 | NDD/GIN/LiNDDA: non-negativity, determinism, anomaly response (3 properties × 3 models) |
| `test_pretrained_classifiers_spec.py` | 6 | ONCET/WVNT: bounded output, determinism, output not constant (3 × 2) — skips cleanly when checkpoints absent |
| `test_hfer_spec.py` | 4 | Non-negativity, amplitude scale invariance, high-band > low-band, determinism |
| `test_absslp_spec.py` | 4 | Non-negativity, amplitude monotonicity, zero-features on constant signal, determinism |
| `test_imprint_spec.py` | 3 | Finiteness on noise, anomaly response, determinism |
| `test_absslp.py` | 3 | Pre-existing ABSSLP integration test |
| `test_ndd.py` | 3 | NDD integration test (current API) |

Stale relic files removed in this branch: `tests/test_gin.py` (wrote
against an old GIN constructor and asserted a 6-column feature
DataFrame that current GIN does not produce) and `tests/test_wavenet.py`
(imported a non-existent `DynaSD.WAVENET.load_wavenet_model`). Both are
now superseded by the contract suite + per-model spec tests.

## Notable production fixes from this round

Surfacing these because they affect interpretation of any prior work
that used these code paths:

- **IMPRINT threshold override (commit `ce7bdc2`).**
  `IMPRINT.get_onset_and_spread(..., threshold=X, legacy=False)` (the
  default modern path) silently replaced an explicit `X` with the
  hardcoded pretrained value (7.82 for `threshold_agg='median'`,
  13.80 for `'mean'`). Any prior threshold-sweep / per-subject-tuned
  call through this path collapsed onto the pretrained constant.
  Recommended audit: `grep -rn "IMPRINT.*get_onset_and_spread" <publication-repo>`
  on any caller passing `threshold=` explicitly.

- **ONCET inference dropout (commit `69857d0`).** Pre-fix, ONCET's
  inner `LightweightSeizureDetector` ran in train mode during
  inference (no `model.eval()` call after load), so `Dropout` layers
  were active. Forward outputs were non-deterministic and slightly
  degraded. Now calls `model.eval()` once at construction.

- **ONCET / WVNT `is_fitted` (commit `69857d0`).** Both detectors'
  `fit()` methods didn't set `self.is_fitted = True`, breaking the
  unified contract documented in `DynaSDBase.forward`. Both now
  conform.

- **ONCET / WVNT float window-length bug (commit `69857d0`).**
  `_prepare_*_segment` computed `win_len_idx = w_size * fs` as a raw
  product, which is a float when `w_size=1.0` and crashed
  `np.zeros(...)` with `TypeError`. Both now route through
  `utils._canonical_sample_counts` so the spec R5 sample-count math
  applies uniformly.

- **Spec § 7.4 / R3 formula (commit `a9eecbb`).** Spec doc claimed
  smoothing shift ≈ `filter_w_idx / 2`. Empirical probing showed the
  shift at threshold `0.5` is actually 0 for odd `filter_w_idx` and
  +1 for even (parity rule). Spec doc and worked example rewritten
  to match the corrected formula. Implementation was always correct;
  the docs were misleading.

## What's next

### Tier 1 — small, contained, high-value

All Tier 1 items from the prior handoff are complete. Open items:

1. **Update `docs/extending_models.md`** to reflect Phase G (NDDBase
   internals are now testable as documented helpers) and the Phase H
   contract surface (every detector must set `is_fitted=True` in
   `fit()` and run inference in eval-equivalent mode).

### Tier 2 — bigger but still scoped

2. **Investigate spec doc § 5 / R5 edge cases** as needed when adding
   new detectors. The `_canonical_sample_counts` plumbing applies to
   every detector that windows the input; ONCET and WVNT were
   retroactively fixed in Phase H, but new detectors should use it
   from the start.

### Tier 3 — bigger commitment (mostly retired)

3. **End-to-end simulated-signal suite** as originally documented in
   `docs/testing_strategy.md` § 6 turned out to bottom out in
   per-detector feature-noise scatter and threshold-tuning that
   isn't really about the smoothing/onset chain. The actual concern
   ("does filtering shift the seizure onset time?") is now pinned by
   the 21-test § 7.4 block in `test_windowing_smoothing_spec.py` —
   single-channel bias formula, multi-channel independence, sparse
   drop-outs, dense drop-outs, brief flickers, channel cascades —
   all on synthetic binary inputs that bypass detector noise. The
   per-detector spec tests then bound each detector's mathematical
   behavior on its own.

   A future detector-driven planted-seizure suite (deferred) would
   sit on top of these and verify, per-detector, that real polyspike
   morphologies cross threshold cleanly and produce onsets within ~1
   stride of the planted location. The existing diagnostic findings
   (closed-form detectors cluster within ±1 stride of R3; LiNDDA and
   WVNT need per-detector threshold tuning to suppress unplanted
   false positives) are not committed but documented in this handoff
   for whoever picks it up.

### Tier 4 — release infrastructure

4. CI matrix (GitHub Actions, py3.10/3.11/3.12, core + extras).
5. README rewrite with a quickstart per detector class showing the
   time-indexed `forward(X)` workflow.
6. CHANGELOG, LICENSE review, version pinning policy.
7. TestPyPI release; collect feedback from 1-2 outside users; real
   PyPI release.

## Working-style notes for the next session

- The user prefers stepping through phases one at a time and approving
  each direction before code is written. Specs and design decisions
  are taken seriously and documented before tests are written; tests
  are written before code is changed where possible.
- `submission` tag is the recovery checkpoint; deletion of files is
  preferred via `git mv` to `archive/legacy-models/` (matches the
  pre-existing convention in `archive/`).
- Commits are kept focused — one phase or one logical change per
  commit, with substantive commit messages explaining rationale.
- The user maintains a `claude.md` scratch file in the working tree
  that is not committed; do not stage it.
- Tests are run via `conda run -n stim-env python -m pytest ...`.
