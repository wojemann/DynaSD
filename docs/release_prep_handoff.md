# Release Prep Handoff

Date: 2026-05-05 (updated after Phase F audit follow-ups)
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
conda run -n stim-env python -m pytest tests/test_windowing_smoothing_spec.py \
                                       tests/test_model_api_contract.py \
                                       tests/test_absslp_spec.py \
                                       tests/test_absslp.py \
                                       tests/test_ndd.py
# Expected: 127 passed
```

## What's been done since the `submission` tag

10 commits on `release-prep/cleanup-packaging`. From oldest to newest:

| Hash | Phase | Summary |
|---|---|---|
| `87bba10` | (prior) | Original release-prep scaffolding (archive directory, draft pyproject) |
| `9aa2e30` | (prior) | Initial release-prep docs (testing strategy, extending models, this doc) |
| `892f514` | C | Lock down windowing/smoothing spec; refactor `DynaSD/utils.py` into the new `DynaSD/tools/` subpackage; introduce `DynaSDBase.get_onset_and_spread` validation; rename `MovingWinClips → moving_win_clips`; add 55-test windowing/smoothing spec suite |
| `e7d79bf` | A | Move 7 dead model files to `archive/legacy-models/` (~2400 LOC archived) |
| `d3b1361` | B | NDDBase kwarg validation; refresh `tests/test_ndd.py` to current API |
| `b58b1d4` | D | Document unified `forward(X)` contract on `DynaSDBase`; fix NDD test (was the result of a Phase B mistake) |
| `67544e6` | E | Per-model API contract test + ABSSLP math-property tests; HFER `is_fitted` consistency fix |
| `f42b26b` | E+ | Extend contract coverage to all NN models in `__all__`; standardize IMPRINT `is_fitted`; document Phase F decision in `docs/phase_f_time_index_api.md`; add neurodsp/matplotlib/seaborn to `[test]` extras |
| `8cd0f69` | E++ | GIN and LiNDDA reject unsupported `sequence_length` with ValueError naming the supported set (instead of bare KeyError mid-construction) |
| `ba1efea` | A+ | Archive LiRNDDA and MINDD (not in published paper, no decision boundaries); drop from `__all__` |
| `806189e` | F | Time-indexed DataFrame outputs across all detectors; `forward(X)` row index is now realized window-start times in seconds named `t_sec`; `get_onset_and_spread` preserves input index through pipeline so onset times come out in seconds directly |
| `a2dc277` | F.1 | Audit follow-up: align NDDBase `_aggregate_sequences_to_windows_mse` window grid with `get_win_times`; remove silent length-mismatch guards in `forward()`. Drops at most `(w_size - w_stride)` seconds of trailing sequences for between-boundary inputs; no clinical impact. |
| `6e41b8a` | F.3 | Audit follow-up: pin down the index-preservation contract for `get_onset_and_spread` with two paired tests (time-indexed vs positional `sz_prob`) using `w_stride=2.0` so the two contracts give numerically distinct values (38.0 vs 19). |
| `ed5338c` | F.4 | Audit follow-up: remove IMPRINT input cropping; drop `onset_buffer` / `offset_buffer` / `ictal_buffer` parameters from `__init__`. Aligns IMPRINT's time-index frame of reference with every other detector. No behavior change under default params (the crop was already a no-op). |

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
  spec_windowing_smoothing.md    # locked spec; 11 sections + R1-R6 appendix
  phase_f_time_index_api.md      # design doc for the time-indexed switch (now implemented)
  testing_strategy.md            # broader testing direction; § 6 = deferred end-to-end suite
  extending_models.md
  release_prep_handoff.md        # this file
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
- **R3.** Full window-index → seconds chain documented; total bias
  formula `(filter_w_idx/2 + (rwin_size_idx - rwin_req_idx)) * w_stride`.
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
- `forward(X)` raises if not fit; `fit(X)` sets `is_fitted = True`
- `forward(X) → pd.DataFrame` of shape `(num_wins(len(X), fs, w_size, w_stride), n_channels)`
- columns are `X.columns`; index is `get_win_index(len(X))` named `"t_sec"`
- `model(X)` is exactly `model.forward(X)` via `DynaSDBase.__call__`

## Test suite

127 deterministic tests passing across:
- `tests/test_windowing_smoothing_spec.py` (57) — windowing, smoothing,
  onset detection math validated against the spec, including the
  Phase F.3 paired tests for time-indexed vs. positional `sz_prob`.
- `tests/test_model_api_contract.py` (66) — 8 contract checks × 6 models
  (ABSSLP, HFER, IMPRINT, NDD, GIN, LiNDDA) + 6 NDDBase kwarg-validation
  tests + 2 unsupported-sequence_length ValueError tests + 4 Phase F.1
  regression tests probing length agreement at between-boundary inputs.
- `tests/test_absslp_spec.py` (4) — ABSSLP non-negativity, amplitude
  monotonicity, determinism, zero-features for level-aligned inputs.
- `tests/test_absslp.py` (3) — pre-existing ABSSLP integration test.
- `tests/test_ndd.py` (3) — NDD integration test (refreshed to current API).

Pre-existing failures not addressed:
- `tests/test_gin.py` may now import after `neurodsp` was added to `[test]`
  extras — needs verification. Whether its assertions actually pass
  against current GIN is unknown.
- `tests/test_wavenet.py` (3 tests) skip because `WAVENET.load_wavenet_model`
  doesn't exist; needs investigation or test refresh.

## What's next

### Tier 1 — small, contained, high-value

1. **IMPRINT `fit()` returns `self`** while every other model returns `None`.
   1-line consistency fix.
2. **Audit `tests/test_gin.py`.** Now that `neurodsp` is in `[test]` deps,
   verify the test imports cleanly and check whether assertions pass
   against current GIN. Either fix or refresh.
3. **`examples/example_utils.py` has a duplicate of `plot_ieeg_data`** —
   delete and import from `DynaSD.tools.viz`.
4. **HFER + IMPRINT math-property tests** mirroring the ABSSLP pattern.
   Each detector should have a small file pinning its mathematical
   contract.

### Tier 2 — bigger but still scoped

5. **Phase G — NDDBase deeper unit tests.** Currently NDDBase is only
   covered by (a) kwarg-validation tests and (b) the F.1 length-agreement
   regression tests. The internals that produce the per-window features
   are not directly unit-tested. Concrete sub-targets for a new
   `tests/test_nddbase_spec.py` file modeled on `tests/test_absslp_spec.py`:

   - `_prepare_multistep_sequences_vectorized` ([NDDBase.py:135-304](DynaSD/NDDBase.py#L135-L304)):
     closed-form assertions on `n_sequences`, `input_data` shape,
     `target_data` shape, and `seq_positions` content for representative
     `(sequence_length, forecast_length, stride)` combinations,
     including the n-samples-vs-total_seq_length boundary case.
   - `_aggregate_sequences_to_windows_mse` ([NDDBase.py:638-703](DynaSD/NDDBase.py#L638-L703)):
     feed a deterministic synthetic `seq_results` dict and assert the
     window-MSE values equal a hand-computed expected. The Haoer-overlap
     window-assignment criterion at lines 688-689 is subtle (uses
     `>=` for start, strict `<` for end) and worth pinning down.
   - **Sequence cache** ([NDDBase.py:65-95, 158-225](DynaSD/NDDBase.py#L65-L225)):
     fit twice with same data, assert no recomputation; mutate input,
     assert recomputation; mutate sequence_length / forecast_length,
     assert cache miss.

   Estimated 6-10 tests in a single focused commit. No code changes
   anticipated — purely test additions.

6. **Phase H — Extend per-model contract test to ONCET and WVNT** once a
   small pretrained-checkpoint fixture is available (or stub
   `_load_model` so the test can run without real weights).

### Tier 3 — bigger commitment

7. **Deferred simulated-signal end-to-end suite** documented in
   `docs/testing_strategy.md` § 6. Per-model planted-seizure fixtures
   with documented detection tolerances. This is the "does it actually
   detect seizures" evidence beyond structural correctness.

### Tier 4 — release infrastructure

8. CI matrix (GitHub Actions, py3.10/3.11/3.12, core + extras).
9. README rewrite with a quickstart per detector class showing the
   time-indexed `forward(X)` workflow.
10. CHANGELOG, LICENSE review, version pinning policy.
11. TestPyPI release; collect feedback from 1-2 outside users; real
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
