# Changelog

All notable changes to DynaSD are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-05-06

Initial public release. Stabilizes the windowing/smoothing/onset
contract, unifies the per-detector API, and ships a reproducible
quickstart.

### Added
- Locked windowing/smoothing/onset spec at
  [`docs/spec_windowing_smoothing.md`](docs/spec_windowing_smoothing.md).
- Per-detector contract suite (`tests/test_model_api_contract.py`) and
  per-model property tests (`tests/test_<detector>_spec.py`).
- Time-indexed `forward(X)` outputs: row index is the realized
  window-start time in seconds (named ``t_sec``).
- `DynaSDBase.get_win_index(n_samples)` helper for time-indexed
  inference DataFrames.
- Bundled synthetic seizure fixture: `DynaSD.load_example_seizure()`
  returns a 60 s, 8-channel, fs=256 Hz polyspike recording with
  ground-truth onset/focal-channel labels.
- Quickstart notebook at [`examples/quickstart.ipynb`](examples/quickstart.ipynb)
  showing the four-step `load → fit → forward → annotate` pipeline.
- Pretrained-checkpoint fetch-on-first-use: `ONCET` and `WVNT` download
  their weights from the `v0.1.0-checkpoints` GitHub Release into a
  per-user cache, with SHA-256 verification. Override with
  `DYNASD_CACHE_DIR` or an explicit `checkpoint_path` /
  `model_path` argument.
- Optional-dependency extras: `[torch]`, `[tensorflow]`, `[viz]`,
  `[test]`, `[docs]`, `[dev]`, and a `[all]` umbrella covering torch +
  tensorflow + viz.
- MIT license.

### Changed
- All detector classes now share the `fit` → `forward(X)` →
  `get_onset_and_spread(sz_prob, ...)` interface; `forward` returns a
  channel-named DataFrame with row count equal to `num_wins(len(X), fs,
  w_size, w_stride)`.
- Sample-count math is computed once per `(fs, w_size, w_stride)` triple
  and reused everywhere downstream; non-integer products emit a single
  `UserWarning` per parameter rather than silently drifting.
- Spread-detection convolution pads at the END of the output (not the
  start), with documented onset-bias formula at threshold 0.5.

### Fixed
- `NDDBase.forward` row count now matches `num_wins(len(X))` for inputs
  whose length falls between window boundaries (previously could return
  an extra row and silently drop the time index).
- `IMPRINT` no longer overrides the user's threshold inside
  `get_onset_and_spread`.

### Documentation
- README rewritten with quickstart pointer and per-model extras table.
- [`docs/extending_models.md`](docs/extending_models.md) reflects the
  current contract and test layout.
- [`docs/testing_strategy.md`](docs/testing_strategy.md) reconciled with
  the actual `tests/` tree.
- Historical handoff/design memos moved to [`docs/dev/`](docs/dev/).

[Unreleased]: https://github.com/wojemann/DynaSD/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wojemann/DynaSD/releases/tag/v0.1.0
