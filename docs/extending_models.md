# Extending DynaSD Models

This guide is for contributors adding new detector classes that build on
shared DynaSD APIs.

## 1. Base class choices

- Use `DynaSDBase` for general detectors that score windows directly from
  the raw signal (see `HFER`, `ABSSLP`, `IMPRINT`).
- Use `NDDBase` for NDD-family detectors that share NDD's
  multistep-sequence preparation, batched torch inference, and
  per-window MSE aggregation (see `NDD`, `GIN`, `LiNDDA`).

Before implementing a new model, review:

- [`dynasd/base.py`](../dynasd/base.py) — windowing helpers, onset/spread
  pipeline, scaler plumbing.
- [`dynasd/NDDBase.py`](../dynasd/NDDBase.py) — sequence prep, training
  loop, sequence-to-window aggregator.
- [`docs/spec_windowing_smoothing.md`](spec_windowing_smoothing.md) — the
  locked windowing/smoothing/onset spec every detector must conform to.

## 2. Minimum model contract

A new detector class must implement:

- constructor accepting at least `fs`, `w_size`, `w_stride`;
- `fit(X) -> None` that calibrates per-channel reference statistics on a
  baseline DataFrame and sets `self.is_fitted = True`;
- `forward(X) -> pandas.DataFrame` returning per-window scores indexed
  by `self.get_win_index(len(X))` (named ``t_sec``, values are realized
  window-start times in seconds), with columns matching `X.columns`;
- `__call__` should equal `forward` (the base class wires this up).

Calling `forward` before `fit` must raise loudly. The contract suite
[`tests/test_model_api_contract.py`](../tests/test_model_api_contract.py)
verifies these invariants for every class in `DynaSD.__all__`.

## 3. Shared windowing/onset utilities

Use the helpers exposed on `DynaSDBase` instead of recomputing window
geometry:

- `self.get_win_times(n_samples)` — 1D array of realized window-start
  times in seconds.
- `self.get_win_index(n_samples)` — `pandas.Index` (named ``t_sec``)
  ready to attach to a `forward` output.
- `self.get_onset_and_spread(sz_prob, threshold, filter_w, rwin_size, rwin_req)`
  — the canonical smoothing → threshold → spread → onset pipeline. Do
  **not** reimplement this per detector; if you need a custom onset
  rule, document why.

When `sz_prob` is fed in time-indexed (the default after `forward`),
`get_onset_and_spread` returns onset labels in seconds directly.

## 4. Dependency guidelines

- Avoid eager top-level imports of `torch` / `tensorflow`. Wrap in a
  module-level `try/except ImportError` with a clear, actionable error
  message at construction time if the dep is missing.
- Document the required install extra (`[torch]`, `[tensorflow]`) in the
  class docstring and the README's "Available models" table.

## 5. Validation and error handling

- Validate `fs`, `w_size`, `w_stride` values early in `__init__`.
- Validate `X` shape and type (`pandas.DataFrame`) at the start of `fit`
  and `forward`.
- Reject unknown `**kwargs` with a `TypeError` naming the offender (see
  `NDDBase.__init__` for the pattern).
- Keep all duration parameters in seconds; convert to sample/window
  counts internally using the canonical math in `dynasd/utils.py`.

## 6. Testing requirements for new models

For every new detector class, add:

- A parametrized entry in `tests/test_model_api_contract.py` so the
  shared contract suite covers it.
- A `tests/test_<name>_spec.py` file with mathematical-property tests
  appropriate to the detector (e.g. non-negativity, scale invariance,
  determinism, monotone behavior under amplitude shifts).
- For `NDDBase` subclasses: rely on
  [`tests/test_nddbase_internals.py`](../tests/test_nddbase_internals.py)
  for sequence-prep / aggregator coverage.

Use seeded synthetic fixtures and assert exact values where feasible
rather than artifact-style "the plot looks reasonable" checks.

## 7. Documentation requirements

For each new model, include:

- a short class docstring describing the method, intended use case, and
  any pretrained checkpoint requirements;
- the install extra in the README table;
- a usage snippet if the constructor takes meaningful non-default kwargs.

## 8. Contributor checklist

- [ ] Class follows the `fit` → `forward` → `get_onset_and_spread` contract.
- [ ] Optional dependencies are isolated and produce clear errors when
      missing.
- [ ] Entry added to `MODELS` in `tests/test_model_api_contract.py`.
- [ ] Mathematical-property tests in `tests/test_<name>_spec.py`.
- [ ] README "Available models" table updated.
- [ ] Class docstring documents the install extra and any preprocessing
      assumptions.
