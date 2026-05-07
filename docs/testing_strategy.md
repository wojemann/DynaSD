# Testing Strategy

This document defines the testing goals for DynaSD as a published
package. The locked spec these tests validate against lives in
[`spec_windowing_smoothing.md`](spec_windowing_smoothing.md).

## 1. Scope and priorities

Primary priorities:

1. Mathematical correctness of windowing, smoothing, and onset/spread math.
2. Per-model API contract (every class in `DynaSD.__all__` follows the
   same `fit` → `forward` → `get_onset_and_spread` interface).
3. Per-detector mathematical properties (non-negativity, scale
   invariance, determinism, monotone behavior under amplitude shifts).
4. Dependency isolation (core detectors importable with no heavy
   frameworks installed).

## 2. Test layout

Flat layout under `tests/`:

```text
tests/
  test_windowing_smoothing_spec.py     # spec § 2-10 closed-form coverage
  test_model_api_contract.py           # shared fit/forward/index contract
  test_nddbase_internals.py            # sequence prep + aggregator
  test_<detector>_spec.py              # per-detector property tests
  synthetic_data_generator.py          # neurodsp-based seizure fixtures
  data_generators.py                   # legacy fixture helpers
  visualization_utils.py               # legacy plotting helpers
```

`test_absslp.py` and `test_ndd.py` are legacy unittest-style modules
retained for backwards compatibility; new tests should follow the
`test_*_spec.py` naming pattern.

## 3. Spec coverage

Every section of [`spec_windowing_smoothing.md`](spec_windowing_smoothing.md)
has at least one parametrized test in
`tests/test_windowing_smoothing_spec.py`:

- canonical sample counts (integer + non-integer products),
- window count and indexing (boundary and partial-final-window cases),
- realized window timestamps,
- seconds → window-count conversion,
- the smoothing → threshold → spread → onset pipeline including the
  bias formula at threshold 0.5,
- multi-channel independence and realistic post-thresholded patterns,
- validation errors and cross-function invariants.

## 4. Per-model contract

`tests/test_model_api_contract.py` parametrizes every detector exported
from `DynaSD.__all__` and verifies:

- constructor accepts `(fs, w_size, w_stride)`;
- `forward` before `fit` raises;
- `fit` sets `is_fitted = True`;
- `forward(X)` returns a `(num_wins, n_channels)` DataFrame with
  channel-name columns and a time-indexed row index named ``t_sec``;
- `model(X)` equals `model.forward(X)`.

NDD-family classes additionally pass kwarg-validation, training-param
absorption, and `forward`-row-count regression tests.

## 5. Per-detector mathematical properties

Each detector has a `test_<detector>_spec.py` with a small set of
mathematical-property tests appropriate to the algorithm:

- `ABSSLP`: non-negativity, zero on per-window-constant input,
  amplitude monotonicity.
- `HFER`: non-negativity, scale invariance, high-band > low-band,
  determinism.
- `IMPRINT`: finiteness on seeded noise, amplitude monotonicity,
  determinism.
- `NDD` / `GIN` / `LiNDDA`: non-negativity, determinism (under fixed
  torch seed), amplitude monotonicity.
- `ONCET` / `WVNT`: bounded to `[0, 1]`, determinism, non-degeneracy.

Pretrained-classifier tests skip cleanly if the checkpoint is
unavailable; override paths via `DYNASD_ONCET_CHECKPOINT` /
`DYNASD_ONCET_CONFIG` / `DYNASD_WVNT_CHECKPOINT`.

## 6. Dependency isolation (planned)

CI matrix should run separate jobs for:

- `core`: no torch, no tensorflow.
- `[torch]`: NDD-family + ONCET only.
- `[tensorflow]`: WVNT only.
- `[test]` (full): everything.

Each job runs the import smoke test plus the subset of contract tests
its extras unlock.

## 7. Pass/fail policy

- No regressions in spec or contract suites on every PR.
- Per-detector property tests must pass under their declared extras.
- Deterministic fixtures only; no flake from randomness.

## 8. Running locally

```bash
# Closed-form spec + contract + internals (fast, no NN training).
pytest tests/test_windowing_smoothing_spec.py \
       tests/test_model_api_contract.py \
       tests/test_nddbase_internals.py

# Full suite (includes NDD-family fits + pretrained-checkpoint tests).
pytest
```
