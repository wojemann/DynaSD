# DynaSD test suite

Pytest-driven test suite for the **Dyna**mic **S**eizure **D**etection
package. Validates the locked windowing/smoothing spec, the per-detector
API contract, and the math properties of every detector exported from
`dynasd.__all__`.

## Layout

```text
tests/
  test_windowing_smoothing_spec.py     # spec § 2-10 closed-form coverage
  test_model_api_contract.py           # shared fit / forward / index contract
  test_nddbase_internals.py            # sequence prep + aggregator
  test_<detector>_spec.py              # per-detector property tests
  synthetic_data_generator.py          # neurodsp-based seizure fixtures
  data_generators.py                   # legacy fixture helpers
  visualization_utils.py               # legacy plotting helpers
  test_absslp.py / test_ndd.py         # legacy unittest-style modules (kept for compat)
```

## Running

```bash
# Fast subset — closed-form spec + property tests; no torch/tensorflow needed.
pytest tests/test_windowing_smoothing_spec.py \
       tests/test_absslp_spec.py \
       tests/test_hfer_spec.py \
       tests/test_imprint_spec.py

# Contract + NDD-family — requires the [torch] extra (and [tensorflow] for WVNT).
pip install -e ".[all,test]"
pytest tests/test_model_api_contract.py \
       tests/test_nddbase_internals.py \
       tests/test_ndd_family_spec.py

# Pretrained-checkpoint suite — downloads weights from the project's
# GitHub Release on first run; cache lives at ~/Library/Caches/dynasd
# (macOS), ~/.cache/dynasd (Linux). Override with DYNASD_CACHE_DIR.
pytest tests/test_pretrained_classifiers_spec.py

# Full suite.
pytest
```

## What's covered

| File | What it validates |
|---|---|
| `test_windowing_smoothing_spec.py` | Every section of [`docs/spec_windowing_smoothing.md`](../docs/spec_windowing_smoothing.md): canonical sample counts, window count + indexing, realized timestamps, seconds → window-count conversion, full smoothing/threshold/spread/onset pipeline including the bias formula at threshold 0.5, multi-channel independence, edge cases. |
| `test_model_api_contract.py` | Every detector in `dynasd.__all__` accepts `(fs, w_size, w_stride)`, raises before `fit`, sets `is_fitted = True` after `fit`, returns a time-indexed DataFrame from `forward(X)` with row index named `t_sec` and `(num_wins, n_channels)` shape. |
| `test_nddbase_internals.py` | `NDDBase` sequence-prep shape and content, sequence cache behavior, sequence → window aggregator overlap rule and `sqrt(mean(mse))` aggregation. |
| `test_absslp_spec.py` | ABSSLP non-negativity, zero on per-window-constant input, monotonic response under amplitude. |
| `test_hfer_spec.py` | HFER non-negativity, scale invariance, high-band > low-band, determinism. |
| `test_imprint_spec.py` | IMPRINT finiteness on seeded noise, monotonic response under amplitude, determinism. |
| `test_ndd_family_spec.py` | `NDD` / `GIN` / `LiNDDA`: non-negativity, determinism (under fixed torch seed), monotonic response under amplitude. |
| `test_pretrained_classifiers_spec.py` | `ONCET` / `WVNT`: output bounded to `[0, 1]`, determinism, non-degeneracy. Skips cleanly if checkpoints aren't cached. |

## Test data

The `dynasd.load_example_seizure` fixture (a 60 s, 8-channel, 256 Hz
synthetic polyspike recording) is shipped with the package and used by
the quickstart notebook. The test suite builds its own seeded synthetic
inputs per test rather than relying on a single canonical fixture so
that math properties can be probed independently.

`synthetic_data_generator.py` wraps `neurodsp` to produce baseline + planted-seizure
recordings used by a couple of the legacy tests. New tests should
prefer seeded `numpy` arrays or `np.arange`-based step functions over
the heavier neurodsp generators when possible.

## Adding tests for a new detector

1. Add a build helper + parametrize entry in `MODELS` inside
   `test_model_api_contract.py`.
2. Add a `test_<name>_spec.py` with detector-specific math properties.
3. If the detector is an `NDDBase` subclass, the existing
   `test_nddbase_internals.py` already covers shared internals.

See [`docs/extending_models.md`](../docs/extending_models.md) for the
full contract a new detector must satisfy.
