# Testing Strategy (Release Draft)

This document defines the testing goals for preparing DynaSD as a reliable shared package.

## 1. Scope and priorities

Primary priorities:
1. Timing-focused unit tests for smoothing and windowing correctness.
2. Dependency isolation (models importable with minimal extras).
3. Baseline package quality checks expected in Python package releases.

## 2. Timing functionality tests (smoothing and windowing)

### Core timing checks

For smoothing and windowing utilities, verify:
- expected number of output windows for `(n_samples, fs, w_size, w_stride)`,
- correct window start/stop indexing across boundary conditions,
- monotonic and aligned window timestamps,
- smoothing outputs have expected length and valid index alignment,
- behavior under very short input, exact-window input, and partial-final-window input.

### Required test inputs

- Deterministic synthetic signals (seeded) covering:
  - constant segments,
  - step changes,
  - impulse/noisy segments,
  - short arrays and edge-length arrays.
- Fixed fixtures committed for regression of timing/index behavior.

### Pass/fail policy

- No regressions in expected index/timestamp behavior for fixtures.
- No regressions in window count calculations.
- Timing-related helper outputs remain deterministic for identical inputs.

## 3. Dependency isolation tests

Each model should declare and enforce its dependency requirements cleanly.

### Required checks

- Core import smoke test with no optional extras:
  - import package and core/base APIs.
- Optional model import behavior:
  - if dependency is missing, error message must be explicit and actionable,
  - if dependency is installed, import and minimal inference path works.

### CI strategy

- Run separate CI jobs:
  - `core` job (no heavy frameworks),
  - `torch` extras job,
  - `tensorflow` extras job,
  - optional test job including `ieeg` for remote-data-related tests.

## 4. Standard package quality tests

- Unit tests for common utilities and shared base-class logic.
- Integration tests for representative model inference paths.
- API smoke tests for public constructors and core method signatures.
- Serialization/roundtrip tests where applicable.
- Test against supported Python versions (minimum + latest).

## 5. Proposed test layout

```text
tests/
  unit/
  integration/
  dependency/
  fixtures/
```

## 6. End-to-end behavioral tests on simulated signals (deferred)

The unit tests covered in section 2 validate **mathematical correctness** of
windowing, indexing, and smoothing math against synthetic inputs (e.g.
`np.arange`, zeros, step changes, impulse trains). They do **not** validate
that detection models produce sensible onset/spread results on realistic
input.

A separate test suite is required that runs each detection model end-to-end
on a deterministic **simulated iEEG-like signal** with a planted seizure
event of known onset time, channel coverage, and morphology. Each model that
claims to detect such events must produce onset detections within a
documented tolerance of the planted onset, on the planted channels, and not
on the unplanted channels. This is the only way to validate that the
windowing/smoothing/onset-detection chain produces clinically/scientifically
meaningful output, not just mathematically correct output.

**This work is deferred until after the code-consolidation pass is complete**
(redundant logic removed, common base behavior unified across model classes).
Testing detection behavior against an unstable target is wasted effort. Once
consolidation lands, simulated-signal tests will be added under
`tests/integration/`, with the simulated signal stored as a deterministic
fixture or generated from a seeded synthesis routine, and tolerance bounds
documented per model.

## 7. Immediate next actions

1. Migrate test entrypoint to `pytest` as canonical runner.
2. Add timing helper assertions for window count/index/timestamp checks.
3. Create first dependency-isolation smoke tests.
4. Add CI matrix for core + extras.
