# Extending DynaSD Models (Draft)

This guide is for contributors adding new detector classes that build on shared DynaSD APIs.

## 1. Base class choices

- Use `DynaSDBase` for general detector interfaces and shared preprocessing behavior.
- Use `NDDBase` for NDD-specific model families and training/inference conventions.

Before implementing a new model, review:
- `DynaSD/base.py`
- `DynaSD/NDDBase.py`

## 2. Minimum model contract

A new model class should clearly implement:
- constructor with explicit defaults (`fs`, `w_size`, `w_stride` where relevant),
- `fit(...)` when training/statistics are required,
- `forward(...)` for feature/probability inference,
- `__str__(...)` for readable model identity.

Keep outputs consistent with existing detectors (window-by-channel DataFrame, when applicable).

## 3. Dependency guidelines

- Avoid importing heavy optional frameworks at package import time.
- If a model depends on optional frameworks (`torch`, `tensorflow`), either:
  - perform local imports in model methods, or
  - guard imports with clear `ImportError` instructions.
- Document required install extra in class docstring and README.

## 4. Validation and parameter handling

- Validate sampling/window parameters early in `__init__`.
- Validate input data type and shape with explicit errors.
- Keep assumptions explicit (units, expected sampling rate, channel axis).

## 5. Testing requirements for new models

Each new model should include:
- unit tests for initialization and input validation,
- integration tests for fit/forward behavior on synthetic data,
- onset timing tests using shared metric helpers,
- dependency-isolation tests for missing/installed optional dependencies.

## 6. Documentation requirements

For each new model, include:
- short description of method and intended use case,
- required dependencies and extras,
- basic usage snippet,
- notes on expected onset-time behavior and limitations.

## 7. Contributor checklist

- [ ] Model follows base-class conventions
- [ ] Optional dependencies are isolated and documented
- [ ] Tests pass locally and in CI matrix
- [ ] README/docs updated with usage and dependency extra
