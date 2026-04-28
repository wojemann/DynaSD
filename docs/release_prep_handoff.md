# Release Prep Handoff

Date: 2026-04-28  
Active branch: `release-prep/cleanup-packaging`  
Checkpoint commit: `87bba10`  
Commit message: `chore: archive legacy artifacts and draft release scaffolding`

## What was completed

1. Archived legacy/ad hoc root files to keep release surface cleaner:
   - `archive/legacy-tests/performance_test.py`
   - `archive/legacy-tests/test_caching_fix.py`
   - `archive/legacy-tests/test_partial_overlap.py`
   - `archive/legacy-tests/test_robust_caching.py`
   - `archive/legacy-tests/test_same_data_subsets.py`
   - `archive/notes/final_optimization_recommendations.md`
   - `archive/notes/optimization_guide.md`
2. Added `archive/README.md` to document archive purpose.
3. Reworked `README.md` into release-oriented structure and restored header image.
4. Added initial packaging draft in `pyproject.toml`.
5. Added contributor/testing draft docs:
   - `docs/testing_strategy.md`
   - `docs/extending_models.md`

## Decisions made during drafting

- Keep work isolated on `release-prep/cleanup-packaging` for safer iteration.
- Treat `ieeg` as test/dev-only support for remote data workflows, not required core runtime functionality.
- Do not gate new models on benchmark performance.
- Prioritize tests for timing behavior of smoothing/windowing utilities.
- Keep `docs/extending_models.md` direction as-is.

## Current testing philosophy

Focus on deterministic timing/unit behavior rather than model benchmark enforcement:
- window count correctness for `(n_samples, fs, w_size, w_stride)`,
- boundary/index alignment for generated windows,
- timestamp monotonicity and alignment,
- smoothing output length/index behavior,
- edge cases (very short input, exact-window input, partial-final-window input).

## Immediate next steps

1. Add first `pytest` timing tests in `tests/` targeting smoothing/windowing helpers.
2. Make package imports lighter (reduce eager heavy imports in `DynaSD/__init__.py`).
3. Decide whether to keep `setup.py` temporarily or retire it in favor of `pyproject.toml`.
4. Add a minimal CI test run (`pytest`) once basic tests are in place.

## Resume checklist on another machine

```bash
git fetch --all
git checkout release-prep/cleanup-packaging
git log --oneline -n 10
git status
```

If branch is missing locally:

```bash
git checkout -b release-prep/cleanup-packaging origin/release-prep/cleanup-packaging
```
