# CLAUDE.md

Project-specific guidance for Claude Code working on DynaSD.

## Reference docs

- [docs/spec_windowing_smoothing.md](docs/spec_windowing_smoothing.md) —
  locked spec for the windowing/smoothing/onset math; tests validate
  against this and code conforms to it.
- [docs/testing_strategy.md](docs/testing_strategy.md) — current test
  layout and per-detector property-test pattern.
- [docs/extending_models.md](docs/extending_models.md) — contract for
  adding new detector classes.
- [docs/dev/](docs/dev/) — historical handoff / design memos; do not
  need to be kept in sync with current code.

## Key code areas

- [dynasd/base.py](dynasd/base.py) — `DynaSDBase`, including
  `get_win_times`, `get_win_index`, `get_onset_and_spread`.
- [dynasd/NDDBase.py](dynasd/NDDBase.py) — shared sequence prep,
  training loop, and per-window MSE aggregation for NDD-family models.
- [dynasd/utils.py](dynasd/utils.py) — `num_wins`, `moving_win_clips`,
  canonical sample-count math.
- [dynasd/__init__.py](dynasd/__init__.py) — public exports.
- [tests/test_windowing_smoothing_spec.py](tests/test_windowing_smoothing_spec.py)
  and [tests/test_model_api_contract.py](tests/test_model_api_contract.py)
  — first stop when changing shared logic.

## Commands

```bash
# Dev install
python -m pip install -e ".[test]"

# Fast subset (no NN training, no pretrained checkpoints)
pytest tests/test_windowing_smoothing_spec.py \
       tests/test_model_api_contract.py \
       tests/test_nddbase_internals.py

# Full suite
pytest
```

## Guardrails

- Preserve public API compatibility unless explicitly asked to break it.
- Keep changes scoped; avoid broad refactors unless requested.
- Prefer explicit errors for missing optional dependencies (`torch`,
  `tensorflow`); avoid eager top-level imports of them in package code.
- Don't modify archived legacy files unless explicitly asked
  (`archive/legacy-tests/`).
- Don't revert unrelated working-tree changes.
- Don't reintroduce historical stage labels (R1-R6, Phase F, etc.) into
  user-facing docstrings or docs — those have been deliberately stripped.
- The repo lives on a case-insensitive filesystem (macOS). Treat
  `claude.md` and `CLAUDE.md` as the same file; don't `rm` one assuming
  the other survives.

## Working loop

1. Make minimal targeted edits.
2. Run the closest test file first.
3. Run the broader spec/contract suite if shared logic changed.
4. Report what changed, what was run, residual risk.
