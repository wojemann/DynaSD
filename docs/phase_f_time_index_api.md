# Phase F: Time-indexed DataFrame outputs (deferred)

Status: **deferred — recommended for the next release-prep iteration before
external publication. Recorded here so the decision is not lost.**

## Decision under consideration

Switch the index of every DataFrame returned by a detector's
``forward(X)`` from a **positional integer index** (``0..n_windows-1``)
to the **realized window start times in seconds** (``i * step_samples / fs``,
matching :meth:`DynaSDBase.get_win_times`).

Likewise switch ``DynaSDBase.get_onset_and_spread``'s outputs:

- ``sz_idxs_df`` would carry onset times directly in **seconds** (the ``idxmax``
  on a time-indexed boolean DataFrame returns the row label, i.e. the time).
- ``sz_clf_ff`` would be indexed by realized window start times throughout.

## Why this is worth doing

- Self-describing output: a user who calls ``model(X)`` gets a DataFrame
  whose index is *the time the row corresponds to*. No parallel
  ``get_win_times`` call required.
- ``df.plot()`` produces a meaningful x-axis automatically.
- ``df.loc[5.0:10.0]`` gives the features in the 5–10s window cleanly.
- ``onset_time = sz_idxs_df["ch0"].iloc[0]`` is in seconds with no
  ``* step_samples / fs`` conversion downstream.
- Aligns the runtime API with the spec's "realized timestamps" convention
  (R5).

## Why it's a breaking change (and why we're doing it pre-publication)

Pandas distinguishes positional access (``.iloc``) from label access
(``.loc``) and from raw integer indexing on the legacy ``[]`` operator.
The shift from int-positional to float-time index will:

- Quietly change the meaning of any ``df.loc[i]`` call where ``i`` was
  previously a window count and is now misinterpreted as a time.
- Change the dtype reported by ``df.index`` (integer → float).
- Change ``df.iloc[k].name`` from an integer to a float.
- Affect any code that joins/concatenates inference outputs along the
  row axis using the default integer index.

Pre-publication is the right time for this break:

- The ``submission`` tag preserves legacy positional-index behavior for
  any external code that relied on it.
- All callers live in this repo and can be migrated atomically.
- Once published, this becomes a 1.x → 2.x migration with deprecation
  warnings and dual-contract maintenance — substantially more painful.

## Touch points (estimated)

| Area | Files |
|---|---|
| Per-model ``forward`` setting the index | `ABSSLP.py`, `HFER.py`, `IMPRINT.py`, `ONCET.py`, `WAVENET.py`, `NDDBase.py` (covers all 5 NN models) |
| ``get_onset_and_spread`` index handling | `base.py` |
| Spec doc | `docs/spec_windowing_smoothing.md` § 5, § 7, § 9, § 11 |
| Tests asserting onset values in **window indices** | `tests/test_windowing_smoothing_spec.py` (every assertion of the form `onset == 49` becomes `onset == 49 * w_stride`) |
| Tests asserting `index.dtype` | `tests/test_model_api_contract.py` (add a new contract test that the index is float and equal to `get_win_times(len(X))`) |
| In-repo callers | `tests/visualization_utils.py`, anything in `examples/` that does `df.iloc[...]` (likely fine), `df.loc[...]` (needs review), or `df.index` (needs update) |

Estimated as a single focused commit; no model algorithm changes required.

## Recommended approach when this is taken on

1. Update the spec doc first (declare the contract change), then code, then
   tests — same order used for Phase C.
2. Hard switch, no opt-out kwarg. Avoid maintaining two contracts.
3. Set ``df.index.name = "t_sec"`` (or a similar conventional name) so
   plots and ``repr(df)`` self-label the axis.
4. Consider exposing a tiny helper ``DynaSDBase.get_win_index(n_samples) ->
   pd.Index`` that returns a properly-named ``pd.Index`` of realized
   window start times. Each model's ``forward()`` then ends with
   ``out.index = self.get_win_index(len(X))`` — one consistent line per
   model.

## Out of scope for Phase F

- Changing the DataFrame's row order or column semantics.
- Changing the spec's window-count math, validation, or smoothing pipeline.
- Adding multi-index (channel × time) representations.

These can be considered later if the time-indexed output proves
insufficient for some workflow; the present recommendation is the smallest
useful step that gives users a self-describing inference output.
