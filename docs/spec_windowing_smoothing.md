# Windowing & Smoothing Spec

Status: **locked.** All design decisions resolved. This document is the source
of truth that tests must validate against and that code in `DynaSD/utils.py`
and `DynaSD/base.py` must conform to.

Scope:
- `DynaSD/utils.py::num_wins`
- `DynaSD/utils.py::MovingWinClips`
- `DynaSD/base.py::DynaSDBase.get_win_times`
- `DynaSD/base.py::DynaSDBase.get_onset_and_spread`

Rationale and the alternatives considered for each decision are recorded in
**Appendix A** (decisions R1–R6).

---

## 1. Definitions and units

| Symbol | Meaning | Units |
|---|---|---|
| `fs` | Sampling frequency of the input signal | Hz |
| `w_size` | Window length | seconds |
| `w_stride` | Window stride (step between consecutive window starts) | seconds |
| `n_samples` | Length of the input signal | samples |
| `D` | A generic duration parameter (`filter_w`, `rwin_size`, `rwin_req`) | seconds |
| `t_i` | Real-time timestamp assigned to window `i` | seconds |

`w_size`, `w_stride`, and all duration parameters (`filter_w`, `rwin_size`,
`rwin_req`) are user-facing and expressed in **seconds**. All conversions to
sample counts or window counts happen internally.

---

## 2. Canonical sample counts

For any `(fs, w_size, w_stride)` triple, the package computes once:

```
win_samples  = int(round(w_size  * fs))
step_samples = int(round(w_stride * fs))
```

These two integers are the canonical representation of the windowing geometry
and are reused everywhere downstream. No per-window float arithmetic is
permitted.

**Non-integer products.** If `w_size * fs` or `w_stride * fs` is not within
`1e-9` of an integer, the package emits exactly one `UserWarning` per affected
parameter, naming both the requested value and the realized rounded value, and
then proceeds with the rounded integer. Example:

> `UserWarning: w_stride * fs = 170.667 samples is non-integer; rounding to 171 samples per step. Realized stride will be 0.334s instead of requested 0.333s, with possible cumulative drift across windows.`

---

## 3. Window count

For input of length `n_samples`:

```
n_windows = (n_samples - win_samples) // step_samples + 1     # if n_samples >= win_samples
```

If `n_samples < win_samples` (input shorter than one window), all windowing
entry points raise `ValueError`. See section 9.

---

## 4. Window indexing (sample bounds)

Window `i ∈ [0, n_windows)` covers samples:

```
[i * step_samples, i * step_samples + win_samples)
```

Length of every window slice is exactly `win_samples`. Stride between
consecutive window starts is exactly `step_samples`. No off-by-one between
window `i`'s end and window `i+1`'s start.

`MovingWinClips(x, fs, w_size, w_stride)` returns an array of shape
`(n_windows, win_samples)` whose row `i` is `x[i * step_samples : i * step_samples + win_samples]`.

---

## 5. Window timestamp

Window `i` is timestamped by its **start time** (R2):

```
t_i = i * step_samples / fs                # seconds
```

This is the **realized** start time corresponding to the actual sample
positions used by `MovingWinClips`. For integer `w_size * fs` and
`w_stride * fs`, `t_i == i * w_stride` exactly. For non-integer products,
realized timestamps may drift by at most `0.5 / fs` seconds per window from
the requested stride; the realized value is reported, never the requested one.

`DynaSDBase` exposes window timestamps via two helpers:

- ``get_win_times(n_samples) → np.ndarray`` — bare 1D array of realized
  window-start times in seconds.
- ``get_win_index(n_samples) → pandas.Index`` (named ``t_sec``) —
  the same values wrapped as a labeled :class:`pandas.Index`, suitable for
  use as the row index of inference DataFrames.

**Inference DataFrames are time-indexed (Phase F).** Every detector's
``forward(X)`` returns a DataFrame whose row index is
``self.get_win_index(len(X))`` — i.e. the realized window-start times in
seconds with index name ``"t_sec"``. Concretely, ``out.iloc[i]`` is the
features at the ``i``-th window and ``out.loc[5.0]`` is the features at
the window starting 5.0 seconds into the input.

The choice of start (vs. center vs. end) is **acausal** and deliberate: it
gives the earliest moment from which a window's data could detect activity,
serving the package's "earliest-detection" intent. See R2 for the full
rationale.

---

## 6. Seconds → window-count conversion

For any duration `D` (one of `filter_w`, `rwin_size`, `rwin_req`):

```
n_idx(D) = floor((D - w_size) / w_stride) + 1
```

This counts the number of windows whose **start AND end** both fall within a
`D`-second span. Each counted window's data is fully contained within the
span; the union of counted windows' data is exactly `[t, t + D)`.

Applied identically to all three duration parameters in
`get_onset_and_spread`:

```
filter_w_idx  = floor((filter_w  - w_size) / w_stride) + 1
rwin_size_idx = floor((rwin_size - w_size) / w_stride) + 1
rwin_req_idx  = floor((rwin_req  - w_size) / w_stride) + 1
```

For overlapping windows (`w_size > w_stride`), the formula remains valid but
counted windows physically overlap in real time. A single instant of activity
can fire `w_size / w_stride` overlapping windows, so the count of fired
counted-windows is not a clean proxy for "seconds of contiguous true
activity." This is an intrinsic property of overlapping detection and is
**not** corrected by the formula. Users requiring clean
seconds-of-activity semantics should set `w_size = w_stride`.

---

## 7. Smoothing pipeline (`get_onset_and_spread`)

### 7.1 Inputs

- `sz_prob`: `pandas.DataFrame` of shape `(n_windows, n_channels)`. Each row
  is the per-channel seizure probability for one window. The row index is
  preserved through the pipeline; when callers feed the time-indexed output
  of ``model.forward(X)`` (Phase F), the resulting onset times come out in
  seconds directly.
- `threshold`: scalar (or class attribute `self.threshold`).
- `filter_w` (default `10.0`): smoothing window in seconds.
- `rwin_size` (default `5.0`): spread-detection lookahead in seconds.
- `rwin_req` (default `4.0`): spread-detection required activity in seconds.
- Class state: `self.fs`, `self.w_size`, `self.w_stride`.

### 7.2 Step-by-step

**S1.** Convert duration parameters to window counts (section 6):
```
filter_w_idx, rwin_size_idx, rwin_req_idx
```

**S2.** Smooth probabilities along the window axis:
```python
sz_prob_smooth = scipy.ndimage.uniform_filter1d(
    sz_prob, size=filter_w_idx, mode='nearest', axis=0, origin=0
)
```
Output shape equals input shape. `mode='nearest'` extends edges with the edge
value (no zero-padding bias).

**S3.** Threshold:
```
sz_clf = sz_prob_smooth > threshold       # boolean (n_windows, n_channels)
```

**S4.** Identify candidate seizing channels:
```
seized_idxs = sz_clf.any(axis=0)
```

**S5.** Spread (sustained-activity) detection via forward-looking convolution.
For each channel, compute:
```python
sliding_sums = np.convolve(sz_clf[col].astype(int),
                           np.ones(rwin_size_idx),
                           mode='valid')
sz_spread[:, col] = (sliding_sums >= rwin_req_idx)
```
Output of `mode='valid'` has length `n_windows - rwin_size_idx + 1`. Restore
original length by appending the **last valid row** at the END,
`rwin_size_idx - 1` times. The result is `sz_clf_ff`, shape
`(n_windows, n_channels)`.

**S6.** Onset detection. For each channel:
- If channel is in `seized_idxs` and has any True row in `sz_clf_ff`:
  `onset[col] = sz_clf_ff[col].idxmax()`
- Otherwise: `onset[col] = NaN` and the column in `sz_clf_ff` is set to
  all zeros.

The value of `onset[col]` is whatever ``sz_prob`` was indexed by. When
callers feed the time-indexed output of ``model.forward(X)`` (Phase F),
``idxmax`` returns the onset time in **seconds** directly — no further
``* step_samples / fs`` conversion required. When callers pass a
positional-indexed DataFrame, ``idxmax`` returns a window index, and
the caller multiplies by ``step_samples / fs`` if seconds are wanted.

### 7.3 Returns

- Default: `sz_idxs_df` — a single-row DataFrame whose columns are channel
  names and values are onset labels (NaN for non-seizing channels). The
  label type matches the row index of the input ``sz_prob``: time in
  seconds when the input is time-indexed (the standard case post-Phase F),
  or window indices when the input is positional.
- If `ret_smooth_mat=True`: also return `sz_clf_ff` — the post-spread,
  post-padding matrix, indexed by the same labels as ``sz_prob`` (time in
  seconds in the standard case).

### 7.4 Reported onset is biased earlier than first raw threshold crossing

The pipeline introduces two shifts versus the first individual
above-threshold raw sample:

1. **Smoothing shift** (S2): a centered moving average around a sharp
   upward step does not move the threshold-midpoint crossing. For a step
   from `0` to `1` at row `k` filtered with `uniform_filter1d(size=N,
   origin=0)` and tested against threshold `0.5`:
   - **Odd `N`** (symmetric window): the smoothed signal first strictly
     exceeds `0.5` at row `k` — no shift.
   - **Even `N`** (right-biased window): the smoothed signal first strictly
     exceeds `0.5` at row `k + 1` — a +1 row *later* than raw, owing to
     the strict `>` comparison and the asymmetric ceiling at exactly half.

   For thresholds far from the ramp midpoint, the shift is roughly
   ``±filter_w_idx/2`` — the smoothed value clears low thresholds earlier
   than raw and high thresholds later. In practice the spec test suite
   pins behavior at threshold ``0.5`` because that's where the post-
   filter signal carries the most information about onset timing; tune
   threshold close to the per-detector smoothed midpoint to keep
   absolute-time interpretation clean.

2. **Spread shift** (S5): for contiguous above-threshold activity beginning
   at row `k` (post-smoothing), the earliest True row in `sz_spread` is
   exactly `k - (rwin_size_idx - rwin_req_idx)`.

Total bias for a clean step at threshold `0.5`:
```
total_shift_windows = filter_offset - (rwin_size_idx - rwin_req_idx)
                      where filter_offset = +1 if filter_w_idx even else 0
total_shift_sec     = total_shift_windows * w_stride
```

Worked example (`w_size=1, w_stride=1, filter_w=10, rwin_size=5, rwin_req=4`,
threshold `0.5`, planted step at row `k=50`):
- `filter_w_idx=10` → even → `filter_offset = +1`
- `rwin_size_idx=5`, `rwin_req_idx=4` → spread shift = 1 row earlier
- Net: detected at row `50 + 1 - 1 = 50` — same as the planted step.

End-padding does not affect onset detection: `idxmax` returns the first
True row, and any True row inside the padded region implies the
corresponding unpadded channel was already detected from a valid row
earlier.

---

## 8. Validation and error handling

`get_onset_and_spread` validates parameters before any computation. All
violations raise `ValueError` with a message naming the offending parameter,
its value, and the constraint:

```
filter_w  >= w_size
rwin_size >= w_size
rwin_req  >= w_size
rwin_req  <= rwin_size
```

Example message:
> `ValueError: rwin_req (0.5s) must be >= w_size (1.0s); cannot fit a window in less than w_size seconds.`

The three windowing entry points (`num_wins`, `MovingWinClips`,
`get_win_times`) each raise `ValueError` when input duration is shorter than
one window:

> `ValueError: Input length 3.0s is shorter than window size 10.0s; cannot produce any windows.`

Non-integer `w_size * fs` or `w_stride * fs` emits `UserWarning` (not an
error); see section 2.

---

## 9. Cross-function invariants

For any valid `(fs, w_size, w_stride, n_samples)`:

1. **Window-count agreement.**
   `num_wins(n_samples, fs, w_size, w_stride) == len(get_win_times(n_samples))`.

2. **Shape consistency.**
   `MovingWinClips(x, fs, w_size, w_stride).shape == (num_wins(len(x), fs, w_size, w_stride), win_samples)`.

3. **Constant stride.**
   `np.diff(get_win_times(n_samples))` is a constant array equal to
   `step_samples / fs` (within float tolerance).

4. **Constant window length.**
   Every row of `MovingWinClips`'s output has length exactly `win_samples`.

5. **Realized vs. requested timestamps.**
   For integer-product `(w_size * fs, w_stride * fs)`,
   `get_win_times(n_samples)[i] == i * w_stride` exactly.

6. **Boundary alignment.**
   For any window `i`, `MovingWinClips`'s row `i` equals
   `x[i * step_samples : i * step_samples + win_samples]` exactly.

---

## 10. Edge cases

| Case | Required behavior |
|---|---|
| `n_samples < win_samples` | `ValueError` from any windowing entry point (section 8). |
| `n_samples == win_samples` | Exactly one window. `n_windows == 1`, `t == [0.0]`, `MovingWinClips` row 0 = full input. |
| `n_samples` partial-final-window | The final partial window is discarded. `n_windows == (n_samples - win_samples) // step_samples + 1`. No padding; trailing samples are not represented. |
| `filter_w == w_size` | `filter_w_idx == 1`. `uniform_filter1d` with size 1 is a no-op (smoothing inactive). Permitted. |
| `rwin_req == w_size` | `rwin_req_idx == 1`. Spread detection requires at least 1 fired window. Permitted. |
| `rwin_req == rwin_size` | `rwin_req_idx == rwin_size_idx`. Strictest spread criterion: every counted window must fire. Permitted. |
| Channel above threshold for fewer than `rwin_req_idx` consecutive fired windows | Channel is not flagged as seizing; onset is `NaN`; column in `sz_clf_ff` is zeroed. |
| Activity only in last `rwin_size_idx - 1` windows | Activity may fail to satisfy spread criterion (no full lookahead). Channel is not flagged. End-padding cannot resurrect it. |
| All channels non-seizing | Returns single-row DataFrame with all `NaN`s; `sz_clf_ff` (if requested) is all zeros. |

---

## 11. Test requirements (consolidated)

Tests live in `tests/` and must validate every section above. Each bullet is
one or more `pytest` cases.

**Section 2 (canonical sample counts):**
- Integer products produce exact `win_samples`/`step_samples`.
- Non-integer products emit `UserWarning` once per affected parameter.
- Warning message includes both requested and realized values.
- `warnings.filterwarnings("error")` correctly upgrades to exception.

**Section 3 (window count):**
- Closed-form check on representative parameter sets.
- `n_samples == win_samples` → 1 window.
- Partial-final-window inputs return floor count, no rounding up.

**Section 4 (window indexing):**
- `MovingWinClips` row `i` exactly equals
  `x[i * step_samples : i * step_samples + win_samples]`.
- All rows have identical length.
- Stride between consecutive row start indices equals `step_samples` exactly.

**Section 5 (timestamps):**
- Integer products: `t[i] == i * w_stride` for all `i`.
- Non-integer products: `t[i] == i * step_samples / fs`, may drift from
  requested stride by at most `0.5 / fs` per window.
- `np.diff(t)` is constant.
- ``forward(X)``'s row index is ``self.get_win_index(len(X))``, named
  ``t_sec``, with values numerically equal to ``get_win_times(len(X))``.

**Section 6 (seconds → window-count):**
- Closed-form check for representative `(D, w_size, w_stride)` sets, including
  `w_size > w_stride` (overlapping) cases.
- `D == w_size` → idx == 1.

**Section 7.2 (smoothing pipeline):**
- S2: `uniform_filter1d` output length equals input length.
- S5: `mode='valid'` length is `n_windows - rwin_size_idx + 1`; padding makes
  result length equal to input length.
- S5: padded rows equal the last valid row.
- S6: `idxmax` returns first True row; non-seizing channels report `NaN`.
- S7: onset times equal `onset_window_idx * step_samples / fs`.

**Section 7.4 (bias):**
- Synthetic fixture with sharp upward step in `sz_prob`: assert reported onset
  matches the documented total-shift formula within tolerance.
- Fixture for seizure starting at row 0 (early-edge): smoothing edge mode
  preserves correct onset.
- Fixture for seizure ending in last `rwin_size_idx - 1` windows: channel is
  correctly flagged as not seizing (or as seizing earlier).
- Fixture with mid-seizure detection gap: onset reflects first sustained span.

**Section 8 (validation):**
- Each constraint violation raises `ValueError`.
- Each error message names the offending parameter.
- `n_samples < win_samples` raises from each of the three windowing entry
  points.

**Section 9 (cross-function invariants):**
- Each invariant validated on a parameter sweep including overlapping windows
  and non-integer-product configurations.

**Section 10 (edge cases):**
- One test per row of the table.

---

## Appendix A: Resolved decisions (rationale)

The following resolutions document the design decisions and the alternatives
considered. They are the historical record of *why* the contracts above are
what they are. Tests are written against the contracts in sections 1–11, not
directly against this appendix.

### R1. Spread convolution padding: pad at END

The spread-detection logic in `get_onset_and_spread` convolves the binary
`sz_clf` matrix with a length-`rwin_size_idx` kernel of ones in `mode='valid'`,
producing output `rwin_size_idx - 1` rows shorter than the input. Those
missing rows are appended at the END of the output, filled with the last valid
row. Stale inline comment at base.py:66 ("Pad at the START") must be fixed.

End-padding does not affect onset detection (`idxmax` returns the first True
row; if the first True lived in the padded region, no valid row was True and
the channel was not flagged as seizing in the first place).

**Documented semantic effect of valid forward-looking convolution on onset
times.** The reported onset window index is shifted **earlier** than the first
individual above-threshold window. For contiguous above-threshold activity
beginning at input row `k`, the reported onset is approximately
`k - (rwin_size_idx - rwin_req_idx)`. With default parameters
(`rwin_size=5s`, `rwin_req=4s`, `w_stride≈1s`) this is ~1 window earlier;
larger gaps between `rwin_size` and `rwin_req` widen the bias.

This is an intentional consequence of the spread-detection definition:
*"the earliest window from which sustained above-threshold activity is
observable looking forward."* No corrective shift is applied in code, because
no shift correctly handles all edge cases (sparse activity, mid-seizure gaps,
seizures near the end of the recording, etc.). Downstream consumers must
account for this offset explicitly when interpreting onset times in absolute
seconds.

### R2. Window timestamps represent window START times

`get_win_times(n_samples)` returns `np.arange(n_windows) * w_stride`. Window
index `i` corresponds to absolute time `t_i = i * w_stride` seconds, which is
the **start** of the real-time interval `[i * w_stride, i * w_stride + w_size)`
that the window covers.

**Causality note.** Strictly speaking, the **causal** choice would be to use
the window's END time as its timestamp (`t_i = i * w_stride + w_size`), since
the window's value can only be computed once all `w_size` seconds of data have
been observed. Assigning the timestamp to the window's START is therefore
**acausal**: the timestamp at time `t` is being assigned to a value that
summarizes data from `t` into the future. There is no causal-inference
argument for this choice.

We adopt the start-time convention anyway because it is **convenient for the
research/clinical question this package is built to answer:** *"what is the
earliest moment from which we can claim seizure activity is present?"* The
start of a window is the earliest sample whose data contributed to the
detection of that window, so reporting onset as a window-start time gives the
most-anticipatory plausible answer. The package is explicitly trading
causality for earliest-detection semantics, and consumers must understand
this when interpreting onset times in absolute seconds.

Combined with the spread-detection forward-looking bias documented in
R3, reported onset times sit a small number of windows away from the
first individual above-threshold sample (typically zero net shift when
threshold is chosen at the smoothed-step midpoint, and a few windows of
forward-looking shift otherwise). This is a deliberate property of the
pipeline, not a bug.

### R3. Full window-index → absolute-seconds chain

This is the canonical walkthrough that all of `get_onset_and_spread`'s
behavior must follow. Sections 7 and 9 of the contract above codify it. Tests
will validate each link of this chain independently.

**Total bias of `onset_time_sec` vs. true first above-threshold sample.**
For a clean step input at threshold `0.5` (the canonical case captured by
the spec test suite), the reported onset is shifted versus the planted step
by:

```
total_shift_windows = filter_offset - (rwin_size_idx - rwin_req_idx)
                      where filter_offset = +1 if filter_w_idx even else 0
total_shift_sec     = total_shift_windows * w_stride
```

A centered moving average around a step does **not** move the
threshold-midpoint crossing — by symmetry the smoothed value crosses
`0.5` at the planted step itself for odd `filter_w_idx`, and one row
later for even `filter_w_idx` (because the strict ``>`` comparison
breaks the half-and-half tie on the right side). For thresholds far from
the smoothed-step midpoint, the smoothing shift grows toward
``±filter_w_idx/2``; tune threshold close to the per-detector smoothed
midpoint to keep the relationship between planted and detected onset
clean.

With default parameters (`w_size = 1s`, `w_stride = 1s`,
`filter_w = 10s`, `rwin_size = 5s`, `rwin_req = 4s`, threshold `0.5`):
- `filter_w_idx = 10` (even) → `filter_offset = +1`
- spread shift = `rwin_size_idx − rwin_req_idx` = 1 window earlier
- **net total = +1 − 1 = 0 windows: detected onset == planted onset**

Plus the windowing convention itself (R2) means each window's timestamp
already refers to neural activity in the following `w_size` seconds.

The spread shift is **intentional and forward-looking**. Downstream
clinical/research consumers must apply their own offset if they need
the timestamp of the first individual above-threshold sample rather
than the forward-looking onset.

### R4. Input shorter than one window: raise `ValueError`

When `n_samples / fs < w_size`, the windowing helpers cannot produce any
valid windows. All public windowing functions raise `ValueError` with a
message identifying the offending lengths in seconds.

Rationale: silent return of empty arrays or negative counts is unsafe for a
package being prepared for external use. Failing loud at the boundary forces
the caller to handle invalid input explicitly. Batch consumers can wrap calls
in `try/except ValueError`; the cost of that is one line and is preferable to
silent downstream nonsense.

### R5. Sample-count-first windowing math

To eliminate per-window float drift in `MovingWinClips`, guarantee constant
stride and constant window length, and ensure all windowing helpers agree
exactly, the canonical math is computed in **integer samples** once and
re-used everywhere. See section 2 for the formal contract.

**Trade-off considered.** Computing `step_samples` once (`int(round(w_stride * fs))`)
gives perfect per-window invariants (constant stride, constant window length)
but can drift from "true" continuous-time positions by up to `0.5 / fs` seconds
per window when `w_stride * fs` is non-integer. The previous per-window
truncation approach tracked continuous-time positions more closely on average
but produced inconsistent stride and window-length values across windows. We
prioritized invariants over continuous-time tracking because invariants are
testable, downstream code implicitly assumes them, and the previous approach
contained a latent shape-mismatch bug in `MovingWinClips`.

**Non-integer products: warn, don't error.** Hard-rejecting non-integer
products would force users to round their parameters preemptively. A
`UserWarning` lets pipelines continue with the realized values reported
honestly, and users who want hard errors can elevate via
`warnings.filterwarnings("error", ...)`.

### R6. Seconds → window-count via start-and-end containment

The conversions of `filter_w`, `rwin_size`, and `rwin_req` from seconds to
window counts use the rule:

> *A window counts toward a duration `D` if and only if both its start and its
> end fall within the `D`-second span.*

Under this rule, the count of qualifying windows is
`floor((D - w_size) / w_stride) + 1`. This matches the existing code (no
behavioral change vs. `submission` tag for existing parameter sets).

**Conceptual model.** Each "qualifying" window's data is *fully contained*
within the lookback / smoothing span — no window's data extends past the
boundary. Each qualifying window represents exactly `w_size` seconds of
evidence whose data lies entirely inside the span. The lookback span
`[t_i, t_i + D)` exactly matches the union of qualifying windows' real-time
coverage.

**Alternatives considered.** A simpler "round to nearest window-step"
formulation (`max(1, round(D / w_stride))`) was rejected because it does not
preserve the exact-coverage property when windows overlap (`w_size > w_stride`).
Under that alternative, the union of counted windows extends `w_size - w_stride`
seconds past the span. The start-and-end containment rule keeps the lookback
span exactly equal to its named duration.

**On overlap and double-counting.** When `w_size > w_stride`, qualifying
windows overlap in real time and a single instant of activity can fire
multiple windows. This is intrinsic to overlapping detection and is **not**
something the formula tries to correct. Users requiring clean
seconds-of-activity semantics should set `w_size = w_stride`. The formula
itself is well-defined and exact for the lookback-span definition; the
ambiguity is in the user's interpretation of "fired window count" as
"seconds of activity," which is downstream of the formula.

**Validation falls out of the formula.** For each duration to produce a
positive window count, `D >= w_size` is required. Plus
`rwin_req <= rwin_size` to keep the criterion satisfiable. See section 8.
