"""
Core windowing utilities used by ``DynaSDBase`` and all DynaSD model classes.

These are the only helpers the package imports at runtime when fitting or
running inference with a model. Tooling for data loading, signal/channel
preprocessing, visualization, and statistics lives in :mod:`dynasd.tools`
and is intentionally kept out of this module so that the runtime surface
remains small and free of heavy optional dependencies.

See ``docs/spec_windowing_smoothing.md`` for the full contract these
functions must satisfy.
"""

import warnings

import numpy as np


_NON_INTEGER_TOL = 1e-9


def _canonical_sample_counts(fs, w_size, w_stride):
    """Return canonical ``(win_samples, step_samples)`` integer counts for a
    given ``(fs, w_size, w_stride)`` triple, per spec section 2.

    If ``w_size * fs`` or ``w_stride * fs`` is not within ``_NON_INTEGER_TOL``
    of an integer, emit a :class:`UserWarning` naming the affected parameter
    and the realized (rounded) value, then proceed with the rounded integer.
    """
    win_product = w_size * fs
    step_product = w_stride * fs

    win_samples = int(round(win_product))
    step_samples = int(round(step_product))

    if abs(win_product - round(win_product)) > _NON_INTEGER_TOL:
        warnings.warn(
            f"w_size * fs = {win_product:.6g} samples is non-integer; "
            f"rounding to {win_samples} samples per window. "
            f"Realized window length will be {win_samples / fs:.6g}s "
            f"instead of requested {w_size:.6g}s.",
            UserWarning,
            stacklevel=3,
        )
    if abs(step_product - round(step_product)) > _NON_INTEGER_TOL:
        warnings.warn(
            f"w_stride * fs = {step_product:.6g} samples is non-integer; "
            f"rounding to {step_samples} samples per step. "
            f"Realized stride will be {step_samples / fs:.6g}s "
            f"instead of requested {w_stride:.6g}s, with possible "
            f"cumulative drift across windows.",
            UserWarning,
            stacklevel=3,
        )

    return win_samples, step_samples


def num_wins(n_samples, fs, w_size, w_stride):
    """Number of windows fitting in a signal of ``n_samples`` samples at
    sampling rate ``fs`` Hz, with window length ``w_size`` seconds and stride
    ``w_stride`` seconds.

    Raises :class:`ValueError` if input is shorter than one window.
    """
    win_samples, step_samples = _canonical_sample_counts(fs, w_size, w_stride)
    if n_samples < win_samples:
        raise ValueError(
            f"Input length {n_samples / fs:.3g}s is shorter than window size "
            f"{w_size:.3g}s; cannot produce any windows."
        )
    return (n_samples - win_samples) // step_samples + 1


def moving_win_clips(x, fs, w_size, w_stride):
    """Slice ``x`` into windows of length ``w_size`` seconds at stride
    ``w_stride`` seconds.

    Returns an array of shape ``(n_windows, win_samples)`` whose row ``i`` is
    exactly ``x[i*step:i*step+win]``.

    Raises :class:`ValueError` if input is shorter than one window.
    """
    win_samples, step_samples = _canonical_sample_counts(fs, w_size, w_stride)
    if len(x) < win_samples:
        raise ValueError(
            f"Input length {len(x) / fs:.3g}s is shorter than window size "
            f"{w_size:.3g}s; cannot produce any windows."
        )
    n_windows = (len(x) - win_samples) // step_samples + 1
    samples = np.empty((n_windows, win_samples))
    for i in range(n_windows):
        start = i * step_samples
        samples[i, :] = x[start : start + win_samples]
    return samples
