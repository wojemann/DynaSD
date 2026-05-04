"""
Multi-channel iEEG plotting helpers.

These depend on matplotlib and seaborn — both are optional dependencies
of the package, so import this module only when you need plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns


def set_plot_params():
    """Apply DynaSD's default matplotlib rcParams (font sizes, line widths)."""
    plt.rcParams["image.cmap"] = "magma"
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["lines.linewidth"] = 2

    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3

    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1


def _shade_y_ticks_background(ax, y_ticks, colors, alpha=1):
    """Add colored horizontal bands behind plot data, centered on y-ticks.

    Used internally by :func:`plot_ieeg_data` to visually group channels.
    """
    if isinstance(colors, str):
        colors = [colors] * len(y_ticks)
    if len(y_ticks) != len(colors):
        raise ValueError("The length of y_ticks and colors must be the same.")

    sorted_indices = np.argsort(y_ticks)
    y_ticks = np.array(y_ticks)[sorted_indices]
    colors = np.array(colors)[sorted_indices]

    for i in range(len(y_ticks)):
        if colors[i] is not None:
            if i == 0:
                lower_bound = y_ticks[i] - (y_ticks[i + 1] - y_ticks[i]) / 2
            else:
                lower_bound = (y_ticks[i - 1] + y_ticks[i]) / 2

            if i == len(y_ticks) - 1:
                upper_bound = y_ticks[i] + (y_ticks[i] - y_ticks[i - 1]) / 2
            else:
                upper_bound = (y_ticks[i] + y_ticks[i + 1]) / 2

            ax.axhspan(lower_bound, upper_bound, color=colors[i], alpha=alpha, linewidth=0)


def plot_ieeg_data(
    data,
    fs=None,
    t=None,
    t_offset=0,
    colors=None,
    plot_color="k",
    shade_color=None,
    shade_alpha=0.3,
    empty=False,
    dr=None,
    fig_size=None,
    minmax=False,
):
    """Multi-channel iEEG "waterfall" plot.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        iEEG data; column names (if a DataFrame) are used as channel labels.
    fs : float, optional
        Sampling frequency in Hz; required if ``t`` is not given.
    t : numpy.ndarray, optional
        Explicit time vector. If None, generated from ``fs``.
    t_offset : float, optional
        Time offset added to the time vector.
    colors : list, optional
        Per-channel colors for y-tick labels.
    plot_color : str, optional
        Color of the data traces.
    shade_color : list or str, optional
        Background shading per channel; see :func:`_shade_y_ticks_background`.
    shade_alpha : float, optional
        Alpha for background shading.
    empty : bool, optional
        Remove plot borders and ticks if True.
    dr : float, optional
        Vertical spacing between channels; auto if None.
    fig_size : tuple, optional
        Figure size; auto if None.
    minmax : bool, optional
        Z-score-normalize data before plotting.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    if minmax:
        data = data.apply(sc.stats.zscore)
    if t is None:
        t = np.arange(len(data)) / fs

    t += t_offset
    if data.shape[0] != np.size(t):
        data = data.T

    n_rows = data.shape[1]
    duration = t[-1] - t[0]

    if fig_size is not None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig, ax = plt.subplots(figsize=(duration / 3, n_rows / 5))

    sns.despine()

    ticklocs = []
    ax.set_xlim(t[0], t[-1])

    dmin = data.min().min()
    dmax = data.max().min()

    if dr is None:
        dr = (dmax - dmin) * 0.8

    y0 = dmin - dr
    y1 = (n_rows - 1) * dr + dmax + dr / 2
    ax.set_ylim([y0, y1])
    segs = []

    for i in range(n_rows):
        if isinstance(data, pd.DataFrame):
            segs.append(np.column_stack((t, data.iloc[:, i])))
        elif isinstance(data, np.ndarray):
            segs.append(np.column_stack((t, data[:, i])))
        else:
            print("Data is not in valid format")

    for i in reversed(range(n_rows)):
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    ax.set_yticks(ticklocs)
    if isinstance(data, pd.DataFrame):
        ax.set_yticklabels(data.columns)

    if colors:
        for col, lab in zip(colors, ax.get_yticklabels()):
            if col is None:
                col = "black"
            lab.set_color(col)

    ax.set_xlabel("Time (s)")

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    ax.plot(t, data + ticklocs, color=plot_color, lw=0.4)

    if shade_color is not None:
        _shade_y_ticks_background(ax, ticklocs, shade_color, alpha=shade_alpha)

    if empty:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.tick_params(axis="both", which="both", length=0)

    return fig, ax
