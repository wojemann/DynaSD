import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc
from typing import Union

def _shade_y_ticks_background(ax, y_ticks, colors, alpha=1):
    """
    Add colored background shading to a plot based on y-axis tick positions.
    
    This function creates horizontal colored bands behind plot data, with each band
    centered on a y-tick position. Useful for visually grouping or highlighting
    different data series in multi-channel plots.
    
    Args:
        ax (matplotlib.axes.Axes): The axis object to modify
        y_ticks (list or array): Y-tick values to center the shading bands on
        colors (list, array, or str): Colors for each shading band. Can be:
            - List of colors (one per y-tick)
            - Single color string (applied to all y-ticks)
            - None values skip shading for that y-tick
        alpha (float, optional): Transparency level (0=transparent, 1=opaque). 
            Defaults to 1.
            
    Notes:
        - Shading extends halfway to adjacent y-ticks on each side
        - For edge ticks, extends same distance as to nearest neighbor
        - Uses axhspan() to create horizontal spans across full x-axis width
        - Automatically sorts y-ticks to ensure proper ordering
        
    Raises:
        ValueError: If y_ticks and colors arrays have different lengths (when colors is a list)
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> y_positions = [0, 1, 2, 3]
        >>> channel_colors = ['red', 'blue', None, 'green']  # No shading for channel 2
        >>> shade_y_ticks_background(ax, y_positions, channel_colors, alpha=0.3)
    """
    if isinstance(colors,str):
        colors = [colors]*len(y_ticks)
    if len(y_ticks) != len(colors):
        raise ValueError("The length of y_ticks and colors must be the same.")

    # Sort y_ticks and colors together to ensure proper ordering
    sorted_indices = np.argsort(y_ticks)
    y_ticks = np.array(y_ticks)[sorted_indices]
    colors = np.array(colors)[sorted_indices]

    # Add shading between each pair of y-ticks
    for i in range(len(y_ticks)):
        if colors[i] is not None:
            if i == 0:
                # First tick: shade from this tick down to halfway to the next tick
                lower_bound = y_ticks[i] - (y_ticks[i + 1] - y_ticks[i]) / 2
            else:
                # Shade from halfway between this tick and the previous one
                lower_bound = (y_ticks[i - 1] + y_ticks[i]) / 2

            if i == len(y_ticks) - 1:
                # Last tick: shade up to halfway to the previous tick
                upper_bound = y_ticks[i] + (y_ticks[i] - y_ticks[i - 1]) / 2
            else:
                # Shade up to halfway between this tick and the next one
                upper_bound = (y_ticks[i] + y_ticks[i + 1]) / 2

            # Add a colored rectangle spanning the full x-axis width, avoiding overlap
            ax.axhspan(lower_bound, upper_bound, color=colors[i], alpha=alpha, linewidth=0)

def plot_iEEG_data(
    data: Union[pd.DataFrame, np.ndarray], 
    fs=None,
    t=None,
    t_offset=0,
    colors=None,
    plot_color = 'k',
    shade_color = None,
    shade_alpha = 0.3,
    empty=False,
    dr=None,
    fig_size=None,
    minmax=False
):
    """
    Create a multi-channel iEEG data plot with customizable styling and layout.
    
    This function generates a standard "waterfall" or "butterfly" plot for multi-channel
    iEEG data, where each channel is plotted at a different y-offset for easy visualization.
    Supports various customization options including color coding, background shading,
    and automatic scaling.
    
    Args:
        data (pandas.DataFrame or numpy.ndarray): iEEG data matrix
            - If DataFrame: Uses column names as channel labels
            - Shape should be (time_points, channels) or (channels, time_points)
        fs (float, optional): Sampling frequency in Hz. Required if t is None.
        t (numpy.ndarray, optional): Time vector. If None, generated from fs.
        t_offset (float, optional): Time offset to add to time vector. Defaults to 0.
        colors (list, optional): List of colors for y-tick labels (channel names).
        plot_color (str, optional): Color for the data traces. Defaults to 'k' (black).
        shade_color (list, optional): Colors for background shading. See shade_y_ticks_background().
        shade_alpha (float, optional): Alpha transparency for background shading. Defaults to 0.3.
        empty (bool, optional): If True, removes plot borders and ticks for clean appearance.
        dr (float, optional): Vertical spacing between channels. If None, auto-calculated.
        fig_size (tuple, optional): Figure size (width, height). If None, auto-calculated.
        minmax (bool, optional): If True, z-score normalizes the data. Defaults to False.
        
    Returns:
        tuple: (fig, ax) - matplotlib figure and axis objects
        
    Notes:
        - Automatically transposes data if dimensions don't match time vector
        - Auto-calculates figure size based on duration and channel count
        - Channel labels appear as y-tick labels (if DataFrame input)
        - Supports both normalized and raw data display
        - Can overlay colored background shading for channel grouping
        
    Example:
        >>> data = pd.DataFrame(ieeg_data, columns=channel_names)
        >>> fig, ax = plot_iEEG_data(data, fs=500, colors=channel_colors)
        >>> plt.show()
    """
    if minmax:
        data = data.apply(sc.stats.zscore)
    if t is None:
        t = np.arange(len(data))/fs

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
        dr = (dmax - dmin) * 0.8  # Crowd them a bit.
    # if minmax and (dr is None):
    #     dr = 1

    y0 = dmin - dr
    y1 = (n_rows-1) * dr + dmax + dr/2
    ax.set_ylim([y0,y1])
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

    # # Set the yticks to use axes coordinates on the y axis
    ax.set_yticks(ticklocs)
    if isinstance(data, pd.DataFrame):
        ax.set_yticklabels(data.columns)

    if colors:
        for col, lab in zip(colors, ax.get_yticklabels()):
            if col is None:
                col = 'black'
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

        # Optionally, remove grid lines if present
        ax.grid(False)

        # Keep tick labels but remove tick markers
        ax.tick_params(axis='both', which='both', length=0)

    return fig, ax