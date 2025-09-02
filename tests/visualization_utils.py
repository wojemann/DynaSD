"""
Visualization utilities for DynaSD test results.

This module provides functions to create heatmaps and other visualizations
of model outputs for testing and validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, Tuple


def create_feature_heatmap(
    features: Union[np.ndarray, pd.DataFrame],
    time_points: Optional[np.ndarray] = None,
    channel_names: Optional[list] = None,
    title: str = "Feature Heatmap",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "viridis",
    show_colorbar: bool = True
) -> plt.Figure:
    """
    Create a heatmap visualization of features/probabilities.
    
    Parameters:
    -----------
    features : np.ndarray or pd.DataFrame
        Feature matrix (channels x time_windows or time_windows x channels)
    time_points : np.ndarray, optional
        Time points for x-axis. If None, uses window indices
    channel_names : list, optional
        Channel names for y-axis. If None, uses generic names
    title : str, default="Feature Heatmap"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved
    figsize : tuple, default=(12, 8)
        Figure size (width, height) in inches
    cmap : str, default="viridis"
        Colormap for heatmap
    show_colorbar : bool, default=True
        Whether to show colorbar
        
    Returns:
    --------
    plt.Figure
        Generated figure object
    """
    # Convert to DataFrame if needed and ensure proper orientation
    if isinstance(features, pd.DataFrame):
        data = features.T if features.shape[0] > features.shape[1] else features
        if channel_names is None:
            channel_names = data.index.tolist()
    else:
        # Assume features is channels x time_windows
        if features.ndim == 1:
            features = features.reshape(1, -1)
        data = features
        if channel_names is None:
            channel_names = [f'CH{i+1:02d}' for i in range(data.shape[0])]
    
    # Create time axis
    if time_points is None:
        time_points = np.arange(data.shape[1])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(
        data,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest'
    )
    
    # Set labels and title
    ax.set_xlabel('Time Windows')
    ax.set_ylabel('Channels')
    ax.set_title(title)
    
    # Set y-axis ticks and labels
    ax.set_yticks(range(len(channel_names)))
    ax.set_yticklabels(channel_names)
    
    # Set x-axis ticks (show every 10th point for readability)
    n_time = len(time_points)
    step = max(1, n_time // 10)
    x_ticks = range(0, n_time, step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{time_points[i]:.1f}' for i in x_ticks])
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Feature Value')
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    return fig


def create_comparison_plot(
    baseline_features: Union[np.ndarray, pd.DataFrame],
    seizure_features: Union[np.ndarray, pd.DataFrame],
    baseline_times: Optional[np.ndarray] = None,
    seizure_times: Optional[np.ndarray] = None,
    channel_names: Optional[list] = None,
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create side-by-side heatmap comparison of baseline vs seizure features.
    
    Parameters:
    -----------
    baseline_features : np.ndarray or pd.DataFrame
        Baseline feature matrix
    seizure_features : np.ndarray or pd.DataFrame
        Seizure feature matrix
    baseline_times : np.ndarray, optional
        Time points for baseline data
    seizure_times : np.ndarray, optional
        Time points for seizure data
    channel_names : list, optional
        Channel names
    model_name : str, default="Model"
        Name of the model for titles
    save_path : str or Path, optional
        Path to save the figure
    figsize : tuple, default=(15, 10)
        Figure size
        
    Returns:
    --------
    plt.Figure
        Generated figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data for both plots
    def prepare_data(features):
        if isinstance(features, pd.DataFrame):
            return features.T if features.shape[0] > features.shape[1] else features
        else:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            return features
    
    baseline_data = prepare_data(baseline_features)
    seizure_data = prepare_data(seizure_features)
    
    if channel_names is None:
        channel_names = [f'CH{i+1:02d}' for i in range(baseline_data.shape[0])]
    
    # Find global color scale for consistent comparison
    vmin = min(np.min(baseline_data), np.min(seizure_data))
    vmax = max(np.max(baseline_data), np.max(seizure_data))
    
    # Baseline heatmap
    im1 = ax1.imshow(baseline_data, aspect='auto', cmap='viridis', 
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title(f'{model_name} - Baseline Data')
    ax1.set_xlabel('Time Windows')
    ax1.set_ylabel('Channels')
    ax1.set_yticks(range(len(channel_names)))
    ax1.set_yticklabels(channel_names)
    
    # Seizure heatmap
    im2 = ax2.imshow(seizure_data, aspect='auto', cmap='viridis',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax2.set_title(f'{model_name} - Seizure Data')
    ax2.set_xlabel('Time Windows')
    ax2.set_ylabel('Channels')
    ax2.set_yticks(range(len(channel_names)))
    ax2.set_yticklabels(channel_names)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Feature Value')
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def create_summary_stats_plot(
    features: Union[np.ndarray, pd.DataFrame],
    channel_names: Optional[list] = None,
    title: str = "Feature Summary Statistics",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create summary statistics plot (mean, std, min, max per channel).
    
    Parameters:
    -----------
    features : np.ndarray or pd.DataFrame
        Feature matrix
    channel_names : list, optional
        Channel names
    title : str
        Plot title
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Generated figure object
    """
    # Prepare data
    if isinstance(features, pd.DataFrame):
        data = features.T if features.shape[0] > features.shape[1] else features
    else:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        data = features
    
    if channel_names is None:
        channel_names = [f'CH{i+1:02d}' for i in range(data.shape[0])]
    
    # Calculate statistics
    stats = pd.DataFrame({
        'Mean': np.mean(data, axis=1),
        'Std': np.std(data, axis=1),
        'Min': np.min(data, axis=1),
        'Max': np.max(data, axis=1)
    }, index=channel_names)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    stats.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Channels')
    ax.set_ylabel('Feature Value')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary stats plot saved to: {save_path}")
    
    return fig


def create_comprehensive_model_plot(
    baseline_data: pd.DataFrame,
    seizure_data: pd.DataFrame,
    baseline_features: Union[np.ndarray, pd.DataFrame],
    seizure_features: Union[np.ndarray, pd.DataFrame],
    model_name: str,
    fs: int = 128,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive 4-subplot figure showing baseline/seizure data and model features.
    
    Parameters:
    -----------
    baseline_data : pd.DataFrame
        Baseline EEG data (samples x channels)
    seizure_data : pd.DataFrame
        Seizure EEG data (samples x channels)
    baseline_features : pd.DataFrame
        Baseline model features/probabilities (windows x channels format)
    seizure_features : pd.DataFrame  
        Seizure model features/probabilities (windows x channels format)
    model_name : str
        Name of the model (e.g., "ABSSLP", "NDD", "WaveNet")
    fs : int, default=128
        Sampling frequency in Hz
    save_path : str or Path, optional
        Path to save the figure
    figsize : tuple, default=(16, 12)
        Figure size
        
    Returns:
    --------
    plt.Figure
        Generated figure object
    """
    # Import plot_iEEG_data from DynaSD utils
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from DynaSD.utils import plot_iEEG_data
    
    # Create 2x2 subplot layout
    fig = plt.figure(figsize=figsize)
    
    # Top left: Baseline iEEG data using plot_iEEG_data
    ax1 = plt.subplot(2, 2, 1)

    # Create a smaller subset for visualization (first 4 channels, 30 seconds)
    baseline_subset = baseline_data.iloc[:,:]
    baseline_fig, baseline_ax = plot_iEEG_data(
        baseline_subset,
        fs=fs,
        plot_color='blue',
        fig_size=(6, 4),
        minmax=False
    )
    # Copy the plot content to our subplot
    for line in baseline_ax.get_lines():
        ax1.plot(line.get_xdata(), line.get_ydata(), color='blue', linewidth=0.5)
    ax1.set_title(f'{model_name} - Baseline EEG Data', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Channels')
    plt.close(baseline_fig)  # Close the temporary figure
    
    # Top right: Seizure iEEG data using plot_iEEG_data  
    ax2 = plt.subplot(2, 2, 2)
    # Create a smaller subset for visualization (first 4 channels, 30 seconds)
    seizure_subset = seizure_data.iloc[:, :]
    seizure_fig, seizure_ax = plot_iEEG_data(
        seizure_subset,
        fs=fs,
        plot_color='red',
        fig_size=(6, 4),
        minmax=False
    )
    # Copy the plot content to our subplot
    for line in seizure_ax.get_lines():
        ax2.plot(line.get_xdata(), line.get_ydata(), color='red', linewidth=0.5)
    ax2.set_title(f'{model_name} - Seizure EEG Data', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Channels')
    plt.close(seizure_fig)  # Close the temporary figure
    
    # Prepare feature data for heatmaps
    # All models should now return DataFrame with windows as rows, channels as columns
    def prepare_features(features, data):
        if isinstance(features, pd.DataFrame):
            # Features should be windows x channels, so transpose to get channels x windows for heatmap
            return features.T.values, features.columns.tolist()
        else:
            # If numpy array, convert to expected format
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            # Assume features is windows x channels, so transpose
            if features.shape[1] == len(data.columns):
                return features.T, data.columns.tolist()
            else:
                # Assume features is channels x windows (old format)
                return features, data.columns.tolist()
    
    baseline_feat_data, baseline_channels = prepare_features(baseline_features, baseline_data)
    seizure_feat_data, seizure_channels = prepare_features(seizure_features, seizure_data)
    
    # Find global color scale for consistent comparison
    vmin = min(np.min(baseline_feat_data), np.min(seizure_feat_data))
    vmax = max(np.max(baseline_feat_data), np.max(seizure_feat_data))
    
    # Bottom left: Baseline features heatmap
    ax3 = plt.subplot(2, 2, 3)
    im1 = ax3.imshow(baseline_feat_data, aspect='auto', cmap='Blues', 
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax3.set_title(f'{model_name} - Baseline Features', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Windows')
    ax3.set_ylabel('Channels')
    ax3.set_yticks(range(len(baseline_channels)))
    ax3.set_yticklabels(baseline_channels)
    
    # Bottom right: Seizure features heatmap
    ax4 = plt.subplot(2, 2, 4)
    im2 = ax4.imshow(seizure_feat_data, aspect='auto', cmap='Reds',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax4.set_title(f'{model_name} - Seizure Features', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time Windows')
    ax4.set_ylabel('Channels')
    ax4.set_yticks(range(len(seizure_channels)))
    ax4.set_yticklabels(seizure_channels)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Feature Value', fontsize=12)
    
    # plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for colorbar
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive model plot saved to: {save_path}")
    
    return fig 