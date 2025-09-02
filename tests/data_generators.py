"""
Synthetic data generators for testing DynaSD models.

This module provides functions to generate realistic baseline and seizure-like
signals for unit testing of the DynaSD pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def generate_baseline_data(
    n_samples: int = 10240,  # 80 seconds at 128 Hz
    n_channels: int = 8,
    fs: int = 128,
    channel_names: List[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic baseline (interictal) EEG data.
    
    Creates random signal with characteristics similar to normal brain activity:
    - Low frequency background activity (1-4 Hz)
    - Alpha rhythm (8-12 Hz) 
    - Beta activity (13-30 Hz)
    - Small amount of noise
    
    Parameters:
    -----------
    n_samples : int, default=10240
        Number of time samples (80 seconds at 128 Hz)
    n_channels : int, default=8
        Number of EEG channels
    fs : int, default=128
        Sampling frequency in Hz
    channel_names : List[str], optional
        Custom channel names. If None, uses default naming
    seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Synthetic baseline data with samples x channels format
    """
    np.random.seed(seed)
    
    if channel_names is None:
        channel_names = [f'CH{i+1:02d}' for i in range(n_channels)]
    
    # Time vector
    t = np.arange(n_samples) / fs
    
    # Initialize data array
    data = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Base signal components with channel-specific variations
        base_freq = 2 + np.random.uniform(-0.5, 0.5)  # 1-4 Hz delta
        alpha_freq = 10 + np.random.uniform(-2, 2)    # 8-12 Hz alpha
        beta_freq = 20 + np.random.uniform(-5, 5)     # 15-25 Hz beta
        
        # Generate signal components
        delta = 0.8 * np.sin(2 * np.pi * base_freq * t + np.random.uniform(0, 2*np.pi))
        alpha = 0.5 * np.sin(2 * np.pi * alpha_freq * t + np.random.uniform(0, 2*np.pi))
        beta = 0.3 * np.sin(2 * np.pi * beta_freq * t + np.random.uniform(0, 2*np.pi))
        
        # Add noise and combine
        noise = np.random.normal(0, 0.2, n_samples)
        
        # Scale to realistic EEG amplitude range (µV)
        signal = (delta + alpha + beta + noise) * (50 + np.random.uniform(-20, 20))
        
        data[:, ch] = signal
    
    return pd.DataFrame(data, columns=channel_names)


def generate_seizure_data(
    n_samples: int = 5120,  # 40 seconds at 128 Hz  
    n_channels: int = 8,
    fs: int = 128,
    channel_names: List[str] = None,
    seizure_onset: float = 0.3,  # Seizure starts at 30% of signal
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic seizure EEG data.
    
    Creates random signal that transitions from baseline to seizure-like activity:
    - Initial baseline period
    - Seizure onset with increasing amplitude and frequency changes
    - Rhythmic seizure activity with higher amplitudes
    
    Parameters:
    -----------
    n_samples : int, default=5120
        Number of time samples (40 seconds at 128 Hz)
    n_channels : int, default=8
        Number of EEG channels
    fs : int, default=128
        Sampling frequency in Hz
    channel_names : List[str], optional
        Custom channel names. If None, uses default naming
    seizure_onset : float, default=0.3
        Fraction of signal where seizure begins (0-1)
    seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Synthetic seizure data with samples x channels format
    """
    np.random.seed(seed + 100)  # Different seed for seizure data
    
    if channel_names is None:
        channel_names = [f'CH{i+1:02d}' for i in range(n_channels)]
    
    # Time vector and seizure onset point
    t = np.arange(n_samples) / fs
    onset_sample = int(seizure_onset * n_samples)
    
    # Initialize data array
    data = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Pre-seizure baseline (similar to baseline_data but shorter)
        baseline_freq = 2 + np.random.uniform(-0.5, 0.5)
        alpha_freq = 10 + np.random.uniform(-2, 2)
        
        # Pre-seizure signal
        pre_seizure = (
            0.6 * np.sin(2 * np.pi * baseline_freq * t[:onset_sample] + np.random.uniform(0, 2*np.pi)) +
            0.4 * np.sin(2 * np.pi * alpha_freq * t[:onset_sample] + np.random.uniform(0, 2*np.pi)) +
            np.random.normal(0, 0.15, onset_sample)
        ) * (40 + np.random.uniform(-15, 15))
        
        # Seizure characteristics
        seizure_freq = 4 + np.random.uniform(-1, 3)  # 3-7 Hz seizure rhythm
        seizure_samples = n_samples - onset_sample
        
        # Seizure signal with ramping amplitude
        seizure_t = t[onset_sample:]
        ramp = np.linspace(1, 3, seizure_samples)  # Amplitude increases
        
        seizure_signal = (
            ramp * 2.0 * np.sin(2 * np.pi * seizure_freq * seizure_t + np.random.uniform(0, 2*np.pi)) +
            ramp * 0.8 * np.sin(2 * np.pi * (seizure_freq * 2) * seizure_t + np.random.uniform(0, 2*np.pi)) +
            np.random.normal(0, 0.3, seizure_samples)
        ) * (80 + np.random.uniform(-30, 30))
        
        # Combine pre-seizure and seizure periods
        data[:onset_sample, ch] = pre_seizure
        data[onset_sample:, ch] = seizure_signal
    
    return pd.DataFrame(data, columns=channel_names)


def generate_test_datasets(
    baseline_duration: float = 80.0,  # seconds
    seizure_duration: float = 40.0,   # seconds
    fs: int = 128,
    n_channels: int = 8,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate paired baseline and seizure datasets for testing.
    
    Parameters:
    -----------
    baseline_duration : float, default=80.0
        Duration of baseline data in seconds
    seizure_duration : float, default=40.0
        Duration of seizure data in seconds
    fs : int, default=128
        Sampling frequency in Hz
    n_channels : int, default=8
        Number of channels
    seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (baseline_data, seizure_data) pair
    """
    baseline_samples = int(baseline_duration * fs)
    seizure_samples = int(seizure_duration * fs)
    
    # Use same channel names for both datasets
    channel_names = [f'CH{i+1:02d}' for i in range(n_channels)]
    
    baseline_data = generate_baseline_data(
        n_samples=baseline_samples,
        n_channels=n_channels,
        fs=fs,
        channel_names=channel_names,
        seed=seed
    )
    
    seizure_data = generate_seizure_data(
        n_samples=seizure_samples,
        n_channels=n_channels,
        fs=fs,
        channel_names=channel_names,
        seizure_onset=0.3,
        seed=seed
    )
    
    return baseline_data, seizure_data 