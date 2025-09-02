import numpy as np
import pandas as pd
from neurodsp.sim.combined import sim_combined, sim_peak_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.utils import create_times, set_random_seed


class SyntheticSeizureGenerator:
    """
    Generate synthetic multivariate neural signals with realistic baseline and seizure patterns.
    
    Uses neurodsp library to create neurologically plausible signals with different
    seizure onset patterns including polyspikes, spike-waves, and focal seizures.
    """
    
    def __init__(self, fs=256, random_seed=42):
        """
        Initialize the synthetic data generator.
        
        Args:
            fs (int): Sampling frequency in Hz
            random_seed (int): Random seed for reproducibility
        """
        self.fs = fs
        self.random_seed = random_seed
        set_random_seed(random_seed)
        
    def _create_neural_signal(self, peaks, bws, heights, exponent, scale, n_seconds):
        """
        Create a single channel neural signal with specified spectral properties.
        
        Args:
            peaks (list): Peak frequencies in Hz
            bws (list): Bandwidth for each peak
            heights (list): Height/amplitude for each peak
            exponent (float): Powerlaw exponent for aperiodic component
            scale (float): Overall scaling factor
            n_seconds (float): Duration in seconds
            
        Returns:
            numpy.ndarray: Generated signal
        """
        # Start with aperiodic powerlaw background
        sig = sim_powerlaw(n_seconds, self.fs, exponent=exponent, f_range=[0.5, None])
        
        # Add oscillatory peaks
        for peak, bw, height in zip(peaks, bws, heights):
            sig = sim_peak_oscillation(sig, self.fs, peak, bw, height)
        
        sig *= scale
        return sig
    
    def generate_baseline_signal(self, n_seconds=10, n_channels=8):
        """
        Generate multivariate baseline neural activity.
        
        Creates signals with typical alpha/beta oscillations and 1/f background
        characteristic of normal brain activity.
        
        Args:
            n_seconds (float): Duration in seconds
            n_channels (int): Number of channels to simulate
            
        Returns:
            pandas.DataFrame: Baseline signals (samples x channels)
        """
        signals = []
        
        for ch in range(n_channels):
            # Add slight variability across channels
            alpha_freq = 10 + np.random.normal(0, 1)  # 8-12 Hz alpha
            beta_freq = 25 + np.random.normal(0, 2)   # 20-30 Hz beta
            
            peaks = [alpha_freq, beta_freq]
            bws = [3 + np.random.uniform(-0.5, 0.5), 5 + np.random.uniform(-1, 1)]
            heights = [1 + np.random.uniform(-0.2, 0.2), 0.5 + np.random.uniform(-0.1, 0.1)]
            exponent = -1.5 + np.random.uniform(-0.2, 0.2)  # Typical 1/f slope
            scale = 1 + np.random.uniform(-0.1, 0.1)
            
            sig = self._create_neural_signal(peaks, bws, heights, exponent, scale, n_seconds)
            signals.append(sig)
        
        signals = np.column_stack(signals)
        
        # Create channel names
        ch_names = [f'Ch{i+1:02d}' for i in range(n_channels)]
        
        return pd.DataFrame(signals, columns=ch_names)
    
    def generate_polyspike_seizure(self, n_seconds=5, n_channels=8, focal_channels=None):
        """
        Generate polyspike seizure pattern.
        
        Characterized by multiple high-frequency spike components,
        typical of generalized tonic-clonic seizures.
        
        Args:
            n_seconds (float): Duration in seconds
            n_channels (int): Number of channels
            focal_channels (list): Channels with stronger seizure activity (None for all)
            
        Returns:
            pandas.DataFrame: Polyspike seizure signals
        """
        if focal_channels is None:
            focal_channels = list(range(n_channels))
        
        signals = []
        
        for ch in range(n_channels):
            if ch in focal_channels:
                # Strong polyspike activity
                peaks = [15, 25, 35]  # Multiple spike frequencies
                bws = [3, 4, 5]
                heights = [1.5, 1.2, 0.8]  # High amplitude
                exponent = -0.5  # Flatter spectrum during seizure
                scale = 3 + np.random.uniform(-0.5, 0.5)
            # else:
            #     # Weaker propagated activity
            #     peaks = [15, 25]
            #     bws = [4, 6]
            #     heights = [0.8, 0.6]
            #     exponent = -1.0
            #     scale = 1.5 + np.random.uniform(-0.3, 0.3)
            
            sig = self._create_neural_signal(peaks, bws, heights, exponent, scale, n_seconds)
            signals.append(sig)
        
        signals = np.column_stack(signals)
        ch_names = [f'Ch{i+1:02d}' for i in range(n_channels)]
        
        return pd.DataFrame(signals, columns=ch_names)
    
    def generate_spike_wave_seizure(self, n_seconds=5, n_channels=8, focal_channels=None):
        """
        Generate spike-wave seizure pattern.
        
        Characterized by 3-4 Hz spike-wave complexes,
        typical of absence seizures.
        
        Args:
            n_seconds (float): Duration in seconds
            n_channels (int): Number of channels
            focal_channels (list): Channels with stronger activity
            
        Returns:
            pandas.DataFrame: Spike-wave seizure signals
        """
        if focal_channels is None:
            focal_channels = list(range(n_channels))
        
        signals = []
        
        for ch in range(n_channels):
            if ch in focal_channels:
                # Classic 3 Hz spike-wave
                peaks = [3, 3.5]
                bws = [1, 1.5]
                heights = [1.2, 1.0]
                exponent = -1.2
                scale = 2.5 + np.random.uniform(-0.3, 0.3)
            else:
                # Weaker activity
                peaks = [3]
                bws = [1.5]
                heights = [0.8]
                exponent = -1.4
                scale = 1.2 + np.random.uniform(-0.2, 0.2)
            
            sig = self._create_neural_signal(peaks, bws, heights, exponent, scale, n_seconds)
            signals.append(sig)
        
        signals = np.column_stack(signals)
        ch_names = [f'Ch{i+1:02d}' for i in range(n_channels)]
        
        return pd.DataFrame(signals, columns=ch_names)
    
    def generate_focal_seizure(self, n_seconds=5, n_channels=8, focal_channels=None):
        """
        Generate focal seizure pattern.
        
        Characterized by high-frequency activity in focal channels
        with gradual propagation to other regions.
        
        Args:
            n_seconds (float): Duration in seconds
            n_channels (int): Number of channels
            focal_channels (list): Primary focal channels
            
        Returns:
            pandas.DataFrame: Focal seizure signals
        """
        if focal_channels is None:
            focal_channels = [0, 1]  # Default to first 2 channels
        
        signals = []
        
        for ch in range(n_channels):
            if ch in focal_channels:
                # High-frequency seizure activity
                peaks = [20, 40, 60]
                bws = [10, 15, 20]
                heights = [2.0, 1.5, 1.0]
                exponent = 0.5  # Very flat spectrum
                scale = 0.5 + np.random.uniform(-0.1, 0.1)
            # elif ch in [ch for ch in range(n_channels) if ch not in focal_channels][:2]:
            #     # Adjacent propagation
            #     peaks = [20, 40]
            #     bws = [12, 18]
            #     heights = [1.2, 0.8]
            #     exponent = -0.4
            #     scale = 1 + np.random.uniform(-0.3, 0.3)
            else:
                # Distant propagation or baseline
                peaks = [10, 25]
                bws = [5, 8]
                heights = [0.8, 0.5]
                exponent = -1.3
                scale = 1 + np.random.uniform(-0.2, 0.2)
            
            sig = self._create_neural_signal(peaks, bws, heights, exponent, scale, n_seconds)
            signals.append(sig)
        
        signals = np.column_stack(signals)
        ch_names = [f'Ch{i+1:02d}' for i in range(n_channels)]
        
        return pd.DataFrame(signals, columns=ch_names)
    
    def generate_combined_signal(self, baseline_duration=10, seizure_duration=5, 
                                seizure_type='polyspike', n_channels=8, 
                                focal_channels=None, transition_duration=1):
        """
        Generate combined baseline + seizure signal.
        
        Args:
            baseline_duration (float): Duration of baseline in seconds
            seizure_duration (float): Duration of seizure in seconds
            seizure_type (str): Type of seizure ('polyspike', 'spike_wave', 'focal')
            n_channels (int): Number of channels
            focal_channels (list): Focal channels for seizure
            transition_duration (float): Smooth transition duration in seconds
            
        Returns:
            tuple: (combined_signal_df, seizure_start_time, seizure_end_time)
        """
        # Generate baseline
        baseline = self.generate_baseline_signal(baseline_duration, n_channels)
        
        # Generate seizure
        if seizure_type == 'polyspike':
            seizure = self.generate_polyspike_seizure(seizure_duration, n_channels, focal_channels)
        elif seizure_type == 'spike_wave':
            seizure = self.generate_spike_wave_seizure(seizure_duration, n_channels, focal_channels)
        elif seizure_type == 'focal':
            seizure = self.generate_focal_seizure(seizure_duration, n_channels, focal_channels)
        else:
            raise ValueError(f"Unknown seizure type: {seizure_type}")
        
        # Create smooth transition
        if transition_duration > 0:
            n_transition = int(transition_duration * self.fs)
            transition_window = np.hanning(2 * n_transition)
            fade_out = transition_window[:n_transition]
            fade_in = transition_window[n_transition:]
            
            # Apply fade to end of baseline and start of seizure
            baseline.iloc[-n_transition:, :] *= fade_out.reshape(-1, 1)
            seizure.iloc[:n_transition, :] *= fade_in.reshape(-1, 1)
        
        # Concatenate signals
        combined = pd.concat([baseline, seizure], ignore_index=True)
        
        seizure_start_time = baseline_duration
        seizure_end_time = baseline_duration + seizure_duration
        
        return combined, seizure_start_time, seizure_end_time
    
    def add_noise(self, signal, snr_db=20):
        """
        Add realistic noise to signal.
        
        Args:
            signal (pandas.DataFrame): Input signal
            snr_db (float): Signal-to-noise ratio in dB
            
        Returns:
            pandas.DataFrame: Noisy signal
        """
        signal_power = np.mean(signal.var())
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        noisy_signal = signal + noise
        
        return pd.DataFrame(noisy_signal, columns=signal.columns) 