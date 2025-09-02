import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from .utils import MovingWinClips

class AbsoluteSlope:
    """
    Absolute slope-based seizure detection algorithm.
    
    Computes seizure probability based on the absolute value of signal derivatives.
    Uses sliding windows to calculate the mean absolute slope for each channel,
    normalized by baseline standard deviations. Higher slopes typically indicate
    seizure activity with rapid voltage changes.
    
    Features:
    - RobustScaler normalization for artifact resistance
    - Configurable window size and stride
    - Baseline normalization using interictal data statistics
    """
    def __init__(self, win_size=1, stride=0.5, fs=256):
        self.function = lambda x: np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)
        self.win_size = win_size
        self.stride = stride
        self.fs = fs
    
    def __str__(self) -> str:
        return "AbsSlp"
        
    def fit(self, x):
        """
        Fit the model on interictal data.
        
        Parameters:
        -----------
        x : pd.DataFrame
            Training data with samples x channels format
        """
        # x should be samples x channels df
        self.scaler = RobustScaler().fit(x)
        nx = self.scaler.transform(x)
        self.inter = pd.DataFrame(nx, columns=x.columns)
        self.nstds = np.std(nx, axis=0)

    def get_times(self, x):
        """
        Get time points for each window.
        
        Parameters:
        -----------
        x : pd.DataFrame
            Input data with samples x channels format
            
        Returns:
        --------
        np.ndarray
            Array of time points for each window
        """
        # x should be samples x channels df
        time_mat = MovingWinClips(np.arange(len(x))/self.fs, self.fs, self.win_size, self.stride)
        return np.ceil(time_mat[:, -1])

    def _apply_windowed_function(self, x):
        """
        Apply the absolute slope function to windowed clips on each channel.
        
        Parameters:
        -----------
        x : np.ndarray
            Input data with channels x samples format
            
        Returns:
        --------
        np.ndarray
            Slopes for each channel and window
        """
        n_channels, n_samples = x.shape
        
        # Calculate number of windows
        from .utils import num_wins
        n_windows = num_wins(n_samples, self.fs, self.win_size, self.stride)
        
        # Initialize output array
        slopes = np.zeros((n_channels, n_windows))
        
        # Apply function to each channel
        for ch in range(n_channels):
            # Get windowed clips for this channel
            clips = MovingWinClips(x[ch, :], self.fs, self.win_size, self.stride)
            
            # Apply the slope function to each window
            for i, clip in enumerate(clips):
                slopes[ch, i] = self.function(clip)
        
        return slopes

    def forward(self, x):
        """
        Apply the fitted model to new data.
        
        Parameters:
        -----------
        x : pd.DataFrame
            Input data with samples x channels format
            
        Returns:
        --------
        np.ndarray
            Normalized slope features
        """
        # x is samples x channels df
        self.data = x
        x_scaled = self.scaler.transform(x)
        x_scaled = x_scaled.T  # Convert to channels x samples
        
        # Replace ft_extract with windowed function application
        slopes = self._apply_windowed_function(x_scaled)
        
        # Scale by standard deviations and sampling frequency
        scaled_slopes = slopes / self.nstds.reshape(-1, 1) * self.fs
        scaled_slopes = scaled_slopes.squeeze()
        
        return scaled_slopes / 1000
    
    def __call__(self, *args):
        return self.forward(*args) 