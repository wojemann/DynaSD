import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from .utils import moving_win_clips
from .base import DynaSDBase

class ABSSLP(DynaSDBase):
    """
    ABSSLP (Absolute Slope).

    Per-window mean absolute slope, normalized by per-channel baseline
    standard deviation and scaled by sampling frequency.

    Reference
    ---------
    Schindler, K., Leung, H., Elger, C. E., and Lehnertz, K., 2007.
    Assessing seizure dynamics by analysing the correlation structure of
    multichannel intracranial EEG. Brain, 130(1), pp.65-77.
    """
    def __init__(self, w_size=1, w_stride=0.5, fs=256):
        """
        Parameters
        ----------
        w_size : float, default=1
            Window size (seconds) for per-window slope.
        w_stride : float, default=0.5
            Window stride (seconds) between consecutive windows.
        fs : int, default=256
            Sampling frequency (Hz) of input iEEG.
        """
        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride)
        self.function = lambda x: np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)

    def __str__(self) -> str:
        return "AbsSlp"

    def fit(self, X):
        """
        Fit scaler and store baseline per-channel standard deviations.

        Parameters
        ----------
        X : pandas.DataFrame
            Baseline (interictal) iEEG of shape (n_samples, n_channels)
            with channel names as columns.
        """
        self._fit_scaler(X)
        nx = self.scaler.transform(X)
        self.inter = pd.DataFrame(nx, columns=X.columns)
        self.nstds = np.std(nx, axis=0)
        self.is_fitted = True

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
        n_windows = num_wins(n_samples, self.fs, self.w_size, self.w_stride)
        
        # Initialize output array
        slopes = np.zeros((n_channels, n_windows))
        
        # Apply function to each channel
        for ch in range(n_channels):
            # Get windowed clips for this channel
            clips = moving_win_clips(x[ch, :], self.fs, self.w_size, self.w_stride)
            
            # Apply the slope function to each window
            for i, clip in enumerate(clips):
                slopes[ch, i] = self.function(clip)
        
        return slopes

    def forward(self, X):
        """
        Run inference on new data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input iEEG of shape (n_samples, n_channels) with channel
            names as columns.

        Returns
        -------
        pandas.DataFrame
            Per-window normalized absolute slope of shape
            (n_windows, n_channels). Row index = realized window-start
            times (seconds). Columns = X.columns.
        """
        assert self.is_fitted, "Must fit model before running inference"

        x_scaled_df = self._scaler_transform(X)
        x_scaled = x_scaled_df.values.T

        slopes = self._apply_windowed_function(x_scaled)

        scaled_slopes = slopes / self.nstds.reshape(-1, 1) * self.fs
        scaled_slopes = scaled_slopes.squeeze()

        # Index is realized window-start times in seconds (spec section 5).
        features_df = pd.DataFrame(
            scaled_slopes.T,
            columns=X.columns,
            index=self.get_win_index(len(X)),
        )

        return features_df
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def _get_pretrained_threshold(self):
        if self.threshold_agg == 'median':
            # threshold = 242.8681542 # f1 median threshold
            threshold = 249.5898951 # f1 median threshold from plateau method
        elif self.threshold_agg == 'mean':
            # threshold =  265.9610556 # phi mean threshold
            threshold = 291.1243992 # phi mean threshold from plateau method
        self._threshold = threshold
        return self._threshold
        