import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import welch
from sklearn.preprocessing import RobustScaler

from .base import DynaSDBase
from .utils import MovingWinClips

class HFER(DynaSDBase):
    """ 
    High frequency energy ratio onset detection algorithm

    Based on the paper: Bartolomei et al., 2008 - https://doi.org/10.1093/brain/awn111

    Computes the signal energy ratio across winows between high and low frequency bands of the EEG
    and detects the onset of the seizure for each channel based on the Page-Hinkley algorithm.
    """

    def __init__(self, fs=256, w_size=1, w_stride=0.5):
        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride)
        self.threshold = None
        self.freq_bands = {
            'theta': (3.5, 7.4),
            'alpha': (7.4, 12.4),
            'beta': (12.4, 24),
            'gamma': (24, 97),
        }
        
        # Algorithm specific parameters
        self.v = -0.5
        self.lambda_thresh = 15.0 
        
        self.tau = 1.0
        self.H = 5.0
        
    def get_threshold(self):
        return self.lambda_thresh

    def fit(self, x):
        """
        Fit the model on interictal data.
        """
        self.scaler = RobustScaler().fit(x)
        nx = self.scaler.transform(x)
        self.inter = pd.DataFrame(nx, columns=x.columns)

    def forward(self, X):
        channels = X.columns.to_list()
        
        all_features = []
        
        # Process each channel seperately
        for channel in channels:
            
            # calculate energy ratio
            channel_data = X[channel]
            er_data = self._compute_single_channel_ER(channel_data)
            all_features.append(er_data)

        # stack all channel results
        feature_master = np.array(all_features).T
        return pd.DataFrame(feature_master, columns=channels)

    def _compute_single_channel_ER(self, channel_data):
        windows = MovingWinClips(channel_data, self.fs, self.w_size, self.w_stride)
        band_powers = self._compute_band_powers(windows)
        
        band_names = list(self.freq_bands.keys())
        theta_idx = band_names.index('theta')
        alpha_idx = band_names.index('alpha') 
        beta_idx = band_names.index('beta')
        gamma_idx = band_names.index('gamma')

        # return high to low frequency energy ratio
        return (band_powers[:, beta_idx] + band_powers[:, gamma_idx]) / \
               (band_powers[:, theta_idx] + band_powers[:, alpha_idx])
    
    def _compute_band_powers(self, window_data):

        # Compute PSDs for all windows
        freqs, psds = welch(window_data, fs=self.fs, axis=1)
        
        # get frequency masks
        if not hasattr(self, '_freq_masks'):
            self._freq_masks = {}
            for band_name, (low, high) in self.freq_bands.items():
                self._freq_masks[band_name] = np.logical_and(freqs >= low, freqs <= high)
        
        # compute band powers for all windows
        dx = freqs[1] - freqs[0]
        all_band_powers = []
        
        for band_name in self.freq_bands.keys():
            mask = self._freq_masks[band_name]
            band_powers = simpson(psds[:, mask], dx=dx, axis=1)
            all_band_powers.append(band_powers)
        
        return np.column_stack(all_band_powers)