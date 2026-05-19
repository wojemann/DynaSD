import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import welch
from sklearn.preprocessing import RobustScaler

from .base import DynaSDBase
from .utils import moving_win_clips

class HFER(DynaSDBase):
    """
    HFER (High Frequency Energy Ratio).

    Per-window ratio of high-band (beta+gamma) to low-band (theta+alpha)
    spectral energy. Core DynaSD pipeline uses the raw ratio as the
    per-window feature; the original paper also defines a Page-Hinkley
    onset rule whose parameters are preserved on the instance for
    advanced/optional use outside the core pipeline.

    Reference
    ---------
    Bartolomei, F., Chauvel, P., and Wendling, F., 2008. Epileptogenicity
    of brain structures in human temporal lobe epilepsy: a quantified
    study from intracerebral EEG. Brain, 131(7), pp.1818-1830.
    """

    def __init__(self, fs=256, w_size=1, w_stride=0.5):
        """
        Parameters
        ----------
        fs : int, default=256
            Sampling frequency (Hz) of input iEEG.
        w_size : float, default=1
            Window size (seconds) for per-window energy ratio.
        w_stride : float, default=0.5
            Window stride (seconds) between consecutive windows.

        Notes
        -----
        The instance also exposes Page-Hinkley parameters from the
        original paper (``v``, ``lambda_thresh``, ``tau``, ``H``).
        These are not consumed by the core DynaSD pipeline (which uses
        the per-window ratio directly via :meth:`forward`) but are
        preserved for users who want to reproduce the paper's onset rule.
        """
        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride)
        self.threshold = None
        self.freq_bands = {
            'theta': (3.5, 7.4),
            'alpha': (7.4, 12.4),
            'beta': (12.4, 24),
            'gamma': (24, 97),
        }

        # Page-Hinkley parameters from Bartolomei et al. 2008. Not used by
        # core DynaSD pipeline; preserved for reproducing the original
        # paper's onset rule.
        self.v = -0.5
        self.lambda_thresh = 15.0
        self.tau = 1.0
        self.H = 5.0

    def get_threshold(self):
        return self.lambda_thresh

    def fit(self, X):
        """
        Fit scaler on baseline data.

        Parameters
        ----------
        X : pandas.DataFrame
            Baseline (interictal) iEEG of shape (n_samples, n_channels)
            with channel names as columns.
        """
        self.scaler = RobustScaler().fit(X)
        nx = self.scaler.transform(X)
        self.inter = pd.DataFrame(nx, columns=X.columns)
        self.is_fitted = True

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
            Per-window high/low band energy ratio of shape
            (n_windows, n_channels). Row index = realized window-start
            times (seconds). Columns = X.columns.
        """
        assert self.is_fitted, "Must fit model before running inference"
        channels = X.columns.to_list()
        
        all_features = []
        
        # Process each channel seperately
        for channel in channels:
            
            # calculate energy ratio
            channel_data = X[channel]
            er_data = self._compute_single_channel_ER(channel_data)
            all_features.append(er_data)

        # stack all channel results; index is realized window-start times
        # in seconds (spec section 5).
        feature_master = np.array(all_features).T
        return pd.DataFrame(
            feature_master,
            columns=channels,
            index=self.get_win_index(len(X)),
        )

    def _compute_single_channel_ER(self, channel_data):
        windows = moving_win_clips(channel_data, self.fs, self.w_size, self.w_stride)
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

    def _get_pretrained_threshold(self):
        if self.threshold_agg == 'median':
            # threshold = 17.5158932 # f1 median threshold
            threshold = 17.71236242 # f1 median threshold from plateau method
        elif self.threshold_agg == 'mean':
            # threshold = 61.48649156 # phi mean threshold
            threshold = 70.21385149 # phi mean threshold from plateau method
        self._threshold = threshold
        return self._threshold