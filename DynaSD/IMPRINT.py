from scipy.special import erfcinv
from .base import DynaSDBase
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import welch
from .utils import MovingWinClips


class IMPRINT(DynaSDBase):
    def __init__(self, fs, onset_buffer=0, offset_buffer=0, w_size=1, w_stride=0.125):
        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride)
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 70),
            'high_gamma': (70, 120)
        }

        self.min_sz_dur = 9 # minimum seizure duration in seconds for inclusion

        self.window_size = 1
        self.det = 7 # count of windows beyond first detected onset which will also be considered as onset

        self.onset_buffer = onset_buffer  # Number of seconds to shift back clinically labelled onset time
        self.offset_buffer = offset_buffer  # Number of seconds to extend beyond seizure end

        self.prop_rec = 0.8 # proportion of the rec_thresh that is required for activity to be considered in onset
        self.rec_thresh = 9 # number of seconds for which activity mst persist to be considered a seizure
        self.rec_type = "sec" # type of threshold ot validation activity is ictal ("sec" or "prop")
        self.mad_thresh = 5 # MAD threshold for detection of activity
        self.ictal_buffer = 10 # Number of seconds to shift back clinically labelled onset time to ensure 'true' onset is captured
        self.channel_labels = None

        # Constant for MAD calculation
        self.mc = -1/(np.sqrt(2) * erfcinv(3/2))

    def _compute_line_length(self, window_data):
        differences = np.abs(np.diff(window_data, axis=1))
        return np.sum(differences, axis=1) / differences.shape[1]
    
    def _compute_energy(self, window_data, demean=True):
        if demean:
            window_data = window_data - np.mean(window_data, axis=1, keepdims=True)
        
        energy = np.sum(window_data**2, axis=1) / window_data.shape[1]
        return energy
        
    def _compute_band_powers(self, window_data):

        # Compute PSDs for all windows at once
        freqs, psds = welch(window_data, fs=self.fs, axis=1)
        
        # Pre-compute frequency masks
        if not hasattr(self, '_freq_masks'):
            self._freq_masks = {}
            for band_name, (low, high) in self.freq_bands.items():
                self._freq_masks[band_name] = np.logical_and(freqs >= low, freqs <= high)
        
        # Compute band powers for all windows
        dx = freqs[1] - freqs[0]
        all_band_powers = []
        
        for band_name in self.freq_bands.keys():
            mask = self._freq_masks[band_name]
            band_powers = simpson(psds[:, mask], dx=dx, axis=1)
            all_band_powers.append(band_powers)
        
        return np.column_stack(all_band_powers)

    def get_threshold(self, X=None):
        return self.mad_thresh
    
    def check_inclusion(self, X):
        include_data = True
        if len(X) < self.min_sz_dur * self.fs:
            print(f"Data length {len(X)} is less than minimum seizure duration of {self.min_sz_dur} seconds.")
            include_data = False
        
        return include_data
    
    def calc_features(self, X):
            """
            Calculate features for all channels.
            """
            data_ch = X.columns.to_list()
            data_np = X.to_numpy()
            
            n_channels = len(data_ch)
            channel_features = []
            
            # Calculate window times
            time_mat = MovingWinClips(np.arange(len(X))/self.fs, self.fs, self.w_size, self.w_stride)
            win_times = time_mat[:, 0]

            # Process each channel
            all_features = []
            
            for k in range(n_channels):
                # Extract all windows for this channel at once
                windows = MovingWinClips(data_np[:, k], self.fs, self.w_size, self.w_stride)
                n_windows = windows.shape[0]
                
                # Vectorized computations
                line_lengths = self._compute_line_length(windows)
                energies = self._compute_energy(windows)
                band_powers = self._compute_band_powers(windows)
                
                # Combine features: (n_windows, n_features)
                channel_features = np.column_stack([line_lengths, energies, band_powers])
                all_features.append(channel_features)
            
            # Shape: (n_channels, n_windows, n_features)
            features = np.log(np.array(all_features))
            
            return features, win_times
    
    def mahal_single(self, Y, X):
        """
        Python equivalent of MATLAB's mahal() function:
        D2(I) = (Y(I,:)-MU) * SIGMA^(-1) * (Y(I,:)-MU)'
        """
        m = np.mean(X, axis=0)
        C = X - m
        Q, R = np.linalg.qr(C) 
        ri = np.linalg.solve(R.T, (Y - m).T)
        return np.sum(ri * ri, axis=0) * (X.shape[0] - 1)

    def ictal_mad_score(self, ictal_features):
        """
        Calculate MAD scores for ictal data.
        """
        n_channels, n_windows = ictal_features.shape[:2]
        mahal_mad = np.empty((n_channels, n_windows))
        
        for channel in range(n_channels):
            # Get cleaned reference for this channel
            clean_ref = self.clean_reference_features[channel]
            
            # Calculate ictal Mahalanobis distances against clean reference
            ictal_mahal = self.mahal_single(ictal_features[channel], clean_ref)
            
            # Calculate baseline Mahalanobis distances
            ref_med = self.reference_stats[channel]['median']
            ref_smad = self.reference_stats[channel]['scaled_mad']
            
            if ref_smad == 0 or np.isnan(ref_smad):
                mahal_mad[channel, :] = np.nan
                continue
            
            # Calculate MAD scores
            mahal_mad[channel, :] = (ictal_mahal - ref_med) / ref_smad
        
        return mahal_mad

    def fit(self, X):
        """
        Fit the model using preictal data.
        """
        # Check inclusion criteria
        if not self.check_inclusion(X):
            raise ValueError("Preictal data does not meet inclusion criteria")

        self.channel_labels = X.columns
        
        # Calculate preictal features
        self.preictal_features, self.preictal_times = self.calc_features(X)
        
        # For each channel, clean and store the reference features
        self.clean_reference_features = {}
        self.reference_stats = {}
        
        for channel in range(self.preictal_features.shape[0]):
            channel_features = self.preictal_features[channel]
            
            # Calculate initial Mahalanobis distances
            initial_mahal = self.mahal_single(channel_features, channel_features)
            
            # Calculate MAD scores to identify outliers
            pre_med = np.nanmedian(initial_mahal)
            pre_smad = self.mc * np.nanmedian(np.abs(initial_mahal - pre_med))
            
            if pre_smad == 0 or np.isnan(pre_smad):
                clean_ref_features = channel_features
            else:
                mad_scores = (initial_mahal - pre_med) / pre_smad
                
                # Remove outlier windows
                outlier_mask = mad_scores >= self.mad_thresh
                clean_ref_features = channel_features[~outlier_mask]
                
                if len(clean_ref_features) < 2:
                    clean_ref_features = channel_features  # fallback
            
            # Store cleaned reference features
            self.clean_reference_features[channel] = clean_ref_features
            
            # Calculate and cache reference statistics on cleaned features
            if len(clean_ref_features) >= 2:
                ref_mahal = self.mahal_single(clean_ref_features, clean_ref_features)
                ref_med = np.nanmedian(ref_mahal)
                ref_smad = self.mc * np.nanmedian(np.abs(ref_mahal - ref_med))
            else:
                ref_med = 0
                ref_smad = 1
                
            self.reference_stats[channel] = {
                'median': ref_med,
                'scaled_mad': ref_smad
            }
        
        return self

    def forward(self, X):
        """
        Detect seizure onset in ictal data using the fitted preictal baseline.
        """

        # Check that model has been fitted
        if not hasattr(self, 'clean_reference_features'):
            raise ValueError("Model must be fitted before calling forward()")
        
        # Start analysis ictal buffer seconds before seizure onset
        analysis_start_time = self.onset_buffer - self.ictal_buffer
        
        # Get time array
        time_array = np.arange(len(X)) / self.fs
        
        # Extract data from analysis start time onward
        analysis_mask = time_array >= analysis_start_time
        X = X[analysis_mask].copy()
        
        # Check inclusion criteria  
        if not self.check_inclusion(X):
            raise ValueError("Analysis data does not meet inclusion criteria")
        
        # Calculate ictal features
        ictal_features, window_times = self.calc_features(X)
        
        # Calculate MAD scores
        mahal_mad = self.ictal_mad_score(ictal_features)

        feature_df = pd.DataFrame(mahal_mad.T, columns = X.columns)

        return feature_df
    
    # moving sums function
    def moving_sums(self, significant, recruitment_threshold):
        n = len(significant)
        ms_a = np.zeros(n)  # forward
        ms_b = np.zeros(n)  # backward  
        ms_c = np.zeros(n)  # center
        
        for i in range(n):
            # Forward: current + next (recruitment_threshold-1)
            end_idx = min(i + recruitment_threshold, n)
            ms_a[i] = np.sum(significant[i:end_idx])
            
            # Backward: previous (recruitment_threshold-1) + current
            start_idx = max(i - recruitment_threshold + 1, 0)
            ms_b[i] = np.sum(significant[start_idx:i+1])
            
            # Center: half window before + half after
            half = recruitment_threshold // 2
            center_start = max(i - half, 0)
            center_end = min(i + half + 1, n)
            ms_c[i] = np.sum(significant[center_start:center_end])
        
        return ms_a, ms_b, ms_c

    def get_onset_and_spread(self,sz_prob,threshold=None,
                            ret_smooth_mat=False,
                            filter_w = None, # seconds
                            rwin_size = None, # seconds
                            rwin_req = None # seconds
                            ):
    
        """
        Imprint-based onset detection with dsosd-compatible interface
        """
        
        # Use imprint parameters if not provided
        if threshold is None:
            threshold = self.mad_thresh
        if rwin_size is None:
            rwin_size = self.rec_thresh
        if rwin_req is None:
            rwin_req = self.rec_thresh * self.prop_rec
        
        # Calculate window parameters
        wl = self.w_stride
        recruitment_threshold = int(rwin_size / wl)
        
        # Apply imprint algorithm
        n_windows, n_channels = sz_prob.shape
        imprint = np.zeros((n_channels, n_windows), dtype=bool)
        
        # Process each channel with imprint logic
        for ch_idx, channel in enumerate(sz_prob.columns):
            # Skip NaN channels
            if np.all(np.isnan(sz_prob[channel].values)):
                continue
            
            # Binary threshold
            significant = sz_prob[channel].values >= threshold
            
            # Moving sums
            ms_a, ms_b, ms_c = self.moving_sums(significant, recruitment_threshold)
            
            # Imprint creation
            req_threshold = rwin_req / wl
            imprint[ch_idx, :] = ((ms_a >= req_threshold) | 
                                (ms_b >= req_threshold) | 
                                (ms_c >= req_threshold))
        
        # Extract onset indices
        seized_channels = []
        onset_indices = []
        
        for ch_idx, channel in enumerate(sz_prob.columns):
            channel_detections = np.where(imprint[ch_idx, :])[0]
            if len(channel_detections) > 0:
                seized_channels.append(channel)
                onset_indices.append(channel_detections[0])  # First detection
        
        # Create output in dsosd format
        if len(seized_channels) > 0:
            # Sort by onset time
            sz_order = np.argsort(onset_indices)
            sz_idxs_arr = np.array(onset_indices)[sz_order]
            sz_ch_arr = np.array(seized_channels)[sz_order]
            
            # Create DataFrame 
            sz_idxs_df = pd.DataFrame(sz_idxs_arr.reshape(1, -1), columns=sz_ch_arr)
        else:
            sz_idxs_df = pd.DataFrame()
        
        # Create imprint matrix (smoothed matrix equivalent)
        sz_clf_ff = pd.DataFrame(imprint.T.astype(float), columns=sz_prob.columns)
        
        if ret_smooth_mat:
            return sz_idxs_df, sz_clf_ff
        else:
            return sz_idxs_df