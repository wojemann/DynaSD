import numpy as np
import pandas as pd
import scipy as sc
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

class DynaSDBase:
    def __init__(self, fs, w_size, w_stride, scaler_class=RobustScaler, scaler_kwargs={}):
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.is_fitted = False
        self.scaler_class = scaler_class
        self.scaler_kwargs = scaler_kwargs  # Additional keyword arguments for the scaler
    def _fit_scaler(self, x):
        self.scaler = self.scaler_class(**self.scaler_kwargs).fit(x)

    def _scaler_transform(self, x):
        col_names = x.columns
        return pd.DataFrame(self.scaler.transform(x),columns=col_names)
    
    def get_win_times(self, n_samples):
        win_len = int(self.w_size * self.fs)
        step = int(self.w_stride * self.fs)
        n_windows = (n_samples - win_len) // step + 1
        return np.arange(n_windows) * self.w_stride
    
    def get_onset_and_spread(self,sz_prob,threshold=None,
                             ret_smooth_mat = False,
                             filter_w = 10, # seconds
                             rwin_size = 5, # seconds
                             rwin_req = 4 # seconds
                             ): 
        if threshold is None:
            threshold = self.threshold

        sz_clf = (sz_prob>threshold).reset_index(drop=True)
        filter_w_idx = np.floor((filter_w - self.w_size)/self.w_stride).astype(int) + 1
        sz_clf = pd.DataFrame(sc.ndimage.median_filter(sz_clf,size=filter_w_idx,mode='nearest',axes=0,origin=0),columns=sz_prob.columns)
        seized_idxs = np.any(sz_clf,axis=0)
        rwin_size_idx = np.floor((rwin_size - self.w_size)/self.w_stride).astype(int) + 1
        rwin_req_idx = np.floor((rwin_req - self.w_size)/self.w_stride).astype(int) + 1
        
        # Use convolution for faster sliding window computation
        # Create convolution kernel (array of ones for sliding sum)
        kernel = np.ones(rwin_size_idx)
        
        # Apply convolution to each column to get sliding sums, then apply threshold
        sz_spread_data = np.zeros((len(sz_clf) - rwin_size_idx + 1, sz_clf.shape[1]))
        for i, col in enumerate(sz_clf.columns):
            # Convolve with 'valid' mode to match rolling behavior after dropna()
            sliding_sums = np.convolve(sz_clf[col].astype(int), kernel, mode='valid')
            # Apply threshold condition (equivalent to >= rwin_req_idx)
            sz_spread_data[:, i] = (sliding_sums >= rwin_req_idx).astype(int)
        
        sz_spread_idxs_all = pd.DataFrame(sz_spread_data, columns=sz_clf.columns)
        
        # Filling in missing values due to smoothing at end of feature matrix
        missing_rows = rwin_size_idx-1
        last_valid_row = sz_spread_idxs_all.iloc[-1]  # Last row of the smoothed matrix
        padding = pd.DataFrame([last_valid_row] * missing_rows, columns=sz_spread_idxs_all.columns)
        
        # Append the propagated values to restore alignment
        sz_spread_idxs_all_padded = pd.concat([sz_spread_idxs_all, padding], ignore_index=True)

        sz_clf_ff = sz_spread_idxs_all_padded.copy()

        # Forward-fill in window space for each channel separately
        for ch in sz_clf_ff.columns:
            for j in range(len(sz_clf_ff) - rwin_size_idx):
                if sz_spread_idxs_all_padded.at[j, ch]:  # If window j is classified as true
                    future_sum = np.sum(sz_spread_idxs_all_padded.loc[j:j + rwin_size_idx, ch])  # Count future true windows
                    if future_sum >= rwin_req_idx:  # If requirement met, propagate forward
                        sz_clf_ff.loc[j:j + rwin_size_idx, ch] = 1
        
        sz_spread_idxs = sz_clf_ff.loc[:,seized_idxs]
        extended_seized_idxs = np.any(sz_spread_idxs,axis=0)
        first_sz_idxs = sz_spread_idxs.loc[:,extended_seized_idxs].idxmax(axis=0)
        
        if sum(extended_seized_idxs) > 0:
            # Get indices into the sz_prob matrix and times since start of matrix that the seizure started
            sz_idxs_arr = np.array(first_sz_idxs)
            sz_order = np.argsort(first_sz_idxs)
            sz_idxs_arr = first_sz_idxs.iloc[sz_order].to_numpy()
            sz_ch_arr = first_sz_idxs.index[sz_order].to_numpy()
            
        else:
            sz_ch_arr = []
            sz_idxs_arr = np.array([])

        sz_idxs_df = pd.DataFrame(sz_idxs_arr.reshape(1,-1),columns=sz_ch_arr)

        if ret_smooth_mat:
            
            return sz_idxs_df,sz_clf_ff
        else:
            return sz_idxs_df    
    
    def _compute_gaussian_boundaries(self, sz_prob, verbose=False, seed=100):
        """
        Helper function to compute Gaussian mixture boundaries for each channel.
        
        Returns:
        --------
        np.array
            Array of boundary values for each channel (may contain NaN)
        """
        all_gbounds = []
        all_chs = []
        
        for i_ch in range(sz_prob.shape[1]):
            # Transform data to log space for better Gaussian fitting
            X = sz_prob.iloc[:,i_ch].to_numpy()
            X_f = np.log(X.reshape(-1,1)+1e-10)
            X_f = X_f[X_f < np.percentile(X_f,99.99)].reshape(-1,1)
            
            # Fit Gaussian mixtures with 1 and 2 components to test for bimodality
            bics = []
            for n in range(1,3):
                gmm = GaussianMixture(n_components=n, random_state=seed)
                gmm.fit(X_f)
                bics.append(gmm.bic(X_f))
            
            # If 1-component model is better, channel is unimodal (no clear threshold)
            if bics[0] < bics[1]:
                all_gbounds.append(np.nan)
                if verbose:
                    print(f'{sz_prob.columns[i_ch]}: unimodal channel')
                all_chs.append(sz_prob.columns[i_ch])
                continue

            # For bimodal data, find intersection point between two Gaussians
            means = gmm.means_.flatten()
            mu1, mu2 = means
            sigma1, sigma2 = np.sqrt(gmm.covariances_.flatten())
            pi1, pi2 = gmm.weights_

            # Solve quadratic equation for Gaussian intersection points
            A = (1 / (2 * sigma1**2)) - (1 / (2 * sigma2**2))
            B = (mu2 / sigma2**2) - (mu1 / sigma1**2)
            C = ((mu1**2 / (2 * sigma1**2)) - (mu2**2 / (2 * sigma2**2))
                - np.log((pi1 * sigma2) / (pi2 * sigma1)))

            boundaries = np.roots([A, B, C])
            meets_criteria = np.exp(boundaries[(boundaries > min(mu1,mu2)) & (boundaries < max(mu1,mu2))])
            
            if len(meets_criteria) < 1:
                all_gbounds.append(np.nan)
                if verbose:
                    print(f'{sz_prob.columns[i_ch]}: overlapping gaussians')
            else:
                boundary = meets_criteria[0]
                all_gbounds.append(boundary)
            all_chs.append(sz_prob.columns[i_ch])
        
        return np.array(all_gbounds)

    def _aggregate_threshold(self, boundaries, method):
        """
        Helper function to aggregate channel boundaries into final threshold.
        
        Parameters:
        -----------
        boundaries : np.array
            Array of boundary values (may contain NaN)
        method : str
            Aggregation method
            
        Returns:
        --------
        float
            Final threshold value
        """
        if method == 'mean':
            return np.nanmean(boundaries) + np.nanstd(boundaries)
        elif method == 'automean':
            if np.sum(boundaries > 1.194797045747334) == 0:
                return 1.479305740987984
            else:
                return np.nanmean(boundaries[boundaries > 1.1748070328441302])
        elif method == 'automedian':
            if np.sum(boundaries > 1.1725) == 0:
                return 1.479305740987984
            else:
                return np.nanmedian(boundaries[boundaries > 1.1748070328441302])
        elif method == 'meanover':
            return np.nanmean(boundaries[boundaries > np.nanmean(boundaries)])
        elif method == 'medianover':
            return np.nanmedian(boundaries[boundaries > np.nanmedian(boundaries)])
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def get_threshold(self, sz_prob, method='automedian', verbose=False, seed=100):
        """
        Calculate seizure detection threshold using specified method.
        
        Parameters:
        -----------
        sz_prob : pandas.DataFrame
            Seizure probability matrix with channels as columns
        method : str, default='automedian'
            Threshold calculation method:
            - 'cval': Constant value (1.479305740987984)
            - 'automedian': Gaussian mixture boundaries with auto median aggregation
            - 'automean': Gaussian mixture boundaries with auto mean aggregation  
            - 'mean': Gaussian mixture boundaries with mean + std aggregation
            - 'meanover': Mean of boundaries above overall mean
            - 'medianover': Median of boundaries above overall median
        verbose : bool, default=False
            Print diagnostic information during processing
        seed : int, default=100
            Random seed for reproducible Gaussian mixture fitting
            
        Returns:
        --------
        float
            Calculated threshold value
        """
        
        # For constant value method, return fixed threshold immediately
        if method == 'cval':
            self.threshold = 1.479305740987984
            return self.threshold
            
        # For Gaussian-based methods, compute boundaries and aggregate
        elif method in ['automedian', 'automean', 'mean', 'meanover', 'medianover']:
            # Compute Gaussian mixture boundaries for each channel
            boundaries = self._compute_gaussian_boundaries(sz_prob, verbose=verbose, seed=seed)
            
            # Aggregate boundaries into final threshold using specified method
            threshold = self._aggregate_threshold(boundaries, method)
            
            self.threshold = threshold
            return threshold
            
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'cval', 'automedian', 'automean', 'mean', 'meanover', 'medianover'")
    
    def fit(self,X):
        print("Must define a fit function")
        return None

    def forward(self,X):
        print("Must define a forward function")
        assert self.is_fitted, "Must fit model before running inference"
        return None
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def __str__(self):
        print('Base')
