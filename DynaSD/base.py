from sys import int_info
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
    
    # def get_win_times(self, n_samples):
    #     win_len = int(self.w_size * self.fs)
    #     step = int(self.w_stride * self.fs)
    #     n_windows = (n_samples - win_len) // step + 1
    #     return np.arange(n_windows) * self.w_stride

    def get_win_times(self, n_samples):
        data_len = n_samples / self.fs
        n_windows = np.floor((data_len - self.w_size)/self.w_stride) + 1
        return np.arange(n_windows) * self.w_stride
    
    def get_onset_and_spread(self, sz_prob, threshold=None,
                            ret_smooth_mat=False,
                            filter_w=10,  # seconds
                            rwin_size=5,  # seconds
                            rwin_req=4    # seconds
                            ):
        if threshold is None:
            threshold = self.threshold
        
        filter_w_idx = np.floor((filter_w - self.w_size)/self.w_stride).astype(int) + 1

        sz_prob = pd.DataFrame(sc.ndimage.uniform_filter1d(sz_prob, size=filter_w_idx, mode='nearest', axis=0, origin=0), columns=sz_prob.columns)

        sz_clf = (sz_prob > threshold).reset_index(drop=True)
        # sz_clf = pd.DataFrame(sc.ndimage.median_filter(sz_clf, size=filter_w_idx, mode='nearest', axes=0, origin=0), columns=sz_prob.columns)
        seized_idxs = np.any(sz_clf, axis=0)
        rwin_size_idx = np.floor((rwin_size - self.w_size)/self.w_stride).astype(int) + 1
        rwin_req_idx = np.floor((rwin_req - self.w_size)/self.w_stride).astype(int) + 1
        
        # Use convolution for faster sliding window computation
        if len(sz_clf) > rwin_size_idx - 1:
            kernel = np.ones(rwin_size_idx)
            sz_spread_data = np.zeros((len(sz_clf) - rwin_size_idx + 1, sz_clf.shape[1]))

            for i, col in enumerate(sz_clf.columns):
                sliding_sums = np.convolve(sz_clf[col].astype(int), kernel, mode='valid')
                sz_spread_data[:, i] = (sliding_sums >= rwin_req_idx).astype(int)

            sz_spread_idxs_all = pd.DataFrame(sz_spread_data, columns=sz_clf.columns)

            # Pad at the END with the last valid row (not zeros)
            missing_rows = rwin_size_idx - 1
            if len(sz_spread_idxs_all) > 0:
                last_valid_row = sz_spread_idxs_all.iloc[-1]
                padding = pd.DataFrame([last_valid_row] * missing_rows, columns=sz_spread_idxs_all.columns)
                sz_spread_idxs_all_padded = pd.concat([sz_spread_idxs_all, padding], ignore_index=True)
            
            else:
                # Handle edge case where convolution produces no output
                sz_spread_idxs_all_padded = pd.DataFrame(np.zeros((len(sz_clf), len(sz_clf.columns))), columns=sz_clf.columns)
            sz_clf_ff = sz_spread_idxs_all_padded # * sz_clf # This effectively undoes all of the convolution that we just did
        else:
            sz_clf_ff = sz_clf
        
        # Forward-fill logic continues unchanged...
        sz_spread_idxs = sz_clf_ff.loc[:,seized_idxs]
        extended_seized_idxs = np.any(sz_spread_idxs,axis=0)
        first_sz_idxs = sz_spread_idxs.loc[:,extended_seized_idxs].idxmax(axis=0)
        
        if sum(extended_seized_idxs) > 0:
            # Get indices into the sz_prob matrix and times since start of matrix that the seizure started
            sz_idxs_arr = np.array(first_sz_idxs)
            sz_order = np.argsort(first_sz_idxs)
            sz_idxs_arr = first_sz_idxs.iloc[sz_order].to_numpy()
            sz_ch_arr = first_sz_idxs.index[sz_order].to_numpy()
            # print(sz_prob.columns[seized_idxs])
            
        else:
            sz_ch_arr = []
            sz_idxs_arr = np.array([])

        sz_idxs_df = pd.DataFrame(sz_idxs_arr.reshape(1,-1),columns=sz_ch_arr)

        # Filling in non-seizing channels with 0 or na
        undetected_chs = [col for col in sz_prob.columns if col not in sz_ch_arr]
        sz_idxs_df[undetected_chs] = np.nan
        sz_clf_ff[undetected_chs] = 0


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
            X = sz_prob.iloc[:,i_ch].fillna(method='ffill').to_numpy()
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
    
    def _get_pretrained_threshold(self):
        return None
    
    def _aggregate_threshold(self, boundaries, method):
        """
        Helper function to aggregate channel boundaries into final threshold.
        To utilize this function, you need to set the self._boundary attribute in the subclass as well as the self._get_pretrained_threshold() function.
        """

        if method == 'mean':
            return np.nanmean(boundaries) + np.nanstd(boundaries)
        elif method == 'automean':
            if np.sum(boundaries > self._boundary) == 0:
                return self._get_pretrained_threshold()
            else:
                return np.nanmean(boundaries[boundaries > self._boundary])
        elif method == 'automedian':
            if np.sum(boundaries > self._boundary) == 0:
                return self._get_pretrained_threshold()
            else:
                return np.nanmedian(boundaries[boundaries > self._boundary])
        elif method == 'meanover':
            return np.nanmean(boundaries[boundaries > np.nanmean(boundaries)])
        elif method == 'medianover':
            return np.nanmedian(boundaries[boundaries > np.nanmedian(boundaries)])
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def get_threshold(self, sz_prob, method='automedian', verbose=False, seed=100, threshold_agg='median'):
        """
        Calculate seizure detection threshold using specified method.
        
        Parameters:
        -----------
        sz_prob : pandas.DataFrame
            Seizure probability matrix with channels as columns
        method : str, default='automedian'
            Threshold calculation method:
            - 'pretrained': Constant value trained on a large seizure onset annotated dataset
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
        self.threshold_agg = threshold_agg
        if method == 'pretrained':
            self.threshold = self._get_pretrained_threshold()
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
            raise ValueError(f"Unknown method '{method}'. Choose from: 'pretrained', 'automedian', 'automean', 'mean', 'meanover', 'medianover'")
    
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
