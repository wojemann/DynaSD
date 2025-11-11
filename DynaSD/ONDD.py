from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import scipy as sc
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class DSOSDbase:
    def __init__(self, fs, w_size, w_stride):
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.is_fitted = False
    
    def _fit_scaler(self, x):
        self.scaler = RobustScaler().fit(x)

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
        sz_spread_idxs_all = sz_clf.rolling(window=rwin_size_idx,center=False).apply(lambda x: (x == 1).sum()>=rwin_req_idx).dropna().reset_index(drop=True)
        
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
    
    def get_gaussian_threshold(self,sz_prob,seed=100):
        gmm = GaussianMixture(n_components=2,random_state=seed)
        X_f = np.log(sz_prob.to_numpy().reshape(-1,1)+1e-10)
        X_f = X_f[X_f < np.percentile(X_f,99.9)].reshape(-1,1)
        gmm.fit(X_f)
        means = gmm.means_.flatten()
        mu1, mu2 = means

        sigma1, sigma2 = np.sqrt(gmm.covariances_.flatten())
        pi1, pi2 = gmm.weights_

        # Coefficients of the quadratic equation
        A = (1 / (2 * sigma1**2)) - (1 / (2 * sigma2**2))
        B = (mu2 / sigma2**2) - (mu1 / sigma1**2)
        C = ((mu1**2 / (2 * sigma1**2)) - (mu2**2 / (2 * sigma2**2))
            - np.log((pi1 * sigma2) / (pi2 * sigma1)))

        # Solve for x (intersection points)
        boundaries = np.roots([A, B, C])
        meets_criteria = np.exp(boundaries[(boundaries > min(mu1,mu2)) & (boundaries < max(mu1,mu2))])
        if len(meets_criteria) < 1:
            threshold = max(np.exp(means))
        else:
            threshold = meets_criteria[0]
        self.threshold = threshold
        return threshold
    
    def get_gaussianx_threshold(self,sz_prob,noise_floor='automedian',verbose=False,seed=100):
        all_gbounds = []
        all_chs = []
        all_max_means = []
        for i_ch in range(sz_prob.shape[1]):
            X = sz_prob.iloc[:,i_ch].to_numpy()
            X_f = np.log(X.reshape(-1,1)+1e-10)
            X_f = X_f[X_f < np.percentile(X_f,99.99)].reshape(-1,1)
            bics = []
            for n in range(1,3):
                gmm = GaussianMixture(n_components=n,random_state=seed)
                gmm.fit(X_f)
                bics.append(gmm.bic(X_f))
            if bics[0]<bics[1]:
                all_gbounds.append(np.nan)
                all_chs.append(sz_prob.columns[i_ch])
                all_max_means.append(np.nan)
                if verbose:
                    print(f'{sz_prob.columns[i_ch]}: unimodal channel')
                continue

            means = gmm.means_.flatten()
            mu1, mu2 = means
            
            sigma1, sigma2 = np.sqrt(gmm.covariances_.flatten())
            pi1, pi2 = gmm.weights_

            # Coefficients of the quadratic equation
            A = (1 / (2 * sigma1**2)) - (1 / (2 * sigma2**2))
            B = (mu2 / sigma2**2) - (mu1 / sigma1**2)
            C = ((mu1**2 / (2 * sigma1**2)) - (mu2**2 / (2 * sigma2**2))
                - np.log((pi1 * sigma2) / (pi2 * sigma1)))

            # Solve for x (intersection points)
            boundaries = np.roots([A, B, C])
            meets_criteria = np.exp(boundaries[(boundaries > min(mu1,mu2)) & (boundaries < max(mu1,mu2))])
            if len(meets_criteria) < 1:
                all_gbounds.append(np.nan)
                all_max_means.append(np.nan)
                if verbose:
                    print(f'{sz_prob.columns[i_ch]}: overlapping gaussians')
            else:
                boundary = meets_criteria[0]
                all_gbounds.append(boundary)
                all_max_means.append(max(np.exp(means)))
            all_chs.append(sz_prob.columns[i_ch])
        all_gbounds = np.array(all_gbounds)
        if noise_floor == 'mean':
            threshold = np.nanmean(all_gbounds)+np.nanstd(all_gbounds)
        elif noise_floor == 'automean':
            if np.sum(all_gbounds > 1.194797045747334) == 0:
                threshold = 1.479305740987984
            else:
                threshold = np.nanmean(all_gbounds[all_gbounds > 1.1748070328441302])
        elif noise_floor == 'automedian':
            if np.sum(all_gbounds > 1.1725) == 0:
                threshold = 1.479305740987984
            else:
                threshold = np.nanmedian(all_gbounds[all_gbounds > 1.1748070328441302])
        elif noise_floor == 'meanover':
            threshold = np.nanmean(all_gbounds[all_gbounds > np.nanmean(all_gbounds)])
        elif noise_floor == 'medianover':
            threshold = np.nanmedian(all_gbounds[all_gbounds > np.nanmedian(all_gbounds)])
        else:
            threshold = np.nanmean(all_gbounds[all_gbounds > noise_floor])
        self.threshold = threshold
        return threshold

    def get_bayesian_threshold(self,sz_prob,percentile=90):
        all_baybs = []
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            for i_c in range(sz_prob.shape[1]):
                gmm = BayesianGaussianMixture(n_components=3)
                X_f = np.log(sz_prob.iloc[:,i_c].to_numpy().reshape(-1,1)+1e-10)
                X_f = X_f[X_f < np.percentile(X_f,99.9)].reshape(-1,1)
                gmm.fit(X_f)
                means = gmm.means_.flatten()
                mu_weights = gmm.weights_ > 0.05

                if sum(mu_weights) > 2:
                    sorted_means = np.sort(means)
                    largest_two_means = sorted_means[-2:]  # Last two in sorted list
                elif sum(mu_weights) == 1:
                    # print(f'{sz_prob.columns[i_c]}: no seizure activity detected')
                    all_baybs.append(max(means))
                    continue
                else:
                    largest_two_means = means[mu_weights]
                mu1, mu2 = largest_two_means  # Assign the two largest means
                index1, index2 = np.where(means == mu1)[0][0], np.where(means == mu2)[0][0]


                # Extract corresponding variances and weights
                sigma1, sigma2 = np.sqrt(gmm.covariances_.flatten()[[index1, index2]])
                pi1, pi2 = gmm.weights_[[index1, index2]]

                # Solve for intersection points between these two Gaussians
                A = (1 / (2 * sigma1**2)) - (1 / (2 * sigma2**2))
                B = (mu2 / sigma2**2) - (mu1 / sigma1**2)
                C = ((mu1**2 / (2 * sigma1**2)) - (mu2**2 / (2 * sigma2**2))
                    - np.log((pi1 * sigma2) / (pi2 * sigma1)))

                boundaries = np.roots([A, B, C])
                meets_criteria = np.exp(boundaries[(boundaries > min(mu1,mu2)) & (boundaries < max(mu1,mu2))])
                if len(meets_criteria) < 1:
                    bayboundary = max(np.exp(boundaries))
                else:
                    bayboundary = meets_criteria[0]
                all_baybs.append(np.abs(bayboundary))
        self.threshold = np.percentile(all_baybs,percentile)
        return self.threshold
    
    def get_cval_threshold(self):
            self.threshold = 1.479305740987984
            return self.threshold
    
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

class LiNDD(DSOSDbase):
    def __init__(self, fs = 128, w_size=1, w_stride = 0.5):
        super().__init__(fs=fs,w_size=w_size,w_stride=w_stride)
        self.model = LinearRegression(fit_intercept=False)
    
    def fit(self, X):
        self._fit_scaler(X)
        nX = self._scaler_transform(X)
        self.model.fit(nX.iloc[:-1,:],nX.iloc[1:,:])

    def forward(self, X):
        ch_names = X.columns
        nX = self._scaler_transform(X)
        y = self.model.predict(nX.iloc[:-1,:])
        se = pd.DataFrame((nX.iloc[1:,:]-y)**2,columns=ch_names)
        mse = se.rolling(int(self.w_size*self.fs),min_periods=int(self.w_size*self.fs),center=False).mean()
        mse_wins = mse.iloc[::int(self.w_stride*self.fs)].reset_index(drop=True)
        mse_wins = mse_wins[~mse_wins.isna().any(axis=1)]
        # smooth_mse_wins = pd.DataFrame(sc.ndimage.uniform_filter1d(mse_wins,20,axis=0,mode='constant'),columns=ch_names)
        return mse_wins

class NDDmodel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NDDmodel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1,:])
        return out
    
    def __str__(self):
         return "NDD" 

class NDDIso(nn.Module):
    def __init__(self, num_channels, hidden_size):
        super(NDDIso, self).__init__()
        self.num_channels = num_channels
        self.lstms = nn.ModuleList([nn.LSTM(1, hidden_size, batch_first=True) for _ in range(num_channels)])
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_channels)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_channels):
            out, _ = self.lstms[i](x[:, :, i].unsqueeze(-1))  # LSTM input shape: (batch_size, seq_len, 1)
            out = self.fcs[i](out[:, -1, :])  # FC input shape: (batch_size, hidden_size)
            outputs.append(out.unsqueeze(1))  # Add channel dimension back

        # Concatenate outputs along channel dimension
        output = torch.cat(outputs, dim=1).squeeze()  # shape: (batch_size, num_channels, 1)
        return output
    
    def __str__(self):
         return "NDDIso"

class NDD(DSOSDbase):
    def __init__(self, hidden_size = 10, fs = 128,
                  train_win = 12, pred_win = 1,
                  w_size = 1, w_stride = 0.5,
                  num_epochs = 10, batch_size = 'full',
                  lr = 0.01,
                  model_class = NDDmodel,
                  use_cuda = False,
                  val=False,
                  ):
        super().__init__(fs=fs,w_size=w_size,w_stride=w_stride)
        self.hidden_size = hidden_size
        self.train_win = train_win
        self.pred_win = pred_win
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.use_cuda = use_cuda
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_class = model_class
        self.val=val
        self.is_fitted = False
        
        

        if self.use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU instead.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    
    def _prepare_segment(self, data, ret_time=False):
        data_ch = data.columns.to_list()
        data_np = data.to_numpy()

        j = int(self.w_size*self.fs-(self.train_win+self.pred_win)+1)

        nwins = num_wins(data_np.shape[0],self.fs,self.w_size,self.w_stride)
        data_mat = torch.zeros((nwins,j,(self.train_win+self.pred_win),data_np.shape[1]))
        for k in range(len(data_ch)): # Iterating through channels
            samples = MovingWinClips(data_np[:,k],self.fs,self.w_size,self.w_stride)
            for i in range(samples.shape[0]):
                clip = samples[i,:]
                mat = torch.tensor(hankel(clip[:j],clip[-(self.train_win+self.pred_win):]))
                data_mat[i,:,:,k] = mat
        time_mat = MovingWinClips(np.arange(len(data))/self.fs,self.fs,self.w_size,self.w_stride)
        win_times = time_mat[:,0]
        data_flat = data_mat.reshape((-1,self.train_win + self.pred_win,len(data_ch)))
        input_data = data_flat[:,:-1,:].float()
        target_data = data_flat[:,-1,:].float()

        if ret_time:
            return input_data, target_data, win_times
        else:
            return input_data, target_data
    
    def _train_model(self,dataloader,criterion,optimizer):
        # Training loop
        tbar = tqdm(range(self.num_epochs),leave=False)
        train_batches = len(dataloader)-5
        all_loss = []
        all_val_loss = []
        self.model.train()
        for e in tbar:
            epoch_loss = []
            val_loss=[]
            for i,(inputs, targets) in enumerate(dataloader):
                if (i < train_batches) or (not self.val):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                    del inputs, targets, outputs
                elif self.val:
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss.append(loss.item())

            all_loss.append(np.mean(epoch_loss))
            if self.val:
                all_val_loss.append(np.mean(val_loss))

            if (e % 5 == 0) and (e > 0):
                if len(epoch_loss) == 0:
                    print(e,"no training loss")
                tbar.set_description(f"{np.mean(epoch_loss):.4f}")
                del loss
        self.train_loss=all_loss
        if self.val:
            self.val_loss=all_val_loss

    def _repair_data(self,outputs,X):
        nwins = num_wins(X.shape[0],self.fs,self.w_size,self.w_size)
        nchannels = X.shape[1]
        repaired = outputs.reshape((nwins,self.w_size*self.fs-(self.train_win + self.pred_win)+1,nchannels))
        return repaired

    def fit(self, X):
        input_size = X.shape[1]
        # Initialize the model
        self.model = self.model_class(input_size, self.hidden_size)
        self.model = self.model.to(self.device)

        # Scale the training data
        self._fit_scaler(X)
        X_z = self._scaler_transform(X)

        # Prepare input and target data for the LSTM
        input_data,target_data = self._prepare_segment(X_z)

        dataset = TensorDataset(input_data, target_data)
        if self.batch_size == 'full':
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Train the model, this will just modify the model object, no returns
        self._train_model(dataloader,criterion,optimizer)
        self.is_fitted = True

    def forward(self, X):
        assert self.is_fitted, "Must fit model before running inference"
        X_z = self._scaler_transform(X)
        input_data,target_data, time_wins = self._prepare_segment(X_z,ret_time=True)
        self.time_wins = time_wins
        dataset = TensorDataset(input_data,target_data)
        if self.batch_size == 'full':
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        with torch.no_grad():
            self.model.eval()
            mse_distribution = []
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                mse = (outputs-targets)**2
                mse_distribution.append(mse)
                del inputs, targets, outputs, mse
        raw_mdl_outputs = torch.cat(mse_distribution).cpu().numpy()
        mdl_outs = raw_mdl_outputs.reshape((len(time_wins),-1,raw_mdl_outputs.shape[1]))
        raw_loss_mat = np.sqrt(np.mean(mdl_outs,axis=1)).T
        # loss_mat = sc.ndimage.uniform_filter1d(raw_loss_mat,20,origin=0,axis=1,mode='constant')
        # loss_mat = sc.ndimage.median_filter(raw_loss_mat,size=10,mode='constant',axes=1)
        self.feature_df = pd.DataFrame(raw_loss_mat.T,columns = X.columns)
        return self.feature_df