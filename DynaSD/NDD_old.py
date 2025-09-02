from .NDDBase import NDDBase
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from scipy.linalg import hankel
from .utils import num_wins, MovingWinClips

class NDDModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NDDModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1,:])
        return out
    
    def __str__(self):
         return "NDD"

class NDD(NDDBase):
    def __init__(self, hidden_size = 10, fs = 128,
                  train_win = 12, pred_win = 1,
                  w_size = 1, w_stride = 0.5,
                  num_epochs = 10, batch_size = 'full',
                  lr = 0.01,
                  model_class = NDDModel,
                  use_cuda = False,
                  val=False,
                  ):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda)
        self.hidden_size = hidden_size
        self.train_win = train_win
        self.pred_win = pred_win
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_class = model_class
        self.val = val
    
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
        self.feature_df = pd.DataFrame(raw_loss_mat.T,columns = X.columns)
        return self.feature_df
    
    def predict(self, X):
        """Use the shared single-step prediction from NDDBase"""
        return self.predict_singlestep(X)

'''
class ImprovedNDDModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(ImprovedNDDModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        output = self.fc(lstm_out[:, -1, :])
        return output
    
    def __str__(self):
        return "ImprovedNDD"

class ImprovedNDD(DynaSDBase):
    def __init__(self, 
                 hidden_size=10, 
                 num_layers=1,
                 dropout=0.0,
                 fs=128,
                 train_win=12, 
                 pred_win=1,
                 w_size=1, 
                 w_stride=0.5,
                 num_epochs=10, 
                 batch_size='full',
                 lr=0.01,
                 val_split=0.2,  # More flexible validation
                 use_cuda=False,
                 model_class=None,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, **kwargs)
        
        # Model architecture parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.train_win = train_win
        self.pred_win = pred_win
        
        # Training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_split = val_split
        
        # Device setup
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU instead.")
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Model class
        self.model_class = model_class or ImprovedNDDModel
        self.model = None
        self.is_fitted = False
        
        # Training history
        self.train_loss = []
        self.val_loss = []
    
    def _create_hankel_matrices_vectorized(self, data):
        """Vectorized Hankel matrix creation for better performance"""
        n_samples, n_channels = data.shape
        j = int(self.w_size * self.fs - (self.train_win + self.pred_win) + 1)
        
        # Get moving window clips for all channels at once
        nwins = num_wins(n_samples, self.fs, self.w_size, self.w_stride)
        data_tensor = torch.zeros((nwins, j, self.train_win + self.pred_win, n_channels))
        
        for k in range(n_channels):
            clips = MovingWinClips(data[:, k], self.fs, self.w_size, self.w_stride)
            for i in range(clips.shape[0]):
                clip = clips[i, :]
                hankel_mat = torch.tensor(
                    hankel(clip[:j], clip[-(self.train_win + self.pred_win):]),
                    dtype=torch.float32
                )
                data_tensor[i, :, :, k] = hankel_mat
                
        return data_tensor
    
    def _prepare_segment(self, data, ret_time=False):
        """Improved segment preparation with better memory management"""
        data_ch = data.columns.tolist()
        data_np = data.to_numpy()
        
        # Create Hankel matrices
        data_tensor = self._create_hankel_matrices_vectorized(data_np)
        
        # Prepare time information if needed
        if ret_time:
            time_mat = MovingWinClips(
                np.arange(len(data)) / self.fs, 
                self.fs, self.w_size, self.w_stride
            )
            win_times = time_mat[:, 0]
        
        # Flatten and split
        data_flat = data_tensor.reshape((-1, self.train_win + self.pred_win, len(data_ch)))
        input_data = data_flat[:, :-self.pred_win, :].float()
        target_data = data_flat[:, -self.pred_win:, :].float()
        
        if self.pred_win == 1:
            target_data = target_data.squeeze(1)  # Remove singleton dimension
        
        return (input_data, target_data, win_times) if ret_time else (input_data, target_data)
    
    def _create_data_loaders(self, input_data, target_data):
        """Create train/validation data loaders"""
        dataset = TensorDataset(input_data, target_data)
        
        if self.val_split > 0:
            n_samples = len(dataset)
            n_train = int(n_samples * (1 - self.val_split))
            n_val = n_samples - n_train
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val]
            )
        else:
            train_dataset = dataset
            val_dataset = None
        
        # Determine batch size
        batch_size = len(train_dataset) if self.batch_size == 'full' else self.batch_size
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        return train_loader, val_loader
    
    def _train_epoch(self, data_loader, criterion, optimizer, is_training=True):
        """Train or validate for one epoch"""
        if is_training:
            self.model.train()
        else:
            self.model.eval()
            
        epoch_losses = []
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if is_training:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_training):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                if is_training:
                    loss.backward()
                    optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def fit(self, X):
        """Improved fit method with better validation and memory management"""
        input_size = X.shape[1]
        
        # Initialize model
        self.model = self.model_class(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Scale data
        self._fit_scaler(X)
        X_scaled = self._scaler_transform(X)
        
        # Prepare data
        input_data, target_data = self._prepare_segment(X_scaled)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(input_data, target_data)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.train_loss = []
        self.val_loss = []
        
        pbar = tqdm(range(self.num_epochs), desc="Training", leave=False)
        for epoch in pbar:
            # Training
            train_loss = self._train_epoch(train_loader, criterion, optimizer, is_training=True)
            self.train_loss.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._train_epoch(val_loader, criterion, optimizer, is_training=False)
                self.val_loss.append(val_loss)
                pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
            else:
                pbar.set_postfix({'train_loss': f'{train_loss:.4f}'})
        
        self.is_fitted = True
    
    def forward(self, X):
        """Improved forward method"""
        assert self.is_fitted, "Must fit model before running inference"
        
        X_scaled = self._scaler_transform(X)
        input_data, target_data, time_wins = self._prepare_segment(X_scaled, ret_time=True)
        
        self.time_wins = time_wins
        
        # Create data loader for inference
        dataset = TensorDataset(input_data, target_data)
        batch_size = len(dataset) if self.batch_size == 'full' else self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Run inference
        self.model.eval()
        mse_results = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                mse = torch.mean((outputs - targets) ** 2, dim=1, keepdim=True)  # Per sample MSE
                mse_results.append(mse.cpu())
        
        # Combine results
        all_mse = torch.cat(mse_results, dim=0).numpy()
        
        # Reshape to match window structure
        mse_reshaped = all_mse.reshape((len(time_wins), -1))
        feature_values = np.sqrt(np.mean(mse_reshaped, axis=1))
        
        # Create feature dataframe
        self.feature_df = pd.DataFrame(
            feature_values.reshape(-1, 1), 
            columns=[f'RMSE_{self.model}']
        )
        
        return self.feature_df

'''