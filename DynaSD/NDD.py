from .NDDBase import NDDBase
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .utils import num_wins, MovingWinClips

class NDDLSTM(nn.Module):
    """
    LSTM model for multi-step forecasting with input_length=12, forecast_horizon=1.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(NDDLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        
        self.projection = nn.Linear(hidden_size, input_size)
    
    def forward(self, input_sequence, forecast_steps):
        """
        Forward pass with autoregressive forecasting.
        
        Args:
            input_sequence: (batch_size, 12, input_size) - input sequence of length 12
            forecast_steps: int - number of steps to forecast (should be 1)
        
        Returns:
            forecasts: (batch_size, 1, input_size) - forecasted sequence
        """
        # Process input sequence to build up hidden state
        _, (hidden, cell) = self.lstm(input_sequence)
        
        # Generate single forecast step
        # Use the last hidden state to predict the next timestep
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        prediction = self.projection(last_hidden)  # (batch_size, input_size)
        
        # Reshape to match expected output format (batch_size, 1, input_size)
        return prediction.unsqueeze(1)
    
    def __str__(self):
        return f"NDDLSTM_h{self.hidden_size}_l{self.num_layers}"

class NDD(NDDBase):
    """
    NDD (Neural Dynamic Divergence) - Multi-step LSTM model.
    Uses input_length=12, forecast_horizon=1 for single-step prediction within multi-step framework.
    """

    def __init__(self, 
                 hidden_size=10,
                 num_layers=1,
                 fs=256,
                 input_length=12,      # Input sequence length
                 forecast_horizon=1,   # Forecast horizon (single step)
                 w_size=1, 
                 w_stride=0.5,
                 num_epochs=10,
                 batch_size='full',
                 lr=0.01,
                 lambda_zcr=0.1,        # Weight for zero-crossing rate loss
                 use_cuda=False,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.sequence_length = input_length + forecast_horizon  # Total sequence length for data prep
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_zcr = lambda_zcr
    
    def _prepare_sequences(self, data, ret_positions=False):
        """
        Prepare sequences for NDD with input_length=12, forecast_horizon=1.
        Creates sequences with stride=1 to maximize data usage.
        """
        data_np = data.to_numpy()
        n_samples, n_channels = data_np.shape
        
        # Create sequences with stride=1 for maximum data usage
        stride = 1
        total_seq_length = self.input_length + self.forecast_horizon  # 12 + 1 = 13
        
        # Calculate how many sequences we can create
        n_sequences = n_samples - total_seq_length + 1
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough data for even one sequence. Need at least {total_seq_length} samples.")
        
        if self.verbose:
            print(f"Creating {n_sequences} overlapping sequences for NDD (input=12, forecast=1)")
        
        all_inputs = []
        all_targets = []
        seq_positions = []  # Track where each sequence starts in original data
        
        for seq_idx in range(n_sequences):
            seq_start = seq_idx * stride
            input_end = seq_start + self.input_length
            target_end = input_end + self.forecast_horizon
            
            # Extract input and target sequences
            input_seq = data_np[seq_start:input_end, :]  # (12, n_channels)
            target_seq = data_np[input_end:target_end, :]  # (1, n_channels)
            
            all_inputs.append(input_seq)
            all_targets.append(target_seq)
            seq_positions.append({
                'seq_idx': seq_idx,
                'input_start': seq_start,
                'input_end': input_end,
                'target_start': input_end,
                'target_end': target_end,
                'seq_time_start': seq_start / self.fs,
                'seq_time_end': input_end / self.fs,
                'input_time_start': seq_start / self.fs,
                'target_time_start': input_end / self.fs
            })
        
        input_data = torch.FloatTensor(np.array(all_inputs))
        target_data = torch.FloatTensor(np.array(all_targets))
        
        if ret_positions:
            return input_data, target_data, seq_positions
        else:
            return input_data, target_data

    def fit(self, X):
        """Fit the LSTM forecasting model using shared training loop"""
        input_size = X.shape[1]
        
        # Initialize model
        self.model = NDDLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        if self.verbose:
            print(f"  Model: {self.model}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Use shared training loop
        self._train_model_multistep(
            X=X,
            model=self.model,
            sequence_length=self.forecast_horizon,  # Use forecast_horizon for model calls
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            lambda_zcr=self.lambda_zcr,
            early_stopping=self.early_stopping,
            val_split=self.val_split,
            patience=self.patience,
            tolerance=self.tolerance
        )
        
        self.is_fitted = True

    def forward(self, X):
        """
        Run inference and return features aggregated into windows.
        1. Create sequences from continuous data and get per-channel losses
        2. Aggregate sequence-level losses into w_size/w_stride windows
        
        Returns:
            tuple: (mse_df, zcr_df, combined_df, corr_df)
        """
        assert self.is_fitted, "Must fit model before running inference"
        
        X_scaled = self._scaler_transform(X)
        input_data, target_data, seq_positions = self._prepare_sequences(X_scaled, ret_positions=True)
        
        # Create dataset and dataloader
        dataset = TensorDataset(input_data, target_data)
        batch_size = len(dataset) if self.batch_size == 'full' else self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Run inference to get sequence-level predictions
        self.model.eval()
        seq_results = []
        
        with torch.no_grad():
            batch_start = 0
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions
                predictions = self.model(inputs, self.forecast_horizon)
                
                # Calculate per-channel losses for each sequence in batch
                batch_size_actual, seq_len, n_channels = predictions.shape
                
                for batch_idx in range(batch_size_actual):
                    seq_idx = batch_start + batch_idx
                    seq_pos = seq_positions[seq_idx]
                    
                    # MSE per channel for this sequence
                    mse = torch.mean((predictions[batch_idx] - targets[batch_idx]) ** 2, dim=0).cpu().numpy()
                    
                    # ZCR difference per channel for this sequence
                    zcr_diffs = []
                    for ch in range(n_channels):
                        pred_zcr = zero_crossing_rate(predictions[batch_idx, :, ch].cpu().numpy())
                        target_zcr = zero_crossing_rate(targets[batch_idx, :, ch].cpu().numpy())
                        zcr_diffs.append((pred_zcr - target_zcr) ** 2)
                    zcr_diffs = np.array(zcr_diffs) * 100
                    
                    # Combined loss per channel
                    combined = mse + self.lambda_zcr * zcr_diffs
                    
                    # Store results with temporal position and sequences for correlation
                    seq_results.append({
                        'seq_idx': seq_idx,
                        'target_start_time': seq_pos['target_time_start'],
                        'target_end_time': seq_pos['target_time_start'] + self.forecast_horizon / self.fs,
                        'mse': mse,
                        'zcr': zcr_diffs,
                        'combined': combined,
                        'predicted_seq': predictions[batch_idx].cpu().numpy(),
                        'target_seq': targets[batch_idx].cpu().numpy()
                    })
                
                batch_start += batch_size_actual
        
        # Now aggregate sequence results into windows
        mse_df, zcr_df, combined_df, corr_df = self._aggregate_sequences_to_windows(seq_results, X)
        
        # Store window times for compatibility with other models
        nwins = num_wins(len(X), self.fs, self.w_size, self.w_stride)
        self.time_wins = np.array([win_idx * self.w_stride for win_idx in range(nwins)])
        
        # Store for backward compatibility
        self.mse_df = mse_df
        self.zcr_df = zcr_df  
        self.combined_df = combined_df
        self.corr_df = corr_df
        
        return mse_df, zcr_df, combined_df, corr_df
    
    def predict(self, X):
        """Use the shared multi-step prediction from NDDBase"""
        return self.predict_multistep(X)
