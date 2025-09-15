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

class MLPForecaster(nn.Module):
    """
    Multi-layer perceptron for multi-step forecasting.
    Takes flattened input sequence and predicts flattened forecast sequence.
    """
    def __init__(self, input_size, sequence_length, hidden_sizes=[128, 64], dropout=0.1, use_batch_norm=True):
        super(MLPForecaster, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.use_batch_norm = use_batch_norm
        
        # Input: (batch_size, sequence_length * input_size)
        # Output: (batch_size, sequence_length * input_size) 
        input_dim = sequence_length * input_size
        output_dim = input_size
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_dim, hidden_size))
            
            # Add batch normalization after each linear layer (except the last)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights properly to prevent flat outputs
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to maintain signal dynamics"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                
                # Initialize bias to small values to prevent zero outputs
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.01, 0.01)
    
    def forward(self, input_sequence, forecast_steps):
        """
        Forward pass for linear forecasting.
        
        Args:
            input_sequence: (batch_size, sequence_length, input_size)
            forecast_steps: int (should equal sequence_length)
        
        Returns:
            forecasts: (batch_size, sequence_length, input_size)
        """
        preds = []
        current_input = input_sequence.clone()
        for _ in range(forecast_steps):
            batch_size = current_input.size(0)
            # Flatten current input
            flattened_input = current_input.view(batch_size, -1)
            # MLP prediction for next time step (output_dim = input_size)
            next_pred = self.mlp(flattened_input)  # (batch_size, input_size)
            preds.append(next_pred.unsqueeze(1))  # (batch_size, 1, input_size)
            # Roll input: remove first time step, append prediction
            if self.sequence_length > 1:
                current_input = torch.cat([current_input[:, 1:, :], next_pred.unsqueeze(1)], dim=1)
            else:
                current_input = next_pred.unsqueeze(1)
        # Concatenate predictions along time dimension
        forecasts = torch.cat(preds, dim=1)  # (batch_size, forecast_steps, input_size)
        return forecasts
    
    def __str__(self):
        return f"MLPForecaster_{self.input_size}ch_{self.sequence_length}seq"

class MINDD(NDDBase):
    """
    MINDA (Multi-layer Integrated Neural Dynamic Analysis) - MLP benchmark for GIN.
    Uses multi-layer perceptron for multi-step forecasting instead of RNNs.
    """

    def __init__(self, 
                 fs=256,
                 sequence_length=16,    # Both input and forecast length (constrained to be equal)
                 forecast_length=8,
                 hidden_sizes=[128, 64], # MLP hidden layer sizes
                 dropout=0.1,           # Dropout rate
                 use_batch_norm=True,   # Whether to use batch normalization
                 w_size=1, 
                 w_stride=0.5,
                 num_epochs=10,
                 batch_size='full',
                 lr=0.01,
                 use_cuda=False,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters - input_length == forecast_horizon for simplicity
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.train = True
    
    def fit(self, X):
        """Fit the MLP forecasting model using shared training loop"""
        input_size = X.shape[1]
        
        # Initialize model
        self.model = MLPForecaster(
            input_size=input_size,
            sequence_length=self.sequence_length,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        if self.verbose:
            print(f"  Model: {self.model}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Use shared training loop
        self._train_model_multistep(
            X=X,
            model=self.model,
            sequence_length=self.sequence_length,
            forecast_length=1, # Train with teacher forcing
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            early_stopping=self.early_stopping,
            val_split=self.val_split,
            patience=self.patience,
            tolerance=self.tolerance
        )
        
    # def _get_features(self, X):
    #     """
    #     Run inference and return features aggregated into windows.
    #     1. Create sequences from continuous data and get per-channel losses
    #     2. Aggregate sequence-level losses into w_size/w_stride windows
        
    #     Returns:
    #         tuple: (mse_df, corr_df)
    #             - mse_df: DataFrame with MSE values, columns = channel names
    #             - corr_df: DataFrame with correlation values, columns = channel names
    #     """
    #     X_scaled = self._scaler_transform(X)
    #     input_data, target_data, seq_positions = self._prepare_multistep_sequences(X_scaled, self.sequence_length, self.forecast_length, ret_positions=True)
        
    #     # Create dataset and dataloader
    #     dataset = TensorDataset(input_data, target_data)
    #     batch_size = len(dataset) if self.batch_size == 'full' else self.batch_size
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    #     # Run inference to get sequence-level predictions
    #     self.model.eval()
    #     seq_results = []
        
    #     with torch.no_grad():
    #         batch_start = 0
    #         for inputs, targets in dataloader:
    #             inputs = inputs.to(self.device)
    #             targets = targets.to(self.device)
                
    #             # Get predictions
    #             predictions = self.model(inputs, self.forecast_length)
                
    #             # Calculate per-channel losses for each sequence in batch
    #             batch_size_actual, seq_len, n_channels = predictions.shape
                
    #             for batch_idx in range(batch_size_actual):
    #                 seq_idx = batch_start + batch_idx
    #                 seq_pos = seq_positions[seq_idx]
                    
    #                 # MSE per channel for this sequence
    #                 mse = torch.mean((predictions[batch_idx] - targets[batch_idx]) ** 2, dim=0).cpu().numpy()
                    
    #                 # Store results with temporal position and sequences for correlation
    #                 seq_results.append({
    #                     'seq_idx': seq_idx,
    #                     'target_start_time': seq_pos['target_time_start'],
    #                     'target_end_time': seq_pos['target_time_start'] + self.forecast_length / self.fs,
    #                     'mse': np.sqrt(mse),
    #                     'predicted_seq': predictions[batch_idx].cpu().numpy(),
    #                     'target_seq': targets[batch_idx].cpu().numpy()
    #                 })
                
    #             batch_start += batch_size_actual
        
    #     # Now aggregate sequence results into windows
    #     mse_df, corr_df = self._aggregate_sequences_to_windows(seq_results, X)

    #     return mse_df, corr_df
    
    def forward(self, X):
        """
        Run inference and return features aggregated into windows.
        1. Create sequences from continuous data and get per-channel losses
        2. Aggregate sequence-level losses into w_size/w_stride windows
        
        Returns:
            ndd_df: DataFrame with NDD values, columns = channel names
        """
        assert self.is_fitted, "Must fit model before running inference"
        mse_df, corr_df = self._get_features(X)
        ndd = pd.DataFrame()
        for ch in X.columns:
            mse_y = mse_df[ch].to_numpy().reshape(-1,1)
            corr_y = corr_df[ch].to_numpy().reshape(-1,1)
            f = np.concatenate((mse_y,corr_y),axis=1)
            m = self.dist_params[ch]['m']
            R = self.dist_params[ch]['R']
            ri = np.linalg.solve(R.T, (f - m).T)
            ndd[ch] = np.sum(ri * ri, axis=0) * (self.dist_params[ch]['n'] - 1)
        
        # Store window times for compatibility with other models
        nwins = num_wins(len(X), self.fs, self.w_size, self.w_stride)
        self.time_wins = np.array([win_idx * self.w_stride for win_idx in range(nwins)])
        
        # Store for backward compatibility
        self.mse_df = mse_df
        self.corr_df = corr_df
        self.ndd_df = ndd

        return self.ndd_df
    
    def predict(self, X):
        """Use the shared multi-step prediction from NDDBase"""
        return self.predict_multistep(X)
    
    def __str__(self):
        return "MINDD"