from .NDDBase import NDDBase, zero_crossing_rate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .utils import num_wins, MovingWinClips

class LinearForecaster(nn.Module):
    """
    Simple linear regression model for multi-step forecasting.
    Takes flattened input sequence and predicts flattened forecast sequence.
    """
    def __init__(self, input_size, sequence_length):
        super(LinearForecaster, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        # Input: (batch_size, sequence_length * input_size)
        # Output: (batch_size, sequence_length * input_size) 
        input_dim = sequence_length * input_size
        output_dim = sequence_length * input_size
        
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Initialize weights properly to prevent flat outputs
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to maintain signal dynamics"""
        # Xavier/Glorot initialization for better gradient flow
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        
        # Initialize bias to small values to prevent zero outputs
        nn.init.uniform_(self.linear.bias, -0.01, 0.01)
    
    def forward(self, input_sequence, forecast_steps):
        """
        Forward pass for linear forecasting.
        
        Args:
            input_sequence: (batch_size, sequence_length, input_size)
            forecast_steps: int (should equal sequence_length)
        
        Returns:
            forecasts: (batch_size, sequence_length, input_size)
        """
        batch_size = input_sequence.size(0)
        
        # Flatten input sequence
        flattened_input = input_sequence.view(batch_size, -1)
        
        # Linear prediction
        flattened_output = self.linear(flattened_input)
        
        # Reshape back to sequence format
        forecasts = flattened_output.view(batch_size, self.sequence_length, self.input_size)
        
        return forecasts
    
    def __str__(self):
        return f"LinearForecaster_{self.input_size}ch_{self.sequence_length}seq"

class LiNDDA(NDDBase):
    """
    LiNDDA (Linear Neural Dynamic Divergence Analysis) - Linear regression benchmark for GIN.
    Uses simple linear regression for multi-step forecasting instead of RNNs.
    """

    def __init__(self, 
                 fs=256,
                 sequence_length=16,    # Both input and forecast length (constrained to be equal)
                 w_size=1, 
                 w_stride=0.5,
                 num_epochs=10,
                 batch_size='full',
                 lr=0.01,
                 lambda_zcr=0.1,        # Weight for zero-crossing rate loss
                 use_cuda=False,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters - input_length == forecast_horizon for simplicity
        self.sequence_length = sequence_length
        self.input_length = sequence_length      # For backward compatibility
        self.forecast_horizon = sequence_length  # For backward compatibility
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_zcr = lambda_zcr
    
    def _prepare_sequences(self, data, ret_positions=False):
        """Use the shared multi-step sequence preparation from NDDBase"""
        return self._prepare_multistep_sequences(data, self.sequence_length, ret_positions)
    
    def fit(self, X):
        """Fit the Linear forecasting model using shared training loop"""
        input_size = X.shape[1]
        
        # Initialize model
        self.model = LinearForecaster(
            input_size=input_size,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        print(f"  Model: {self.model}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Use shared training loop
        self._train_model_multistep(
            X=X,
            model=self.model,
            sequence_length=self.sequence_length,
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
                        'target_end_time': seq_pos['target_time_start'] + self.sequence_length / self.fs,
                        'mse': mse,
                        'zcr': zcr_diffs,
                        'combined': combined,
                        'predicted_seq': predictions[batch_idx].cpu().numpy(),
                        'target_seq': targets[batch_idx].cpu().numpy()
                    })
                
                batch_start += batch_size
        
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
