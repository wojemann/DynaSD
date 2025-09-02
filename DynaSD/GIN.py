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

# class MultiStepGRU(nn.Module):
#     """
#     Simple GRU model for multi-step forecasting.
#     """
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(MultiStepGRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size, 
#             num_layers=num_layers,
#             batch_first=True
#         )
        
#         self.projection = nn.Linear(hidden_size, input_size)
    
#     def forward(self, input_sequence, forecast_steps):
#         """
#         Forward pass with autoregressive forecasting.
        
#         Args:
#             input_sequence: (batch_size, n, input_size) - input sequence of length n
#             forecast_steps: int - number of steps p to forecast
        
#         Returns:
#             forecasts: (batch_size, p, input_size) - forecasted sequence
#         """
#         # Phase 1: Process input sequence to build up hidden state
#         _, hidden = self.gru(input_sequence)
        
#         # Phase 2: Generate forecasts autoregressively
#         forecasts = []
#         current_input = input_sequence[:, -1:, :]  # Last timestep as starting point
#         current_hidden = hidden
        
#         for _ in range(forecast_steps):
#             # GRU forward step
#             gru_output, current_hidden = self.gru(current_input, current_hidden)
            
#             # Project to output space
#             prediction = self.projection(gru_output)
            
#             forecasts.append(prediction.squeeze(1))  # Remove time dimension
#             current_input = prediction  # Use prediction as next input
        
#         # Stack forecasts: (batch_size, forecast_steps, input_size)
#         return torch.stack(forecasts, dim=1)
    
#     def __str__(self):
#         return f"MultiStepGRU_h{self.hidden_size}_l{self.num_layers}"

class MultiStepGRU(nn.Module):
    """GRU with residual connections to bypass saturation"""
    def __init__(self, input_size, hidden_size, num_layers=1, residual_init=0.5):
        super(MultiStepGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        
        # Skip connection projections
        self.input_projection = nn.Linear(input_size, input_size)
        self.skip_weight = nn.Parameter(torch.tensor(residual_init))  # Learnable skip weight with custom initialization
        
        self.projection = nn.Linear(hidden_size, input_size)
    
    def forward(self, input_sequence, forecast_steps):
        # Phase 1: Process input sequence
        gru_output, hidden = self.gru(input_sequence)
        
        # Phase 2: Autoregressive forecasting with skip connections
        forecasts = []
        current_input = input_sequence[:, -1:, :]
        current_hidden = hidden
        
        for _ in range(forecast_steps):
            # GRU forward step
            gru_output, current_hidden = self.gru(current_input, current_hidden)
            
            # Project to output space
            prediction = self.projection(gru_output)
            
            # Add residual connection from input
            skip_contribution = self.input_projection(current_input)
            prediction = prediction + self.skip_weight * skip_contribution
            
            forecasts.append(prediction.squeeze(1))
            current_input = prediction
        
        return torch.stack(forecasts, dim=1)

class GIN(NDDBase):
    """
    GIN (Gated recurrent unit with Integrated spatio-temporal modeling of Neural dynamic divergence) 
    simplified model following NDD pattern.
    """

    def __init__(self, 
                 hidden_size=10,
                 num_layers=1,
                 fs=256,
                 sequence_length=16,    # Both input and forecast length (constrained to be equal)
                 w_size=1, 
                 w_stride=0.5,
                 num_epochs=10,
                 batch_size='full',
                 lr=0.01,
                 lambda_zcr=0.1,        # Weight for zero-crossing rate loss
                 residual_init=0.5,     # Initial value for residual connection weight
                 use_cuda=False,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters - input_length == forecast_horizon for simplicity
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.input_length = sequence_length      # For backward compatibility
        self.forecast_horizon = sequence_length  # For backward compatibility
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_zcr = lambda_zcr
        self.residual_init = residual_init
    
    def _prepare_sequences(self, data, ret_positions=False):
        """Use the shared multi-step sequence preparation from NDDBase"""
        return self._prepare_multistep_sequences(data, self.sequence_length, ret_positions)
    
    def fit(self, X):
        """Fit the GRU forecasting model using shared training loop"""
        input_size = X.shape[1]
        
        # Initialize model
        self.model = MultiStepGRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            residual_init=self.residual_init
        ).to(self.device)
        
        print(f"  Model: {self.model}")
        
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
            tuple: (mse_df, zcr_df, combined_df)
                - mse_df: DataFrame with MSE values, columns = channel names
                - zcr_df: DataFrame with ZCR values, columns = channel names  
                - combined_df: DataFrame with combined loss values, columns = channel names
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
                batch_size, seq_len, n_channels = predictions.shape
                
                for batch_idx in range(batch_size):
                    seq_idx = batch_start + batch_idx
                    seq_pos = seq_positions[seq_idx]
                    
                    # MSE per channel for this sequence
                    mse = torch.mean((predictions[batch_idx] - targets[batch_idx]) ** 2, dim=0).cpu().numpy()
                    # mse = [np.corrcoef(a,b)[0,1] for a,b in zip(predictions[batch_idx].cpu().numpy().T,targets[batch_idx].cpu().numpy().T)]
                    # ZCR difference per channel for this sequence
                    zcr_diffs = []
                    for ch in range(n_channels):
                        pred_zcr = zero_crossing_rate(predictions[batch_idx, :, ch].cpu().numpy())
                        target_zcr = zero_crossing_rate(targets[batch_idx, :, ch].cpu().numpy())
                        zcr_diffs.append((pred_zcr - target_zcr) ** 2)
                    zcr_diffs = np.array(zcr_diffs)*100
                    
                    # Combined loss per channel
                    combined = mse + self.lambda_zcr * zcr_diffs
                    
                    # Store results with temporal position
                    seq_results.append({
                        'seq_idx': seq_idx,
                        'target_start_time': seq_pos['target_time_start'],
                        'target_end_time': seq_pos['target_time_start'] + self.sequence_length / self.fs,
                        'predicted_seq': predictions[batch_idx].cpu().numpy(),
                        'target_seq': targets[batch_idx].cpu().numpy(),
                        'mse': mse,
                        'zcr': zcr_diffs,
                        'combined': combined
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