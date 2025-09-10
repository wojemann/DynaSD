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
        
        # Initialize weights properly to prevent vanishing gradients
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent vanishing gradients and state collapse"""
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param, gain=1.0)  # Orthogonal helps with long sequences
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set update gate bias to 1 for better gradient flow
                if 'bias_ih' in name:
                    param.data[param.size(0)//3:2*param.size(0)//3] = 1.0
        
        # Initialize linear layer weights
        nn.init.xavier_uniform_(self.input_projection.weight, gain=1.0)
        nn.init.zeros_(self.input_projection.bias)
        
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, input_sequence, forecast_steps):
        # Phase 1: Process input sequence
        gru_output, hidden = self.gru(input_sequence)
        
        # Phase 2: Autoregressive forecasting
        forecasts = []
        current_hidden = hidden
        
        # Start with the first prediction from the final hidden state
        first_prediction = self.projection(gru_output[:, -1:, :])
        forecasts.append(first_prediction.squeeze(1))
        current_input = first_prediction
        
        for _ in range(forecast_steps - 1):  # Note: forecast_steps - 1
            gru_output, current_hidden = self.gru(current_input, current_hidden)
            prediction = self.projection(gru_output)
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
                 sequence_length=16,
                 forecast_length=16,
                 w_size=1, 
                 w_stride=0.5,
                 num_epochs=10,
                 batch_size='full',
                 lr=0.01,
                 residual_init=0.5,     # Initial value for residual connection weight
                 use_cuda=False,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters - input_length == forecast_horizon for simplicity
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.residual_init = residual_init
    
    def _prepare_sequences(self, data, ret_positions=False):
        """Use the shared multi-step sequence preparation from NDDBase"""
        return self._prepare_multistep_sequences(data, self.sequence_length, self.forecast_length, ret_positions)
    
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
    #         tuple: (mse_df, zcr_df, combined_df)
    #             - mse_df: DataFrame with MSE values, columns = channel names
    #             - zcr_df: DataFrame with ZCR values, columns = channel names  
    #             - combined_df: DataFrame with combined loss values, columns = channel names
    #     """        
    #     X_scaled = self._scaler_transform(X)
    #     input_data, target_data, seq_positions = self._prepare_sequences(X_scaled, ret_positions=True)
        
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
    #             batch_size, _, _ = predictions.shape
                
    #             for batch_idx in range(batch_size):
    #                 seq_idx = batch_start + batch_idx
    #                 seq_pos = seq_positions[seq_idx]
                    
    #                 # MSE per channel for this sequence
    #                 mse = torch.mean((predictions[batch_idx] - targets[batch_idx]) ** 2, dim=0).cpu().numpy()
                    
    #                 # Store results with temporal position
    #                 seq_results.append({
    #                     'seq_idx': seq_idx,
    #                     'target_start_time': seq_pos['target_time_start'],
    #                     'target_end_time': seq_pos['target_time_start'] + self.forecast_length / self.fs,
    #                     'predicted_seq': predictions[batch_idx].cpu().numpy(),
    #                     'target_seq': targets[batch_idx].cpu().numpy(),
    #                     'mse': mse,
    #                 })
                
    #             batch_start += batch_size
        
    #     # Now aggregate sequence results into windows
    #     mse_df, corr_df = self._aggregate_sequences_to_windows(seq_results, X)
    #     return mse_df, corr_df
    
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