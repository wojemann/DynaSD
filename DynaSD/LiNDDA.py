from .NDDBase import NDDBase
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .utils import num_wins, MovingWinClips

# class SkLinearForecaster(LinearRegression):
#     def __init__(self, input_size, sequence_length):
#         super(SkLinearForecaster, self).__init__()
#         self.input_size = input_size
#         self.sequence_length = sequence_length
#         self.model = LinearRegression()
    
#     def forward(self, input_sequence, forecast_steps):
#         """
#         Forward pass for linear forecasting.
        
#         Args:
#             input_sequence: (batch_size, sequence_length, input_size)
#             forecast_steps: int (should equal sequence_length)
        
#         Returns:
#             forecasts: (batch_size, sequence_length, input_size)
#         """
#         preds = []
#         current_input = input_sequence.clone()
#         for _ in range(forecast_steps):
#             batch_size = current_input.size(0)
#             flattened_input = current_input.view(batch_size, -1)
#             next_pred = self.model.predict(flattened_input)
#             preds.append(next_pred.unsqueeze(1))
#             if self.sequence_length > 1:
#                 current_input = torch.cat([current_input[:, 1:, :], next_pred.unsqueeze(1)], dim=1)
#             else:
#                 current_input = next_pred.unsqueeze(1)
#         forecasts = torch.cat(preds, dim=1)
#         return forecasts

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
        output_dim = input_size
        
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
        preds = []
        current_input = input_sequence.clone()
        for _ in range(forecast_steps):
            batch_size = current_input.size(0)
            # Flatten current input
            flattened_input = current_input.view(batch_size, -1)
            # Linear prediction for next time step (output_dim = input_size)
            next_pred = self.linear(flattened_input)  # (batch_size, input_size)
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
        return f"LinearForecaster_{self.input_size}ch_{self.sequence_length}seq"

class LiNDDA(NDDBase):
    """
    LiNDDA (Linear Neural Dynamic Divergence Analysis) - Linear regression benchmark for GIN.
    Uses simple linear regression for multi-step forecasting instead of RNNs.
    """

    def __init__(self, 
                 fs=256,
                 sequence_length=16, 
                 forecast_length=16,
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
        self.forecast_length = forecast_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.train = True
    
    def fit(self, X):
        """Fit the Linear forecasting model using shared training loop"""
        input_size = X.shape[1]

        self.model = LinearForecaster(
            input_size=input_size,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        # Initialize model
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
    
    def predict(self, X):
        """Use the shared multi-step prediction from NDDBase"""
        return self.predict_multistep(X)

    def __str__(self):
        return "LiNDDA"