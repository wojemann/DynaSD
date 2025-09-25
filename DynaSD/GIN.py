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
    def __init__(self, input_size, hidden_size, num_layers=1, num_stacks=1, residual_init=0.5):
        super(MultiStepGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stacks = num_stacks
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        
        # Skip connection projections
        self.input_projection = nn.Linear(input_size, input_size)
        self.skip_weight = nn.Parameter(torch.tensor(residual_init))  # Learnable skip weight with custom initialization
        
        # Per-timestep input projection with nonlinearity and normalization (D -> D)
        # Build a stack of num_stacks layers, each: Linear(input_size, input_size*2) -> GELU -> LayerNorm(input_size*2)
        # except the last, which is Linear(input_size*2, input_size) -> GELU -> LayerNorm(input_size)
        input_stack_layers = []
        for i in range(self.num_stacks):
            if i == 0:
                in_dim = self.input_size
            else:
                in_dim = self.input_size * 2
            if i == self.num_stacks - 1:
                out_dim = self.input_size
            else:
                out_dim = self.input_size * 2
            input_stack_layers.append(nn.Linear(in_dim, out_dim))
            input_stack_layers.append(nn.GELU())
            input_stack_layers.append(nn.LayerNorm(out_dim))
        self.input_stack = nn.Sequential(*input_stack_layers)
        
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
        
        # Initialize input_stack's Linear layer
        input_stack_linear = self.input_stack[0]
        nn.init.xavier_uniform_(input_stack_linear.weight, gain=1.0)
        nn.init.zeros_(input_stack_linear.bias)
        
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, input_sequence, forecast_steps):
        # Phase 1: Process input sequence via per-timestep projection
        projected_input = self.input_stack(input_sequence)  # (B, T, D)
        gru_output, hidden = self.gru(projected_input)
        
        # Phase 2: Autoregressive forecasting
        forecasts = []
        current_hidden = hidden
        
        # Start with the first prediction from the final hidden state
        first_prediction = self.projection(gru_output[:, -1:, :])  # (B, 1, D)
        # Residual skip from the last data-space input
        last_data_step = input_sequence[:, -1:, :]  # (B, 1, D)
        first_prediction = first_prediction + self.skip_weight * self.input_projection(last_data_step)
        forecasts.append(first_prediction.squeeze(1))  # (B, D)
        current_input_data_space = first_prediction  # keep in data space (B, 1, D)
        
        for _ in range(forecast_steps - 1):  # Note: forecast_steps - 1
            # Feed the projected current input to the GRU
            gru_in = self.input_stack(current_input_data_space)  # (B, 1, D)
            gru_output_step, current_hidden = self.gru(gru_in, current_hidden)
            prediction = self.projection(gru_output_step)  # (B, 1, D)
            # Residual skip from current data-space input
            prediction = prediction + self.skip_weight * self.input_projection(current_input_data_space)
            forecasts.append(prediction.squeeze(1))
            # Next step uses the data-space prediction
            current_input_data_space = prediction
            
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
                 batch_size=1024,
                 lr=0.01,
                 residual_init=0.5,     # Initial value for residual connection weight
                 use_cuda=False,
                 num_stacks=1,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters - input_length == forecast_horizon for simplicity
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stacks = num_stacks
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
            num_stacks=self.num_stacks,
            residual_init=self.residual_init
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
    
    def predict(self, X):
        """Use the shared multi-step prediction from NDDBase"""
        return self.predict_multistep(X)
    
    def __str__(self):
        return "GIN"