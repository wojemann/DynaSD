from .NDDBase import NDDBase
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from .utils import num_wins


class LinearRNNLayer(nn.Module):
    """Single Linear RNN layer without gating mechanisms"""
    def __init__(self, input_size, hidden_size, decay_init=0.9):
        super(LinearRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear transformations for this layer
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Learnable decay parameter for this layer
        self.decay = nn.Parameter(torch.tensor(decay_init))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for this layer"""
        nn.init.xavier_uniform_(self.input_linear.weight, gain=1.0)
        nn.init.zeros_(self.input_linear.bias)
        nn.init.orthogonal_(self.hidden_linear.weight, gain=0.8)
        
        # Initialize decay parameter to stable range
        with torch.no_grad():
            self.decay.data.clamp_(0.1, 0.99)
    
    def forward(self, input_x, hidden_state):
        """Forward pass for single time step"""
        # Linear recurrence: h_t = decay * W_h * h_{t-1} + W_i * x_t
        new_hidden = (self.decay * self.hidden_linear(hidden_state) + 
                     self.input_linear(input_x))
        return new_hidden


class MultiStepLinearRNN(nn.Module):
    """Linear RNN without gating mechanisms to preserve high-frequency dynamics"""
    def __init__(self, input_size, hidden_size, num_layers=1, num_stacks=1, decay_init=0.9, residual_init=0.5):
        super(MultiStepLinearRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stacks = num_stacks
        # Create layers with proper input sizing
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer takes input_size, subsequent layers take hidden_size
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(LinearRNNLayer(layer_input_size, hidden_size, decay_init))
        
        # Per-timestep input projection with nonlinearity and normalization (D -> D)
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
        
        # Skip connection projections
        self.input_projection = nn.Linear(input_size, input_size)
        self.skip_weight = nn.Parameter(torch.tensor(residual_init))  # Learnable skip weight with custom initialization
        
        # Output projection (same as MultiStepGRU)
        self.projection = nn.Linear(hidden_size, input_size)
        
        # Initialize weights properly to prevent vanishing gradients
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for projection layers"""
        # Note: Individual layer weights are initialized in LinearRNNLayer.__init__
        
        # Initialize input_stack's Linear layer
        input_stack_linear = self.input_stack[0]
        nn.init.xavier_uniform_(input_stack_linear.weight, gain=1.0)
        nn.init.zeros_(input_stack_linear.bias)
        
        # Initialize skip connection weights
        nn.init.xavier_uniform_(self.input_projection.weight, gain=1.0)
        nn.init.zeros_(self.input_projection.bias)
        
        # Initialize projection layer weights
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)

    def forward(self, input_sequence, forecast_steps):
        batch_size, seq_len, _ = input_sequence.shape
        
        # Phase 1: Process input sequence through per-timestep projection
        projected_input = self.input_stack(input_sequence)  # (B, T, D)
        
        # Phase 2: Process projected sequence through all layers
        # Initialize hidden states for all layers
        hidden_states = [torch.zeros(batch_size, self.hidden_size, device=input_sequence.device)
                        for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            # Pass projected input through each layer sequentially
            layer_input = projected_input[:, t, :]
            for layer_idx in range(self.num_layers):
                # For layer 0, use projected sequence input; for others, use previous layer's output
                current_input = layer_input if layer_idx == 0 else hidden_states[layer_idx-1]
                # Update this layer's hidden state
                hidden_states[layer_idx] = self.layers[layer_idx](current_input, hidden_states[layer_idx])
        
        # Phase 3: Autoregressive forecasting with skip connections
        forecasts = []
        
        # First prediction: Project the final hidden state from the last timestep
        first_prediction = self.projection(hidden_states[-1])  # (B, D)
        # Residual skip from the last data-space input
        last_data_step = input_sequence[:, -1, :]  # (B, D)
        first_prediction = first_prediction + self.skip_weight * self.input_projection(last_data_step)
        forecasts.append(first_prediction)
        
        # Use first prediction as input for remaining steps
        current_input_data_space = first_prediction  # keep in data space (B, D)
        
        for _ in range(forecast_steps - 1):  # Note: forecast_steps - 1
            # Project the current data-space input
            projected_current = self.input_stack(current_input_data_space.unsqueeze(1)).squeeze(1)  # (B, D)
            
            # Pass through all layers
            layer_input = projected_current
            for layer_idx in range(self.num_layers):
                # For layer 0, use projected current input; for others, use previous layer's output
                step_input = layer_input if layer_idx == 0 else hidden_states[layer_idx-1]
                # Update this layer's hidden state
                hidden_states[layer_idx] = self.layers[layer_idx](step_input, hidden_states[layer_idx])
            
            # Project from final layer's output to prediction space
            prediction = self.projection(hidden_states[-1])  # (B, D)
            # Residual skip from current data-space input
            prediction = prediction + self.skip_weight * self.input_projection(current_input_data_space)
            forecasts.append(prediction)
            # Next step uses the data-space prediction
            current_input_data_space = prediction
        
        return torch.stack(forecasts, dim=1)

class LiRNDDA(NDDBase):
    """
    LinearRNN (Linear Recurrent Neural Network) model for EEG forecasting.
    Removes gating mechanisms to preserve high-frequency dynamics that are 
    often suppressed in GRU/LSTM architectures.
    """

    def __init__(self, 
                 hidden_size=64,
                 num_layers=1,
                 num_stacks=1,
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
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters
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
        """Fit the Linear RNN forecasting model using shared training loop"""
        input_size = X.shape[1]
        
        # Initialize model
        self.model = MultiStepLinearRNN(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_stacks=self.num_stacks,
            residual_init=self.residual_init
        ).to(self.device)
        
        if self.verbose:
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
    
    def predict(self, X):
        """Use the shared multi-step prediction from NDDBase"""
        return self.predict_multistep(X)
    
    def __str__(self):
        return "LiRNDDA"
