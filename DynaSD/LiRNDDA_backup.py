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
    def __init__(self, input_size, hidden_size, num_layers=1, decay_init=0.9):
        super(MultiStepLinearRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create layers with proper input sizing
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer takes input_size, subsequent layers take hidden_size
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(LinearRNNLayer(layer_input_size, hidden_size, decay_init))
        
        # Output projection (same as MultiStepGRU)
        self.projection = nn.Linear(hidden_size, input_size)
        
        # Initialize weights properly to prevent vanishing gradients
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for projection layers"""
        # Note: Individual layer weights are initialized in LinearRNNLayer.__init__
        
        # Initialize projection layer weights
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)

    def forward(self, input_sequence, forecast_steps):
        batch_size, seq_len, _ = input_sequence.shape
        
        # Phase 1: Process input sequence through all layers
        # Initialize hidden states for all layers
        hidden_states = [torch.zeros(batch_size, self.hidden_size, device=input_sequence.device)
                        for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            # Pass input through each layer sequentially
            layer_input = input_sequence[:, t, :]
            for layer_idx in range(self.num_layers):
                # For layer 0, use sequence input; for others, use previous layer's output
                current_input = layer_input if layer_idx == 0 else hidden_states[layer_idx-1]
                # Update this layer's hidden state
                hidden_states[layer_idx] = self.layers[layer_idx](current_input, hidden_states[layer_idx])
        
        # Phase 2: Autoregressive forecasting with skip connections
        forecasts = []
        
        # First prediction: Project the final hidden state from the last timestep
        first_prediction = self.projection(hidden_states[-1])
        # Add residual connection if needed
        # skip_contribution = self.input_projection(input_sequence[:, -1, :])
        # first_prediction = first_prediction + self.skip_weight * skip_contribution
        forecasts.append(first_prediction)
        
        # Use first prediction as input for remaining steps
        current_input = first_prediction
        
        for _ in range(forecast_steps - 1):  # Note: forecast_steps - 1
            # Pass through all layers
            layer_input = current_input
            for layer_idx in range(self.num_layers):
                # For layer 0, use current input; for others, use previous layer's output
                step_input = layer_input if layer_idx == 0 else hidden_states[layer_idx-1]
                # Update this layer's hidden state
                hidden_states[layer_idx] = self.layers[layer_idx](step_input, hidden_states[layer_idx])
            
            # Project from final layer's output to prediction space
            prediction = self.projection(hidden_states[-1])
            # Add residual connection from input
            # skip_contribution = self.input_projection(current_input)
            # prediction = prediction + self.skip_weight * skip_contribution
            forecasts.append(prediction)
            current_input = prediction
        
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
                 fs=256,
                 sequence_length=16,
                 forecast_length=16,
                 w_size=1, 
                 w_stride=0.5,
                 num_epochs=10,
                 batch_size='full',
                 lr=0.01,
                 use_cuda=False,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
    
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
    #             batch_size_actual, _, _ = predictions.shape
                
    #             for batch_idx in range(batch_size_actual):
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
                
    #             batch_start += batch_size_actual
        
    #     # Now aggregate sequence results into windows
    #     mse_df, corr_df = self._aggregate_sequences_to_windows(seq_results, X)
    #     return mse_df, corr_df
    
    def forward(self, X):
        """
        Run inference and return neural dynamic divergence features.
        
        Returns:
            DataFrame: NDD values with columns = channel names
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
        return "LiRNDDA"