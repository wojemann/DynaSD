from .NDDBase import NDDBase
import torch
import torch.nn as nn
import numpy as np

class MultiStepGRU(nn.Module):
    """GRU for autoregressive forecasting without skip connections or input stacks"""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(MultiStepGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        
        self.projection = nn.Linear(hidden_size, input_size)
    
    def forward(self, input_sequence, forecast_steps):
        # Phase 1: Process input sequence
        gru_output, hidden = self.gru(input_sequence)
        
        # Phase 2: Autoregressive forecasting
        forecasts = []
        current_hidden = hidden
        
        # Start with the first prediction from the final hidden state
        first_prediction = self.projection(gru_output[:, -1:, :])  # (B, 1, D)
        forecasts.append(first_prediction.squeeze(1))  # (B, D)
        current_input = first_prediction  # (B, 1, D)
        
        for _ in range(forecast_steps - 1):
            # Feed the current prediction to the GRU
            gru_output_step, current_hidden = self.gru(current_input, current_hidden)
            prediction = self.projection(gru_output_step)  # (B, 1, D)
            forecasts.append(prediction.squeeze(1))
            # Next step uses the prediction
            current_input = prediction
            
        return torch.stack(forecasts, dim=1)

class NDD(NDDBase):
    """
    NDD (Neural Dynamic Divergence) - Multi-step GRU model with window-based sequence generation.
    FIXED VERSION: Now matches the old NDD behavior for proper seizure detection.
    """

    def __init__(self, 
                 hidden_size=10,
                 num_layers=1,
                 fs=256,
                 sequence_length=12,  # This is train_win in old version
                 forecast_length=1,   # This is pred_win in old version
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
        self.sequence_length = sequence_length  # train_win
        self.forecast_length = forecast_length  # pred_win
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
    
    def _prepare_windowed_sequences(self, data, ret_positions=False):
        """
        FIXED: Generate sequences window-by-window to match old NDD behavior.
        
        This creates dense overlapping sequences within each window, exactly
        matching the old Hankel matrix approach in model.py.
        
        Args:
            data: Input DataFrame (scaled)
            ret_positions: Whether to return position information
            
        Returns:
            tuple: (input_data, target_data, [positions])
        """
        data_np = data.to_numpy()
        n_samples, n_channels = data_np.shape
        
        # Calculate window parameters (matching old version)
        win_length = int(self.w_size * self.fs)
        win_stride = int(self.w_stride * self.fs)
        n_windows = (n_samples - win_length) // win_stride + 1
        
        # Sequences per window (matching old J calculation)
        # J = w_size*fs - (train_win + pred_win) + 1
        sequences_per_window = win_length - (self.sequence_length + self.forecast_length) + 1
        
        if self.verbose:
            print(f"  Generating {n_windows} windows with {sequences_per_window} sequences each")
            print(f"  Total sequences: {n_windows * sequences_per_window}")
        
        # Pre-allocate arrays for efficiency
        total_sequences = n_windows * sequences_per_window
        all_inputs = np.zeros((total_sequences, self.sequence_length, n_channels), dtype=np.float32)
        all_targets = np.zeros((total_sequences, self.forecast_length, n_channels), dtype=np.float32)
        
        if ret_positions:
            all_positions = []
        
        seq_idx_global = 0
        
        # Generate sequences window-by-window
        for win_idx in range(n_windows):
            win_start = win_idx * win_stride
            win_end = win_start + win_length
            window_data = data_np[win_start:win_end]
            
            # Generate all sequences within this window (Hankel-like structure)
            for seq_idx_local in range(sequences_per_window):
                # Input sequence
                input_seq = window_data[seq_idx_local:seq_idx_local + self.sequence_length]
                # Target sequence (next forecast_length samples)
                target_seq = window_data[
                    seq_idx_local + self.sequence_length:
                    seq_idx_local + self.sequence_length + self.forecast_length
                ]
                
                all_inputs[seq_idx_global] = input_seq
                all_targets[seq_idx_global] = target_seq
                
                if ret_positions:
                    all_positions.append({
                        'window_idx': win_idx,
                        'sequence_idx_in_window': seq_idx_local,
                        'input_start': win_start + seq_idx_local,
                        'input_end': win_start + seq_idx_local + self.sequence_length,
                        'target_start': win_start + seq_idx_local + self.sequence_length,
                        'target_end': win_start + seq_idx_local + self.sequence_length + self.forecast_length
                    })
                
                seq_idx_global += 1
        
        # Convert to tensors
        input_data = torch.tensor(all_inputs, dtype=torch.float32)
        target_data = torch.tensor(all_targets, dtype=torch.float32)
        
        if ret_positions:
            return input_data, target_data, all_positions
        else:
            return input_data, target_data

    def fit(self, X):
        """Fit the GRU forecasting model using window-based sequence generation"""
        input_size = X.shape[1]
        
        # Initialize model
        self.model = MultiStepGRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        if self.verbose:
            print(f"  Model: {self.model}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Scale the data
        self._fit_scaler(X)
        X_scaled = self._scaler_transform(X)
        
        # FIXED: Use window-based sequence generation
        input_data, target_data = self._prepare_windowed_sequences(X_scaled, ret_positions=False)
        
        if self.verbose:
            print(f"  Input shape: {input_data.shape}")
            print(f"  Target shape: {target_data.shape}")
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(input_data, target_data)
        batch_size = len(dataset) if self.batch_size == 'full' else self.batch_size
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False  # Don't shuffle to maintain window order
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = []
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass - train with teacher forcing (forecast_length=1)
                outputs = self.model(inputs, forecast_steps=1)
                
                # Calculate loss (outputs: B x 1 x C, targets: B x 1 x C)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
            
            if self.verbose and (epoch % 5 == 0 or epoch == self.num_epochs - 1):
                print(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {np.mean(epoch_loss):.6f}")
        
        self.is_fitted = True

    def predict(self, X):
        """
        FIXED: Predict using window-based sequence generation and proper aggregation.
        Matches the old NDD forward() method behavior.
        """
        assert self.is_fitted, "Must fit model before making predictions"
        
        # Scale the data
        X_scaled = self._scaler_transform(X)
        
        # FIXED: Use window-based sequence generation
        input_data, target_data, seq_positions = self._prepare_windowed_sequences(
            X_scaled, ret_positions=True
        )
        
        if self.verbose:
            print(f"  Running inference on {len(input_data)} sequences")
        
        # Run inference
        self.model.eval()
        all_predictions = []
        all_mse = []
        
        with torch.no_grad():
            dataset = torch.utils.data.TensorDataset(input_data, target_data)
            batch_size = len(dataset) if self.batch_size == 'full' else self.batch_size
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Generate predictions
                predictions = self.model(inputs, forecast_steps=self.forecast_length)
                
                # Calculate MSE per sample
                mse = (predictions - targets) ** 2
                
                all_predictions.append(predictions.cpu())
                all_mse.append(mse.cpu())
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0).numpy()  # (n_sequences, forecast_length, n_channels)
        all_mse = torch.cat(all_mse, dim=0).numpy()  # (n_sequences, forecast_length, n_channels)
        
        # FIXED: Aggregate into windows matching old version
        # Calculate window parameters
        win_length = int(self.w_size * self.fs)
        win_stride = int(self.w_stride * self.fs)
        sequences_per_window = win_length - (self.sequence_length + self.forecast_length) + 1
        n_windows = len(all_mse) // sequences_per_window
        n_channels = X.shape[1]
        
        # Reshape MSE to (n_windows, sequences_per_window, forecast_length, n_channels)
        mse_windowed = all_mse.reshape(n_windows, sequences_per_window, self.forecast_length, n_channels)
        
        # Average over forecast_length and sequences_per_window dimensions
        # This matches: np.sqrt(np.mean(mdl_outs, axis=1))
        window_mse = np.sqrt(np.mean(mse_windowed, axis=(1, 2)))  # (n_windows, n_channels)
        
        # Calculate window times
        time_wins = np.arange(n_windows) * self.w_stride
        self.time_wins = time_wins
        
        # Create feature DataFrame matching old version
        import pandas as pd
        self.feature_df = pd.DataFrame(window_mse, columns=X.columns)
        
        if self.verbose:
            print(f"  Output shape: {self.feature_df.shape}")
            print(f"  Number of windows: {n_windows}")
        
        return self.feature_df
    
    def __str__(self):
        return "NDD"
    
    def _get_pretrained_threshold(self):
        return 0.947901

    def _aggregate_threshold(self, boundaries, method):
        """
        Helper function to aggregate channel boundaries into final threshold.
        """
        boundary = 0.535674
        if method == 'mean':
            return np.nanmean(boundaries) + np.nanstd(boundaries)
        elif method == 'automean':
            if np.sum(boundaries > boundary) == 0:
                return self._get_pretrained_threshold()
            else:
                return np.nanmean(boundaries[boundaries > boundary])
        elif method == 'automedian':
            if np.sum(boundaries > boundary) == 0:
                return self._get_pretrained_threshold()
            else:
                return np.nanmedian(boundaries[boundaries > boundary])
        elif method == 'meanover':
            return np.nanmean(boundaries[boundaries > np.nanmean(boundaries)])
        elif method == 'medianover':
            return np.nanmedian(boundaries[boundaries > np.nanmedian(boundaries)])
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
