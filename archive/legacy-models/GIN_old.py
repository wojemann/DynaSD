from .base import DynaSDBase
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .utils import num_wins, MovingWinClips

def zero_crossing_rate(signal):
    """
    Compute zero-crossing rate for a batch of signals.
    
    Args:
        signal: (batch_size, seq_len, n_channels) tensor
    
    Returns:
        zcr: (batch_size, n_channels) tensor of zero-crossing rates
    """
    # Compute sign changes
    sign_changes = torch.diff(torch.sign(signal), dim=1)  # (batch, seq_len-1, n_channels)
    
    # Count non-zero sign changes (zero-crossings)
    zero_crossings = torch.sum(sign_changes != 0, dim=1).float()  # (batch, n_channels)
    
    # Normalize by sequence length
    seq_len = signal.size(1)
    zcr = zero_crossings / (seq_len - 1)
    
    return zcr

class CombinedLoss(nn.Module):
    """
    Combined loss function with MSE and zero-crossing rate difference.
    """
    def __init__(self, lambda_zcr=0.3):
        super().__init__()
        self.lambda_zcr = lambda_zcr
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Compute combined loss.
        
        Args:
            predictions: (batch_size, seq_len, n_channels)
            targets: (batch_size, seq_len, n_channels)
        """
        # MSE loss (time domain)
        mse = self.mse_loss(predictions, targets)
        
        # Zero-crossing rate loss (frequency domain proxy)
        pred_zcr = zero_crossing_rate(predictions)  # (batch, n_channels)
        target_zcr = zero_crossing_rate(targets)    # (batch, n_channels)
        
        # Mean squared difference in zero-crossing rates
        zcr_loss = torch.mean((pred_zcr - target_zcr) ** 2)
        
        # Combined loss
        total_loss = mse + self.lambda_zcr * zcr_loss
        
        return total_loss, mse, zcr_loss

class MultiStepGRU(nn.Module):
    """
    GRU model for multi-step forecasting with explicit projection layer control.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(MultiStepGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.projection = nn.Linear(hidden_size, input_size)
    
    def forward(self, input_sequence, forecast_steps):
        """
        Forward pass with autoregressive forecasting.
        
        Args:
            input_sequence: (batch_size, n, input_size) - input sequence of length n
            forecast_steps: int - number of steps p to forecast
        
        Returns:
            forecasts: (batch_size, p, input_size) - forecasted sequence
        """
        # Phase 1: Process input sequence to build up hidden state
        _, hidden = self.gru(input_sequence)
        
        # Phase 2: Generate forecasts autoregressively
        forecasts = []
        current_input = input_sequence[:, -1:, :]  # Last timestep as starting point
        current_hidden = hidden
        
        for _ in range(forecast_steps):
            # GRU forward step
            gru_output, current_hidden = self.gru(current_input, current_hidden)
            
            # Project to output space
            prediction = self.projection(gru_output)
            
            forecasts.append(prediction.squeeze(1))  # Remove time dimension
            current_input = prediction  # Use prediction as next input
        
        # Stack forecasts: (batch_size, forecast_steps, input_size)
        return torch.stack(forecasts, dim=1)
    
    def __str__(self):
        return f"MultiStepGRU_h{self.hidden_size}_l{self.num_layers}"

class GIN(DynaSDBase):
    """
    GIN (Gated recurent unit with Integrated spatio-temporal modeling of Neural dynamic divergence) model with GRU and combined time/frequency domain loss.
    """

    def __init__(self, 
                 # Model architecture
                 hidden_ratio=0.5,
                 num_layers=1,
                 dropout=0.0,
                 
                 # Sequence parameters
                 input_length=16,       # n: Length of input sequence
                 forecast_horizon=16,   # p: Length of forecast sequence
                 stride=16,             # Stride for sequence generation
                 
                 # Data parameters
                 fs=256,
                 w_size=1, 
                 w_stride=0.5,
                 
                 # Training parameters
                 num_epochs=50,
                 batch_size=32,
                 lr=0.001,
                 
                 # Loss function parameters
                 lambda_zcr=0.1,        # Weight for zero-crossing rate loss
                 
                 # Other parameters
                 use_cuda=False,
                 grad_clip_norm=1.0,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, **kwargs)
        
        # Store all parameters
        self.hidden_ratio = hidden_ratio
        self.input_length = input_length  # n
        self.forecast_horizon = forecast_horizon  # p
        self.stride = stride

        self.num_layers = num_layers
        self.dropout = dropout

        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_zcr = lambda_zcr
        self.grad_clip_norm = grad_clip_norm
        
        # Device setup
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU instead.")
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Model and training state
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.is_fitted = False
        
        # Training history
        self.train_losses = []
        self.train_mse_losses = []
        self.train_zcr_losses = []
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate model parameters for compatibility and to minimize data loss"""
        total_length = self.input_length + self.forecast_horizon
        window_samples = int(self.w_size * self.fs)
        window_stride_samples = int(self.w_stride * self.fs)
        
        # Basic validation
        if total_length > window_samples:
            raise ValueError(
                f"input_length ({self.input_length}) + forecast_horizon "
                f"({self.forecast_horizon}) = {total_length} must be <= "
                f"w_size * fs = {window_samples}"
            )
        
        if self.stride <= 0:
            raise ValueError("stride must be positive")
            
        if self.input_length <= 0 or self.forecast_horizon <= 0:
            raise ValueError("input_length and forecast_horizon must be positive")
            
        if self.lambda_zcr < 0:
            raise ValueError("lambda_zcr must be non-negative")
        
        # Data efficiency warnings and recommendations
        sequences_per_window = (window_samples - total_length) // self.stride + 1
        if sequences_per_window <= 0:
            raise ValueError(
                f"No sequences can be created with current parameters. "
                f"Window has {window_samples} samples, need {total_length}, stride={self.stride}"
            )
        
        # Check for data wastage
        unused_samples_per_window = (window_samples - total_length) % self.stride
        if unused_samples_per_window > 0:
            waste_percentage = (unused_samples_per_window / window_samples) * 100
            if waste_percentage > 10:  # Warn if wasting more than 10% of data
                warnings.warn(
                    f"Current parameters waste {unused_samples_per_window} samples "
                    f"({waste_percentage:.1f}%) per window. Consider adjusting stride "
                    f"from {self.stride} to {(window_samples - total_length) // sequences_per_window} "
                    f"to minimize data loss."
                )
        
        # Optimal stride suggestion
        optimal_stride = (window_samples - total_length) // sequences_per_window
        if optimal_stride != self.stride and optimal_stride > 0:
            efficiency_improvement = (unused_samples_per_window / self.stride) * 100
            if efficiency_improvement > 5:  # Only suggest if meaningful improvement
                print(f"INFO: For maximum data utilization, consider stride={optimal_stride} "
                      f"(current: {self.stride}). This would create {sequences_per_window} "
                      f"sequences per window with minimal waste.")
        
        # Store calculated values for reference
        self._window_samples = window_samples
        self._sequences_per_window = sequences_per_window
        self._unused_samples_per_window = unused_samples_per_window
    
    def _create_sequences(self, data):
        """
        Create sequences for training/inference with specified stride.
        """
        data_np = data.to_numpy()
        n_samples, n_channels = data_np.shape
        
        # Get moving window clips
        nwins = num_wins(n_samples, self.fs, self.w_size, self.w_stride)
        win_length = int(self.w_size * self.fs)
        
        all_inputs = []
        all_targets = []
        all_win_info = []
        
        for win_idx in range(nwins):
            win_start = int(win_idx * self.w_stride * self.fs)
            win_end = win_start + win_length
            
            if win_end > n_samples:
                break
                
            win_data = data_np[win_start:win_end, :]
            win_time = win_start / self.fs
            
            # Create sequences within this window using specified stride
            total_seq_length = self.input_length + self.forecast_horizon
            
            for seq_start in range(0, len(win_data) - total_seq_length + 1, self.stride):
                seq_end = seq_start + total_seq_length
                
                if seq_end <= len(win_data):
                    full_sequence = win_data[seq_start:seq_end, :]
                    
                    # Split into input and target
                    input_seq = full_sequence[:self.input_length, :]
                    target_seq = full_sequence[self.input_length:, :]
                    
                    all_inputs.append(input_seq)
                    all_targets.append(target_seq)
                    all_win_info.append({
                        'win_idx': win_idx,
                        'win_time': win_time,
                        'seq_start': seq_start,
                        'seq_time': win_time + seq_start / self.fs,
                        'global_start': win_start + seq_start,
                        'global_input_range': (win_start + seq_start, win_start + seq_start + self.input_length),
                        'global_forecast_range': (win_start + seq_start + self.input_length, win_start + seq_end),
                        'input_range': (seq_start, seq_start + self.input_length),
                        'forecast_range': (seq_start + self.input_length, seq_end)
                    })
        
        if len(all_inputs) == 0:
            raise ValueError(
                f"No valid sequences created. Check parameters: "
                f"w_size={self.w_size}s ({win_length} samples), "
                f"total_seq_length={total_seq_length}, stride={self.stride}"
            )
        
        inputs = torch.FloatTensor(np.array(all_inputs))
        targets = torch.FloatTensor(np.array(all_targets))
        
        return inputs, targets, all_win_info
    
    def _create_data_loaders(self, inputs, targets):
        """Create training data loader"""
        dataset = TensorDataset(inputs, targets)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True if len(dataset) > self.batch_size else False
        )
        
        return train_loader
    
    def fit(self, X):
        """Fit the multi-step forecasting model"""
        print(f"Fitting MultiStepForecaster:")
        print(f"  Architecture: n={self.input_length}, p={self.forecast_horizon}, stride={self.stride}")
        print(f"  Hidden size ratio: {self.hidden_ratio}, Layers: {self.num_layers}")
        print(f"  Lambda ZCR: {self.lambda_zcr}")
        
        input_size = X.shape[1]
        hidden_size = int(np.floor(self.hidden_ratio * input_size))
        # Initialize model
        self.model = MultiStepGRU(
            input_size=input_size,
            hidden_size= hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        
        print(f"  Model: {self.model}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup training components
        self.criterion = CombinedLoss(lambda_zcr=self.lambda_zcr)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
        )
        
        # Scale data
        self._fit_scaler(X)
        X_scaled = self._scaler_transform(X)
        
        # Create sequences
        inputs, targets, win_info = self._create_sequences(X_scaled)
        print(f"  Created {len(inputs)} sequences from {len(set(info['win_idx'] for info in win_info))} windows")
        
        # Calculate sequences per window for verification
        seqs_per_win = len(inputs) / len(set(info['win_idx'] for info in win_info))
        print(f"  Average sequences per window: {seqs_per_win:.1f}")
        
        # Create data loader
        train_loader = self._create_data_loaders(inputs, targets)
        print(f"  Train batches: {len(train_loader)}")
        
        # Training loop
        self.train_losses = []
        self.train_mse_losses = []
        self.train_zcr_losses = []
        
        print("Starting training...")
        pbar = tqdm(range(self.num_epochs), desc="Training")
        
        for epoch in pbar:
            # Training epoch
            self.model.train()
            epoch_losses = []
            epoch_mse_losses = []
            epoch_zcr_losses = []
            
            for inputs_batch, targets_batch in train_loader:
                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(inputs_batch, self.forecast_horizon)
                
                # Compute combined loss
                total_loss, mse_loss, zcr_loss = self.criterion(predictions, targets_batch)
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.grad_clip_norm
                    )
                
                self.optimizer.step()
                
                epoch_losses.append(total_loss.item())
                epoch_mse_losses.append(mse_loss.item())
                epoch_zcr_losses.append(zcr_loss.item())
            
            # Record epoch averages
            train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            train_mse = np.mean(epoch_mse_losses) if epoch_mse_losses else float('inf')
            train_zcr = np.mean(epoch_zcr_losses) if epoch_zcr_losses else float('inf')
            
            self.train_losses.append(train_loss)
            self.train_mse_losses.append(train_mse)
            self.train_zcr_losses.append(train_zcr)
            
            pbar.set_postfix({
                'total': f'{train_loss:.4f}',
                'mse': f'{train_mse:.4f}',
                'zcr': f'{train_zcr:.4f}'
            })
        
        self.is_fitted = True
        final_loss = self.train_losses[-1]
        print(f"Training completed. Final loss: {final_loss:.4f}")
        
        # Store info for later analysis
        self.last_win_info = win_info
    
    def forward(self, X, aggregate_within_windows=True):
        """
        Run inference and return forecasting errors as features per channel.
        
        Args:
            X: Input data
            aggregate_within_windows: If True, aggregate sequence-level errors within each window
                                    to create window-level features. If False, return sequence-level features.
        
        Returns:
            If aggregate_within_windows=False:
                Dict[str, pd.DataFrame]: Dictionary with error types as keys ('MSE', 'MAE', etc.)
                and DataFrames with channels as columns as values.
                
            If aggregate_within_windows=True:
                Dict[str, Dict[str, pd.DataFrame]]: Nested dictionary structure:
                {aggregation_type: {error_type: DataFrame}}
                e.g., result['Mean']['MSE'] gives mean MSE per channel across windows
        """
        assert self.is_fitted, "Must fit model before running inference"
        
        X_scaled = self._scaler_transform(X)
        inputs, targets, win_info = self._create_sequences(X_scaled)
        
        # Run inference
        self.model.eval()
        all_errors = []
        
        with torch.no_grad():
            dataset = TensorDataset(inputs, targets)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            for input_batch, target_batch in dataloader:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                # Get predictions
                predictions = self.model(input_batch, self.forecast_horizon)
                
                # Compute error metrics PER CHANNEL (batch_size, n_channels)
                mse = torch.mean((predictions - target_batch) ** 2, dim=1)  # Only average over time dimension
                mae = torch.mean(torch.abs(predictions - target_batch), dim=1)  # Only average over time dimension
                rmse = torch.sqrt(mse)
                
                # Zero-crossing rate difference per channel
                pred_zcr = zero_crossing_rate(predictions)  # (batch_size, n_channels)
                target_zcr = zero_crossing_rate(target_batch)  # (batch_size, n_channels)
                zcr_mse = (pred_zcr - target_zcr) ** 2  # Keep per channel, don't average
                
                # Max error per channel
                max_error = torch.max(torch.abs(predictions - target_batch), dim=1)[0]  # Max over time, keep channels
                
                # Combined loss per sample (we'll compute per channel separately)
                # For now, compute total combined loss per sample (this will be expanded per channel below)
                combined_per_sample = torch.zeros(predictions.size(0)).to(self.device)
                for i in range(predictions.size(0)):
                    sample_loss, _, _ = self.criterion(
                        predictions[i:i+1], target_batch[i:i+1]
                    )
                    combined_per_sample[i] = sample_loss
                
                # Reshape all metrics to (batch_size * n_channels,) for storage
                batch_size, n_channels = mse.shape
                mse_flat = mse.flatten()
                mae_flat = mae.flatten()
                rmse_flat = rmse.flatten()
                zcr_mse_flat = zcr_mse.flatten()
                max_error_flat = max_error.flatten()
                
                # Repeat combined loss for each channel (since it's computed per sample)
                combined_flat = combined_per_sample.repeat_interleave(n_channels)
                
                # Stack all metrics
                errors = torch.stack([
                    mse_flat, mae_flat, rmse_flat, zcr_mse_flat, max_error_flat, combined_flat
                ], dim=1).cpu()
                all_errors.append(errors)
        
        # Combine all errors
        all_errors = torch.cat(all_errors, dim=0).numpy()
        
        if aggregate_within_windows:
            # Group sequences by window and aggregate
            return self._aggregate_window_features(all_errors, win_info, inputs.shape[-1])
        else:
            # Return sequence-level features
            return self._create_sequence_features(all_errors, win_info, inputs.shape[-1])
    
    def _create_sequence_features(self, all_errors, win_info, n_channels):
        """Create feature dictionary with error types as keys and channel DataFrames as values"""
        # Reshape errors to have one row per sequence, with all channel metrics as columns
        n_sequences = len(win_info)
        errors_reshaped = all_errors.reshape(n_sequences, -1)
        
        # Define error types and their order
        error_types = ['MSE', 'MAE', 'RMSE', 'ZCR_MSE', 'MaxError', 'CombinedLoss']
        n_error_types = len(error_types)
        
        # Create channel column names
        channel_columns = [f'Ch{ch}' for ch in range(n_channels)]
        
        # Create dictionary of DataFrames
        feature_dict = {}
        
        for i, error_type in enumerate(error_types):
            # Extract data for this error type across all channels
            start_idx = i * n_channels
            end_idx = (i + 1) * n_channels
            error_data = errors_reshaped[:, start_idx:end_idx]
            
            # Create DataFrame for this error type
            feature_dict[error_type] = pd.DataFrame(
                error_data, 
                columns=channel_columns,
                index=range(n_sequences)
            )
        
        self.feature_dict = feature_dict
        self.last_inference_info = win_info
        
        return feature_dict
    
    def _aggregate_window_features(self, all_errors, win_info, n_channels):
        """
        Aggregate sequence-level features within each window to create window-level features.
        Returns nested dictionary: {aggregation_type: {error_type: DataFrame}}
        """
        # Reshape errors to have one row per sequence, with all channel metrics as columns
        n_sequences = len(win_info)
        errors_reshaped = all_errors.reshape(n_sequences, -1)
        
        # Define error types and their order
        error_types = ['MSE', 'MAE', 'RMSE', 'ZCR_MSE', 'MaxError', 'CombinedLoss']
        n_error_types = len(error_types)
        channel_columns = [f'Ch{ch}' for ch in range(n_channels)]
        
        # Create a DataFrame with sequence-level errors and window information
        temp_df = pd.DataFrame(errors_reshaped)
        temp_df['win_idx'] = [info['win_idx'] for info in win_info]
        temp_df['win_time'] = [info['win_time'] for info in win_info]
        temp_df['seq_time'] = [info['seq_time'] for info in win_info]
        
        # Prepare aggregation functions
        agg_funcs = {
            'Mean': np.mean,
            'Max': np.max,
            'Min': np.min,
            'Std': np.std
        }
        
        # Initialize nested dictionary structure
        feature_dict = {agg_type: {} for agg_type in agg_funcs.keys()}
        window_info_list = []
        
        # Get unique windows for consistent ordering
        unique_windows = sorted(temp_df['win_idx'].unique())
        
        for agg_name, agg_func in agg_funcs.items():
            # Initialize error type dictionaries for this aggregation
            for error_type in error_types:
                feature_dict[agg_name][error_type] = []
        
        # Process each window
        for win_idx in unique_windows:
            win_data = temp_df[temp_df['win_idx'] == win_idx]
            
            # Store window metadata (only once)
            if len(window_info_list) < len(unique_windows):
                window_info_list.append({
                    'win_idx': win_idx,
                    'win_time': win_data['win_time'].iloc[0],
                    'n_sequences': len(win_data),
                    'seq_time_start': win_data['seq_time'].min(),
                    'seq_time_end': win_data['seq_time'].max()
                })
            
            # Get error metrics (excluding window metadata columns)
            error_cols = [col for col in temp_df.columns if col not in ['win_idx', 'win_time', 'seq_time']]
            win_errors = win_data[error_cols].values
            
            # Apply each aggregation function
            for agg_name, agg_func in agg_funcs.items():
                aggregated_errors = agg_func(win_errors, axis=0)
                
                # Split aggregated errors by error type and store
                for i, error_type in enumerate(error_types):
                    start_idx = i * n_channels
                    end_idx = (i + 1) * n_channels
                    error_data = aggregated_errors[start_idx:end_idx]
                    feature_dict[agg_name][error_type].append(error_data)
        
        # Convert lists to DataFrames
        n_windows = len(unique_windows)
        for agg_name in agg_funcs.keys():
            for error_type in error_types:
                feature_dict[agg_name][error_type] = pd.DataFrame(
                    np.array(feature_dict[agg_name][error_type]),
                    columns=channel_columns,
                    index=range(n_windows)
                )
        
        self.feature_dict = feature_dict
        self.last_inference_info = window_info_list  # Now contains window-level info
        
        return feature_dict
    
    def predict(self, X, return_details=False):
        """Generate actual forecasts"""
        assert self.is_fitted, "Must fit model before making predictions"
        
        X_scaled = self._scaler_transform(X)
        inputs, _, win_info = self._create_sequences(X_scaled)
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            dataset = TensorDataset(inputs, torch.zeros_like(inputs[:, :1, :]))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            for input_batch, _ in dataloader:
                input_batch = input_batch.to(self.device)
                predictions = self.model(input_batch, self.forecast_horizon)
                all_predictions.append(predictions.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        
        if return_details:
            return all_predictions, win_info
        else:
            return all_predictions
    
    def get_temporal_alignment(self):
        """
        Return temporal alignment information for the last inference.
        
        Returns:
            pd.DataFrame: DataFrame with temporal alignment info for each sequence/window.
                        Columns include global sample indices, times, and window info.
        """
        if not hasattr(self, 'last_inference_info'):
            raise ValueError("No inference has been run yet. Call forward() first.")
        
        alignment_data = []
        for i, info in enumerate(self.last_inference_info):
            # Handle both sequence-level and window-level info
            if 'n_sequences' in info:  # Window-level aggregated data
                alignment_data.append({
                    'index': i,
                    'win_idx': info['win_idx'],
                    'win_start_time': info['win_time'],
                    'n_sequences_in_window': info['n_sequences'],
                    'seq_time_start': info['seq_time_start'],
                    'seq_time_end': info['seq_time_end'],
                    'type': 'window_aggregated'
                })
            else:  # Sequence-level data
                alignment_data.append({
                    'index': i,
                    'win_idx': info['win_idx'],
                    'win_start_time': info['win_time'],
                    'seq_start_time': info['seq_time'],
                    'global_input_start': info['global_input_range'][0],
                    'global_input_end': info['global_input_range'][1],
                    'global_forecast_start': info['global_forecast_range'][0],
                    'global_forecast_end': info['global_forecast_range'][1],
                    'input_length': self.input_length,
                    'forecast_horizon': self.forecast_horizon,
                    'input_time_start': info['seq_time'],
                    'input_time_end': info['seq_time'] + self.input_length / self.fs,
                    'forecast_time_start': info['seq_time'] + self.input_length / self.fs,
                    'forecast_time_end': info['seq_time'] + (self.input_length + self.forecast_horizon) / self.fs,
                    'type': 'sequence_level'
                })
        
        return pd.DataFrame(alignment_data)
    
    def get_feature_summary(self):
        """
        Return a summary of the feature structure from the last forward() call.
        
        Returns:
            dict: Summary information about the feature structure
        """
        if not hasattr(self, 'feature_dict'):
            raise ValueError("No features computed yet. Call forward() first.")
        
        if isinstance(list(self.feature_dict.values())[0], dict):
            # Nested structure (aggregated)
            summary = {
                'structure': 'aggregated',
                'aggregation_types': list(self.feature_dict.keys()),
                'error_types': list(list(self.feature_dict.values())[0].keys()),
                'n_windows': len(list(list(self.feature_dict.values())[0].values())[0]),
                'n_channels': len(list(list(self.feature_dict.values())[0].values())[0].columns),
                'channel_names': list(list(list(self.feature_dict.values())[0].values())[0].columns)
            }
        else:
            # Flat structure (sequence-level)
            summary = {
                'structure': 'sequence_level',
                'error_types': list(self.feature_dict.keys()),
                'n_sequences': len(list(self.feature_dict.values())[0]),
                'n_channels': len(list(self.feature_dict.values())[0].columns),
                'channel_names': list(list(self.feature_dict.values())[0].columns)
            }
        
        return summary
    
    def get_loss_history(self):
        """Return detailed loss history"""
        if not self.is_fitted:
            return None
        
        history = {
            'train_total': self.train_losses,
            'train_mse': self.train_mse_losses,
            'train_zcr': self.train_zcr_losses,
        }
        
        return history