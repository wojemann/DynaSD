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

class NDDBase(DynaSDBase):
    """
    Base class for neural dynamic divergence models.
    Extends DynaSDBase with common neural forecasting functionality.
    """
    
    def __init__(self, fs=256, w_size=1, w_stride=0.5, use_cuda=False, **kwargs):
        # Extract training-specific parameters before passing to parent
        training_params = ['early_stopping', 'val_split', 'patience', 'tolerance']
        training_kwargs = {}
        
        for param in training_params:
            if param in kwargs:
                training_kwargs[param] = kwargs.pop(param)
        
        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, **kwargs)
        
        # Device setup
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU instead.")
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Model state
        self.model = None
        self.is_fitted = False
        
        # Store training parameters with defaults
        self.early_stopping = training_kwargs.get('early_stopping', False)
        self.val_split = training_kwargs.get('val_split', 0.2)
        self.patience = training_kwargs.get('patience', 5)
        self.tolerance = training_kwargs.get('tolerance', 1e-4)
        
    def _train_model_multistep(self, X, model, sequence_length, forecast_length, num_epochs, batch_size, lr, 
                              early_stopping=False, val_split=0.2, patience=5, tolerance=1e-4):
        """
        Standardized training loop for multi-step forecasting models.
        
        Args:
            X: Input DataFrame
            model: PyTorch model to train
            sequence_length: Length of input/forecast sequences
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            early_stopping: Whether to use early stopping
            val_split: Fraction of data to use for validation
            patience: Number of epochs to wait before early stopping
            tolerance: Minimum improvement threshold for early stopping
        """
        print(f"Training {model.__class__.__name__} model:")
        print(f"  Sequence length: {sequence_length}, Forecast length: {forecast_length}")
        print(f"  Early stopping: {early_stopping}")
        
        input_size = X.shape[1]
        
        # Scale data
        self._fit_scaler(X)
        X_scaled = self._scaler_transform(X)
        
        # Prepare sequences from continuous data
        input_data, target_data = self._prepare_multistep_sequences(X_scaled, sequence_length, forecast_length)
        print(f"  Created {len(input_data)} sequences")
        
        # Split data for validation if early stopping is enabled
        if early_stopping:
            val_size = int(len(input_data) * val_split)
            train_size = len(input_data) - val_size
            
            # Random split
            indices = torch.randperm(len(input_data))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            train_inputs = input_data[train_indices]
            train_targets = target_data[train_indices]
            val_inputs = input_data[val_indices]
            val_targets = target_data[val_indices]
            
            print(f"  Training sequences: {len(train_inputs)}, Validation sequences: {len(val_inputs)}")

        else:
            train_inputs = input_data
            train_targets = target_data
            val_inputs = None
            val_targets = None
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(train_inputs, train_targets)
        train_batch_size = len(train_dataset) if batch_size == 'full' else batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        
        if early_stopping:
            val_dataset = TensorDataset(val_inputs, val_targets)
            val_batch_size = len(val_dataset) if batch_size == 'full' else batch_size
            val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        
        # Setup training
        mse_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        print("Starting training...")
        pbar = tqdm(range(num_epochs), desc="Training")
        
        for epoch in pbar:
            model.train()
            epoch_losses = []
            
            # Training phase
            for inputs, targets in train_dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(inputs, forecast_length)
                
                # MSE loss (averaged across channels for training)
                mse_loss = mse_criterion(predictions, targets)
                mse_loss.backward()
                optimizer.step()
                
                epoch_losses.append(mse_loss.item())
            
            avg_train_loss = np.mean(epoch_losses)
            
            # Validation phase
            if early_stopping:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for inputs, targets in val_dataloader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        predictions = model(inputs, forecast_length)
                        mse_loss = mse_criterion(predictions, targets)
                        val_losses.append(mse_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                
                # Early stopping check
                if avg_val_loss < best_val_loss - tolerance:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                pbar.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}',
                    'patience': patience_counter
                })
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            else:
                pbar.set_postfix({'loss': f'{avg_train_loss:.4f}'})

        self.is_fitted = True
        mse,corr = self._get_features(X)
        print(mse.iloc[-5:,:].head())
        print(mse.shape)
        dist_params = {ch:dict() for ch in X.columns}
        for ch in X.columns:
            mse_x = mse[ch].to_numpy().reshape(-1,1)
            corr_x = corr[ch].to_numpy().reshape(-1,1)
            f = np.concatenate((mse_x,corr_x),axis=1)
            m = np.mean(f,axis=0)
            C = f - m
            _, R = np.linalg.qr(C) 
            dist_params[ch]['m'] = m
            dist_params[ch]['R'] = R
            dist_params[ch]['n'] = f.shape[0]

        self.dist_params = dist_params
 
        print("Training completed")
        
    # def _aggregate_sequences_to_windows(self, seq_results, X):
    #     """
    #     Aggregate sequence-level results into w_size/w_stride windows.
        
    #     Args:
    #         seq_results: List of dictionaries with sequence results
    #         X: Original input DataFrame
            
    #     Returns:
    #         tuple: (mse_df, zcr_df, combined_df, corr_df) - Four DataFrames with channel names as columns
    #     """
    #     n_samples = len(X)
    #     n_channels = X.shape[1]
        
    #     # Get window times using base class utility
    #     nwins = num_wins(n_samples, self.fs, self.w_size, self.w_stride)
        
    #     # Calculate window start times
    #     window_times = []
    #     for win_idx in range(nwins):
    #         win_start_time = win_idx * self.w_stride
    #         win_end_time = win_start_time + self.w_size
    #         window_times.append((win_start_time, win_end_time))
        
    #     # Aggregate sequences into windows
    #     window_features = {
    #         'mse': np.full((nwins, n_channels), np.nan),
    #         'corr': np.full((nwins, n_channels), np.nan)
    #     }
        
    #     for win_idx, (win_start, win_end) in enumerate(window_times):
    #         # Find sequences that fall within this window
    #         sequences_in_window = []
    #         for seq_result in seq_results:
    #             seq_start = seq_result['target_start_time']
    #             seq_end = seq_result['target_end_time']
                
    #             # Check if sequence overlaps with window
    #             if seq_start < win_end and seq_end > win_start:
    #                 sequences_in_window.append(seq_result)
            
    #         # Aggregate if we have sequences in this window
    #         if sequences_in_window:
    #             # mse_values = np.array([s['mse'] for s in sequences_in_window])
                
    #             # # Take mean across sequences in window
    #             # window_features['mse'][win_idx] = np.mean(mse_values, axis=0)
                
    #             # Correlation requires per-channel predicted/target sequences
    #             if 'predicted_seq' in sequences_in_window[0] and 'target_seq' in sequences_in_window[0]:
    #                 try:
    #                     combined_predicted = np.concatenate([s['predicted_seq'] for s in sequences_in_window], axis=0).T
    #                     combined_target = np.concatenate([s['target_seq'] for s in sequences_in_window], axis=0).T
    #                     window_features['corr'][win_idx] = np.array([
    #                         np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0
    #                         for a, b in zip(combined_predicted, combined_target)
    #                     ])
    #                     window_features['mse'][win_idx] = np.sqrt(np.mean((combined_predicted - combined_target) ** 2, axis=1))
    #                 except Exception:
    #                     # Fallback: leave as NaN if concatenation fails
    #                     pass
        
    #     # Create separate DataFrames for each feature type
    #     mse_df = pd.DataFrame(window_features['mse'], columns=X.columns)
    #     corr_df = pd.DataFrame(window_features['corr'], columns=X.columns)
        
    #     return mse_df, corr_df
    def _aggregate_sequences_to_windows(self, seq_results, X):
        """
        Aggregate sequence-level results into w_size/w_stride windows with consistent alignment.
        
        Key fix: Calculate windows based on actual sequence coverage, not total data length.
        This ensures consistent windowing whether called from fit() or forward().
        
        Args:
            seq_results: List of dictionaries with sequence results
            X: Original input DataFrame
            
        Returns:
            tuple: (mse_df, corr_df) - DataFrames with channel names as columns
        """
        n_channels = X.shape[1]
        
        # Find the actual time range covered by sequences
        if not seq_results:
            raise ValueError("No sequences provided for aggregation")
        
        
        max_sequence_time = max(s['target_end_time'] for s in seq_results) if seq_results else 0
        max_samples_needed = int(max_sequence_time * self.fs) + 1
        nwins = num_wins(max_samples_needed, self.fs, self.w_size, self.w_stride)
        
        window_times = []
        for win_idx in range(nwins):
            win_start_time = win_idx * self.w_stride
            win_end_time = win_start_time + self.w_size
            window_times.append((win_start_time, win_end_time))
        # Create window time pairs
        window_times = [(start, start + self.w_size) for start in window_starts]
        
        # Initialize feature arrays
        window_features = {
            'mse': np.full((nwins, n_channels), np.nan),
            'corr': np.full((nwins, n_channels), np.nan)
        }
        
        # Aggregate sequences into windows using absolute time overlap
        for win_idx, (win_start, win_end) in enumerate(window_times):
            # Find sequences that overlap with this window
            sequences_in_window = []
            
            for seq_result in seq_results:
                seq_start = seq_result['target_start_time']
                seq_end = seq_result['target_end_time']
                
                # Check for time overlap (any overlap counts)
                if seq_start < win_end and seq_end > win_start:
                    sequences_in_window.append(seq_result)
            
            # Aggregate if we have sequences in this window
            if sequences_in_window:
                if 'predicted_seq' in sequences_in_window[0] and 'target_seq' in sequences_in_window[0]:
                    try:
                        # Concatenate all sequence predictions/targets for this window
                        combined_predicted = np.concatenate([s['predicted_seq'] for s in sequences_in_window], axis=0).T
                        combined_target = np.concatenate([s['target_seq'] for s in sequences_in_window], axis=0).T
                        
                        # Calculate per-channel metrics
                        window_features['corr'][win_idx] = np.array([
                            np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0
                            for a, b in zip(combined_predicted, combined_target)
                        ])
                        window_features['mse'][win_idx] = np.sqrt(np.mean((combined_predicted - combined_target) ** 2, axis=1))
                        
                    except Exception:
                        # Fallback: leave as NaN if concatenation fails
                        pass
        
        # Create DataFrames
        mse_df = pd.DataFrame(window_features['mse'], columns=X.columns)
        corr_df = pd.DataFrame(window_features['corr'], columns=X.columns)
        
        # Store consistent window times for external access
        self.window_start_times = np.array(window_starts)
        
        return mse_df, corr_df

    def get_consistent_window_times(self, data_length_samples):
        """
        Get window start times that are consistent regardless of input parameters.
        Use this instead of calculating windows based on num_wins().
        
        Args:
            data_length_samples: Number of samples in the data
            
        Returns:
            np.array: Window start times in seconds
        """
        data_duration = data_length_samples / self.fs
        
        window_starts = []
        current_time = 0.0
        
        while current_time + self.w_size <= data_duration:
            window_starts.append(current_time)
            current_time += self.w_stride
        
        return np.array(window_starts)

    # Modified forward method to use consistent windowing
    def forward_consistent(self, X):
        """
        Modified forward pass using consistent windowing.
        """
        assert self.is_fitted, "Must fit model before running inference"
        mse_df, corr_df = self._get_features_consistent(X)
        
        ndd = pd.DataFrame()
        for ch in X.columns:
            mse_y = mse_df[ch].to_numpy().reshape(-1,1)
            corr_y = corr_df[ch].to_numpy().reshape(-1,1)
            f = np.concatenate((mse_y,corr_y),axis=1)
            m = self.dist_params[ch]['m']
            R = self.dist_params[ch]['R']
            ri = np.linalg.solve(R.T, (f - m).T)
            ndd[ch] = np.sum(ri * ri, axis=0) * (self.dist_params[ch]['n'] - 1)
        
        # Store consistent window times
        self.time_wins = self.window_start_times
        
        # Store for backward compatibility
        self.mse_df = mse_df
        self.corr_df = corr_df
        self.ndd_df = ndd

        return self.ndd_df

    def _get_features_consistent(self, X):
        """
        Modified _get_features using consistent windowing.
        """
        X_scaled = self._scaler_transform(X)
        input_data, target_data, seq_positions = self._prepare_multistep_sequences(X_scaled, self.sequence_length, self.forecast_length, ret_positions=True)
        
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
                predictions = self.model(inputs, self.forecast_length)
                
                # Calculate per-channel losses for each sequence in batch
                batch_size_actual, seq_len, n_channels = predictions.shape
                
                for batch_idx in range(batch_size_actual):
                    seq_idx = batch_start + batch_idx
                    seq_pos = seq_positions[seq_idx]
                    
                    # Store results with temporal position and sequences for correlation
                    seq_results.append({
                        'seq_idx': seq_idx,
                        'seq_start_time': seq_pos['input_time_start'],
                        'seq_end_time': seq_pos['input_time_end'],
                        'target_start_time': seq_pos['target_time_start'],
                        'target_end_time': seq_pos['target_time_end'],
                        'predicted_seq': predictions[batch_idx].cpu().numpy(),
                        'target_seq': targets[batch_idx].cpu().numpy()
                    })
                
                batch_start += batch_size_actual
        
        # Use consistent window aggregation
        mse_df, corr_df = self._aggregate_sequences_to_windows_consistent(seq_results, X)
        return mse_df, corr_df

    def _prepare_multistep_sequences(self, data, sequence_length, forecast_length = 1,ret_positions=False):
        """
        Prepare sequences from continuous data for multi-step forecasting.
        Used by GIN, LiNDDA, MINDA.
        
        Args:
            data: Input DataFrame
            sequence_length: Length of input/forecast sequences
            ret_positions: Whether to return position information
            
        Returns:
            tuple: (input_data, target_data, [seq_positions])
        """
        data_np = data.to_numpy()
        n_samples, _ = data_np.shape
        
        # Create non-overlapping sequences with stride = sequence_length
        stride = forecast_length

        total_seq_length = sequence_length + forecast_length  # input + target
        
        # Calculate how many sequences we can create
        n_sequences = (n_samples - total_seq_length) // stride + 1
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough data for even one sequence. Need at least {total_seq_length} samples.")
        
        print(f"Creating {n_sequences} non-overlapping sequences from continuous data")
        
        all_inputs = []
        all_targets = []
        seq_positions = []  # Track where each sequence starts in original data
        
        for seq_idx in range(n_sequences):
            seq_start = seq_idx * stride
            input_end = seq_start + sequence_length
            target_end = input_end + forecast_length
            
            # Extract input and target sequences
            input_seq = data_np[seq_start:input_end, :]
            target_seq = data_np[input_end:target_end, :]
            
            all_inputs.append(input_seq)
            all_targets.append(target_seq)
            seq_positions.append({
                'seq_idx': seq_idx,
                'input_start': seq_start,
                'input_end': input_end,
                'target_start': input_end,
                'target_end': target_end,
                'seq_time_start': seq_start / self.fs,
                'seq_time_end': input_end / self.fs,
                'input_time_start': seq_start / self.fs,
                'target_time_start': input_end / self.fs
            })
        
        input_data = torch.FloatTensor(np.array(all_inputs))
        target_data = torch.FloatTensor(np.array(all_targets))
        
        if ret_positions:
            return input_data, target_data, seq_positions
        else:
            return input_data, target_data
    
    def predict_multistep(self, X):
        """
        Generate forecasted time series by concatenating multi-step sequence predictions.
        Used by GIN, LiNDDA, MINDA.
        
        Returns:
            DataFrame: Forecasted time series with same shape as input (minus initial sequence)
        """
        assert self.is_fitted, "Must fit model before making predictions"
        assert hasattr(self, 'sequence_length'), "Model must have sequence_length attribute for multi-step prediction"
        
        X_scaled = self._scaler_transform(X)
        input_data, _, seq_positions = self._prepare_multistep_sequences(X_scaled, self.sequence_length, self.forecast_length, ret_positions=True)
        
        # Run inference to get all predictions
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            dataset = TensorDataset(input_data, torch.zeros_like(input_data))  # Dummy targets
            batch_size = len(dataset) if self.batch_size == 'full' else self.batch_size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # In theory this will propagate the forecasting method implemented in the specific model classes
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                if hasattr(self.model, 'forward') and 'forecast_steps' in self.model.forward.__code__.co_varnames:
                    predictions = self.model(inputs, self.forecast_length)
                else:
                    predictions = self.model(inputs)
                all_predictions.append(predictions.cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        
        # Reconstruct time series by concatenating predictions
        n_samples = len(X)
        n_channels = X.shape[1]
        n_sequences = len(seq_positions)
        
        # Initialize output array (we can't predict the first sequence_length samples)
        reconstructed = np.full((n_samples, n_channels), np.nan)
        
        # Fill in predictions for each sequence
        for seq_idx in range(n_sequences):
            seq_pos = seq_positions[seq_idx]
            prediction = all_predictions[seq_idx]  # (sequence_length, n_channels)
            
            # Place prediction in the target region
            start_idx = seq_pos['target_start']
            end_idx = seq_pos['target_end']
            
            # Handle potential overlaps by averaging
            if np.any(~np.isnan(reconstructed[start_idx:end_idx])):
                # There's existing data, average with current prediction
                existing = reconstructed[start_idx:end_idx]
                mask = ~np.isnan(existing)
                reconstructed[start_idx:end_idx][mask] = (existing[mask] + prediction[mask]) / 2
                reconstructed[start_idx:end_idx][~mask] = prediction[~mask]
            else:
                # No existing data, just place prediction
                reconstructed[start_idx:end_idx] = prediction
        
        # Convert back to DataFrame with same column names and index
        predicted_df = pd.DataFrame(
            reconstructed, 
            columns=X.columns,
            index=X.index
        )
        
        # Apply inverse scaling
        predicted_df_scaled = self.scaler.inverse_transform(predicted_df)
        
        return predicted_df_scaled
    
    def predict_singlestep(self, X):
        """
        Generate forecasted time series using single-step prediction approach.
        Used by original NDD model.
        
        Returns:
            DataFrame: Forecasted time series
        """
        assert self.is_fitted, "Must fit model before making predictions"
        assert hasattr(self, '_prepare_segment'), "Model must have _prepare_segment method for single-step prediction"
        
        X_scaled = self._scaler_transform(X)
        input_data, target_data, win_times = self._prepare_segment(X_scaled, ret_time=True)
        
        # Run inference
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            dataset = TensorDataset(input_data, target_data)
            batch_size = len(dataset) if self.batch_size == 'full' else self.batch_size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                predictions = self.model(inputs)
                all_predictions.append(predictions.cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        
        # Reshape to match NDD output structure
        nwins = len(win_times)
        nchannels = X.shape[1]
        j = int(self.w_size * self.fs - (self.train_win + self.pred_win) + 1)
        
        # Reshape predictions to (nwins, j, nchannels)
        predictions_reshaped = all_predictions.reshape((nwins, j, nchannels))
        
        # Create time series reconstruction
        win_length = int(self.w_size * self.fs)
        n_samples = len(X)
        reconstructed = np.full((n_samples, nchannels), np.nan)
        
        for win_idx in range(nwins):
            win_start_sample = int(win_idx * self.w_stride * self.fs)
            
            # Place predictions within the window
            for j_idx in range(j):
                pred_sample_idx = win_start_sample + j_idx + self.train_win
                if pred_sample_idx < n_samples:
                    if np.isnan(reconstructed[pred_sample_idx, 0]):
                        reconstructed[pred_sample_idx, :] = predictions_reshaped[win_idx, j_idx, :]
                    else:
                        # Average overlapping predictions
                        reconstructed[pred_sample_idx, :] = (
                            reconstructed[pred_sample_idx, :] + predictions_reshaped[win_idx, j_idx, :]
                        ) / 2
        
        # Convert back to DataFrame
        predicted_df = pd.DataFrame(
            reconstructed,
            columns=X.columns,
            index=X.index
        )
        
        # Apply inverse scaling
        predicted_df_scaled = self.scaler.inverse_transform(predicted_df)
        
        return predicted_df_scaled 