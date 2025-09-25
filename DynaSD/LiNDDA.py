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

class SkLinearForecaster:
    """
    Sklearn-based linear forecaster that uses LinearRegression for multi-step forecasting.
    """
    def __init__(self, input_size, sequence_length):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.model = LinearRegression()
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length * input_size)
            y: Target values of shape (n_samples, input_size)
        """
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict_step(self, input_sequence):
        """
        Predict single step given input sequence.
        
        Args:
            input_sequence: numpy array of shape (sequence_length, input_size)
        
        Returns:
            prediction: numpy array of shape (input_size,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Flatten input sequence
        flattened_input = input_sequence.flatten().reshape(1, -1)
        prediction = self.model.predict(flattened_input)
        return prediction[0]  # Return single prediction
    
    def predict_multistep(self, input_sequence, forecast_steps):
        """
        Generate multi-step forecasts using autoregressive approach.
        
        Args:
            input_sequence: numpy array of shape (sequence_length, input_size)
            forecast_steps: int, number of steps to forecast
        
        Returns:
            forecasts: numpy array of shape (forecast_steps, input_size)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        forecasts = []
        current_input = input_sequence.copy()
        
        for _ in range(forecast_steps):
            # Predict next step
            next_pred = self.predict_step(current_input)
            forecasts.append(next_pred)
            
            # Update input sequence for next prediction
            if self.sequence_length > 1:
                # Roll the sequence: remove first timestep, add prediction
                current_input = np.vstack([current_input[1:], next_pred])
            else:
                # For sequence_length=1, just replace with prediction
                current_input = next_pred.reshape(1, -1)
        
        return np.array(forecasts)
    
    def __call__(self, input_sequence, forecast_steps):
        """
        Make the model callable like PyTorch models.
        This allows the model to be used with model(inputs, forecast_steps) syntax.
        
        Args:
            input_sequence: torch tensor or numpy array of shape (batch_size, sequence_length, input_size)
            forecast_steps: int, number of steps to forecast
            
        Returns:
            torch tensor of shape (batch_size, forecast_steps, input_size)
        """
        # Handle torch tensor input by converting to numpy
        if hasattr(input_sequence, 'numpy'):
            input_sequence = input_sequence.numpy()
        
        # Handle batch dimension - process each sequence in the batch
        if input_sequence.ndim == 3:  # (batch_size, sequence_length, input_size)
            batch_predictions = []
            for i in range(input_sequence.shape[0]):
                seq_pred = self.predict_multistep(input_sequence[i], forecast_steps)
                batch_predictions.append(seq_pred)
            result = np.array(batch_predictions)  # (batch_size, forecast_steps, input_size)
        else:  # (sequence_length, input_size) - single sequence
            result = self.predict_multistep(input_sequence, forecast_steps)
            result = result[np.newaxis, :]  # Add batch dimension for consistency
        
        # Convert back to torch tensor to match PyTorch model interface
        return torch.FloatTensor(result)
    
    def __str__(self):
        return f"SkLinearForecaster_{self.input_size}ch_{self.sequence_length}seq"

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
                 batch_size=1024,
                 lr=0.01,
                 lambda_zcr=0.1,        # Weight for zero-crossing rate loss
                 use_cuda=False,
                 closeform=False,       # Use sklearn LinearRegression instead of torch
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters - input_length == forecast_horizon for simplicity
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.closeform = closeform
        self.train = True
    
    def fit(self, X):
        """Fit the Linear forecasting model using either torch or sklearn"""
        input_size = X.shape[1]

        if self.closeform:
            # Use sklearn LinearRegression
            self.model = SkLinearForecaster(
                input_size=input_size,
                sequence_length=self.sequence_length
            )
            
            if self.verbose:
                print(f"  Model: {self.model}")
                print(f"  Using sklearn LinearRegression (closeform=True)")
            
            # Prepare training data for sklearn
            self._fit_scaler(X)  # Initialize scaler first
            X_scaled = self._scaler_transform(X)
            input_data, target_data = self._prepare_multistep_sequences(
                X_scaled, self.sequence_length, 1, ret_positions=False  # Train with single-step targets
            )
            
            # Convert to numpy and flatten input sequences
            X_train = input_data.view(input_data.size(0), -1).numpy()  # (n_samples, seq_len * input_size)
            y_train = target_data.squeeze(1).numpy()  # (n_samples, input_size)
            
            # Fit sklearn model
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            mse,corr = self._get_features(X)

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
                dist_params[ch]['mse_m'] = np.mean(mse_x,axis=0)
                dist_params[ch]['mse_std'] = np.std(mse_x,axis=0)
            self.dist_params = dist_params
            
        else:
            # Use torch LinearForecaster
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
    
    def predict_multistep_sklearn(self, X):
        """
        Generate forecasted time series using sklearn model (non-batched approach).
        Similar to predict_multistep but without torch batching.
        
        Returns:
            DataFrame: Forecasted time series with same shape as input (minus initial sequence)
        """
        assert self.is_fitted, "Must fit model before making predictions"
        assert hasattr(self, 'sequence_length'), "Model must have sequence_length attribute for multi-step prediction"
        
        X_scaled = self._scaler_transform(X)
        input_data, _, seq_positions = self._prepare_multistep_sequences(X_scaled, self.sequence_length, self.forecast_length, ret_positions=True)
        
        # Convert to numpy for sklearn processing
        input_data_np = input_data.numpy()
        n_sequences = len(seq_positions)
        
        # Generate predictions for each sequence individually
        all_predictions = []
        for seq_idx in range(n_sequences):
            input_seq = input_data_np[seq_idx]  # (sequence_length, input_size)
            
            # Use sklearn model to predict multiple steps
            prediction = self.model.predict_multistep(input_seq, self.forecast_length)
            all_predictions.append(prediction)
        
        all_predictions = np.array(all_predictions)  # (n_sequences, forecast_length, input_size)
        
        # Reconstruct time series by concatenating predictions
        n_samples = len(X)
        n_channels = X.shape[1]
        n_sequences = len(seq_positions)
        
        # Initialize output array (we can't predict the first sequence_length samples)
        reconstructed = np.full((n_samples, n_channels), np.nan)
        
        # Fill in predictions for each sequence
        for seq_idx in range(n_sequences):
            seq_pos = seq_positions[seq_idx]
            prediction = all_predictions[seq_idx]  # (forecast_length, n_channels)
            
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
    
    def predict(self, X):
        """Use the appropriate prediction method based on closeform setting"""
        if self.closeform:
            return self.predict_multistep_sklearn(X)
        else:
            return self.predict_multistep(X)

    def __str__(self):
        return "LiNDDA"