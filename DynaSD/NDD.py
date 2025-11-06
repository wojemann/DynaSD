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
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        
        self.projection = nn.Linear(hidden_size, input_size)
        
        # Initialize weights properly to prevent vanishing gradients
        # self._init_weights()
    
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
        
        # Initialize projection layer weights
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)
    
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
        
        for _ in range(forecast_steps - 1):  # Note: forecast_steps - 1
            # Feed the current prediction to the GRU
            gru_output_step, current_hidden = self.gru(current_input, current_hidden)
            prediction = self.projection(gru_output_step)  # (B, 1, D)
            forecasts.append(prediction.squeeze(1))
            # Next step uses the prediction
            current_input = prediction
            
        return torch.stack(forecasts, dim=1)

class NDD(NDDBase):
    """
    NDD (Neural Dynamic Divergence) - Multi-step GRU model without skip connections or input stacks.
    Simplified version of GIN.
    """

    def __init__(self, 
                 hidden_size=10,
                 num_layers=1,
                 fs=256,
                 sequence_length=12,
                 forecast_length=1,
                 w_size=1, 
                 w_stride=0.5,
                 num_epochs=10,
                 batch_size=1024,
                 lr=0.01,
                 use_cuda=False,
                 **kwargs):

        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)
        
        # Store parameters - matching GIN structure
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
        """Fit the GRU forecasting model using shared training loop"""
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
        
        # Use shared training loop
        self._train_model_multistep(
            X=X,
            model=self.model,
            sequence_length=self.sequence_length,
            forecast_length=1, # Train with teacher forcing
            num_epochs=self.num_epochs,
            batch_size='full',
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
