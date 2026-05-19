from .NDDBase import NDDBase
import torch
import torch.nn as nn
import numpy as np

class MultiStepLSTM(nn.Module):
    """LSTM for autoregressive forecasting without skip connections or input stacks"""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(MultiStepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
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
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
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
        lstm_output, hidden = self.lstm(input_sequence)

        # Phase 2: Autoregressive forecasting
        forecasts = []
        current_hidden = hidden

        # Start with the first prediction from the final hidden state
        first_prediction = self.projection(lstm_output[:, -1:, :])  # (B, 1, D)
        forecasts.append(first_prediction.squeeze(1))  # (B, D)
        current_input = first_prediction  # (B, 1, D)

        for _ in range(forecast_steps - 1):  # Note: forecast_steps - 1
            # Feed the current prediction to the LSTM
            lstm_output_step, current_hidden = self.lstm(current_input, current_hidden)
            prediction = self.projection(lstm_output_step)  # (B, 1, D)
            forecasts.append(prediction.squeeze(1))
            # Next step uses the prediction
            current_input = prediction

        return torch.stack(forecasts, dim=1)

class NDD(NDDBase):
    """
    NDD (Neural Dynamic Divergence).

    Trains an autoregressive LSTM to forecast the next time step on
    baseline dynamics. At inference, per-window forecasting MSE
    quantifies divergence from the baseline dynamics.

    Reference
    ---------
    Ojemann, W. K. S., Xu, Z., Shi, H., Walsh, K., Pattnaik, A.,
    Sinha, N., Lavelle, S., Aguila, C., Gallagher, R., Revell, A.,
    LaRocque, J. J., Korzun, J., Kulick-Soper, C. V., Zhou, D. J.,
    Galer, P. D., Sinha, S. R., Shinohara, R. T., Davis, K. A.,
    Litt, B., and Conrad, E. C., 2026. Unsupervised seizure annotation
    and detection with neural dynamic divergence. medRxiv, 2026-02-15.
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
                 batch_size='full',
                 lr=0.01,
                 use_cuda=False,
                 **kwargs):
        """
        Parameters
        ----------
        hidden_size : int, default=10
            LSTM hidden state size.
        num_layers : int, default=1
            Number of stacked LSTM layers.
        fs : int, default=256
            Sampling frequency (Hz) of input iEEG.
        sequence_length : int, default=12
            Number of input samples fed to the LSTM per forecast.
        forecast_length : int, default=1
            Number of samples forecast per sequence at inference.
            Training always uses teacher-forced single-step forecasts.
        w_size : float, default=1
            Aggregation window size (seconds) for per-window divergence.
        w_stride : float, default=0.5
            Window stride (seconds) between consecutive windows.
        num_epochs : int, default=10
            Training epochs.
        batch_size : int or 'full', default='full'
            Mini-batch size; 'full' uses the entire training set per
            step.
        lr : float, default=0.01
            Adam learning rate.
        use_cuda : bool, default=False
            If True and CUDA is available, train and infer on GPU.

        Other Parameters
        ----------------
        early_stopping : bool, default=False
            Enable validation-loss early stopping during training.
        val_split : float, default=0.2
            Fraction of training sequences held out for validation when
            ``early_stopping=True``.
        patience : int, default=5
            Epochs without ``tolerance``-level improvement before
            stopping early.
        tolerance : float, default=1e-4
            Minimum validation-loss improvement counted as progress.
        verbose : bool, default=True
            If True, print training diagnostics and tqdm progress.
        num_workers : int, default=0
            DataLoader worker count for training.
        pin_memory : bool, default=use_cuda
            DataLoader ``pin_memory`` flag.
        persistent_workers : bool, default=True
            DataLoader ``persistent_workers`` flag (applied only when
            ``num_workers > 0``).
        prefetch_factor : int, default=3
            DataLoader ``prefetch_factor`` (applied only when
            ``num_workers > 0``).
        grad_accumulation_steps : int, default=1
            Number of mini-batches to accumulate gradients over before
            an optimizer step.
        mrse : bool, default=False
            Reserved flag; see :class:`NDDBase`.
        scaler_class : type, default=sklearn.preprocessing.RobustScaler
            Scaler class used to normalize input data.
        scaler_kwargs : dict, default={}
            Keyword arguments forwarded to ``scaler_class``.
        """
        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, use_cuda=use_cuda, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self._boundary = 1.11771875
    
    def _prepare_sequences(self, data, ret_positions=False):
        """Use the shared multi-step sequence preparation from NDDBase"""
        return self._prepare_multistep_sequences(data, self.sequence_length, self.forecast_length, ret_positions)

    def fit(self, X):
        """
        Fit the LSTM forecaster on baseline data.

        Trains with teacher-forced single-step forecasts via
        :meth:`NDDBase._train_model_multistep`, then computes and stores
        per-channel baseline MSE statistics used by :meth:`forward` to
        z-score test-time prediction errors.

        Parameters
        ----------
        X : pandas.DataFrame
            Baseline (interictal) iEEG of shape (n_samples, n_channels)
            with channel names as columns.
        """
        input_size = X.shape[1]

        # Initialize model
        self.model = MultiStepLSTM(
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
        """
        Generate the forecasted time series for diagnostics.

        Use :meth:`forward` (or ``self(X)``) for seizure detection
        features. This method returns the reconstructed forecast itself
        and is intended for inspecting model behavior, not detection.

        Parameters
        ----------
        X : pandas.DataFrame
            Input iEEG of shape (n_samples, n_channels) with channel
            names as columns.

        Returns
        -------
        pandas.DataFrame
            Forecasted iEEG of shape (n_samples, n_channels) on the
            original scale, with NaNs for the first ``sequence_length``
            samples that have no preceding context. Columns = X.columns.
        """
        return self.predict_multistep(X)
    
    def __str__(self):
        return "NDD"
    
    def _get_pretrained_threshold(self):
        if self.threshold_agg == 'median':
            # return 1.27661184 # f1 median threshold
            threshold = 1.334605646 # f1 median threshold from plateau method
        elif self.threshold_agg == 'mean':
            # return 1.527851971 # phi mean threshold
            threshold = 1.597843694 # phi mean threshold from plateau method
        self._threshold = threshold
        return self._threshold