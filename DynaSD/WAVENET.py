import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from .utils import num_wins, MovingWinClips
from .base import DynaSDBase
from os.path import join as ospj

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppresses all INFO, WARNING, and ERROR messages

# TensorFlow/Keras imports
try:
    from tensorflow.config.experimental import list_physical_devices, set_memory_growth
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True

except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. WaveNet will be non-functional.")

def configure_gpu_memory():
    """
    Configure GPU memory growth for TensorFlow/PyTorch compatibility.
    
    This function sets memory growth to True for all available GPUs to prevent
    TensorFlow from allocating all GPU memory at once, which can cause issues
    when running alongside other GPU-accelerated libraries.
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping GPU configuration")
        return
        
    gpus = list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
            print(f"Configured memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs detected")

def prepare_wavenet_segment(x):
    """
    Prepare data segments for WaveNet input format.
    
    This function processes windowed iEEG data into the format expected by the
    WaveNet model. The exact implementation depends on the specific WaveNet
    architecture requirements.
    
    Parameters:
    -----------
    x : pandas.DataFrame
        Preprocessed and normalized iEEG data
        
    Returns:
    --------
    numpy.ndarray
        Data formatted for WaveNet input
    """
    # TODO: Implement specific data preparation for WaveNet
    # This is a placeholder - actual implementation depends on your WaveNet architecture
    
    # For now, convert to numpy and ensure proper shape
    # Typical WaveNet expects (batch, time, channels) format
    return x.values.reshape(1, -1, x.shape[1])

class WVNT(DynaSDBase):
    """
    WaveNet-based seizure detection wrapper class.
    
    Utilizes a pre-trained convolutional neural network (WaveNet) for seizure detection.
    The model was previously trained on iEEG data to classify seizure vs non-seizure epochs.
    This wrapper handles data preprocessing, windowing, and probability extraction for
    real-time seizure detection applications.
    
    The WaveNet architecture is particularly effective at capturing temporal patterns
    in multi-channel neural data through dilated convolutions and residual connections.
    
    Parameters:
    -----------
    mdl : tensorflow.keras.Model
        Pre-trained WaveNet model loaded from disk
    win_size : float, default=1
        Window size in seconds for analysis
    stride : float, default=0.5  
        Window stride in seconds (overlap control)
    fs : int, default=128
        Sampling frequency for the model input
    """
    def __init__(self, model_path = None, w_size=1, w_stride=0.5, fs=128, batch_size=32, verbose=False, **kwargs):
        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, **kwargs)
        """Initialize WaveNet wrapper with model and windowing parameters."""
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.model_path = model_path
        self.batch_size = batch_size
        self.verbose = verbose
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - cannot load WaveNet model")

        if self.model_path is None:
            self.model_path = ospj('..','checkpoints', 'WaveNet', 'v111.hdf5')
       
        try:
            self.mdl = load_model(self.model_path)
            print(f"Successfully loaded WaveNet model from {self.model_path}")

        except Exception as e:
            raise ValueError(f"Error loading WaveNet model: {e}")
    
    def __str__(self) -> str:
        return "WVNT"
        
    def fit(self, x):
        """
        Fit RobustScaler to training data for normalization.
        
        Parameters:
        -----------
        x : pandas.DataFrame
            Training data (samples x channels)
        """
        self.scaler = RobustScaler().fit(x)

    def forward(self, x):
        """
        Generate seizure detection predictions using WaveNet model.
        
        Processes multi-channel iEEG data through sliding windows, applies
        normalization, and generates seizure probability for each window-channel pair.
        
        Parameters:
        -----------
        x : pandas.DataFrame
            Input iEEG data (samples x channels)
            
        Returns:
        --------
        pd.DataFrame
            Seizure probabilities with windows as rows and channels as columns
        """
        if self.mdl is None:
            raise ValueError("No valid WaveNet model available for prediction")
            
        # Store channel names and calculate dimensions
        chs = x.columns
        nwins = num_wins(len(x), self.fs, self.w_size, self.w_stride)
        nch = len(chs)
        
        # Apply normalization and prepare data for WaveNet
        x_normalized = pd.DataFrame(self.scaler.transform(x), columns=chs)
        x_prepared = self._prepare_wavenet_segment(x_normalized)
        if not self.verbose:
            verbocity = 0
        else:
            verbocity = 1
        # Generate predictions (get seizure probability from class 1)
        y = self.mdl.predict(x_prepared,verbose=verbocity,batch_size=self.batch_size)[:, 1]
        
        # Reshape to windows x channels format and convert to DataFrame
        probabilities = y.reshape(nwins, nch)
        features_df = pd.DataFrame(probabilities, columns=x.columns)
        
        return features_df
        
    def __call__(self, *args):
        """Allow direct calling of the forward method."""
        return self.forward(*args)

    def _prepare_wavenet_segment(self,data):
        """
        Prepare data segments for WaveNet seizure detection model.
        
        Formats multi-channel iEEG data into non-overlapping windows suitable for 
        convolutional neural network processing. Reshapes data for channel-wise analysis.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Multi-channel iEEG data (samples x channels)
        fs : int, default=128
            Sampling frequency in Hz
        w_size : float, default=1
            Window size in seconds  
        w_stride : float, default=0.5
            Window stride in seconds
        ret_time : bool, default=False
            Whether to return time stamps for each window
            
        Returns:
        --------
        data_flat : numpy.ndarray
            Flattened data array (n_windows*n_channels, window_length)
        win_times : numpy.ndarray, optional
            Window start times if ret_time=True
        """
        data_ch = data.columns.to_list()
        n_ch = len(data_ch)
        data_np = data.to_numpy()
        win_len_idx = self.w_size*self.fs
        nwins = num_wins(len(data_np[:,0]),self.fs,self.w_size,self.w_stride)
        data_mat = np.zeros((nwins,win_len_idx,len(data_ch)))
        for k in range(n_ch):
            samples = MovingWinClips(data_np[:,k],self.fs,self.w_size,self.w_stride)
            data_mat[:,:,k] = samples
        data_flat = data_mat.transpose(0,2,1).reshape(-1,win_len_idx)
        return data_flat