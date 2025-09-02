import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from .utils import MovingWinClips, num_wins
from os.path import join as ospj

# TensorFlow/Keras imports
try:
    from tensorflow.config.experimental import list_physical_devices, set_memory_growth
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. WaveNet functionality will be limited.")

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

def load_wavenet_model(model_path):
    """
    Load a pre-trained WaveNet model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
        
    Returns:
    --------
    tensorflow.keras.Model or None
        Loaded WaveNet model, or None if loading fails
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - cannot load WaveNet model")
        return None
        
    try:
        wave_model = load_model(model_path)
        print(f"Successfully loaded WaveNet model from {model_path}")
        return wave_model
    except Exception as e:
        print(f"Error loading WaveNet model: {e}")
        return None

class WVNT:
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
    def __init__(self, mdl, win_size=1, stride=0.5, fs=128):
        """Initialize WaveNet wrapper with model and windowing parameters."""
        self.win_size = win_size
        self.stride = stride
        self.fs = fs
        self.mdl = mdl
        
        if self.mdl is None:
            print("Warning: No valid model provided to WVNT")
    
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

    def get_times(self, x):
        """
        Calculate time stamps for analysis windows.
        
        Parameters:
        -----------
        x : pandas.DataFrame
            Input data (samples x channels)
            
        Returns:
        --------
        numpy.ndarray
            Window end times in seconds
        """
        time_mat = MovingWinClips(np.arange(len(x))/self.fs, self.fs, self.win_size, self.stride)
        return np.ceil(time_mat[:, -1])

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
        nwins = num_wins(len(x), self.fs, 1, 0.5)
        nch = len(chs)
        
        # Apply normalization and prepare data for WaveNet
        x_normalized = pd.DataFrame(self.scaler.transform(x), columns=chs)
        x_prepared = prepare_wavenet_segment(x_normalized)
        
        # Generate predictions (get seizure probability from class 1)
        y = self.mdl.predict(x_prepared)[:, 1]
        
        # Reshape to windows x channels format and convert to DataFrame
        probabilities = y.reshape(nwins, nch)
        features_df = pd.DataFrame(probabilities, columns=x.columns)
        
        return features_df
        
    def __call__(self, *args):
        """Allow direct calling of the forward method."""
        return self.forward(*args)

# Helper function to create WVNT instance with model loading
def create_wavenet_detector(model_path, win_size=1, stride=0.5, fs=128):
    """
    Convenience function to create a WVNT detector with automatic model loading.
    
    Parameters:
    -----------
    model_path : str
        Path to the WaveNet model file
    win_size : float, default=1
        Window size in seconds
    stride : float, default=0.5
        Window stride in seconds  
    fs : int, default=128
        Sampling frequency
        
    Returns:
    --------
    WVNT
        Configured WaveNet detector instance
    """
    # Configure GPU if available
    configure_gpu_memory()
    
    # Load the model
    model = load_wavenet_model(model_path)
    
    # Create and return detector
    return WVNT(model, win_size=win_size, stride=stride, fs=fs)

# Example usage for loading with the specified path structure
def load_wavenet_from_config(prodatapath):
    """
    Load WaveNet model using the standard path structure.
    
    Parameters:
    -----------
    prodatapath : str
        Base path to processed data directory
        
    Returns:
    --------
    WVNT
        WaveNet detector instance
    """
    model_path = ospj(prodatapath, 'WaveNet', 'v111.hdf5')
    return create_wavenet_detector(model_path)
