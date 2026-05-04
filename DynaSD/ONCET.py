import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as ospj
import json
from .base import DynaSDBase
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from .utils import num_wins, moving_win_clips
from tqdm import tqdm
class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for efficient temporal processing.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        causal: bool = True
    ):
        super().__init__()
        
        # Causal padding (don't look into future)
        if causal:
            self.padding = (kernel_size - 1) * dilation
            self.causal = True
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2
            self.causal = False
        
        # Depthwise: each feature processed independently
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise: mix features
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # Depthwise
        x = self.depthwise(x)
        if self.causal:
            x = x[..., :-self.padding]  # Remove future-looking padding
        x = self.bn1(x)
        x = F.gelu(x)
        
        # Pointwise
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.gelu(x)
        
        return x


class SqueezeExcitation1d(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.shape
        # Global context
        y = self.pool(x).view(b, c)
        # Learn channel importance
        y = self.fc(y).view(b, c, 1)
        # Reweight
        return x * y


class TemporalResidualBlock(nn.Module):
    """
    Residual block with depthwise separable convolution.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels,
            kernel_size, dilation
        )
        self.conv2 = DepthwiseSeparableConv1d(
            channels, channels,
            kernel_size, dilation
        )
        self.se = SqueezeExcitation1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.dropout(x)
        return x + residual


class LightweightSeizureDetector(nn.Module):
    """
    Lightweight seizure detector using depthwise separable TCN.
    
    Input: [batch, 1, 256] - single electrode, 1 second at 256Hz
    Output: [batch, 2] - logits for [not_seizing, seizing]
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        base_filters: int = 32,
        num_blocks: int = 3,
        kernel_size: int = 16,
        dilations: list = None,
        dropout_stem: float = 0.1,
        dropout_blocks: float = 0.2,
        dropout_head: float = 0.3
    ):
        super().__init__()
        
        if dilations is None:
            dilations = [2, 4, 8]
        
        assert len(dilations) == num_blocks, "Number of dilations must match num_blocks"
        
        self.num_classes = num_classes
        self.base_filters = base_filters
        
        # Stem: Convert raw EEG to initial features
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Dropout(dropout_stem)
        )
        
        # Temporal processing blocks
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(
                base_filters,
                kernel_size,
                dilation,
                dropout_blocks
            )
            for dilation in dilations
        ])
        
        # Feature compression
        self.compress = DepthwiseSeparableConv1d(
            base_filters,
            base_filters // 2,
            kernel_size=1,
            causal=False
        )
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout_head),
            nn.Linear(base_filters, 32),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(32, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 1, 256] - single electrode signal
            
        Returns:
            logits: [batch, num_classes]
        """
        # Stem
        x = self.stem(x)  # [batch, base_filters, 128]
        
        # Temporal blocks
        for block in self.blocks:
            x = block(x)  # [batch, base_filters, 128]
        
        # Compress
        x = self.compress(x)  # [batch, base_filters//2, 128]
        
        # Global pooling
        gap = self.gap(x).squeeze(-1)  # [batch, base_filters//2]
        gmp = self.gmp(x).squeeze(-1)  # [batch, base_filters//2]
        x = torch.cat([gap, gmp], dim=1)  # [batch, base_filters]
        
        # Classification
        x = self.head(x)  # [batch, num_classes]
        
        return x
    
    def get_seizure_probability(self, x):
        """
        Convenience method for inference.
        
        Returns:
            probs: [batch] - probability of seizure class
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ONCET(DynaSDBase):
    """
    ONCET (OjemanN's Convolutional neural nETwork for seizure detection) model.
    """
    def __init__(self, checkpoint_path = None, config_path = None, w_size=1, w_stride=0.5, fs=256, batch_size=2048, verbose=False, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        super().__init__(fs=fs, w_size=w_size, w_stride=w_stride, **kwargs)
        """Initialize ONCET wrapper with model and windowing parameters."""
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.batch_size = batch_size
        self.verbose = verbose,
        self.device = device

        if self.checkpoint_path is None:
            # self.checkpoint_path = ospj('..','checkpoints', 'ONCET', 'best_model.pth')
            self.checkpoint_path = '/Users/wojemann/Documents/CNT/DynaSD/checkpoints/ONCET/best_model.pth'
        if self.config_path is None:
            # self.config_path = ospj('..','checkpoints', 'ONCET', 'final_training_config.json')
            self.config_path = '/Users/wojemann/Documents/CNT/DynaSD/checkpoints/ONCET/final_training_config.json'
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.config = json.load(open(self.config_path))['model']
            self.model = LightweightSeizureDetector(
                num_classes=2,
                base_filters=self.config['base_filters'],
                num_blocks=self.config['num_blocks'],
                kernel_size=self.config.get('kernel_size', 16),
                dilations=self.config.get('dilations', [2, 4, 8]),
                dropout_stem=self.config.get('dropout_stem', 0.1),
                dropout_blocks=self.config.get('dropout_blocks', 0.2),
                dropout_head=self.config.get('dropout_head', 0.3)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            # print(f"Successfully loaded ONCET model from {self.checkpoint_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading ONCET model: {e} from {self.checkpoint_path}")
            self.model = None
            raise ValueError(f"Error loading ONCET model: {e} from {self.checkpoint_path}")
    
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
        Generate seizure detection predictions using ONCET model.
        
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
        if self.model is None:
            raise ValueError("No valid ONCET model available for prediction")
            
        # Store channel names and calculate dimensions
        chs = x.columns
        nwins = num_wins(len(x), self.fs, self.w_size, self.w_stride)
        nch = len(chs)
        
        # Apply normalization and prepare data for WaveNet
        x_normalized = pd.DataFrame(self.scaler.transform(x), columns=chs)
        x_prepared = self._prepare_oncet_segment(x_normalized)
        if not self.verbose:
            verbocity = 0
        else:
            verbocity = 1
        # Generate predictions (get seizure probability from class 1)
        probabilities = []
        # Batch processing: use DataLoader to process x_prepared in batches
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(x_prepared)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        probabilities = []
        for batch in tqdm(loader, disable=not self.verbose):
            batch_x = batch[0].to(self.device)  # shape: batch x 1 x 256
            y = self.model.get_seizure_probability(batch_x)
            probabilities.append(y.detach().cpu().numpy())
        probabilities = np.concatenate(probabilities, axis=0)
        probabilities = probabilities.reshape(nwins, nch)
        features_df = pd.DataFrame(probabilities, columns=x.columns)
        return features_df
        # y = self.model.get_seizure_probability(x_prepared)
        
        # # Reshape to windows x channels format and convert to DataFrame
        # probabilities = y.detach().cpu().numpy().reshape(nwins, nch)
        # features_df = pd.DataFrame(probabilities, columns=x.columns)
        
        # return features_df

    def _prepare_oncet_segment(self, data):
        """
        Prepare data segments for ONCET model.
        
        Parameters:
        -----------
        x : pandas.DataFrame
            Input data (samples x channels)
        """
        data_ch = data.columns.to_list()
        n_ch = len(data_ch)
        data_np = data.to_numpy()
        win_len_idx = self.w_size*self.fs
        nwins = num_wins(len(data_np[:,0]),self.fs,self.w_size,self.w_stride)
        data_mat = np.zeros((nwins,win_len_idx,len(data_ch)))
        for k in range(n_ch):
            samples = moving_win_clips(data_np[:,k],self.fs,self.w_size,self.w_stride)
            data_mat[:,:,k] = samples
        data_flat = data_mat.transpose(0,2,1).reshape(-1,1,win_len_idx)
        # Should return a tensor of shape (nwins, 1, win_len_idx)
        return torch.from_numpy(data_flat).float().to(self.device)

def get_pretrained_threshold():
    """
    Get pretrained threshold for ONCET model.
    """
    return 0.5