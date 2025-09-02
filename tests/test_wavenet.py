
"""
Unit tests for WVNT (WaveNet) model.

Tests the WVNT convolutional neural network-based seizure detection algorithm 
using synthetic data and generates a single comprehensive visualization.

Note: This test uses mock models since pre-trained WaveNet models may not be available.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import warnings

# Add parent directory to path to import DynaSD modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from DynaSD import WVNT
    from DynaSD.WAVENET import load_wavenet_model
    WAVENET_AVAILABLE = True
except ImportError as e:
    WAVENET_AVAILABLE = False
    IMPORT_ERROR = str(e)

from tests.data_generators import generate_test_datasets
from tests.visualization_utils import create_comprehensive_model_plot


class TestWVNT(unittest.TestCase):
    """Test suite for WVNT model."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not WAVENET_AVAILABLE:
            self.skipTest(f"WaveNet dependencies not available: {IMPORT_ERROR}")
            
        self.fs = 128
        self.n_channels = 8
        self.win_size = 1.0  # 1 second windows
        self.stride = 0.5    # 0.5 second stride
        
        # Create output directory for test results
        self.output_dir = Path(__file__).parent / "results" / "wavenet"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate test datasets
        self.baseline_data, self.seizure_data = generate_test_datasets(
            baseline_duration=80.0,
            seizure_duration=40.0,
            fs=self.fs,
            n_channels=self.n_channels,
            seed=42
        )
        
        # Try to load a WaveNet model (will be None if no model exists)
        self.model_path = None
        self.wavenet_model = load_wavenet_model(self.model_path)
        
        # Initialize WVNT wrapper
        self.model = WVNT(
            mdl=self.wavenet_model,
            win_size=self.win_size,
            stride=self.stride,
            fs=self.fs
        )
    
    def test_model_initialization(self):
        """Test that WVNT model initializes correctly."""
        self.assertEqual(self.model.win_size, self.win_size)
        self.assertEqual(self.model.stride, self.stride)
        self.assertEqual(self.model.fs, self.fs)
        
    def test_data_format(self):
        """Test that synthetic data has correct format."""
        # Check baseline data
        self.assertIsInstance(self.baseline_data, pd.DataFrame)
        self.assertEqual(self.baseline_data.shape[1], self.n_channels)
        self.assertEqual(len(self.baseline_data.columns), self.n_channels)
        
        # Check seizure data  
        self.assertIsInstance(self.seizure_data, pd.DataFrame)
        self.assertEqual(self.seizure_data.shape[1], self.n_channels)
        self.assertEqual(len(self.seizure_data.columns), self.n_channels)
        
        # Check that column names match
        self.assertEqual(list(self.baseline_data.columns), list(self.seizure_data.columns))
        
    def test_comprehensive_model_validation_with_mock(self):
        """Test model with mock WaveNet and generate comprehensive visualization."""
        # Create mock models with different behavior for baseline vs seizure
        class MockWaveNetModel:
            def predict(self, x):
                n_samples = len(x)
                # Generate realistic probabilities: lower for baseline, higher for seizure
                base_prob = 0.2 if hasattr(self, '_is_seizure') and self._is_seizure else 0.05
                noise_scale = 0.3 if hasattr(self, '_is_seizure') and self._is_seizure else 0.15
                
                probs = np.random.random((n_samples, 2)) * noise_scale + base_prob
                probs[:, 1] = np.random.random(n_samples) * noise_scale + base_prob  # Seizure probability
                return probs
        
        # Test with mock model
        mock_model = MockWaveNetModel()
        self.model.mdl = mock_model
        
        # Fit and get baseline predictions
        self.model.fit(self.baseline_data)
        baseline_features = self.model.forward(self.baseline_data)
        
        # Switch to seizure mode for higher probabilities
        mock_model._is_seizure = True
        self.model.mdl = mock_model
        self.model.fit(self.baseline_data)  # Still fit on baseline
        seizure_features = self.model.forward(self.seizure_data)
        
        # Validation checks for standard format (windows x channels)
        self.assertIsInstance(baseline_features, pd.DataFrame)
        self.assertEqual(baseline_features.shape[1], self.n_channels)
        self.assertEqual(list(baseline_features.columns), list(self.baseline_data.columns))
        self.assertIsInstance(seizure_features, pd.DataFrame)
        self.assertEqual(seizure_features.shape[1], self.n_channels)
        self.assertEqual(list(seizure_features.columns), list(self.seizure_data.columns))
        
        # Features should be probabilities (0-1 range)
        self.assertTrue(np.all(baseline_features.values >= 0))
        self.assertTrue(np.all(baseline_features.values <= 1))
        self.assertTrue(np.all(seizure_features.values >= 0))
        self.assertTrue(np.all(seizure_features.values <= 1))
        
        # Create comprehensive plot
        fig = create_comprehensive_model_plot(
            baseline_data=self.baseline_data,
            seizure_data=self.seizure_data,
            baseline_features=baseline_features,
            seizure_features=seizure_features,
            model_name="WaveNet (Mock)",
            fs=self.fs,
            save_path=self.output_dir / "wavenet_comprehensive_analysis.png"
        )
        plt.close(fig)


def run_wavenet_tests():
    """Run all WaveNet tests and generate visualizations."""
    print("Running WaveNet model tests...")
    print("Note: Using mock models since pre-trained WaveNet models may not be available")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWVNT)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All WaveNet tests passed! ({result.testsRun} tests)")
        print(f"📊 Visualization saved to: tests/results/wavenet/wavenet_comprehensive_analysis.png")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for test, error in result.failures + result.errors:
            print(f"   - {test}: {error}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_wavenet_tests() 