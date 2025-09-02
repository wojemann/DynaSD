"""
Unit tests for NDD (Neural Network Detection) model.

Tests the NDD LSTM-based seizure detection algorithm using synthetic data
and generates a single comprehensive visualization.
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

from DynaSD import NDD
from tests.data_generators import generate_test_datasets
from tests.visualization_utils import create_comprehensive_model_plot


class TestNDD(unittest.TestCase):
    """Test suite for NDD model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 128
        self.n_channels = 8
        self.win_size = 1.0  # 1 second windows
        self.stride = 0.5    # 0.5 second stride
        self.hidden_size = 10
        self.train_win = 12
        self.pred_win = 1
        self.num_epochs = 5  # Reduced for faster testing
        
        # Create output directory for test results
        self.output_dir = Path(__file__).parent / "results" / "ndd"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate test datasets
        self.baseline_data, self.seizure_data = generate_test_datasets(
            baseline_duration=80.0,
            seizure_duration=40.0,
            fs=self.fs,
            n_channels=self.n_channels,
            seed=42
        )
        
        # Initialize model
        self.model = NDD(
            hidden_size=self.hidden_size,
            fs=self.fs,
            train_win=self.train_win,
            pred_win=self.pred_win,
            w_size=self.win_size,
            w_stride=self.stride,
            num_epochs=self.num_epochs,
            lr=0.01,
            use_cuda=False,  # Use CPU for testing
            val=False
        )
    
    def test_model_initialization(self):
        """Test that NDD model initializes correctly."""
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.fs, self.fs)
        self.assertEqual(self.model.train_win, self.train_win)
        self.assertEqual(self.model.pred_win, self.pred_win)
        self.assertEqual(self.model.w_size, self.win_size)
        self.assertEqual(self.model.w_stride, self.stride)
        self.assertEqual(self.model.num_epochs, self.num_epochs)
        self.assertFalse(self.model.is_fitted)
        
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
        
    def test_comprehensive_model_validation(self):
        """Test model and generate comprehensive visualization."""
        # Suppress training output for cleaner test results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Fit model and get predictions
            self.model.fit(self.baseline_data)
            baseline_features = self.model.forward(self.baseline_data)
            seizure_features = self.model.forward(self.seizure_data)
        
        # Validation checks for standard format (windows x channels)
        self.assertIsInstance(baseline_features, pd.DataFrame)
        self.assertEqual(baseline_features.shape[1], self.n_channels)
        self.assertEqual(list(baseline_features.columns), list(self.baseline_data.columns))
        self.assertIsInstance(seizure_features, pd.DataFrame)
        self.assertEqual(seizure_features.shape[1], self.n_channels)
        self.assertEqual(list(seizure_features.columns), list(self.seizure_data.columns))
        
        # Features should be positive (MSE values)
        baseline_mean = np.mean(baseline_features.values)
        seizure_mean = np.mean(seizure_features.values)
        self.assertGreater(baseline_mean, 0, "Baseline features should be positive")
        self.assertGreater(seizure_mean, 0, "Seizure features should be positive")
        
        # Create comprehensive plot
        fig = create_comprehensive_model_plot(
            baseline_data=self.baseline_data,
            seizure_data=self.seizure_data,
            baseline_features=baseline_features,
            seizure_features=seizure_features,
            model_name="NDD",
            fs=self.fs,
            save_path=self.output_dir / "ndd_comprehensive_analysis.png"
        )
        plt.close(fig)


def run_ndd_tests():
    """Run all NDD tests and generate visualizations."""
    print("Running NDD model tests...")
    print("Note: This may take a few minutes due to neural network training...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNDD)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All NDD tests passed! ({result.testsRun} tests)")
        print(f"📊 Visualization saved to: tests/results/ndd/ndd_comprehensive_analysis.png")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for test, error in result.failures + result.errors:
            print(f"   - {test}: {error}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_ndd_tests() 