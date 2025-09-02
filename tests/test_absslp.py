"""
Unit tests for ABSSLP (Absolute Slope) model.

Tests the ABSSLP seizure detection algorithm using synthetic data
and generates a single comprehensive visualization.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path to import DynaSD modules
sys.path.append(str(Path(__file__).parent.parent))

from DynaSD import ABSSLP
from tests.data_generators import generate_test_datasets
from tests.visualization_utils import create_comprehensive_model_plot


class TestABSSLP(unittest.TestCase):
    """Test suite for ABSSLP model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 128
        self.n_channels = 8
        self.w_size = 1.0    # 1 second windows  
        self.w_stride = 0.5  # 0.5 second stride
        
        # Create output directory for test results
        self.output_dir = Path(__file__).parent / "results" / "absslp"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate test datasets
        self.baseline_data, self.seizure_data = generate_test_datasets(
            baseline_duration=80.0,
            seizure_duration=40.0,
            fs=self.fs,
            n_channels=self.n_channels,
            seed=42
        )
        
        # Initialize model with updated parameter names
        self.model = ABSSLP(
            w_size=self.w_size,
            w_stride=self.w_stride,
            fs=self.fs
        )
    
    def test_model_initialization(self):
        """Test that ABSSLP model initializes correctly."""
        # Test inherited attributes from DynaSDBase
        self.assertEqual(self.model.w_size, self.w_size)
        self.assertEqual(self.model.w_stride, self.w_stride)
        self.assertEqual(self.model.fs, self.fs)
        self.assertFalse(self.model.is_fitted)
        
        # Test ABSSLP-specific attributes
        self.assertIsNotNone(self.model.function)
        self.assertTrue(hasattr(self.model, 'scaler_class'))
        
        # Test inherited methods are available
        self.assertTrue(hasattr(self.model, '_fit_scaler'))
        self.assertTrue(hasattr(self.model, '_scaler_transform'))
        self.assertTrue(hasattr(self.model, 'get_win_times'))
        
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
        # Test that model starts unfitted
        self.assertFalse(self.model.is_fitted)
        
        # Fit model and check fitted status
        self.model.fit(self.baseline_data)
        self.assertTrue(self.model.is_fitted)
        
        # Get predictions
        baseline_features = self.model.forward(self.baseline_data)
        seizure_features = self.model.forward(self.seizure_data)
        
        # Validation checks for standard format (windows x channels)
        self.assertIsInstance(baseline_features, pd.DataFrame)
        self.assertEqual(baseline_features.shape[1], self.n_channels)
        self.assertEqual(list(baseline_features.columns), list(self.baseline_data.columns))
        self.assertIsInstance(seizure_features, pd.DataFrame)
        self.assertEqual(seizure_features.shape[1], self.n_channels)
        self.assertEqual(list(seizure_features.columns), list(self.seizure_data.columns))
        
        # Features should be reasonable
        baseline_mean = baseline_features.values.mean()
        seizure_mean = seizure_features.values.mean()
        self.assertGreater(seizure_mean, baseline_mean, 
                          "Seizure features should be higher than baseline on average")
        
        # Test inherited get_win_times method
        win_times = self.model.get_win_times(len(self.baseline_data))
        expected_windows = baseline_features.shape[0]
        self.assertEqual(len(win_times), expected_windows)
        
        # Create comprehensive plot
        fig = create_comprehensive_model_plot(
            baseline_data=self.baseline_data,
            seizure_data=self.seizure_data,
            baseline_features=baseline_features,
            seizure_features=seizure_features,
            model_name="ABSSLP",
            fs=self.fs,
            save_path=self.output_dir / "absslp_comprehensive_analysis.png"
        )
        plt.close(fig)


def run_absslp_tests():
    """Run all ABSSLP tests and generate visualizations."""
    print("Running ABSSLP model tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestABSSLP)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All ABSSLP tests passed! ({result.testsRun} tests)")
        print(f"📊 Visualization saved to: tests/results/absslp/absslp_comprehensive_analysis.png")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for test, error in result.failures + result.errors:
            print(f"   - {test}: {error}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_absslp_tests() 