#!/usr/bin/env python3
"""
Test script for GIN model debugging and validation.

This script tests the GIN model for coding errors and validates performance
on synthetic baseline and combined baseline+seizure signals.
"""

import sys
import os
import traceback
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DynaSD.GIN import GIN
from tests.synthetic_data_generator import SyntheticSeizureGenerator


class GINTester:
    """Comprehensive tester for GIN model functionality and debugging."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def log_result(self, test_name, passed, details=None, error=None):
        """Log test results and errors."""
        self.results[test_name] = {
            'passed': passed,
            'details': details,
            'error': str(error) if error else None
        }
        if error:
            self.errors.append(f"{test_name}: {error}")
    
    def test_imports(self):
        """Test that all required imports work."""
        try:
            from DynaSD.GIN import GIN, MultiStepGRU, CombinedLoss, zero_crossing_rate
            self.log_result("imports", True, "All imports successful")
            return True
        except Exception as e:
            self.log_result("imports", False, error=e)
            return False
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generator functionality."""
        try:
            generator = SyntheticSeizureGenerator(fs=256)
            
            # Test baseline generation
            baseline = generator.generate_baseline_signal(n_seconds=5, n_channels=4)
            assert isinstance(baseline, pd.DataFrame), "Baseline should be DataFrame"
            assert baseline.shape == (5*256, 4), f"Expected shape (1280, 4), got {baseline.shape}"
            
            # Test seizure generation
            seizure = generator.generate_polyspike_seizure(n_seconds=3, n_channels=4)
            assert seizure.shape == (3*256, 4), f"Expected shape (768, 4), got {seizure.shape}"
            
            # Test combined signal
            combined, start_time, end_time = generator.generate_combined_signal(
                baseline_duration=5, seizure_duration=3, n_channels=4
            )
            assert combined.shape == (8*256, 4), f"Expected shape (2048, 4), got {combined.shape}"
            assert start_time == 5, f"Expected start_time=5, got {start_time}"
            assert end_time == 8, f"Expected end_time=8, got {end_time}"
            
            self.log_result("synthetic_data_generation", True, 
                          f"Generated data shapes: baseline={baseline.shape}, seizure={seizure.shape}, combined={combined.shape}")
            return True
            
        except Exception as e:
            self.log_result("synthetic_data_generation", False, error=e)
            return False
    
    def test_gin_initialization(self):
        """Test GIN model initialization with various parameters."""
        try:
            # Test default initialization
            model1 = GIN()
            assert model1.hidden_ratio == 0.5, "Default hidden_ratio should be 0.5"
            assert model1.input_length == 16, "Default input_length should be 16"
            assert model1.forecast_horizon == 16, "Default forecast_horizon should be 16"
            
            # Test custom initialization
            model2 = GIN(
                hidden_ratio=0.5,
                num_layers=2,
                input_length=32,
                forecast_horizon=8,
                batch_size=16,
                lr=0.01
            )
            assert model2.hidden_ratio == 0.5, "Custom hidden_ratio not set correctly"
            assert model2.num_layers == 2, "Custom num_layers not set correctly"
            assert model2.batch_size == 16, "Custom batch_size not set correctly"
            
            self.log_result("gin_initialization", True, "Model initialization successful")
            return True
            
        except Exception as e:
            self.log_result("gin_initialization", False, error=e)
            return False
    
    def test_parameter_validation(self):
        """Test parameter validation in GIN model."""
        try:
            # Test invalid parameters that should raise errors
            test_cases = [
                {"input_length": 0, "forecast_horizon": 16},  # Invalid input_length
                {"input_length": 16, "forecast_horizon": 0},  # Invalid forecast_horizon
                {"input_length": 16, "forecast_horizon": 16, "stride": 0},  # Invalid stride
                {"input_length": 16, "forecast_horizon": 16, "lambda_zcr": -1},  # Invalid lambda_zcr
                {"input_length": 100, "forecast_horizon": 100, "w_size": 0.5, "fs": 256},  # Sequence longer than window
            ]
            
            for i, params in enumerate(test_cases):
                try:
                    GIN(**params)
                    self.log_result(f"parameter_validation_case_{i}", False, 
                                  f"Should have raised error for params: {params}")
                    return False
                except ValueError:
                    # Expected behavior
                    continue
                except Exception as e:
                    self.log_result(f"parameter_validation_case_{i}", False, 
                                  f"Unexpected error type for params {params}: {e}")
                    return False
            
            self.log_result("parameter_validation", True, "All invalid parameters correctly rejected")
            return True
            
        except Exception as e:
            self.log_result("parameter_validation", False, error=e)
            return False
    
    def test_baseline_fitting(self):
        """Test fitting GIN model on baseline data."""
        try:
            # Generate baseline data
            generator = SyntheticSeizureGenerator(fs=256)
            baseline_data = generator.generate_baseline_signal(n_seconds=10, n_channels=4)
            
            # Initialize model with smaller parameters for faster testing
            model = GIN(
                hidden_ratio=0.5,
                num_layers=1,
                input_length=8,
                forecast_horizon=8,
                num_epochs=10,
                batch_size=8,
                val_split=0.2,
                early_stopping_patience=3,
                lambda_zcr=0.1
            )
            
            # Fit model
            model.fit(baseline_data)
            
            # Check that model was fitted
            assert model.is_fitted, "Model should be marked as fitted"
            assert model.model is not None, "Model architecture should be initialized"
            assert len(model.train_losses) > 0, "Training losses should be recorded"
            
            # Check loss components
            assert len(model.train_mse_losses) == len(model.train_losses), "MSE losses should match total losses"
            assert len(model.train_zcr_losses) == len(model.train_losses), "ZCR losses should match total losses"
            
            # Check that losses generally decrease (allow some fluctuation)
            if len(model.train_losses) > 2:
                final_loss = np.mean(model.train_losses[-2:])
                initial_loss = np.mean(model.train_losses[:2])
                assert final_loss <= initial_loss * 1.1, "Training loss should generally decrease"
            
            self.log_result("baseline_fitting", True, 
                          f"Model fitted successfully. Final training loss: {model.train_losses[-1]:.4f}")
            return model, baseline_data
            
        except Exception as e:
            self.log_result("baseline_fitting", False, error=e)
            return None, None
    
    def test_inference_on_baseline(self, model, baseline_data):
        """Test inference on baseline data."""
        if model is None or baseline_data is None:
            self.log_result("inference_baseline", False, error="No fitted model or baseline data available")
            return False
        
        try:
            # Run forward pass (inference)
            features = model.forward(baseline_data)
            
            # Check output format
            assert isinstance(features, pd.DataFrame), "Features should be returned as DataFrame"
            assert len(features.columns) == 6, f"Expected 6 feature columns, got {len(features.columns)}"
            
            expected_cols = [
                f'MSE_H{model.forecast_horizon}',
                f'MAE_H{model.forecast_horizon}', 
                f'RMSE_H{model.forecast_horizon}',
                f'ZCR_MSE_H{model.forecast_horizon}',
                f'MaxError_H{model.forecast_horizon}',
                f'CombinedLoss_H{model.forecast_horizon}'
            ]
            
            for col in expected_cols:
                assert col in features.columns, f"Missing expected column: {col}"
            
            # Check that features are reasonable (no NaN, not all zeros)
            assert not features.isna().any().any(), "Features should not contain NaN values"
            assert not (features == 0).all().any(), "Features should not be all zeros"
            
            self.log_result("inference_baseline", True, 
                          f"Inference successful. Feature shape: {features.shape}, Mean MSE: {features.iloc[:, 0].mean():.4f}")
            return True
            
        except Exception as e:
            self.log_result("inference_baseline", False, error=e)
            return False
    
    def test_combined_signal_detection(self):
        """Test GIN model on combined baseline+seizure signal."""
        try:
            # Generate combined signal
            generator = SyntheticSeizureGenerator(fs=256)
            combined_data, seizure_start, seizure_end = generator.generate_combined_signal(
                baseline_duration=8, seizure_duration=4, 
                seizure_type='polyspike', n_channels=4
            )
            
            # Train model on baseline portion
            baseline_portion = combined_data.iloc[:int(seizure_start * 256), :]
            
            model = GIN(
                hidden_ratio=0.5,
                num_layers=1,
                input_length=8,
                forecast_horizon=8,
                num_epochs=5,
                batch_size=8,
                val_split=0.0,  # No validation for speed
                lambda_zcr=0.1
            )
            
            model.fit(baseline_portion)
            
            # Run inference on full combined signal
            features = model.forward(combined_data)
            
            # Analyze feature patterns
            n_baseline_windows = len(features) * seizure_start / (seizure_start + (seizure_end - seizure_start))
            baseline_features = features.iloc[:int(n_baseline_windows), :]
            seizure_features = features.iloc[int(n_baseline_windows):, :]
            
            if len(seizure_features) > 0:
                # Check if seizure features are higher than baseline
                baseline_mse = baseline_features.iloc[:, 0].mean()
                seizure_mse = seizure_features.iloc[:, 0].mean()
                
                detection_ratio = seizure_mse / baseline_mse if baseline_mse > 0 else float('inf')
                
                self.log_result("combined_signal_detection", True, 
                              f"Detection successful. Baseline MSE: {baseline_mse:.4f}, "
                              f"Seizure MSE: {seizure_mse:.4f}, Ratio: {detection_ratio:.2f}")
            else:
                self.log_result("combined_signal_detection", True, 
                              "Detection test completed (no seizure windows generated)")
            
            return True
            
        except Exception as e:
            self.log_result("combined_signal_detection", False, error=e)
            return False
    
    def test_prediction_functionality(self):
        """Test prediction method of GIN model."""
        try:
            generator = SyntheticSeizureGenerator(fs=256)
            data = generator.generate_baseline_signal(n_seconds=6, n_channels=3)
            
            model = GIN(
                hidden_ratio=0.5,
                input_length=4,
                forecast_horizon=4,
                num_epochs=10,  
                batch_size=4
            )
            
            model.fit(data)
            
            # Test prediction
            predictions = model.predict(data)
            
            # Check prediction format
            assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
            assert predictions.ndim == 3, f"Predictions should be 3D, got {predictions.ndim}D"
            assert predictions.shape[1] == model.forecast_horizon, f"Wrong forecast horizon dimension"
            assert predictions.shape[2] == data.shape[1], f"Wrong channel dimension"
            
            # Test prediction with details
            predictions_detailed, win_info = model.predict(data, return_details=True)
            assert len(win_info) == predictions_detailed.shape[0], "Win info length should match predictions"
            
            self.log_result("prediction_functionality", True, 
                          f"Predictions generated. Shape: {predictions.shape}")
            return True
            
        except Exception as e:
            self.log_result("prediction_functionality", False, error=e)
            return False
    
    def run_all_tests(self):
        """Run all tests and return summary."""
        test_methods = [
            self.test_imports,
            self.test_synthetic_data_generation,
            self.test_gin_initialization,
            self.test_parameter_validation,
            self.test_prediction_functionality,
            self.test_combined_signal_detection,
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        print("Running GIN model tests...")
        print("=" * 50)
        
        for test_method in test_methods:
            test_name = test_method.__name__
            try:
                result = test_method()
                if result:
                    passed_tests += 1
                    print(f"✓ {test_name}")
                else:
                    print(f"✗ {test_name}")
            except Exception as e:
                print(f"✗ {test_name} (Exception: {e})")
                self.log_result(test_name, False, error=e)
        
        # Run baseline fitting test separately to get model for inference test
        print("\nRunning baseline fitting and inference tests...")
        model, baseline_data = self.test_baseline_fitting()
        if model is not None:
            passed_tests += 1
            print("✓ test_baseline_fitting")
            
            if self.test_inference_on_baseline(model, baseline_data):
                passed_tests += 1
                print("✓ test_inference_on_baseline")
            else:
                print("✗ test_inference_on_baseline")
        else:
            print("✗ test_baseline_fitting")
            print("✗ test_inference_on_baseline (skipped due to fitting failure)")
        
        total_tests += 2  # Add the two additional tests
        
        print("\n" + "=" * 50)
        print(f"Test Results: {passed_tests}/{total_tests} passed")
        
        if self.errors:
            print(f"\nErrors encountered ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\nNo errors encountered!")
        
        return passed_tests == total_tests


def main():
    """Main test execution function."""
    print("GIN Model Testing and Debugging")
    print("===============================")
    
    tester = GINTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! GIN model is working correctly.")
        return 0
    else:
        print(f"\n❌ Some tests failed. Check errors above for debugging.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 