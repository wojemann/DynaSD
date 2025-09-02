# DynaSD Test Suite

This test suite provides comprehensive unit testing for the DynaSD (Dynamic System Detection) pipeline using synthetic EEG data. It validates all three main models and generates detailed visualizations of their outputs.

## Overview

The test suite includes:

- **Synthetic Data Generation**: Creates realistic baseline and seizure-like EEG signals
- **Model Testing**: Validates all three DynaSD models (ABSSLP, NDD, WaveNet)
- **Visualization**: Generates heatmaps and comparison plots for each model
- **Performance Analysis**: Provides statistical summaries and comparisons

## Tested Models

### 1. ABSSLP (Absolute Slope)
- **Type**: Feature-based detection
- **Method**: Calculates absolute slopes of signal derivatives
- **Output**: Normalized slope features per channel/window
- **Test File**: `test_absslp.py`

### 2. NDD (Neural Network Detection)  
- **Type**: LSTM-based reconstruction
- **Method**: Uses reconstruction error (MSE) for anomaly detection
- **Output**: Reconstruction error features per channel/window
- **Test File**: `test_ndd.py`

### 3. WVNT (WaveNet)
- **Type**: Convolutional neural network
- **Method**: Pre-trained CNN for seizure classification
- **Output**: Seizure probability per channel/window
- **Test File**: `test_wavenet.py`

## Quick Start

### Run All Tests
```bash
cd tests/
python run_all_tests.py
```

### Run Individual Model Tests
```bash
# Test ABSSLP model only
python test_absslp.py

# Test NDD model only  
python test_ndd.py

# Test WaveNet model only
python test_wavenet.py
```

## Test Data

The test suite generates two types of synthetic EEG data:

### Baseline Data (80 seconds, 8 channels)
- **Characteristics**: Normal brain activity patterns
- **Frequencies**: Delta (1-4 Hz), Alpha (8-12 Hz), Beta (13-30 Hz)
- **Amplitude**: Realistic EEG voltage ranges (μV)
- **Use**: Training baseline models and comparison

### Seizure Data (40 seconds, 8 channels)
- **Characteristics**: Seizure-like patterns with onset at 30% of signal
- **Frequencies**: Rhythmic seizure activity (3-7 Hz)
- **Amplitude**: Increased amplitude during seizure period
- **Use**: Testing seizure detection capability

## Generated Outputs

Results are saved to `tests/results/` with subdirectories for each model:

```
tests/results/
├── absslp/
│   ├── absslp_baseline_heatmap.png
│   ├── absslp_seizure_heatmap.png
│   ├── absslp_comparison.png
│   ├── absslp_baseline_stats.png
│   ├── absslp_seizure_stats.png
│   └── absslp_summary_results.csv
├── ndd/
│   ├── ndd_baseline_heatmap.png
│   ├── ndd_seizure_heatmap.png
│   ├── ndd_comparison.png
│   └── ndd_summary_results.csv
└── wavenet/
    ├── wavenet_baseline_heatmap_mock.png
    ├── wavenet_seizure_heatmap_mock.png
    ├── wavenet_comparison_mock.png
    └── wavenet_summary_results_mock.csv
```

### Visualization Types

1. **Baseline Heatmaps**: Features/probabilities for normal EEG data
2. **Seizure Heatmaps**: Features/probabilities for seizure-like data
3. **Comparison Plots**: Side-by-side baseline vs seizure visualization
4. **Summary Statistics**: Statistical analysis per channel
5. **CSV Results**: Numerical summaries for further analysis

## Expected Results

### ABSSLP Model
- **Baseline**: Low slope values (< 1.0)
- **Seizure**: Higher slope values due to rapid voltage changes
- **Visualization**: Blue (baseline) vs Red (seizure) heatmaps

### NDD Model  
- **Baseline**: Lower reconstruction errors after training
- **Seizure**: Higher reconstruction errors for anomalous patterns
- **Visualization**: MSE values across channels and time windows

### WaveNet Model
- **Baseline**: Low seizure probabilities (< 0.5)
- **Seizure**: Higher seizure probabilities (> 0.5)
- **Note**: Uses mock models for testing when pre-trained model unavailable

## Test Configuration

### Data Parameters
- **Sampling Rate**: 128 Hz
- **Channels**: 8 (CH01-CH08)
- **Window Size**: 1.0 seconds
- **Window Stride**: 0.5 seconds
- **Baseline Duration**: 80 seconds
- **Seizure Duration**: 40 seconds

### Model Parameters
- **ABSSLP**: Default windowing parameters
- **NDD**: Reduced epochs (5) for faster testing
- **WaveNet**: Mock models when pre-trained unavailable

## Dependencies

```python
# Core packages
numpy
pandas
matplotlib
seaborn
scikit-learn

# Deep learning (for NDD)
torch
tqdm

# Optional (for WaveNet)
tensorflow  # If using real WaveNet models
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure DynaSD package is in Python path
2. **Missing Dependencies**: Install required packages (see above)
3. **CUDA Warnings**: NDD tests use CPU-only mode for compatibility
4. **WaveNet Skipped**: Normal if no pre-trained model available

### Test Failures

If tests fail, check:
1. Data format requirements (DataFrame with samples × channels)
2. Model initialization parameters
3. Memory availability for neural network training
4. File permissions for saving results

## Interpreting Results

### Successful Tests
- All models should process data without errors
- Visualizations should show clear differences between baseline and seizure
- Features/probabilities should be in expected ranges

### Model Performance
- **ABSSLP**: Should detect rapid signal changes during seizure periods
- **NDD**: Should show higher reconstruction errors for anomalous seizure patterns  
- **WaveNet**: Should output higher probabilities for seizure-like activity

## Extending Tests

### Adding New Models
1. Create test file following existing pattern
2. Implement model-specific test cases
3. Add visualization generation
4. Update main test runner

### Custom Data
Replace synthetic data generation with:
```python
# Load your own EEG data
baseline_data = pd.read_csv('your_baseline.csv')
seizure_data = pd.read_csv('your_seizure.csv')
```

### Additional Metrics
Extend test cases to include:
- Sensitivity/specificity analysis
- ROC curve generation
- Parameter optimization
- Cross-validation

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Include comprehensive docstrings
3. Generate appropriate visualizations
4. Update this README

## License

This test suite is part of the DynaSD package and follows the same license terms. 