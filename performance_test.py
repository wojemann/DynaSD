#!/usr/bin/env python3
"""
Performance testing script for optimized NDD training.
Tests training speed with various configurations on high-end hardware.
"""

import torch
import pandas as pd
import numpy as np
import time
from contextlib import contextmanager

# Import your models
from DynaSD.GIN import GIN
from DynaSD.MINDD import MINDD

@contextmanager
def timer():
    """Context manager for timing code blocks"""
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")

def generate_test_data(n_samples=50000, n_channels=8, fs=256):
    """Generate synthetic EEG-like data for testing"""
    # Create time vector
    t = np.arange(n_samples) / fs
    
    # Generate synthetic signals with various frequencies
    data = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Mix of frequencies typical in EEG
        signal = (np.sin(2 * np.pi * 10 * t) +  # Alpha band
                 0.5 * np.sin(2 * np.pi * 4 * t) +  # Theta band
                 0.3 * np.sin(2 * np.pi * 30 * t) +  # Gamma band
                 0.1 * np.random.randn(n_samples))  # Noise
        
        # Add channel-specific variation
        signal += ch * 0.1 * np.sin(2 * np.pi * 8 * t)
        data[:, ch] = signal
    
    # Create DataFrame
    channels = [f'ch_{i:02d}' for i in range(n_channels)]
    df = pd.DataFrame(data, columns=channels)
    
    return df

def test_performance_configs():
    """Test different performance configurations"""
    
    print("=== NDD Training Performance Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPU threads: {torch.get_num_threads()}")
    print()
    
    # Generate test data
    print("Generating test data...")
    X = generate_test_data(n_samples=50000, n_channels=8, fs=256)
    print(f"Data shape: {X.shape}")
    print()
    
    # Test configurations
    configs = [
        {
            'name': 'Baseline (no optimization)',
            'params': {
                'use_cuda': True,
                'batch_size': 1024,
                'num_epochs': 5,
                'sequence_length': 32,
                'forecast_length': 8,
                'num_workers': 0,
                'pin_memory': False,
                'verbose': False
            }
        },
        {
            'name': 'Optimized DataLoader',
            'params': {
                'use_cuda': True,
                'batch_size': 1024,
                'num_epochs': 5,
                'sequence_length': 32,
                'forecast_length': 8,
                'num_workers': 8,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 3,
                'verbose': False
            }
        },
        {
            'name': 'Large Batch Optimized',
            'params': {
                'use_cuda': True,
                'batch_size': 2048,  # Larger batch for better GPU utilization
                'num_epochs': 5,
                'sequence_length': 32,
                'forecast_length': 8,
                'num_workers': 12,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 3,
                'verbose': False
            }
        },
        {
            'name': 'Long Sequence (optimized)',
            'params': {
                'use_cuda': True,
                'batch_size': 512,  # Smaller batch for longer sequences
                'num_epochs': 3,
                'sequence_length': 128,  # Much longer sequence
                'forecast_length': 32,
                'num_workers': 8,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2,  # Reduce for memory efficiency
                'grad_accumulation_steps': 2,  # Simulate larger effective batch
                'verbose': True
            }
        },
        {
            'name': 'Large Model (200k params)',
            'params': {
                'use_cuda': True,
                'batch_size': 256,
                'num_epochs': 3,
                'sequence_length': 64,
                'forecast_length': 16,
                'hidden_sizes': [512, 256, 128],  # ~200k parameters
                'num_workers': 8,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2,
                'verbose': True
            }
        },
        {
            'name': 'Optimal Configuration (Based on Results)',
            'params': {
                'use_cuda': True,
                'batch_size': 1024,
                'num_epochs': 5,
                'sequence_length': 64,
                'forecast_length': 16,
                'num_workers': 8,  # Sweet spot from results
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 3,
                'verbose': True
            }
        },
        {
            'name': 'Ultra Large Batch Test',
            'params': {
                'use_cuda': True,
                'batch_size': 4096,  # Test much larger batch with available GPU memory
                'num_epochs': 5,
                'sequence_length': 64,
                'forecast_length': 16,
                'num_workers': 8,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2,  # Lower prefetch for large batches
                'verbose': True
            }
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing: {config['name']}")
        print(f"  Config: {config['params']}")
        
        try:
            # Test with MINDD (simpler model)
            model = MINDD(**config['params'])
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Time the training
            with timer():
                model.fit(X)
            
            # Get memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Peak GPU memory: {memory_used:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
            
            results.append({
                'config': config['name'],
                'success': True,
                'memory_gb': memory_used if torch.cuda.is_available() else 0
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'config': config['name'],
                'success': False,
                'error': str(e)
            })
        
        print()
    
    print("=== Results Summary ===")
    for result in results:
        if result['success']:
            print(f"✓ {result['config']}: {result['memory_gb']:.2f} GB")
        else:
            print(f"✗ {result['config']}: {result['error']}")

if __name__ == "__main__":
    test_performance_configs()
