#!/usr/bin/env python3
"""
Performance test for optimized sequence generation in DynaSD models.
Compares old vs new sequence generation methods.
"""

import numpy as np
import pandas as pd
import time
import torch
from DynaSD.NDDBase import NDDBase
from DynaSD.utils import MovingWinClips


def create_test_data(n_samples=10000, n_channels=8, fs=256):
    """Create synthetic test data optimized for many channels"""
    print(f"Generating test data: {n_samples:,} samples × {n_channels} channels...")
    
    t = np.arange(n_samples) / fs
    data = np.zeros((n_samples, n_channels), dtype=np.float32)  # Use float32 for memory efficiency
    
    # Create realistic multi-channel time series efficiently
    # Generate base frequency components once
    base_freqs = np.array([1, 2, 4, 8, 15, 25, 40, 60])  # Common neural frequencies
    base_components = np.array([np.sin(2 * np.pi * freq * t) for freq in base_freqs])
    
    for ch in range(n_channels):
        # Mix different frequency components with channel-specific weights
        weights = np.random.rand(len(base_freqs))
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Weighted combination of frequency components
        signal = np.sum(weights[:, np.newaxis] * base_components, axis=0)
        
        # Add channel-specific phase and amplitude variation
        phase = ch * 0.1
        amplitude = 0.8 + 0.4 * np.random.rand()
        signal = amplitude * np.sin(signal + phase)
        
        # Add realistic noise
        noise_level = 0.1 + 0.05 * np.random.rand()
        signal += noise_level * np.random.randn(n_samples)
        
        data[:, ch] = signal
    
    columns = [f'ch_{i:02d}' for i in range(n_channels)]
    print(f"Data generation complete.")
    return pd.DataFrame(data, columns=columns)


def old_prepare_sequences(data, sequence_length, forecast_length=1):
    """Original inefficient sequence preparation method"""
    data_np = data.to_numpy()
    n_samples, _ = data_np.shape
    stride = forecast_length
    total_seq_length = sequence_length + forecast_length
    n_sequences = (n_samples - total_seq_length) // stride + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough data for even one sequence.")
        
    # OLD INEFFICIENT METHOD: List appending
    all_inputs = []
    all_targets = []
    
    for seq_idx in range(n_sequences):
        seq_start = seq_idx * stride
        input_end = seq_start + sequence_length
        target_end = input_end + forecast_length
        
        input_seq = data_np[seq_start:input_end, :]
        target_seq = data_np[input_end:target_end, :]
        
        all_inputs.append(input_seq)
        all_targets.append(target_seq)
    
    input_data = torch.FloatTensor(np.array(all_inputs))
    target_data = torch.FloatTensor(np.array(all_targets))
    
    return input_data, target_data


def benchmark_sequence_generation():
    """Benchmark old vs new sequence generation methods"""
    print("=" * 60)
    print("SEQUENCE GENERATION PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Realistic test configurations for EEG/neural data
    # Sample size: (180+120+100)*256 = 102,400 samples
    realistic_samples = (180 + 120 + 100) * 256  # 102,400 samples
    
    test_configs = [
        # Short sequences with many channels - typical for real-time processing
        {'n_samples': realistic_samples, 'n_channels': 64, 'seq_len': 1, 'forecast_len': 1},
        {'n_samples': realistic_samples, 'n_channels': 80, 'seq_len': 8, 'forecast_len': 1},
        {'n_samples': realistic_samples, 'n_channels': 96, 'seq_len': 32, 'forecast_len': 16},
        {'n_samples': realistic_samples, 'n_channels': 100, 'seq_len': 128, 'forecast_len': 32},
        
        # Additional test with maximum realistic parameters
        {'n_samples': realistic_samples, 'n_channels': 50, 'seq_len': 8, 'forecast_len': 1},
        {'n_samples': realistic_samples, 'n_channels': 75, 'seq_len': 32, 'forecast_len': 16},
    ]
    
    # Initialize NDDBase for new method
    ndd_base = NDDBase(verbose=False)
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting: {config['n_samples']:,} samples, {config['n_channels']} channels")
        print(f"Sequence length: {config['seq_len']}, Forecast length: {config['forecast_len']}")
        
        # Create test data
        data = create_test_data(config['n_samples'], config['n_channels'])
        print(f"Data created: {data.shape} ({data.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
        
        # Test old method
        try:
            start_time = time.time()
            old_input, old_target = old_prepare_sequences(
                data, config['seq_len'], config['forecast_len']
            )
            old_time = time.time() - start_time
            print(f"  Old method: {old_time:.4f}s ({old_input.shape[0]:,} sequences)")
        except Exception as e:
            old_time = float('inf')
            print(f"  Old method: FAILED ({e})")
        
        # Test new method (first call - no cache)
        ndd_base.clear_sequence_cache()
        start_time = time.time()
        new_input, new_target = ndd_base._prepare_multistep_sequences_vectorized(
            data, config['seq_len'], config['forecast_len']
        )
        new_time_no_cache = time.time() - start_time
        print(f"  New method (no cache): {new_time_no_cache:.4f}s ({new_input.shape[0]:,} sequences)")
        
        # Test new method (second call - with cache)
        start_time = time.time()
        cached_input, cached_target = ndd_base._prepare_multistep_sequences_vectorized(
            data, config['seq_len'], config['forecast_len']
        )
        cached_time = time.time() - start_time
        print(f"  New method (cached): {cached_time:.4f}s")
        
        # Memory usage comparison
        if old_time != float('inf'):
            old_memory = old_input.element_size() * old_input.nelement() + old_target.element_size() * old_target.nelement()
        else:
            old_memory = 0
        new_memory = new_input.element_size() * new_input.nelement() + new_target.element_size() * new_target.nelement()
        print(f"  Memory usage: {new_memory / 1e6:.1f} MB for sequences")
        
        # Verify results are identical (shapes and approximately equal values)
        if old_time != float('inf'):
            assert old_input.shape == new_input.shape, "Input shapes don't match!"
            assert old_target.shape == new_target.shape, "Target shapes don't match!"
            assert torch.allclose(old_input, new_input, atol=1e-6), "Input values don't match!"
            assert torch.allclose(old_target, new_target, atol=1e-6), "Target values don't match!"
            print(f"  ✓ Results verified identical")
            
            speedup = old_time / new_time_no_cache
            cache_speedup = old_time / cached_time
            print(f"  Speedup: {speedup:.1f}x (no cache), {cache_speedup:.1f}x (cached)")
        else:
            speedup = float('inf')
            cache_speedup = float('inf')
            print(f"  New method succeeded where old method failed")
        
        results.append({
            'config': config,
            'old_time': old_time,
            'new_time': new_time_no_cache,
            'cached_time': cached_time,
            'speedup': speedup,
            'cache_speedup': cache_speedup,
            'n_sequences': new_input.shape[0],
            'memory_mb': new_memory / 1e6
        })
    
    print("\n" + "=" * 60)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 60)
    
    for result in results:
        config = result['config']
        print(f"\nConfig: {config['n_samples']:,} samples × {config['n_channels']} channels")
        print(f"        Sequence: {config['seq_len']} → Forecast: {config['forecast_len']}")
        print(f"        Generated: {result['n_sequences']:,} sequences ({result['memory_mb']:.1f} MB)")
        
        if result['old_time'] != float('inf'):
            print(f"        Old method: {result['old_time']:.4f}s")
            print(f"        New method: {result['new_time']:.4f}s")
            print(f"        Cached:     {result['cached_time']:.4f}s")
            print(f"        Speedup:    {result['speedup']:.1f}x (no cache), {result['cache_speedup']:.1f}x (cached)")
        else:
            print(f"        Old method: FAILED")
            print(f"        New method: {result['new_time']:.4f}s (SUCCESS)")
            print(f"        Cached:     {result['cached_time']:.4f}s")
    
    # Calculate statistics
    successful_results = [r for r in results if r['old_time'] != float('inf')]
    if successful_results:
        avg_speedup = np.mean([r['speedup'] for r in successful_results])
        avg_cache_speedup = np.mean([r['cache_speedup'] for r in successful_results])
        print(f"\n" + "=" * 60)
        print(f"PERFORMANCE SUMMARY")
        print(f"Average speedup (no cache): {avg_speedup:.1f}x")
        print(f"Average speedup (cached):   {avg_cache_speedup:.1f}x")
        print(f"Successful optimizations:   {len(successful_results)}/{len(results)}")
        print("=" * 60)


def demonstrate_caching_usage():
    """Demonstrate how to use the caching features"""
    print("=" * 60)
    print("CACHING USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Create model instance with realistic parameters
    model = NDDBase(verbose=True)
    realistic_samples = (180 + 120 + 100) * 256  # 102,400 samples  
    data = create_test_data(realistic_samples, 64)  # 64 channels
    
    print(f"\nUsing realistic data: {data.shape} ({data.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    
    print("\n1. First call - sequences computed and cached:")
    start = time.time()
    input1, target1 = model._prepare_multistep_sequences(data, 32, 16)  # Realistic seq/forecast lengths
    time1 = time.time() - start
    print(f"   Time: {time1:.4f}s ({input1.shape[0]:,} sequences generated)")
    
    print("\n2. Second call with same data - uses cache:")
    start = time.time()
    input2, target2 = model._prepare_multistep_sequences(data, 32, 16)
    time2 = time.time() - start
    print(f"   Time: {time2:.4f}s (speedup: {time1/time2:.0f}x)")
    
    print("\n3. Different parameters - recomputes:")
    start = time.time()
    input3, target3 = model._prepare_multistep_sequences(data, 8, 1)  # Different params
    time3 = time.time() - start
    print(f"   Time: {time3:.4f}s ({input3.shape[0]:,} sequences generated)")
    
    print("\n4. Same different parameters - now cached:")
    start = time.time()
    input4, target4 = model._prepare_multistep_sequences(data, 8, 1)  # Same as #3
    time4 = time.time() - start
    print(f"   Time: {time4:.4f}s (speedup: {time3/time4:.0f}x)")
    
    print("\n5. Cache management:")
    print("   - Clear cache: model.clear_sequence_cache()")
    print("   - Disable caching: model.disable_sequence_cache()")
    print("   - Enable caching: model.enable_sequence_cache()")
    
    # Demonstrate cache clearing
    model.clear_sequence_cache()
    print("\n6. After clearing cache - recomputes:")
    start = time.time()
    input5, target5 = model._prepare_multistep_sequences(data, 32, 16)  # Same as #1
    time5 = time.time() - start
    print(f"   Time: {time5:.4f}s")
    
    print("\n7. Tips for optimal performance with your data:")
    print("   - With 50-100 channels and 102k samples, expect major speedups")
    print("   - Short sequences (1-32) benefit from vectorization")
    print("   - Cache is most beneficial when reusing same data/parameters")
    print("   - Clear cache when switching between different datasets")
    print("   - Memory usage scales with number of sequences, not data size")


if __name__ == "__main__":
    print("DynaSD Sequence Generation Performance Test")
    print("=" * 60)
    print("Testing with REALISTIC neural data parameters:")
    print("• Sample size: 102,400 samples (400 seconds at 256 Hz)")  
    print("• Channels: 50-100 (typical for high-density EEG)")
    print("• Sequence lengths: 1-128 (short sequences for real-time)")
    print("• Forecast lengths: 1-32 (short-term prediction)")
    print("=" * 60)
    
    try:
        benchmark_sequence_generation()
        demonstrate_caching_usage()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST COMPLETED SUCCESSFULLY!")
        print("The optimized sequence generation should show significant speedups,")
        print("especially with your realistic parameters (many channels, short sequences).")
        print("Caching provides additional speedups for repeated operations.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during performance test: {e}")
        import traceback
        traceback.print_exc()
