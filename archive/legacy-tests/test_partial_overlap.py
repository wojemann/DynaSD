"""
Test partial data overlap scenarios to understand cache behavior.
"""

import numpy as np
import pandas as pd
import time
from DynaSD.NDDBase import NDDBase

def test_partial_overlap_scenarios():
    print("Testing cache behavior with partial data overlap...")
    print("=" * 60)
    
    # Create base dataset
    np.random.seed(42)  # For reproducible results
    base_data = np.random.randn(10000, 8)
    base_df = pd.DataFrame(base_data, columns=[f'ch_{i}' for i in range(8)])
    
    print(f"Base dataset shape: {base_df.shape}")
    
    model = NDDBase(verbose=True)
    
    # Scenario 1: Completely identical data
    print("\n" + "=" * 60)
    print("SCENARIO 1: Completely identical data")
    print("=" * 60)
    
    identical_df = base_df.copy()
    
    print("1. First call with base data:")
    start = time.time()
    input1, target1 = model._prepare_multistep_sequences(base_df, 32, 16)
    time1 = time.time() - start
    hash1 = model._get_data_hash(base_df, 32, 16)
    print(f"   Time: {time1:.4f}s, Hash: {hash1[:8]}...")
    
    print("2. Second call with identical data:")
    start = time.time()
    input2, target2 = model._prepare_multistep_sequences(identical_df, 32, 16)
    time2 = time.time() - start
    hash2 = model._get_data_hash(identical_df, 32, 16)
    print(f"   Time: {time2:.4f}s, Hash: {hash2[:8]}...")
    print(f"   Cache hit: {hash1 == hash2}, Speedup: {time1/time2:.1f}x")
    
    # Scenario 2: 50% overlap (first half same, second half different)
    print("\n" + "=" * 60)
    print("SCENARIO 2: 50% overlap (first half same, second half different)")
    print("=" * 60)
    
    model.clear_sequence_cache()
    
    half_overlap_data = base_data.copy()
    half_overlap_data[5000:] = np.random.randn(5000, 8)  # Different second half
    half_overlap_df = pd.DataFrame(half_overlap_data, columns=[f'ch_{i}' for i in range(8)])
    
    print("1. First call with base data:")
    start = time.time()
    input1, target1 = model._prepare_multistep_sequences(base_df, 32, 16)
    time1 = time.time() - start
    hash1 = model._get_data_hash(base_df, 32, 16)
    print(f"   Time: {time1:.4f}s, Hash: {hash1[:8]}...")
    
    print("2. Second call with 50% overlap data:")
    start = time.time()
    input2, target2 = model._prepare_multistep_sequences(half_overlap_df, 32, 16)
    time2 = time.time() - start
    hash2 = model._get_data_hash(half_overlap_df, 32, 16)
    print(f"   Time: {time2:.4f}s, Hash: {hash2[:8]}...")
    print(f"   Cache hit: {hash1 == hash2}, Detection: {'✓ DIFFERENT' if hash1 != hash2 else '✗ MISSED'}")
    
    # Scenario 3: Small change in middle (edge case)
    print("\n" + "=" * 60)
    print("SCENARIO 3: Small change in middle (edge case)")
    print("=" * 60)
    
    model.clear_sequence_cache()
    
    small_change_data = base_data.copy()
    small_change_data[4900:5100, :] += 0.01  # Tiny change in middle
    small_change_df = pd.DataFrame(small_change_data, columns=[f'ch_{i}' for i in range(8)])
    
    print("1. First call with base data:")
    start = time.time()
    input1, target1 = model._prepare_multistep_sequences(base_df, 32, 16)
    time1 = time.time() - start
    hash1 = model._get_data_hash(base_df, 32, 16)
    print(f"   Time: {time1:.4f}s, Hash: {hash1[:8]}...")
    
    print("2. Second call with small middle change:")
    start = time.time()
    input2, target2 = model._prepare_multistep_sequences(small_change_df, 32, 16)
    time2 = time.time() - start
    hash2 = model._get_data_hash(small_change_df, 32, 16)
    print(f"   Time: {time2:.4f}s, Hash: {hash2[:8]}...")
    print(f"   Cache hit: {hash1 == hash2}, Detection: {'✓ DIFFERENT' if hash1 != hash2 else '✗ MISSED'}")
    
    # Scenario 4: Different shape (should always detect)
    print("\n" + "=" * 60)
    print("SCENARIO 4: Different shape")
    print("=" * 60)
    
    model.clear_sequence_cache()
    
    different_shape_data = np.random.randn(8000, 8)  # Different number of samples
    different_shape_df = pd.DataFrame(different_shape_data, columns=[f'ch_{i}' for i in range(8)])
    
    print("1. First call with base data:")
    start = time.time()
    input1, target1 = model._prepare_multistep_sequences(base_df, 32, 16)
    time1 = time.time() - start
    hash1 = model._get_data_hash(base_df, 32, 16)
    print(f"   Time: {time1:.4f}s, Hash: {hash1[:8]}...")
    
    print("2. Second call with different shape:")
    start = time.time()
    input2, target2 = model._prepare_multistep_sequences(different_shape_df, 32, 16)
    time2 = time.time() - start
    hash2 = model._get_data_hash(different_shape_df, 32, 16)
    print(f"   Time: {time2:.4f}s, Hash: {hash2[:8]}...")
    print(f"   Cache hit: {hash1 == hash2}, Detection: {'✓ DIFFERENT' if hash1 != hash2 else '✗ MISSED'}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Current cache detection strategy:")
    print("• ✓ Detects identical data correctly")
    print("• ✓ Detects different shapes")
    print("• ? May miss subtle changes depending on sampling")
    print("• Uses 100 evenly spaced samples + first/last 50 sums")
    print("\nRecommendation: Current strategy is generally safe but could")
    print("be improved for edge cases with subtle changes.")

def demonstrate_improved_hash():
    """Demonstrate a more robust hashing strategy"""
    print("\n" + "=" * 60)
    print("IMPROVED HASH DEMONSTRATION")
    print("=" * 60)
    
    def improved_hash(data, sequence_length, forecast_length, stride=None):
        """More robust hash that's less likely to miss differences"""
        shape_str = f"{data.shape}_{sequence_length}_{forecast_length}_{stride}"
        
        # More comprehensive sampling strategy
        if len(data) > 1000:
            # Sample from beginning, middle, end, and random positions
            n_samples = min(200, len(data) // 10)
            indices = []
            
            # Beginning samples
            indices.extend(range(0, min(50, len(data))))
            
            # End samples  
            indices.extend(range(max(0, len(data) - 50), len(data)))
            
            # Middle samples
            mid_start = len(data) // 2 - 25
            mid_end = len(data) // 2 + 25
            indices.extend(range(max(0, mid_start), min(len(data), mid_end)))
            
            # Random samples for better coverage
            np.random.seed(42)  # Deterministic
            random_indices = np.random.choice(len(data), min(75, len(data)), replace=False)
            indices.extend(random_indices)
            
            # Remove duplicates and sort
            indices = sorted(list(set(indices)))
            sample_data = data.iloc[indices].values.flatten()
        else:
            sample_data = data.values.flatten()
        
        # Use more robust statistics
        stats = [
            np.mean(sample_data),
            np.std(sample_data), 
            np.min(sample_data),
            np.max(sample_data),
            np.sum(sample_data[:len(sample_data)//3]),  # First third
            np.sum(sample_data[len(sample_data)//3:2*len(sample_data)//3]),  # Middle third
            np.sum(sample_data[2*len(sample_data)//3:])  # Last third
        ]
        
        hash_str = shape_str + ''.join(f'{stat:.6f}' for stat in stats)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    # Test improved hash on the scenarios above
    import hashlib
    
    base_data = np.random.randn(10000, 8)
    base_df = pd.DataFrame(base_data, columns=[f'ch_{i}' for i in range(8)])
    
    # Small change scenario
    small_change_data = base_data.copy()
    small_change_data[4900:5100, :] += 0.01
    small_change_df = pd.DataFrame(small_change_data, columns=[f'ch_{i}' for i in range(8)])
    
    old_hash1 = NDDBase()._get_data_hash(base_df, 32, 16)
    old_hash2 = NDDBase()._get_data_hash(small_change_df, 32, 16)
    
    new_hash1 = improved_hash(base_df, 32, 16)
    new_hash2 = improved_hash(small_change_df, 32, 16)
    
    print(f"Small change detection:")
    print(f"  Current method: {old_hash1 == old_hash2} (same hash = missed)")
    print(f"  Improved method: {new_hash1 == new_hash2} (same hash = missed)")
    print(f"  Improvement: {'✓ Better detection' if (old_hash1 == old_hash2) and (new_hash1 != new_hash2) else 'No change needed'}")

if __name__ == "__main__":
    test_partial_overlap_scenarios()
    demonstrate_improved_hash() 