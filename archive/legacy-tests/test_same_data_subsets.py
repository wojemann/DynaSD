"""
Test demonstrating cache miss with same data but different DataFrame subsets.
Shows the problem and potential solutions.
"""

import numpy as np
import pandas as pd
import time
import torch
from DynaSD.NDDBase import NDDBase

def test_same_data_subset_problem():
    print("Testing cache behavior with YOUR EXACT use case...")
    print("=" * 70)
    
    # YOUR EXACT SCENARIO: X is 10000x50, train on X[:5000,:], inference on all X
    np.random.seed(42)
    X = np.random.randn(10000, 50)  # Your full dataset
    X_df = pd.DataFrame(X, columns=[f'ch_{i:02d}' for i in range(50)])
    
    # Your exact workflow
    train_data = X_df.iloc[:5000, :].copy()  # Training: first 5000 samples
    inference_data = X_df.copy()             # Inference: all 10000 samples
    
    model = NDDBase(verbose=True)
    
    print(f"Full dataset X shape: {X_df.shape}")
    print(f"Training data shape: {train_data.shape} (X[:5000, :])")
    print(f"Inference data shape: {inference_data.shape} (all of X)")
    print(f"Training data is subset of inference: {np.array_equal(train_data.values, inference_data.iloc[:5000, :].values)}")
    
    # Test current behavior
    print("\n" + "=" * 70)
    print("CURRENT CACHE BEHAVIOR - YOUR WORKFLOW")
    print("=" * 70)
    
    print("1. TRAINING on X[:5000, :]:")
    start = time.time()
    train_input, train_target = model._prepare_multistep_sequences(train_data, 32, 16)
    train_time = time.time() - start
    train_hash = model._get_data_hash(train_data, 32, 16)
    print(f"   Time: {train_time:.4f}s, Training sequences: {train_input.shape[0]}, Hash: {train_hash[:8]}...")
    
    print("2. INFERENCE on full X (contains all training data):")
    start = time.time()
    inference_input, inference_target = model._prepare_multistep_sequences(inference_data, 32, 16)
    inference_time = time.time() - start
    inference_hash = model._get_data_hash(inference_data, 32, 16)
    print(f"   Time: {inference_time:.4f}s, Inference sequences: {inference_input.shape[0]}, Hash: {inference_hash[:8]}...")
    print(f"   Cache hit: {train_hash == inference_hash} (should be True but ISN'T!)")
    
    # Calculate how many sequences are actually identical
    # Training data generates sequences from samples 0 to 5000-48
    # These should be identical to the first part of inference sequences
    seq_len, forecast_len, stride = 32, 16, 16
    total_len = seq_len + forecast_len  # 48
    
    # Number of training sequences
    train_sequences = train_input.shape[0]
    
    # Check if the first N inference sequences match training sequences
    sequences_match = torch.allclose(train_input, inference_input[:train_sequences])
    targets_match = torch.allclose(train_target, inference_target[:train_sequences])
    
    print(f"   First {train_sequences} inference sequences IDENTICAL to training: {sequences_match and targets_match}")
    print(f"   WASTED COMPUTATION: {train_sequences}/{inference_input.shape[0]} sequences ({train_sequences/inference_input.shape[0]*100:.1f}%) could be reused!")
    
    speedup_potential = train_time / (inference_time * (1 - train_sequences/inference_input.shape[0]))
    print(f"   Potential speedup if cached: {speedup_potential:.1f}x")

def demonstrate_content_based_caching():
    """Demonstrate a content-based caching approach that could work better"""
    print("\n" + "=" * 70)
    print("PROPOSED CONTENT-BASED CACHING")
    print("=" * 70)
    
    def content_based_hash(data, sequence_length, forecast_length, stride=None):
        """Hash based on actual data content, not DataFrame shape"""
        if stride is None:
            stride = forecast_length
            
        param_str = f"{sequence_length}_{forecast_length}_{stride}"
        
        # Hash the actual data content more thoroughly
        data_content = data.values
        
        # Use multiple statistical fingerprints
        content_stats = [
            np.mean(data_content),
            np.std(data_content),
            np.sum(data_content[:100] if len(data_content) > 100 else data_content),  # Start
            np.sum(data_content[-100:] if len(data_content) > 100 else data_content), # End
            np.sum(data_content[len(data_content)//2:len(data_content)//2+100] if len(data_content) > 200 else data_content), # Middle
            data_content.shape[0],  # Include length
            data_content.shape[1],  # Include channels
        ]
        
        # Create hash from parameters + content statistics
        import hashlib
        hash_str = param_str + ''.join(f'{stat:.8f}' for stat in content_stats)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    # Test with YOUR exact scenario
    np.random.seed(42)
    X = np.random.randn(10000, 50)
    X_df = pd.DataFrame(X, columns=[f'ch_{i:02d}' for i in range(50)])
    
    train_data = X_df.iloc[:5000, :].copy()   # X[:5000, :]
    inference_data = X_df.copy()              # Full X
    
    # Current hashing (includes shape, will differ)
    train_hash_old = NDDBase()._get_data_hash(train_data, 32, 16)
    inference_hash_old = NDDBase()._get_data_hash(inference_data, 32, 16)
    
    # Improved content-based hashing
    train_hash_new = content_based_hash(train_data, 32, 16)
    inference_hash_new = content_based_hash(inference_data, 32, 16)
    
    print("YOUR SCENARIO - Hash comparison:")
    print(f"  Training data (5000×50) → Old: {train_hash_old[:8]}..., New: {train_hash_new[:8]}...")
    print(f"  Inference data (10000×50) → Old: {inference_hash_old[:8]}..., New: {inference_hash_new[:8]}...")
    print(f"  Current method match: {train_hash_old == inference_hash_old} ← CACHE MISS!")
    print(f"  Improved method match: {train_hash_new == inference_hash_new} ← Still different (expected)")
    
    print("\nBUT - What if we hash just the OVERLAPPING part?")
    # Hash only the training portion of the inference data
    inference_train_portion = inference_data.iloc[:5000, :].copy()
    inference_portion_hash_old = NDDBase()._get_data_hash(inference_train_portion, 32, 16)
    inference_portion_hash_new = content_based_hash(inference_train_portion, 32, 16)
    
    print(f"  Training data hash: {train_hash_old[:8]}...")
    print(f"  Inference[:5000] hash: {inference_portion_hash_old[:8]}...")
    print(f"  Overlapping portion match: {train_hash_old == inference_portion_hash_old} ← SHOULD match!")
    print(f"  Content identical: {np.array_equal(train_data.values, inference_train_portion.values)}")

def show_solution_options():
    print("\n" + "=" * 70)
    print("SOLUTION OPTIONS FOR YOUR EXACT WORKFLOW")
    print("=" * 70)
    
    print("Your workflow: X(10000×50) → train on X[:5000,:] → inference on full X")
    print("Problem: Training sequences are IDENTICAL to first part of inference sequences")
    print("But current cache misses due to different DataFrame shapes!")
    
    print("\nOption 1: SIMPLE - Clear cache between training/inference")
    print("  model.fit(X[:5000, :])  # Caching works during training")
    print("  model.clear_sequence_cache()  # Clear before inference")
    print("  model.forward(X)  # Still gets 4x vectorization speedup")
    print("  → Easy fix, still much faster than old method")
    
    print("\nOption 2: IMPROVED - Shape-agnostic content-based caching")
    print("  → Modify hash to focus on data content, not DataFrame shape")
    print("  → Would detect when training data is subset of inference data")
    print("  → Partial cache hits for overlapping sequences")
    
    print("\nOption 3: SMART - Prefix-aware caching")
    print("  → Cache sequences with content-based keys")
    print("  → Detect when one dataset is prefix of another")
    print("  → Automatically reuse all possible sequences")
    print("  → Perfect for your train-on-subset, infer-on-full pattern")
    
    print("\nImmediate recommendation:")
    print("  Option 1 gives you immediate 4x speedup with zero code changes")
    print("  If you want the full cache benefit, we can implement Option 3")
    print("  Your use case would benefit enormously from smart caching!")

if __name__ == "__main__":
    test_same_data_subset_problem()
    demonstrate_content_based_caching()
    show_solution_options() 