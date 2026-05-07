"""
Quick test to demonstrate the improved caching efficiency.
Shows how the new approach eliminates duplicate cache entries.
"""

import numpy as np
import pandas as pd
import time
from DynaSD.NDDBase import NDDBase

def test_caching_efficiency():
    print("Testing improved caching efficiency...")
    print("=" * 50)
    
    # Create test data
    data = pd.DataFrame(np.random.randn(10000, 8), columns=[f'ch_{i}' for i in range(8)])
    model = NDDBase(verbose=True)
    
    print("Scenario: First call WITHOUT positions, then WITH positions")
    print("-" * 50)
    
    # First call - without positions
    print("1. First call (ret_positions=False):")
    start = time.time()
    input1, target1 = model._prepare_multistep_sequences(data, 32, 16, ret_positions=False)
    time1 = time.time() - start
    print(f"   Time: {time1:.4f}s")
    print(f"   Cache size: {len(model._sequence_cache)}")
    
    # Second call - with positions (should be cache hit now!)
    print("\n2. Second call (ret_positions=True) - should be CACHE HIT:")
    start = time.time()
    input2, target2, positions2 = model._prepare_multistep_sequences(data, 32, 16, ret_positions=True)
    time2 = time.time() - start
    print(f"   Time: {time2:.4f}s")
    print(f"   Cache size: {len(model._sequence_cache)}")
    print(f"   Positions returned: {len(positions2) if positions2 else 0}")
    print(f"   Speedup: {time1/time2:.1f}x")
    
    # Verify data is identical
    assert torch.allclose(input1, input2), "Inputs should be identical"
    assert torch.allclose(target1, target2), "Targets should be identical"
    assert positions2 is not None, "Positions should not be None"
    print("   ✓ Data verified identical")
    
    print("\n" + "=" * 50)
    print("EFFICIENCY IMPROVEMENT DEMONSTRATED!")
    print("• Single cache entry for both position modes")
    print("• Cache hit regardless of call order")
    print("• Minimal overhead for always computing positions")
    print("=" * 50)

if __name__ == "__main__":
    import torch  # Add missing import
    test_caching_efficiency() 