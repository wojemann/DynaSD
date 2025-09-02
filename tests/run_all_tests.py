#!/usr/bin/env python3
"""
Main test runner for DynaSD model validation.

This script runs comprehensive tests on all DynaSD models using synthetic data
and generates visualizations showing feature/probability heatmaps for each model.

Usage:
    python run_all_tests.py

Results are saved to tests/results/ with separate subdirectories for each model.
"""

import sys
import time
from pathlib import Path
import warnings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tests.test_absslp import run_absslp_tests
from tests.test_ndd import run_ndd_tests  
from tests.test_wavenet import run_wavenet_tests


def print_banner(text: str, char: str = "=", width: int = 80):
    """Print a formatted banner."""
    print(char * width)
    print(f"{text:^{width}}")
    print(char * width)


def print_section(text: str, char: str = "-", width: int = 60):
    """Print a section header."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")


def main():
    """Run all DynaSD model tests."""
    start_time = time.time()
    
    print_banner("DynaSD Model Test Suite", "=", 80)
    print(f"📋 Running comprehensive tests on all DynaSD models")
    print(f"📊 Each model generates ONE comprehensive figure with 4 subplots:")
    print(f"   - Baseline EEG data (using plot_iEEG_data)")
    print(f"   - Seizure EEG data (using plot_iEEG_data)")
    print(f"   - Baseline features heatmap")
    print(f"   - Seizure features heatmap")
    print(f"📁 Results will be saved to: tests/results/")
    print(f"⏱️  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track test results
    test_results = {}
    
    # Test 1: ABSSLP Model
    print_section("Testing ABSSLP Model (Absolute Slope)")
    try:
        test_results['ABSSLP'] = run_absslp_tests()
    except Exception as e:
        print(f"❌ ABSSLP tests failed with exception: {e}")
        test_results['ABSSLP'] = False
    
    # Test 2: NDD Model  
    print_section("Testing NDD Model (Neural Network Detection)")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            test_results['NDD'] = run_ndd_tests()
    except Exception as e:
        print(f"❌ NDD tests failed with exception: {e}")
        test_results['NDD'] = False
    
    # Test 3: WaveNet Model
    print_section("Testing WVNT Model (WaveNet)")
    try:
        test_results['WaveNet'] = run_wavenet_tests()
    except Exception as e:
        print(f"❌ WaveNet tests failed with exception: {e}")
        test_results['WaveNet'] = False
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print_banner("Test Summary", "=", 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"📊 Test Results:")
    for model, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {model:<15} {status}")
    
    print(f"\n📈 Overall Results:")
    print(f"   Passed: {passed_tests}/{total_tests} models")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"   Duration: {duration:.1f} seconds")
    
    # Check if comprehensive figures were created
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        print(f"\n📁 Generated Results:")
        for model_dir in model_dirs:
            comprehensive_files = list(model_dir.glob("*comprehensive_analysis.png"))
            if comprehensive_files:
                print(f"   ✅ {model_dir.name}: {comprehensive_files[0].name}")
            else:
                print(f"   ❌ {model_dir.name}: No comprehensive analysis found")
    
    # Final recommendations
    print(f"\n💡 Next Steps:")
    if passed_tests == total_tests:
        print("   🎉 All tests passed! Your DynaSD pipeline is working correctly.")
        print("   📊 Review the comprehensive analysis figures in each model directory.")
        print("   🔬 Each figure shows: raw data + model features side-by-side.")
        print("   🔬 Consider running with real EEG data for validation.")
    else:
        print("   🔧 Some tests failed. Check the error messages above.")
        print("   📝 Review model configurations and dependencies.")
        print("   🐛 Check logs for detailed error information.")
    
    print(f"\n📂 Find all results in: {results_dir.absolute()}")
    print("="*80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 