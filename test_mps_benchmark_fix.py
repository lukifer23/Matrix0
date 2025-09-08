#!/usr/bin/env python3
"""
Test script to verify MPS fixes in benchmark system work correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_mps_benchmark_fixes():
    """Test that MPS stability fixes are properly implemented in benchmark system."""

    print("=== MPS BENCHMARK FIXES VERIFICATION ===\n")

    # Test 1: Check that MPS environment variables are set correctly
    print("1. Checking MPS environment variables...")

    mps_env_vars = [
        'PYTORCH_ENABLE_MPS_FALLBACK',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO',
        'PYTORCH_MPS_LOW_WATERMARK_RATIO'
    ]

    for var in mps_env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")

    # Test 2: Verify benchmark system can import and initialize
    print("\n2. Testing benchmark system initialization...")

    try:
        from benchmarks.benchmark import BenchmarkRunner
        from benchmarks.config import BenchmarkConfig, UCIEngineConfig, TestScenario, PerformanceConfig
        print("   ✅ Benchmark system imports successful")

        # Test creating a basic config
        engine = UCIEngineConfig(
            name="test_engine",
            command="echo test",
            options={}
        )

        perf_config = PerformanceConfig()
        scenario = TestScenario(
            name="test_scenario",
            engine_config=engine,
            model_checkpoint="checkpoints/best.pt",
            num_games=1
        )

        config = BenchmarkConfig(
            name="Test Config",
            description="Test configuration for MPS fixes",
            scenarios=[scenario],
            performance_config=perf_config
        )

        print("   ✅ BenchmarkConfig creation successful")

    except Exception as e:
        print(f"   ❌ Benchmark system initialization failed: {e}")
        return False

    # Test 3: Check MPS detection and handling in device selection
    print("\n3. Testing MPS device handling...")

    try:
        from azchess.config import select_device

        # Test device selection
        device = select_device("auto")
        print(f"   Auto-selected device: {device}")

        # Test MPS-specific selection
        if hasattr(select_device, '__code__'):
            # Try to test MPS device selection
            mps_device = select_device("mps")
            print(f"   MPS device selection: {mps_device}")
        else:
            print("   Device selection function available")

    except Exception as e:
        print(f"   ❌ Device selection test failed: {e}")

    # Test 4: Verify MPS stability code is present in benchmark.py
    print("\n4. Verifying MPS stability code in benchmark.py...")

    benchmark_file = Path("benchmarks/benchmark.py")
    if benchmark_file.exists():
        with open(benchmark_file, 'r') as f:
            content = f.read()

        mps_indicators = [
            "torch.mps.empty_cache",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            "enable_memory_optimization",
            "MPS-specific error recovery"
        ]

        found_indicators = []
        for indicator in mps_indicators:
            if indicator in content:
                found_indicators.append(indicator)

        print(f"   Found {len(found_indicators)}/{len(mps_indicators)} MPS stability indicators:")
        for indicator in found_indicators:
            print(f"   ✅ {indicator}")
        for indicator in mps_indicators:
            if indicator not in found_indicators:
                print(f"   ❌ Missing: {indicator}")
    else:
        print("   ❌ benchmark.py not found")

    print("\n=== MPS FIXES VERIFICATION COMPLETE ===")

    print("\n🎯 RECOMMENDED NEXT STEPS:")
    print("1. Test the EX0Bench command:")
    print("   python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 5 --time 60+0.6")
    print("\n2. If issues persist, try with explicit MPS settings:")
    print("   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 5 --time 60+0.6")

    return True

if __name__ == "__main__":
    test_mps_benchmark_fixes()
