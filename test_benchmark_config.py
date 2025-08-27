#!/usr/bin/env python3
"""
Quick test script to verify benchmark configuration and dependencies.
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append('.')

try:
    from benchmarks.config import ConfigManager
    print("✅ ConfigManager import successful")
except ImportError as e:
    print(f"❌ ConfigManager import failed: {e}")
    sys.exit(1)

try:
    from benchmarks.uci_bridge import EngineManager
    print("✅ EngineManager import successful")
except ImportError as e:
    print(f"❌ EngineManager import failed: {e}")
    sys.exit(1)

def test_config_loading():
    """Test loading the benchmark configuration."""
    config_path = "benchmarks/configs/model_step_5000_benchmark.yaml"

    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        config = ConfigManager.load_config(config_path)
        print(f"✅ Config loaded successfully: {config.name}")
        print(f"   - Scenarios: {len(config.scenarios)}")
        for scenario in config.scenarios:
            print(f"   - {scenario.name}: {scenario.engine_config.name} vs {scenario.model_checkpoint}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_engine_manager():
    """Test engine manager initialization."""
    try:
        manager = EngineManager()
        print("✅ EngineManager initialized successfully")
        return True
    except Exception as e:
        print(f"❌ EngineManager initialization failed: {e}")
        return False

def test_model_checkpoint():
    """Test if model checkpoint exists and is readable."""
    checkpoint_path = "checkpoints/model_step_5000.pt"

    if not Path(checkpoint_path).exists():
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        return False

    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Model checkpoint loaded successfully")
        if 'model_state_dict' in checkpoint:
            print(f"   - Model parameters: {len(checkpoint['model_state_dict'])}")
        else:
            print(f"   - Model parameters: {len(checkpoint)}")
        return True
    except Exception as e:
        print(f"❌ Model checkpoint loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Benchmark Configuration Test")
    print("=" * 40)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Engine Manager", test_engine_manager),
        ("Model Checkpoint", test_model_checkpoint)
    ]

    all_passed = True
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if not test_func():
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Ready to run benchmark.")
        print("\n📋 Run the benchmark with:")
        print("   python benchmarks/benchmark.py --config benchmarks/configs/model_step_5000_benchmark.yaml")
    else:
        print("❌ Some tests failed. Please fix issues before running benchmark.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
