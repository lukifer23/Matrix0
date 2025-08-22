#!/usr/bin/env python3
"""
Example script showing how to run the Matrix0 benchmark system.
This demonstrates proper usage from the project root directory.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.append('.')

from benchmarks.config import ConfigManager
from benchmarks.uci_bridge import EngineManager
from benchmarks.benchmark import BenchmarkRunner


def main():
    """Run a simple benchmark example."""

    print("🚀 Matrix0 Benchmark System Example")
    print("=" * 40)

    # Load default configuration
    try:
        config = ConfigManager.load_config('benchmarks/configs/default.yaml')
        print(f"✅ Loaded configuration: {config.name}")
        print(f"✅ Description: {config.description}")
        print(f"✅ Scenarios: {len(config.scenarios)}")

        for i, scenario in enumerate(config.scenarios, 1):
            print(f"   {i}. {scenario.name} vs {scenario.engine_config.name}")

    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # Initialize engine manager
    engine_manager = EngineManager()
    print(f"✅ Engine manager initialized")

    # Note: This is just a demonstration
    # In a real run, you would:
    # 1. Check if Matrix0 model exists
    # 2. Start UCI engines
    # 3. Run the benchmark

    print("\n📋 To run a real benchmark:")
    print("1. Ensure you have a trained Matrix0 model in checkpoints/")
    print("2. Install UCI engines (Stockfish, lc0, etc.)")
    print("3. Run: python benchmarks/benchmark.py --config benchmarks/configs/default.yaml")

    print("\n📊 Expected output files:")
    print("• benchmarks/results/matrix0_benchmark_results.json")
    print("• benchmarks/results/matrix0_benchmark_summary.json")
    print("• benchmarks/results/matrix0_benchmark_report.txt")
    print("• benchmarks/results/matrix0_benchmark_report.html (if plotly available)")

    # Clean up
    engine_manager.stop_all()
    print("✅ Example completed successfully")


if __name__ == "__main__":
    main()
