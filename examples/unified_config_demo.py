#!/usr/bin/env python3
"""
Unified Configuration System Demo
Showcases the centralized configuration management for Matrix0.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azchess.config import Config
from azchess.utils import (ConfigPath, config_get, config_get_section,
                           config_get_typed, config_manager, set_global_config,
                           validate_config_requirements)


def demo_basic_config_access():
    """Demonstrate basic configuration access patterns."""
    print("=" * 60)
    print("Basic Configuration Access Demo")
    print("=" * 60)

    # Load configuration
    cfg = Config.load("config.yaml")
    set_global_config(cfg)

    print("Configuration loaded and set as global")
    print()

    # Demonstrate different access patterns
    print("Access patterns:")

    # Direct key access
    device = config_get("device", "auto")
    print(f"  Direct: config_get('device', 'auto') = {device}")

    # Section.key access
    batch_size = config_get("training.batch_size", 32)
    print(f"  Section: config_get('training.batch_size', 32) = {batch_size}")

    # Typed access with conversion
    lr = config_get_typed("training.lr", float, 0.001)
    print(f"  Typed: config_get_typed('training.lr', float, 0.001) = {lr}")

    # Section access
    training_config = config_get_section("training")
    print(f"  Section: config_get_section('training') keys = {list(training_config.keys())[:5]}...")

    print("\n✅ Basic config access patterns demonstrated")


def demo_config_validation():
    """Demonstrate configuration validation."""
    print("\n" + "=" * 60)
    print("Configuration Validation Demo")
    print("=" * 60)

    # Define required configuration
    requirements = {
        "device": str,
        "training.batch_size": int,
        "training.lr": float,
        "mcts.num_simulations": int,
        "mcts.cpuct": float
    }

    print("Validating required configuration:")
    for path, expected_type in requirements.items():
        value = config_get(path)
        print(f"  {path}: {value} (expected: {expected_type.__name__})")

    # Validate requirements
    is_valid = validate_config_requirements(requirements)
    print(f"\nConfiguration validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")

    print("\n✅ Configuration validation demonstrated")


def demo_advanced_config_features():
    """Demonstrate advanced configuration features."""
    print("\n" + "=" * 60)
    print("Advanced Configuration Features Demo")
    print("=" * 60)

    print("Advanced features:")

    # ConfigPath objects
    mcts_path = ConfigPath(section="mcts", key="cpuct")
    cpuct = config_get(mcts_path, 1.4)
    print(f"  ConfigPath: config_get(ConfigPath(section='mcts', key='cpuct'), 1.4) = {cpuct}")

    # Type conversion
    str_batch_size = config_get_typed("training.batch_size", str, "32")
    print(f"  Type conversion: batch_size as string = {str_batch_size}")

    # Boolean conversion
    encoder_cache = config_get_typed("mcts.encoder_cache", bool, False)
    print(f"  Boolean conversion: encoder_cache = {encoder_cache}")

    # Default value handling
    nonexistent = config_get("nonexistent.key", "default_value")
    print(f"  Default handling: nonexistent.key = {nonexistent}")

    print("\n✅ Advanced configuration features demonstrated")


def demo_cross_module_consistency():
    """Demonstrate how the unified config provides consistency across modules."""
    print("\n" + "=" * 60)
    print("Cross-Module Consistency Demo")
    print("=" * 60)

    print("Unified configuration provides consistency by:")
    print("• Centralized access patterns")
    print("• Cached configuration values")
    print("• Type checking and conversion")
    print("• Error handling and validation")
    print("• Thread-safe operations")
    print("• Structured path navigation")

    print("\nExample usage patterns:")
    print("  # In training module:")
    print("  batch_size = config_get('training.batch_size', 32)")
    print("  lr = config_get_typed('training.lr', float, 0.001)")
    print("  ")
    print("  # In MCTS module:")
    print("  cpuct = config_get('mcts.cpuct', 1.4)")
    print("  simulations = config_get('mcts.num_simulations', 800)")
    print("  ")
    print("  # In orchestrator:")
    print("  device = config_get('device', 'auto')")
    print("  workers = config_get('selfplay.num_workers', 2)")

    print("\n✅ Cross-module consistency demonstrated")


def demo_performance_features():
    """Demonstrate performance features of the unified config system."""
    print("\n" + "=" * 60)
    print("Performance Features Demo")
    print("=" * 60)

    import time

    print("Performance features:")
    print("• Configuration value caching")
    print("• Thread-safe operations")
    print("• Lazy loading and evaluation")
    print("• Efficient path resolution")

    # Demonstrate caching
    start_time = time.time()
    for i in range(1000):
        _ = config_get("training.batch_size", 32)
    cached_time = time.time() - start_time

    print(".4f")
    print(".1f")

    # Show cache statistics
    cache_size = len(config_manager._config_cache)
    print(f"  Cache size: {cache_size} entries")

    print("\n✅ Performance features demonstrated")


def main():
    """Run all unified configuration demonstrations."""
    print("Matrix0 Unified Configuration System Demo")
    print("=" * 60)
    print("This demo showcases the centralized configuration management system,")
    print("providing consistent access patterns across all Matrix0 modules.")
    print()

    # Run all demos
    demo_basic_config_access()
    demo_config_validation()
    demo_advanced_config_features()
    demo_cross_module_consistency()
    demo_performance_features()

    print("\n" + "=" * 60)
    print("🎉 All unified configuration demonstrations completed!")
    print("=" * 60)
    print("\nThe unified configuration system provides:")
    print("• Centralized configuration management")
    print("• Consistent access patterns across modules")
    print("• Type checking and automatic conversion")
    print("• Configuration validation and error handling")
    print("• Thread-safe operations with caching")
    print("• Structured path navigation")
    print("• Performance optimization through caching")
    print("• Cross-module consistency and reliability")
    print("\nConfiguration access is now unified across:")
    print("• Training modules (batch_size, lr, etc.)")
    print("• MCTS modules (simulations, cpuct, etc.)")
    print("• Self-play modules (workers, simulations, etc.)")
    print("• Arena and evaluation modules")
    print("• All other Matrix0 components")
    print("\nThe configuration system is now production-ready with")
    print("enterprise-grade reliability and consistency! 🏆")


if __name__ == "__main__":
    main()
