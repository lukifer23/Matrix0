#!/usr/bin/env python3
"""
Demonstration of Unified Utilities in Matrix0
This script shows how to use the newly unified utility modules for consistent behavior.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azchess.utils import \
    check_tensor_health  # Memory management; Device management; Tensor utilities; Checkpoint management; Configuration utilities; Logging utilities
from azchess.utils import (clear_memory_cache, create_tensor,
                           emergency_memory_cleanup, ensure_contiguous,
                           get_checkpoint_info, get_device_info, get_logger,
                           get_memory_usage, load_checkpoint,
                           log_config_summary, safe_config_get,
                           save_checkpoint, select_device, setup_device,
                           setup_logging, validate_device,
                           validate_tensor_shapes)


def demo_memory_management():
    """Demonstrate unified memory management."""
    print("=" * 50)
    print("Memory Management Demo")
    print("=" * 50)

    # Get memory usage
    usage = get_memory_usage()
    print(f"Memory usage: {usage}")

    # Clear cache
    print("Clearing memory cache...")
    clear_memory_cache()

    # Emergency cleanup
    print("Performing emergency memory cleanup...")
    emergency_memory_cleanup()

    print("✅ Memory management demo complete")


def demo_device_management():
    """Demonstrate unified device management."""
    print("\n" + "=" * 50)
    print("Device Management Demo")
    print("=" * 50)

    # Auto-select device
    device = select_device("auto")
    print(f"Auto-selected device: {device}")

    # Validate device
    is_valid = validate_device(device)
    print(f"Device valid: {is_valid}")

    # Get device info
    info = get_device_info(device)
    print(f"Device info: {info}")

    # Setup device
    setup_device(device)
    print(f"Device {device} configured")

    print("✅ Device management demo complete")


def demo_tensor_utilities():
    """Demonstrate unified tensor utilities."""
    print("\n" + "=" * 50)
    print("Tensor Utilities Demo")
    print("=" * 50)

    try:
        import torch

        # Create tensor
        tensor = create_tensor([1, 2, 3, 4, 5], device="cpu", dtype=torch.float32)
        print(f"Created tensor: {tensor}")

        # Check health
        health = check_tensor_health(tensor, "demo_tensor")
        print(f"Tensor health: {health}")

        # Ensure contiguous
        contiguous_tensor = ensure_contiguous(tensor, "demo_tensor")
        print(f"Contiguous tensor: {contiguous_tensor.is_contiguous()}")

        # Validate shapes
        tensor2 = create_tensor([6, 7, 8, 9, 10], device="cpu", dtype=torch.float32)
        shapes_valid = validate_tensor_shapes(tensor, tensor2, names=["tensor1", "tensor2"])
        print(f"Shapes valid: {shapes_valid}")

    except ImportError:
        print("PyTorch not available, skipping tensor demo")

    print("✅ Tensor utilities demo complete")


def demo_checkpoint_management():
    """Demonstrate unified checkpoint management."""
    print("\n" + "=" * 50)
    print("Checkpoint Management Demo")
    print("=" * 50)

    try:
        import tempfile

        import torch

        # Create a simple model for demo
        model = torch.nn.Linear(10, 1)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name

        success = save_checkpoint(
            model=model,
            step=42,
            loss=0.123,
            path=checkpoint_path
        )
        print(f"Checkpoint saved: {success}")

        # Get checkpoint info
        info = get_checkpoint_info(checkpoint_path)
        print(f"Checkpoint info: {info}")

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, model=model)
        print(f"Checkpoint loaded: {checkpoint is not None}")

        # Cleanup
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)

    except ImportError:
        print("PyTorch not available, skipping checkpoint demo")

    print("✅ Checkpoint management demo complete")


def demo_logging_utilities():
    """Demonstrate unified logging utilities."""
    print("\n" + "=" * 50)
    print("Logging Utilities Demo")
    print("=" * 50)

    # Setup logging
    logger = setup_logging(level=20)  # INFO level

    # Get named logger
    demo_logger = get_logger("demo")
    demo_logger.info("This is a demo log message")
    demo_logger.debug("This debug message won't show at INFO level")
    demo_logger.warning("This is a warning message")
    demo_logger.error("This is an error message")

    print("✅ Logging utilities demo complete")


def main():
    """Run all unified utilities demonstrations."""
    print("Matrix0 Unified Utilities Demonstration")
    print("This script demonstrates the newly unified utility modules")

    # Run all demos
    demo_memory_management()
    demo_device_management()
    demo_tensor_utilities()
    demo_checkpoint_management()
    demo_logging_utilities()

    print("\n" + "=" * 60)
    print("🎉 All unified utilities demonstrations completed!")
    print("=" * 60)
    print("\nThese utilities provide:")
    print("• Consistent memory management across modules")
    print("• Unified device selection and validation")
    print("• Standardized tensor operations and validation")
    print("• Centralized checkpoint management")
    print("• Unified logging configuration")
    print("• Safe configuration access patterns")
    print("\nUse these utilities throughout Matrix0 for consistent behavior!")


if __name__ == "__main__":
    main()
