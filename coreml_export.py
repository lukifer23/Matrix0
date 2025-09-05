#!/usr/bin/env python3
"""
CoreML Export for Matrix0 Chess Engine
Exports trained Matrix0 models to CoreML format for optimized Apple Silicon inference.

Usage:
    python coreml_export.py --checkpoint checkpoints/best.pt --output matrix0.mlmodel
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from azchess.config import Config
from azchess.model import PolicyValueNet


def export_to_coreml(checkpoint_path: str, output_path: str, device: str = "cpu"):
    """
    Export Matrix0 model to CoreML format for Apple Silicon optimization.

    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Path for CoreML model output
        device: Device to load model on
    """
    print("🚀 Starting CoreML export for Matrix0...")

    # Check if coremltools is available
    try:
        import coremltools as ct
        print("✅ CoreML tools available")
    except ImportError:
        print("❌ CoreML tools not installed. Install with: pip install coremltools")
        print("Note: CoreML export requires macOS with Xcode command line tools")
        sys.exit(1)

    # Load configuration and model
    print(f"📋 Loading configuration...")
    cfg = Config.load("config.yaml")

    print(f"🧠 Loading model from {checkpoint_path}...")
    model = PolicyValueNet.from_config(cfg.model())

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_ema", checkpoint.get("model", {})))
    model.eval()

    print(f"📊 Model loaded: {model.count_parameters()} parameters")

    # Create example input for tracing
    print("🎯 Creating example input for tracing...")
    example_input = torch.randn(1, 19, 8, 8)  # Matrix0 input shape

    # Trace the model
    print("🔍 Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Define input/output specifications for CoreML
    print("📝 Defining CoreML specifications...")

    # Input: chess board state (19 planes, 8x8)
    input_spec = ct.TensorType(
        name="board_state",
        shape=(1, 19, 8, 8),
        dtype=ct.TensorType.np.float32
    )

    # Outputs: policy logits and value
    policy_spec = ct.TensorType(
        name="policy_logits",
        dtype=ct.TensorType.np.float32
    )

    value_spec = ct.TensorType(
        name="value",
        dtype=ct.TensorType.np.float32
    )

    # Convert to CoreML with optimizations
    print("⚡ Converting to CoreML format...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[input_spec],
        outputs=[policy_spec, value_spec],
        compute_units=ct.ComputeUnit.ALL,  # Use CPU, GPU, and ANE
        minimum_deployment_target=ct.target.iOS15,  # Modern iOS support
    )

    # Add metadata
    print("📋 Adding model metadata...")
    coreml_model.author = "Matrix0 Chess Engine"
    coreml_model.license = "MIT"
    coreml_model.short_description = "AlphaZero-style chess engine with SSL integration"
    coreml_model.version = "2.1"

    coreml_model.input_description["board_state"] = "Chess board state (19 planes, 8x8)"
    coreml_model.output_description["policy_logits"] = "Policy logits for all legal moves (4672)"
    coreml_model.output_description["value"] = "Position evaluation (-1 to 1)"

    # Add user-defined metadata
    coreml_model.user_defined_metadata["model_type"] = "chess_engine"
    coreml_model.user_defined_metadata["architecture"] = "ResNet-24 + SSL"
    coreml_model.user_defined_metadata["ssl_tasks"] = "7"
    coreml_model.user_defined_metadata["parameters"] = str(model.count_parameters())
    coreml_model.user_defined_metadata["source"] = "Matrix0 v2.1"

    # Save the model
    print(f"💾 Saving CoreML model to {output_path}...")
    coreml_model.save(output_path)

    # Get file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"📏 Model size: {file_size:.1f} MB")
    print("✅ CoreML export completed successfully!")
    print(f"🎯 Model ready for Apple Silicon inference optimization")

    return output_path


def benchmark_coreml_inference(coreml_path: str, num_positions: int = 100):
    """
    Benchmark CoreML model inference performance.

    Args:
        coreml_path: Path to CoreML model
        num_positions: Number of test positions
    """
    print(f"🏃 Benchmarking CoreML inference ({num_positions} positions)...")

    try:
        import coremltools as ct
        import numpy as np
        import time

        # Load model
        model = ct.models.MLModel(coreml_path)

        # Generate test positions
        test_positions = []
        for _ in range(num_positions):
            pos = np.random.randn(1, 19, 8, 8).astype(np.float32)
            test_positions.append(pos)

        # Benchmark
        times = []
        for pos in test_positions:
            start = time.time()
            prediction = model.predict({"board_state": pos})
            end = time.time()
            times.append(end - start)

        avg_time = sum(times) / len(times) * 1000  # ms
        print(f"⚡ Average inference time: {avg_time:.2f} ms")
        print(f"🏃 Max inference time: {max(times) * 1000:.2f} ms")
        print(f"🐌 Min inference time: {min(times) * 1000:.2f} ms")
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export Matrix0 to CoreML format")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output path for CoreML model")
    parser.add_argument("--device", default="cpu", help="Device to load model on")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark after export")

    args = parser.parse_args()

    # Export model
    output_path = export_to_coreml(args.checkpoint, args.output, args.device)

    # Optional benchmarking
    if args.benchmark:
        benchmark_coreml_inference(output_path)

    print("\n🎉 CoreML export complete!")
    print(f"📁 Model saved to: {output_path}")
    print("💡 Use this model for optimized inference on Apple Silicon devices")


if __name__ == "__main__":
    main()
