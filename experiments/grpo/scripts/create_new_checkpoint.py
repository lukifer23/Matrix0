#!/usr/bin/env python3
"""
Create fresh baseline checkpoints for all transformer architectures
"""
import torch
import os
from pathlib import Path
import sys

# Add the parent directory to the path to allow imports from the models module
sys.path.append(str(Path(__file__).parent.parent))

from models.large_chess_transformer import MagnusChessTransformerFactory

def create_magnus_checkpoint():
    """Create Magnus transformer checkpoint (~70M parameters)"""
    print("🚀 Creating MAGNUS transformer checkpoint (~70M parameters)...")

    # Create Magnus model
    print("📦 Initializing Magnus transformer model...")
    model = MagnusChessTransformerFactory.create_magnus_chess()

    # Move to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    print(f"🎯 Magnus model created on device: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create checkpoint directory
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create checkpoint
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,  # No optimizer yet
        'loss': float('inf'),
        'config': {
            'model_type': 'magnus_transformer',
            'd_model': 512,
            'nhead': 8,
            'num_layers': 12,
            'dim_feedforward': 2048,
            'architecture': 'MagnusChessTransformer'
        }
    }

    # Save checkpoint
    checkpoint_path = checkpoint_dir / "baseline_magnus_transformer_fresh.pt"
    torch.save(checkpoint, checkpoint_path)

    print(f"💾 Magnus checkpoint saved to: {checkpoint_path}")
    print(f"📁 File size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")

    # Verify checkpoint can be loaded
    print("🔄 Verifying Magnus checkpoint can be loaded...")
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    loaded_model = MagnusChessTransformerFactory.create_magnus_chess()
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    loaded_model = loaded_model.to(device)

    print("✅ Magnus checkpoint verification successful!")
    print("🎉 Magnus baseline checkpoint created and verified!")

def create_medium_checkpoint():
    """Create medium transformer checkpoint for comparison"""
    print("🚀 Creating MEDIUM transformer checkpoint...")

    # Create medium model
    print("📦 Initializing medium transformer model...")
    model = MagnusChessTransformerFactory.create_medium_transformer()

    # Move to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    print(f"🎯 Medium model created on device: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create checkpoint directory
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create checkpoint
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,  # No optimizer yet
        'loss': float('inf'),
        'config': {
            'model_type': 'medium_transformer',
            'd_model': 384,
            'nhead': 6,
            'num_layers': 6,
            'dim_feedforward': 1536
        }
    }

    # Save checkpoint
    checkpoint_path = checkpoint_dir / "baseline_medium_transformer_fresh.pt"
    torch.save(checkpoint, checkpoint_path)

    print(f"💾 Medium checkpoint saved to: {checkpoint_path}")
    print(f"📁 File size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")

    # Verify checkpoint can be loaded
    print("🔄 Verifying medium checkpoint can be loaded...")
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    loaded_model = MagnusChessTransformerFactory.create_medium_transformer()
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    loaded_model = loaded_model.to(device)

    print("✅ Medium checkpoint verification successful!")
    print("🎉 Medium baseline checkpoint created and verified!")

def main():
    """Main function with command line options"""
    import argparse

    parser = argparse.ArgumentParser(description="Create fresh baseline checkpoints for chess transformers")
    parser.add_argument('--model', choices=['magnus', 'medium', 'both'],
                       default='magnus', help='Which model checkpoint to create')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, mps, cuda)')

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🎯 Using device: {device}")

    if args.model == 'magnus':
        create_magnus_checkpoint()
    elif args.model == 'medium':
        create_medium_checkpoint()
    elif args.model == 'both':
        create_magnus_checkpoint()
        print("\n" + "="*50 + "\n")
        create_medium_checkpoint()

if __name__ == "__main__":
    main()