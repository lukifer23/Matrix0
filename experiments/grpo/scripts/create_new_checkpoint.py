#!/usr/bin/env python3
"""
Create a fresh baseline checkpoint with correct architecture
"""
import torch
import os
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.large_chess_transformer import LargeChessTransformerFactory

def create_fresh_checkpoint():
    """Create a new baseline checkpoint with correct relative_pos_emb dimensions"""

    print("🚀 Creating fresh baseline checkpoint...")

    # Create model with correct architecture
    print("📦 Initializing medium transformer model...")
    factory = LargeChessTransformerFactory()
    model = factory.create_medium_large()

    # Move to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    print(f"🎯 Model created on device: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify relative_pos_emb shape
    rel_pos_shape = model.transformer_layers[0].self_attn.relative_pos_emb.shape
    print(f"🔍 Relative pos emb shape: {rel_pos_shape}")

    if rel_pos_shape[0] == 64:
        print("✅ Correct relative_pos_emb shape: [64, nhead, head_dim]")
    else:
        print(f"❌ Incorrect shape: {rel_pos_shape}")
        return

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
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

    print(f"💾 Checkpoint saved to: {checkpoint_path}")
    print(f"📁 File size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")

    # Verify checkpoint can be loaded
    print("🔄 Verifying checkpoint can be loaded...")
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    loaded_model = factory.create_medium_large()
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    loaded_model = loaded_model.to(device)

    print("✅ Checkpoint verification successful!")
    print("🎉 Fresh baseline checkpoint created and verified!")

if __name__ == "__main__":
    create_fresh_checkpoint()
