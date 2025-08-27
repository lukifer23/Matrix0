#!/usr/bin/env python3
"""
Merge trained weights from old checkpoint to new SSL-integrated checkpoint.
This preserves trained policy/value weights while adding fresh SSL heads.
"""

import os
import torch
import logging
from pathlib import Path
from azchess.model.resnet import PolicyValueNet, NetConfig
from azchess.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_checkpoint_safely(path: str, device: str = 'cpu'):
    """Load checkpoint with error handling."""
    try:
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None

def create_weight_mapping(old_state: dict, new_model: PolicyValueNet) -> dict:
    """Create mapping from old parameter names to new ones."""
    mapping = {}
    old_keys = set(old_state.keys())
    new_keys = set(new_model.state_dict().keys())

    # Find common parameters (should be most policy/value weights)
    common_keys = old_keys & new_keys
    logger.info(f"Common parameters: {len(common_keys)}")

    # Check for shape mismatches
    shape_mismatches = []
    for key in common_keys:
        old_shape = old_state[key].shape
        new_shape = new_model.state_dict()[key].shape
        if old_shape != new_shape:
            shape_mismatches.append((key, old_shape, new_shape))

    if shape_mismatches:
        logger.warning(f"Shape mismatches found: {len(shape_mismatches)}")
        for key, old_shape, new_shape in shape_mismatches:
            logger.warning(f"  {key}: {old_shape} vs {new_shape}")

    # Only map parameters that have matching shapes
    valid_mapping = {}
    for key in common_keys:
        old_shape = old_state[key].shape
        new_shape = new_model.state_dict()[key].shape
        if old_shape == new_shape:
            valid_mapping[key] = key

    logger.info(f"Valid parameter mappings: {len(valid_mapping)}")
    return valid_mapping

def merge_checkpoints(old_checkpoint_path: str, new_checkpoint_path: str, output_path: str, config_path: str = "config.yaml"):
    """Merge old trained weights into new SSL-integrated checkpoint."""

    # Load configuration
    cfg = Config.load(config_path)

    # Create new model with full SSL integration
    device = 'cpu'
    new_model = PolicyValueNet.from_config(cfg.model()).to(device)
    logger.info(f"Created new model with {sum(p.numel() for p in new_model.parameters()):,} parameters")

    # Load old checkpoint
    old_checkpoint = load_checkpoint_safely(old_checkpoint_path, device)
    if not old_checkpoint:
        raise ValueError(f"Could not load old checkpoint: {old_checkpoint_path}")

    # Extract old model state
    old_model_state = None
    if 'model_state_dict' in old_checkpoint:
        old_model_state = old_checkpoint['model_state_dict']
    elif 'model' in old_checkpoint:
        old_model_state = old_checkpoint['model']
    else:
        raise ValueError("Could not find model state in old checkpoint")

    logger.info(f"Old checkpoint has {sum(p.numel() for p in old_model_state.values()):,} parameters")

    # Create parameter mapping
    weight_mapping = create_weight_mapping(old_model_state, new_model)

    # Load new checkpoint to get the base SSL-integrated state
    new_checkpoint = load_checkpoint_safely(new_checkpoint_path, device)
    if not new_checkpoint:
        raise ValueError(f"Could not load new checkpoint: {new_checkpoint_path}")

    if 'model_state_dict' in new_checkpoint:
        new_model_state = new_checkpoint['model_state_dict']
    elif 'model' in new_checkpoint:
        new_model_state = new_checkpoint['model']
    else:
        raise ValueError("Could not find model state in new checkpoint")

    # Merge weights: copy old trained weights where possible, keep new SSL heads
    merged_state = new_model_state.copy()  # Start with new SSL-integrated state

    merged_count = 0
    ssl_heads_preserved = 0

    for new_key, old_key in weight_mapping.items():
        # Only copy if it's not an SSL head (preserve fresh SSL initialization)
        if not new_key.startswith(('ssl_', 'SSL_')):
            merged_state[new_key] = old_model_state[old_key].clone()
            merged_count += 1
        else:
            ssl_heads_preserved += 1

    logger.info(f"Merged {merged_count} trained parameters")
    logger.info(f"Preserved {ssl_heads_preserved} fresh SSL head parameters")

    # Count SSL heads in merged state
    ssl_keys = [k for k in merged_state.keys() if k.startswith(('ssl_', 'SSL_'))]
    logger.info(f"SSL-related parameters in merged checkpoint: {len(ssl_keys)}")

    # Create merged checkpoint
    merged_checkpoint = {
        'model_state_dict': merged_state,
        'optimizer_state_dict': None,  # Fresh optimizer for new training
        'scheduler_state_dict': None,  # Fresh scheduler
        'epoch': 0,
        'global_step': 0,
        'best_loss': float('inf'),
        'config': cfg.model() if isinstance(cfg.model(), dict) else cfg.model().__dict__,
        'version': 'v2_merged_ssl',
        'merge_info': {
            'old_checkpoint': str(old_checkpoint_path),
            'new_checkpoint': str(new_checkpoint_path),
            'merged_params': merged_count,
            'ssl_heads_preserved': ssl_heads_preserved,
            'total_params': sum(p.numel() for p in merged_state.values())
        },
        'ssl_info': {
            'ssl_enabled': cfg.model().get('self_supervised', False),
            'ssl_tasks': cfg.model().get('ssl_tasks', []),
            'ssl_head_count': len(cfg.model().get('ssl_tasks', []))
        }
    }

    # Save merged checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_checkpoint, output_path)

    logger.info(f"✅ Successfully created merged checkpoint: {output_path}")
    logger.info(f"   - Total parameters: {merged_checkpoint['merge_info']['total_params']:,}")
    logger.info(f"   - Trained parameters merged: {merged_count}")
    logger.info(f"   - SSL heads preserved: {ssl_heads_preserved}")
    logger.info(f"   - SSL tasks enabled: {cfg.model().get('ssl_tasks', [])}")

    return output_path

def main():
    """Main function to merge checkpoints."""
    import argparse

    parser = argparse.ArgumentParser(description="Merge old checkpoint weights into new SSL-integrated checkpoint")
    parser.add_argument("--old-checkpoint", type=str, default="checkpoints/model_step_4000.pt",
                       help="Path to old checkpoint with trained weights")
    parser.add_argument("--new-checkpoint", type=str, default="checkpoints/v2_base.pt",
                       help="Path to new SSL-integrated checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/v2_merged.pt",
                       help="Path for merged checkpoint output")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file path")

    args = parser.parse_args()

    # Verify input files exist
    if not os.path.exists(args.old_checkpoint):
        logger.error(f"Old checkpoint not found: {args.old_checkpoint}")
        return

    if not os.path.exists(args.new_checkpoint):
        logger.error(f"New checkpoint not found: {args.new_checkpoint}")
        return

    try:
        merge_checkpoints(args.old_checkpoint, args.new_checkpoint, args.output, args.config)
        logger.info("✅ Checkpoint merge completed successfully!")
        logger.info(f"   Use the merged checkpoint: {args.output}")

    except Exception as e:
        logger.error(f"❌ Checkpoint merge failed: {e}")
        raise

if __name__ == "__main__":
    main()
