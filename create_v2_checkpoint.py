#!/usr/bin/env python3
"""
Create a fresh V2 base checkpoint for Matrix0.
This script initializes the V2 model architecture and saves it as a starting checkpoint.
"""

import logging
import os
import sys
from dataclasses import asdict

import torch

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from azchess.config import Config
from azchess.model.resnet import NetConfig, PolicyValueNet


def create_v2_checkpoint():
    """Create a fresh V2 model checkpoint."""

    try:
        # Load configuration from config.yaml via unified Config helper
        config_path = os.path.join(project_root, 'config.yaml')
        cfg = Config.load(config_path)
        model_cfg = cfg.model()

        logger.info("Creating FRESH V2 CHECKPOINT - ALIGNED WITH CURRENT CONFIG!")
        logger.info(f"Model config: {model_cfg}")

        # Create model with V2 architecture
        device = torch.device('cpu')  # Create on CPU first, then move if needed

        # Build NetConfig by intersecting config keys with dataclass fields
        net_kwargs = {name: model_cfg[name] for name in NetConfig.__dataclass_fields__ if name in model_cfg}
        net_config = NetConfig(**net_kwargs)

        # Preserve any extra model attributes (e.g., policy_logit_init_scale) for PolicyValueNet getters
        for key, value in model_cfg.items():
            if key not in NetConfig.__dataclass_fields__:
                setattr(net_config, key, value)

        model = PolicyValueNet(net_config)
        
        # The model now has proper weight initialization in _init_weights method
        # This includes conservative SSL head initialization and policy head stability fixes
        logger.info("Model created with proper weight initialization")

        # Sanity check value head does not collapse at initialization
        with torch.no_grad():
            dummy_states = torch.zeros(2, net_config.planes, 8, 8)
            policy_logits, value_logits = model(dummy_states)
            logger.info(
                "   - Value sanity check (min/max): %.6f / %.6f",
                float(value_logits.min().item()),
                float(value_logits.max().item()),
            )

        # Create checkpoint directory if it doesn't exist
        checkpoints_dir = os.path.join(project_root, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Create checkpoint data (save in a compatible format for all loaders)
        state_dict = model.state_dict()
        checkpoint = {
            'model_state_dict': state_dict,
            'model': state_dict,        # compatibility with loaders expecting 'model'
            'model_ema': state_dict,    # seed EMA with same weights for fresh start
            'optimizer_state_dict': None,  # Fresh start
            'scheduler_state_dict': None,  # Fresh start
            'epoch': 0,
            'global_step': 0,
            'best_loss': float('inf'),
            'config': model_cfg,
            'version': 'v2_fresh_clean',
            'model_config': asdict(net_config)
        }

        # Save the checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, 'v2_fresh_clean.pt')
        torch.save(checkpoint, checkpoint_path)

        # Also save as the main v2_base.pt for compatibility
        main_checkpoint_path = os.path.join(checkpoints_dir, 'v2_base.pt')
        torch.save(checkpoint, main_checkpoint_path)

        # Seed best.pt if none exists so orchestrator/self-play start from the fresh weights
        best_checkpoint_path = os.path.join(checkpoints_dir, 'best.pt')
        if not os.path.exists(best_checkpoint_path):
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"   - Seeded best checkpoint: {best_checkpoint_path}")
        else:
            logger.info(f"   - Preserved existing best checkpoint: {best_checkpoint_path}")

        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        ssl_head_count = len(net_config.ssl_tasks) if net_config.ssl_tasks else 0

        logger.info(f"✅ Successfully created FRESH V2 CHECKPOINT:")
        logger.info(f"   - Fresh checkpoint: {checkpoint_path}")
        logger.info(f"   - Main checkpoint: {main_checkpoint_path}")
        logger.info(f"   - Total parameters: {total_params:,}")
        logger.info(f"   - SSL enabled: {net_config.self_supervised}")
        logger.info(f"   - SSL tasks: {net_config.ssl_tasks} ({ssl_head_count} heads)")
        logger.info(f"   - SSL curriculum: {net_config.ssl_curriculum}")
        logger.info(f"   - SSRL tasks: {net_config.ssrl_tasks}")

        # Log SSL head information if SSL is enabled
        if net_config.self_supervised and hasattr(model, 'ssl_heads'):
            ssl_head_params = {}
            for task_name, head in model.ssl_heads.items():
                ssl_head_params[task_name] = sum(p.numel() for p in head.parameters())
            logger.info(f"   - SSL head parameters: {ssl_head_params}")

        # Test loading the checkpoint
        logger.info("Testing checkpoint loading...")
        loaded_checkpoint = torch.load(main_checkpoint_path, map_location='cpu')
        test_model = PolicyValueNet(net_config)
        test_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        logger.info("✅ Checkpoint loading test successful!")

        # Log initial policy logit scale for transparency
        try:
            initial_scale = (torch.nn.functional.softplus(test_model._policy_logit_scale_raw).item()
                             + test_model._policy_logit_scale_eps)
            logger.info(f"   - Initial policy logit scale: {initial_scale:.4f}")
        except Exception:
            pass

        return checkpoint_path

    except Exception as e:
        logger.error(f"❌ Error creating V2 checkpoint: {e}")
        raise

if __name__ == "__main__":
    create_v2_checkpoint()
