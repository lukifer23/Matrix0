#!/usr/bin/env python3
"""
Create a fresh V2 base checkpoint for Matrix0.
This script initializes the V2 model architecture and saves it as a starting checkpoint.
"""

import logging
import os
import sys

import torch

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from azchess.model.resnet import NetConfig, PolicyValueNet


def create_v2_checkpoint():
    """Create a fresh V2 model checkpoint."""

    try:
        # Load configuration from config.yaml in the same directory
        config_path = os.path.join(project_root, 'config.yaml')

        # Simple config loading without external dependencies
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Get model configuration from the config
        model_cfg = config_data.get('model', {})

        logger.info("Creating FRESH V2 CHECKPOINT - ALIGNED WITH CURRENT CONFIG!")
        logger.info(f"Model config: {model_cfg}")

        # Create model with V2 architecture
        device = torch.device('cpu')  # Create on CPU first, then move if needed

        # Create NetConfig using the same approach as the main training code
        net_config = NetConfig(
            planes=model_cfg.get('planes', 19),
            channels=model_cfg.get('channels', 320),
            blocks=model_cfg.get('blocks', 24),
            policy_size=model_cfg.get('policy_size', 4672),
            se=model_cfg.get('se', True),
            se_ratio=model_cfg.get('se_ratio', 0.25),
            attention=model_cfg.get('attention', True),
            attention_heads=model_cfg.get('attention_heads', 20),
            # Use a sane float default and coerce to float to avoid boolean spillover
            attention_unmasked_mix=float(model_cfg.get('attention_unmasked_mix', 0.2)),
            attention_relbias=model_cfg.get('attention_relbias', True),
            attention_every_k=model_cfg.get('attention_every_k', 3),
            chess_features=model_cfg.get('chess_features', True),
            self_supervised=model_cfg.get('self_supervised', False),  # Match current config
            piece_square_tables=model_cfg.get('piece_square_tables', True),
            wdl=model_cfg.get('wdl', False),
            policy_factor_rank=model_cfg.get('policy_factor_rank', 128),
            norm=model_cfg.get('norm', 'group'),
            activation=model_cfg.get('activation', 'silu'),
            preact=model_cfg.get('preact', True),
            droppath=model_cfg.get('droppath', 0.1),
            aux_policy_from_square=model_cfg.get('aux_policy_from_square', True),
            aux_policy_move_type=model_cfg.get('aux_policy_move_type', True),
            enable_visual=model_cfg.get('enable_visual', False),
            visual_encoder_channels=model_cfg.get('visual_encoder_channels', 64),
            ssl_tasks=model_cfg.get('ssl_tasks', []),  # Match current config (disabled)
            ssl_curriculum=model_cfg.get('ssl_curriculum', True),
            ssrl_tasks=model_cfg.get('ssrl_tasks', [])  # Match current config (disabled)
        )

        model = PolicyValueNet(net_config)
        
        # The model now has proper weight initialization in _init_weights method
        # This includes conservative SSL head initialization and policy head stability fixes
        logger.info("Model created with proper weight initialization")

        # Create checkpoint directory if it doesn't exist
        checkpoints_dir = os.path.join(project_root, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Create checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,  # Fresh start
            'scheduler_state_dict': None,  # Fresh start
            'epoch': 0,
            'global_step': 0,
            'best_loss': float('inf'),
            'config': model_cfg,
            'version': 'v2_fresh_clean',
            'model_config': {
                'channels': net_config.channels,
                'blocks': net_config.blocks,
                'attention_heads': net_config.attention_heads,
                'policy_factor_rank': net_config.policy_factor_rank,
                'ssl_tasks': net_config.ssl_tasks,
                'ssrl_tasks': net_config.ssrl_tasks
            }
        }

        # Save the checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, 'v2_fresh_clean.pt')
        torch.save(checkpoint, checkpoint_path)

        # Also save as the main v2_base.pt for compatibility
        main_checkpoint_path = os.path.join(checkpoints_dir, 'v2_base.pt')
        torch.save(checkpoint, main_checkpoint_path)

        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Successfully created FRESH V2 CHECKPOINT:")
        logger.info(f"   - Fresh checkpoint: {checkpoint_path}")
        logger.info(f"   - Main checkpoint: {main_checkpoint_path}")
        logger.info(f"   - Total parameters: {total_params:,}")
        logger.info(f"   - SSL enabled: {net_config.self_supervised}")
        logger.info(f"   - SSL tasks: {net_config.ssl_tasks}")
        logger.info(f"   - SSRL tasks: {net_config.ssrl_tasks}")

        # Test loading the checkpoint
        logger.info("Testing checkpoint loading...")
        loaded_checkpoint = torch.load(main_checkpoint_path, map_location='cpu')
        test_model = PolicyValueNet(net_config)
        test_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        logger.info("✅ Checkpoint loading test successful!")

        return checkpoint_path

    except Exception as e:
        logger.error(f"❌ Error creating V2 checkpoint: {e}")
        raise

if __name__ == "__main__":
    create_v2_checkpoint()
