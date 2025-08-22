#!/usr/bin/env python3
"""
Create a fresh V2 base checkpoint for Matrix0.
This script initializes the V2 model architecture and saves it as a starting checkpoint.
"""

import os
import sys
import torch
import logging

# Add the project root to the path
sys.path.insert(0, '/Users/admin/Downloads/VSCode/Matrix0')

from azchess.model.resnet import PolicyValueNet, NetConfig
from azchess.config import Config

def create_v2_checkpoint():
    """Create a fresh V2 model checkpoint."""

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config_path = '/Users/admin/Downloads/VSCode/Matrix0/config.yaml'
        config = Config.load(config_path)

        # Get model configuration
        model_cfg = config.model()

        logger.info("Creating 32M PARAMETER BEAST - FULL V2 FEATURES!")
        logger.info(f"Model config: {model_cfg}")

        # Create model with V2 architecture
        device = torch.device('cpu')  # Create on CPU first, then move if needed

        # Create NetConfig for 32M PARAMETER BEAST - FULL V2 FEATURES!
        net_config = NetConfig(
            planes=19,
            channels=320,  # DOUBLED from 160 for massive capacity
            blocks=24,     # +10 blocks for deep strategic reasoning
            policy_size=4672,
            se=True,  # Squeeze-and-Excitation for adaptive feature recalibration
            se_ratio=0.25,
            attention=True,  # Full attention mechanisms
            attention_heads=20,  # +12 heads for rich spatial relationships
            attention_unmasked_mix=0.2,
            attention_relbias=True,
            attention_every_k=3,
            chess_features=True,  # All chess-specific features
            self_supervised=True,  # Full SSL pipeline
            piece_square_tables=True,  # Positional knowledge
            wdl=True,  # Win-Draw-Loss auxiliary head
            policy_factor_rank=128,  # Factorized policy for efficiency
            norm='group',  # GroupNorm for stability
            activation='silu',  # SiLU for better gradients
            preact=True,  # Pre-activation for deeper networks
            droppath=0.2,  # DropPath for regularization
            aux_policy_from_square=True,  # Auxiliary from-square head
            aux_policy_move_type=True,  # Auxiliary move-type head
            enable_visual=False,  # Keep visual disabled for now
            ssl_tasks=['piece', 'threat', 'pin', 'fork', 'control'],  # Full SSL suite
            ssl_curriculum=True,  # Progressive difficulty
            ssrl_tasks=['position', 'material', 'rotation'],  # Full SSRL
            enable_llm_tutor=False  # Keep LLM disabled for now
        )

        model = PolicyValueNet(net_config)
        
        # The model now has proper weight initialization in _init_weights method
        # This includes conservative SSL head initialization and policy head stability fixes
        logger.info("Model created with proper weight initialization")

        # Create checkpoint directory if it doesn't exist
        os.makedirs('/Users/admin/Downloads/VSCode/Matrix0/checkpoints', exist_ok=True)

        # Create checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,  # Fresh start
            'scheduler_state_dict': None,  # Fresh start
            'epoch': 0,
            'global_step': 0,
            'best_loss': float('inf'),
            'config': model_cfg,
            'version': 'v2_large_32m_beast_fixed',
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
        checkpoint_path = '/Users/admin/Downloads/VSCode/Matrix0/checkpoints/v2_large_32m_fixed.pt'
        torch.save(checkpoint, checkpoint_path)

        # Also save as the main v2_base.pt for compatibility
        main_checkpoint_path = '/Users/admin/Downloads/VSCode/Matrix0/checkpoints/v2_base.pt'
        torch.save(checkpoint, main_checkpoint_path)

        logger.info(f"✅ Successfully created 32M PARAMETER BEAST (FIXED):")
        logger.info(f"   - Large checkpoint: {checkpoint_path}")
        logger.info(f"   - Main checkpoint: {main_checkpoint_path}")

        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   - Total parameters: {total_params:,}")

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
