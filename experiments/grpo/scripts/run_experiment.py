#!/usr/bin/env python3
"""
GRPO Experiment Runner

Simple script to run GRPO experiments with different configurations.
Designed for quick iteration and testing of different architectures.
"""

import argparse
import yaml
import torch
import logging
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from experiments.grpo.models.small_resnet import ChessSmallResNet
from experiments.grpo.models.chess_transformer import ChessTransformerFactory
from experiments.grpo.training.grpo_trainer import GRPOTrainer, GRPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str, experiment_name: str) -> Dict[str, Any]:
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    if experiment_name not in configs:
        raise ValueError(f"Experiment '{experiment_name}' not found in config")

    return configs[experiment_name]


def create_model(config: Dict[str, Any], device: str):
    """Create model based on configuration"""
    model_config = config['model']

    if model_config['type'] == 'small_resnet':
        model = ChessSmallResNet.create(
            input_channels=model_config.get('input_channels', 19),
            base_channels=model_config.get('base_channels', 64),
            num_blocks=model_config.get('num_blocks', 4)
        )
    elif model_config['type'] == 'chess_transformer':
        variant = model_config.get('variant', 'small')
        if variant == 'small':
            model = ChessTransformerFactory.create_small(
                input_channels=model_config.get('input_channels', 19),
                d_model=model_config.get('d_model', 128),
                nhead=model_config.get('nhead', 4),
                num_layers=model_config.get('num_layers', 4),
                dim_feedforward=model_config.get('dim_feedforward', 512)
            )
        elif variant == 'medium':
            model = ChessTransformerFactory.create_medium(
                input_channels=model_config.get('input_channels', 19),
                d_model=model_config.get('d_model', 256),
                nhead=model_config.get('nhead', 8),
                num_layers=model_config.get('num_layers', 6),
                dim_feedforward=model_config.get('dim_feedforward', 1024)
            )
        else:
            raise ValueError(f"Unknown transformer variant: {variant}")
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")

    model.to(device)
    return model


def create_grpo_config(config: Dict[str, Any]) -> GRPOConfig:
    """Create GRPO configuration from dict"""
    grpo_config = config['grpo']
    return GRPOConfig(
        group_size=int(grpo_config.get('group_size', 4)),
        clip_epsilon=float(grpo_config.get('clip_epsilon', 0.2)),
        value_loss_coef=float(grpo_config.get('value_loss_coef', 0.5)),
        entropy_coef=float(grpo_config.get('entropy_coef', 0.01)),
        learning_rate=float(grpo_config.get('learning_rate', 1e-4)),
        max_grad_norm=float(grpo_config.get('max_grad_norm', 0.5)),
        ppo_epochs=int(grpo_config.get('ppo_epochs', 4)),
        batch_size=int(grpo_config.get('batch_size', 32))
    )


def run_experiment(experiment_name: str, config_path: str, device: str = "cpu",
                  max_epochs: int = None, quick_test: bool = False):
    """Run a single GRPO experiment"""

    logger.info(f"Starting experiment: {experiment_name}")

    # Load configuration
    config = load_config(config_path, experiment_name)
    logger.info(f"Loaded config: {config}")

    # Create model
    model = create_model(config, device)
    model_info = ChessSmallResNet.get_model_info(model) if hasattr(model, 'conv1') else ChessTransformerFactory.get_model_info(model)
    logger.info(f"Created model: {model_info}")

    # Create GRPO trainer
    grpo_config = create_grpo_config(config)
    trainer = GRPOTrainer(model, grpo_config, device)

    # Training configuration
    training_config = config['training']
    num_games_per_epoch = training_config['num_games_per_epoch']
    max_epochs = max_epochs or training_config.get('max_epochs', 5)

    logger.info(f"Training config: {num_games_per_epoch} games/epoch, {max_epochs} epochs")

    # Quick test mode
    if quick_test:
        logger.info("Running quick test...")
        # Create dummy trajectories for testing
        test_trajectories = []
        for i in range(grpo_config.group_size):
            # Create dummy trajectory
            from experiments.grpo.training.grpo_trainer import Trajectory, TrajectoryStep
            steps = [TrajectoryStep(
                state=torch.randn(19, 8, 8),
                action=i % 4672,
                log_prob=-1.0,
                value=0.0,
                reward=0.1,
                done=False
            )]
            traj = Trajectory(steps=steps, total_reward=0.1, length=1, game_result=0.0)
            test_trajectories.append(traj)

        # Test training
        metrics = trainer.train_on_trajectories(test_trajectories)
        logger.info(f"Quick test metrics: {metrics}")
        return

    # Full training loop (placeholder for now)
    logger.info("Full training not yet implemented - use quick_test mode")

    # TODO: Implement full training loop with:
    # 1. MCTS integration for game generation
    # 2. Trajectory collection
    # 3. GRPO training updates
    # 4. Evaluation and checkpointing


def main():
    parser = argparse.ArgumentParser(description='Run GRPO experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name from config file')
    parser.add_argument('--config', type=str, default='configs/experiment_configs.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda/mps)')
    parser.add_argument('--max-epochs', type=int, default=None,
                       help='Override max epochs')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with dummy data')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Using device: {args.device}")

    # Make config path absolute
    config_path = Path(__file__).parent.parent / args.config

    try:
        run_experiment(
            experiment_name=args.experiment,
            config_path=str(config_path),
            device=args.device,
            max_epochs=args.max_epochs,
            quick_test=args.quick_test
        )
        logger.info("Experiment completed successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
