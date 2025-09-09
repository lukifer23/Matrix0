#!/usr/bin/env python3
"""
GRPO (Generalized Reward-based Policy Optimization) Trainer

Core GRPO implementation with group-based reward normalization for chess.
Designed for integration with MCTS-generated trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """Single step in a trajectory"""
    state: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    legal_mask: Optional[torch.Tensor] = None


@dataclass
class Trajectory:
    """Complete trajectory from a game"""
    steps: List[TrajectoryStep]
    total_reward: float
    length: int
    game_result: float  # +1, 0, or -1


class GRPOConfig:
    """Configuration for GRPO training"""

    def __init__(self,
                 group_size: int = 8,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 learning_rate: float = 1e-4,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 batch_size: int = 64):
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size


class GRPOTrainer:
    """
    GRPO Trainer with group-based reward normalization

    Key innovation: Uses groups of trajectories to normalize rewards,
    reducing variance and improving sample efficiency.
    """

    def __init__(self, model: nn.Module, config: GRPOConfig, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.model.to(device)

        logger.info(f"Initialized GRPO trainer with group_size={config.group_size}")

    def collect_trajectories(self, mcts_engine, num_games: int) -> List[Trajectory]:
        """
        Collect trajectories using MCTS for exploration

        Args:
            mcts_engine: MCTS engine for move selection
            num_games: Number of games to play

        Returns:
            List of trajectories from self-play
        """
        trajectories = []

        for game_idx in range(num_games):
            trajectory = self._play_single_game(mcts_engine)
            trajectories.append(trajectory)
            logger.debug(f"Completed game {game_idx + 1}/{num_games}, length: {trajectory.length}")

        return trajectories

    def _play_single_game(self, mcts_engine) -> Trajectory:
        """Play a single game and collect trajectory"""
        # This would integrate with existing MCTS implementation
        # For now, return a dummy trajectory structure
        steps = []
        total_reward = 0.0

        # TODO: Implement actual game playing with MCTS
        # This would collect: state, action, log_prob, value, reward, done

        # Placeholder: create a simple trajectory
        game_result = np.random.choice([-1, 0, 1])  # Random for now

        return Trajectory(
            steps=steps,
            total_reward=total_reward,
            length=len(steps),
            game_result=game_result
        )

    def train_on_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Train using GRPO with group-based reward normalization

        Args:
            trajectories: List of trajectories to train on

        Returns:
            Dictionary of training metrics
        """
        if len(trajectories) < self.config.group_size:
            logger.warning(f"Only {len(trajectories)} trajectories, need at least {self.config.group_size}")
            return {}

        # Form groups for reward normalization
        groups = self._form_groups(trajectories)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for group in groups:
            group_metrics = self._train_on_group(group)
            total_policy_loss += group_metrics['policy_loss']
            total_value_loss += group_metrics['value_loss']
            total_entropy_loss += group_metrics['entropy_loss']
            num_updates += 1

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'num_groups': len(groups),
            'total_trajectories': len(trajectories)
        }

    def _form_groups(self, trajectories: List[Trajectory]) -> List[List[Trajectory]]:
        """
        Form groups of trajectories for reward normalization

        Strategy: Group by similar reward characteristics to improve normalization
        """
        groups = []

        # Sort by total reward for better grouping
        sorted_trajectories = sorted(trajectories, key=lambda t: t.total_reward)

        # Form groups of group_size
        for i in range(0, len(sorted_trajectories), self.config.group_size):
            group = sorted_trajectories[i:i + self.config.group_size]
            if len(group) >= self.config.group_size // 2:  # Allow partial groups
                groups.append(group)

        logger.info(f"Formed {len(groups)} groups from {len(trajectories)} trajectories")
        return groups

    def _train_on_group(self, group: List[Trajectory]) -> Dict[str, float]:
        """
        Train on a single group using GRPO

        Args:
            group: List of trajectories in this group

        Returns:
            Training metrics for this group
        """
        # Extract rewards for group normalization
        group_rewards = [t.total_reward for t in group]
        reward_mean = np.mean(group_rewards)
        reward_std = np.std(group_rewards) + 1e-8  # Avoid division by zero

        # Normalize rewards within group
        normalized_rewards = [(r - reward_mean) / reward_std for r in group_rewards]

        # TODO: Implement actual GRPO training loop
        # This would include:
        # 1. Compute advantages with group-normalized rewards
        # 2. PPO-style policy updates with clipping
        # 3. Value function updates
        # 4. Entropy regularization

        # Placeholder metrics
        return {
            'policy_loss': 0.5,  # Placeholder
            'value_loss': 0.3,   # Placeholder
            'entropy_loss': 0.1  # Placeholder
        }

    def compute_group_advantage(self, trajectories: List[Trajectory],
                              reward_mean: float, reward_std: float) -> torch.Tensor:
        """
        Compute advantages using group-based reward normalization

        Args:
            trajectories: Group of trajectories
            reward_mean: Mean reward in group
            reward_std: Std reward in group

        Returns:
            Normalized advantages tensor
        """
        # TODO: Implement advantage computation
        # This should compute GAE (Generalized Advantage Estimation)
        # with group-normalized rewards

        # Placeholder
        return torch.randn(len(trajectories), 10)  # (batch, seq_len)

    def grpo_policy_loss(self, old_logprobs: torch.Tensor, new_logprobs: torch.Tensor,
                        advantages: torch.Tensor, clip_epsilon: float) -> torch.Tensor:
        """
        Compute GRPO policy loss with group normalization

        Args:
            old_logprobs: Log probabilities from old policy
            new_logprobs: Log probabilities from new policy
            advantages: Group-normalized advantages
            clip_epsilon: Clipping parameter

        Returns:
            GRPO policy loss
        """
        ratio = torch.exp(new_logprobs - old_logprobs)

        # PPO-style clipping
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

        # GRPO loss (similar to PPO but with group-normalized advantages)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        return policy_loss

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")


class GRPOEvaluator:
    """Evaluator for GRPO-trained models"""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_game(self, game_trajectory: Trajectory) -> Dict[str, float]:
        """Evaluate a single game"""
        # TODO: Implement game evaluation
        # This would replay the game and compute metrics

        return {
            'game_length': game_trajectory.length,
            'final_result': game_trajectory.game_result,
            'total_reward': game_trajectory.total_reward
        }

    def evaluate_games(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Evaluate multiple games"""
        results = [self.evaluate_game(traj) for traj in trajectories]

        return {
            'avg_game_length': np.mean([r['game_length'] for r in results]),
            'win_rate': np.mean([1 if r['final_result'] > 0 else 0 for r in results]),
            'draw_rate': np.mean([1 if r['final_result'] == 0 else 0 for r in results]),
            'loss_rate': np.mean([1 if r['final_result'] < 0 else 0 for r in results]),
            'avg_reward': np.mean([r['total_reward'] for r in results])
        }


if __name__ == "__main__":
    # Test GRPO trainer setup
    from models.small_resnet import ChessSmallResNet

    print("=== GRPO Trainer Test ===")

    # Create small model
    model = ChessSmallResNet.create()
    config = GRPOConfig(group_size=4)  # Smaller for testing
    trainer = GRPOTrainer(model, config)

    print(f"Model parameters: {ChessSmallResNet.get_parameter_count(model):,}")
    print(f"GRPO config: group_size={config.group_size}, lr={config.learning_rate}")
    print("✅ GRPO trainer initialized successfully!")
