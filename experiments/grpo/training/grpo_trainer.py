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
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda


class GRPOTrainer:
    """
    GRPO Trainer with group-based reward normalization
    """

    def __init__(self, model: nn.Module, config: GRPOConfig, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.model.to(device)

        logger.info(f"Initialized GRPO trainer with group_size={config.group_size}")

    def train_on_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Train using GRPO with group-based reward normalization

        Args:
            trajectories: List of trajectories to train on

        Returns:
            Dictionary of training metrics
        """
        if not trajectories:
            logger.warning("No trajectories provided for training")
            return {}

        if len(trajectories) < self.config.group_size:
            logger.warning(f"Only {len(trajectories)} trajectories, need at least {self.config.group_size}")
            # Use all trajectories as a single group if we don't have enough
            groups = [trajectories]
        else:
            # Form groups for reward normalization
            groups = self._form_groups(trajectories)

        logger.info(f"Starting GRPO training on {len(trajectories)} trajectories in {len(groups)} groups")

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for group_idx, group in enumerate(groups):
            logger.debug(f"Training on group {group_idx + 1}/{len(groups)} ({len(group)} trajectories)")
            group_metrics = self._train_on_group(group)
            total_policy_loss += group_metrics['policy_loss']
            total_value_loss += group_metrics['value_loss']
            total_entropy_loss += group_metrics['entropy_loss']
            num_updates += 1

        avg_metrics = {
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0.0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0.0,
            'entropy_loss': total_entropy_loss / num_updates if num_updates > 0 else 0.0,
            'num_groups': len(groups),
            'total_trajectories': len(trajectories),
            'trajectories_per_group': len(trajectories) / len(groups) if groups else 0
        }

        logger.info(f"GRPO training completed: Policy Loss: {avg_metrics['policy_loss']:.4f}, "
                   f"Value Loss: {avg_metrics['value_loss']:.4f}")
        return avg_metrics

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
            if len(group) > 0:  # Include all groups, even partial ones
                groups.append(group)

        logger.info(f"Formed {len(groups)} groups from {len(trajectories)} trajectories")
        return groups

    def _train_on_group(self, group: List[Trajectory]) -> Dict[str, float]:
        """
        Train on a single group using GRPO.

        Each trajectory's advantages (and value targets) are first scaled by the
        trajectory's normalized total reward ``(R - reward_mean) / reward_std``.
        After concatenating steps across the group, advantages are standardized
        again to zero mean and unit variance. This preserves the relative impact
        of reward scaling while keeping optimization numerically stable.

        Args:
            group: List of trajectories in this group

        Returns:
            Training metrics for this group
        """
        # Process trajectories to compute advantages
        all_advantages = []
        all_returns = []
        all_log_probs = []
        all_states = []
        all_actions = []

        # Group-based reward statistics used to normalize each trajectory
        group_rewards = [t.total_reward for t in group]
        reward_mean = np.mean(group_rewards)
        reward_std = np.std(group_rewards) + 1e-8

        for trajectory in group:
            rewards = [step.reward for step in trajectory.steps]
            values = [step.value for step in trajectory.steps]
            log_probs = [step.log_prob for step in trajectory.steps]
            states = [step.state for step in trajectory.steps]
            actions = [step.action for step in trajectory.steps]

            advantages, returns = self.compute_gae(rewards, values, self.config.gamma, self.config.gae_lambda)

            # Scale by normalized total reward so higher rewarded trajectories
            # have proportionally larger advantages/returns.
            norm_reward = (trajectory.total_reward - reward_mean) / reward_std
            advantages = advantages * norm_reward
            returns = returns * norm_reward

            all_advantages.append(advantages)
            all_returns.append(returns)
            all_log_probs.extend(log_probs)
            all_states.extend(states)
            all_actions.extend(actions)

        if not all_states:
            logger.warning("No states found in group")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}

        # Concatenate all data
        advantages_tensor = torch.cat(all_advantages).to(self.device)
        returns_tensor = torch.cat(all_returns).to(self.device)
        old_log_probs_tensor = torch.tensor(all_log_probs, device=self.device)
        states_tensor = torch.cat(all_states).to(self.device)
        actions_tensor = torch.tensor(all_actions, device=self.device)

        # Standardize advantages after trajectory-level normalization.
        # This keeps optimization stable while still reflecting reward differences.
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Store losses for averaging
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(self.config.ppo_epochs):
            for i in range(0, len(states_tensor), self.config.batch_size):
                batch_states = states_tensor[i:i+self.config.batch_size]
                batch_actions = actions_tensor[i:i+self.config.batch_size]
                batch_old_log_probs = old_log_probs_tensor[i:i+self.config.batch_size]
                batch_advantages = advantages_tensor[i:i+self.config.batch_size]
                batch_returns = returns_tensor[i:i+self.config.batch_size]

                # Get new policy and value
                new_policy_logits, new_values = self.model(batch_states)
                new_policy = F.softmax(new_policy_logits, dim=-1)
                new_log_probs = torch.log(new_policy.gather(1, batch_actions.unsqueeze(1)).squeeze(1) + 1e-8)

                # GRPO Policy loss with group normalization
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)

                # Entropy loss
                entropy = -torch.sum(new_policy * torch.log(new_policy + 1e-8), dim=-1).mean()

                # Total loss
                total_loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'value_loss': np.mean(value_losses) if value_losses else 0.0,
            'entropy_loss': np.mean(entropy_losses) if entropy_losses else 0.0
        }

    def compute_gae(self, rewards: List[float], values: List[float], gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = []
        last_advantage = 0
        last_value = values[-1]

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * last_value - values[i]
            last_advantage = delta + gamma * gae_lambda * last_advantage
            advantages.insert(0, last_advantage)
            last_value = values[i]

        returns = [adv + val for adv, val in zip(advantages, values)]

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

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
        if not game_trajectory.steps:
            return {
                'game_length': 0,
                'final_result': game_trajectory.game_result,
                'total_reward': game_trajectory.total_reward,
                'move_accuracy': 0.0,
                'avg_value_error': 0.0,
            }

        correct_moves = 0
        value_errors = []

        with torch.no_grad():
            for idx, step in enumerate(game_trajectory.steps):
                state = step.state.to(self.device)
                if state.dim() == 1:
                    state = state.unsqueeze(0)

                policy_logits, value_pred = self.model(state)

                # Apply legal move mask if provided
                if step.legal_mask is not None:
                    mask = step.legal_mask.to(self.device)
                    if mask.dim() == 1:
                        mask = mask.unsqueeze(0)
                    masked_logits = policy_logits.clone()
                    masked_logits[mask == 0] = -1e9
                else:
                    masked_logits = policy_logits

                predicted_action = int(torch.argmax(masked_logits, dim=-1).item())
                if predicted_action == step.action:
                    correct_moves += 1

                # Determine target value from current player's perspective
                result = game_trajectory.game_result
                if idx % 2 == 1:
                    result = -result

                pred_val = value_pred.squeeze().item()
                value_errors.append((pred_val - result) ** 2)

        move_accuracy = correct_moves / len(game_trajectory.steps)
        avg_value_error = float(np.mean(value_errors)) if value_errors else 0.0

        return {
            'game_length': game_trajectory.length,
            'final_result': game_trajectory.game_result,
            'total_reward': game_trajectory.total_reward,
            'move_accuracy': move_accuracy,
            'avg_value_error': avg_value_error,
        }

    def evaluate_games(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Evaluate multiple games"""
        results = [self.evaluate_game(traj) for traj in trajectories]

        return {
            'avg_game_length': np.mean([r['game_length'] for r in results]) if results else 0.0,
            'win_rate': np.mean([1 if r['final_result'] > 0 else 0 for r in results]) if results else 0.0,
            'draw_rate': np.mean([1 if r['final_result'] == 0 else 0 for r in results]) if results else 0.0,
            'loss_rate': np.mean([1 if r['final_result'] < 0 else 0 for r in results]) if results else 0.0,
            'avg_reward': np.mean([r['total_reward'] for r in results]) if results else 0.0,
            'avg_move_accuracy': np.mean([r['move_accuracy'] for r in results]) if results else 0.0,
            'avg_value_error': np.mean([r['avg_value_error'] for r in results]) if results else 0.0,
        }


if __name__ == "__main__":
    # Test GRPO trainer setup
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from models.large_chess_transformer import MagnusChessTransformerFactory

    print("=== GRPO Trainer Test ===")

    # Create Magnus transformer model
    model = MagnusChessTransformerFactory.create_magnus_chess()
    config = GRPOConfig(group_size=4)  # Smaller for testing
    trainer = GRPOTrainer(model, config)

    print(f"Model parameters: {MagnusChessTransformerFactory.get_model_info(model)}")
    print(f"GRPO config: group_size={config.group_size}, lr={config.learning_rate}")
    print("✅ GRPO trainer initialized successfully!")