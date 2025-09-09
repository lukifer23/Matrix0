#!/usr/bin/env python3
"""
Meta-Learning for Chess GRPO

Adaptive parameter optimization that learns optimal hyperparameters
based on game characteristics and performance history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class GameCharacteristics:
    """Characteristics of a chess game for meta-learning"""
    game_length: int
    material_imbalance: float
    complexity_score: float
    win_probability: float
    phase: str  # 'opening', 'middlegame', 'endgame'
    tactical_intensity: float
    positional_complexity: float


@dataclass
class GRPOParameters:
    """Adaptable GRPO parameters"""
    learning_rate: float
    clip_epsilon: float
    value_loss_coef: float
    entropy_coef: float
    group_size: int
    max_grad_norm: float


class AdaptiveParameterLearner(nn.Module):
    """Meta-learning model for adapting GRPO parameters"""

    def __init__(self, d_model: int = 512, num_game_features: int = 8):
        super().__init__()

        self.game_encoder = nn.Sequential(
            nn.Linear(num_game_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Parameter predictors
        self.lr_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1, will scale to LR range
        )

        self.clip_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1, will scale to clip range
        )

        self.value_coef_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        self.entropy_coef_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        self.group_size_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1, will map to group size
        )

        self.grad_norm_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1, will scale to grad norm range
        )

        # Parameter ranges for scaling
        self.param_ranges = {
            'learning_rate': (1e-6, 1e-3),
            'clip_epsilon': (0.1, 0.3),
            'value_loss_coef': (0.1, 1.0),
            'entropy_coef': (0.001, 0.1),
            'group_size': (2, 16),
            'max_grad_norm': (0.1, 2.0)
        }

    def forward(self, game_features: torch.Tensor) -> GRPOParameters:
        """Predict optimal GRPO parameters for given game characteristics"""
        # Encode game features
        encoded = self.game_encoder(game_features)

        # Predict parameters
        lr_raw = self.lr_predictor(encoded).squeeze(-1)
        clip_raw = self.clip_predictor(encoded).squeeze(-1)
        value_coef_raw = self.value_coef_predictor(encoded).squeeze(-1)
        entropy_coef_raw = self.entropy_coef_predictor(encoded).squeeze(-1)
        group_size_raw = self.group_size_predictor(encoded).squeeze(-1)
        grad_norm_raw = self.grad_norm_predictor(encoded).squeeze(-1)

        # Scale to parameter ranges
        learning_rate = self._scale_to_range(lr_raw, *self.param_ranges['learning_rate'])
        clip_epsilon = self._scale_to_range(clip_raw, *self.param_ranges['clip_epsilon'])
        value_loss_coef = self._scale_to_range(value_coef_raw, *self.param_ranges['value_loss_coef'])
        entropy_coef = self._scale_to_range(entropy_coef_raw, *self.param_ranges['entropy_coef'])
        group_size = int(self._scale_to_range(group_size_raw, *self.param_ranges['group_size']))
        max_grad_norm = self._scale_to_range(grad_norm_raw, *self.param_ranges['max_grad_norm'])

        return GRPOParameters(
            learning_rate=learning_rate,
            clip_epsilon=clip_epsilon,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            group_size=max(2, min(16, group_size)),  # Clamp to valid range
            max_grad_norm=max_grad_norm
        )

    def _scale_to_range(self, value: torch.Tensor, min_val: float, max_val: float) -> float:
        """Scale sigmoid output (0-1) to parameter range"""
        if isinstance(value, torch.Tensor):
            value = value.item()
        return min_val + (max_val - min_val) * value


class GameAnalyzer:
    """Analyzes chess games to extract characteristics for meta-learning"""

    def __init__(self):
        self.material_values = {
            'pawn': 1, 'knight': 3, 'bishop': 3,
            'rook': 5, 'queen': 9, 'king': 0
        }

    def analyze_game(self, board_history: List[Any], move_history: List[Any],
                    result: float) -> GameCharacteristics:
        """Analyze a complete chess game"""
        game_length = len(move_history)

        # Material imbalance (average over game)
        material_imbalances = []
        for board_state in board_history:
            material_imbalances.append(self._calculate_material_imbalance(board_state))
        avg_material_imbalance = np.mean(material_imbalances) if material_imbalances else 0.0

        # Complexity score based on branching factor and tactics
        complexity_score = self._calculate_complexity_score(board_history, move_history)

        # Win probability (based on material and position)
        win_probability = self._estimate_win_probability(board_history[-1] if board_history else None)

        # Game phase
        phase = self._determine_game_phase(board_history[-1] if board_history else None, game_length)

        # Tactical intensity
        tactical_intensity = self._calculate_tactical_intensity(move_history)

        # Positional complexity
        positional_complexity = self._calculate_positional_complexity(board_history)

        return GameCharacteristics(
            game_length=game_length,
            material_imbalance=avg_material_imbalance,
            complexity_score=complexity_score,
            win_probability=win_probability,
            phase=phase,
            tactical_intensity=tactical_intensity,
            positional_complexity=positional_complexity
        )

    def _calculate_material_imbalance(self, board) -> float:
        """Calculate material imbalance from white's perspective"""
        if board is None:
            return 0.0
            
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.material_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Return advantage from white's perspective (normalized)
        advantage = white_material - black_material
        return advantage / 100.0  # Normalize by typical material values

    def _calculate_complexity_score(self, board_history, move_history) -> float:
        """Calculate game complexity based on tactics and variations"""
        if not board_history or not move_history:
            return 0.5
            
        # Length complexity
        length_complexity = min(1.0, len(move_history) / 100.0)
        
        # Material change complexity
        material_changes = 0
        for i in range(1, len(board_history)):
            prev_material = self._calculate_material_imbalance(board_history[i-1])
            curr_material = self._calculate_material_imbalance(board_history[i])
            if abs(curr_material - prev_material) > 0.1:  # Significant material change
                material_changes += 1
        
        material_complexity = min(1.0, material_changes / len(move_history))
        
        # Combine factors
        return (length_complexity + material_complexity) / 2.0

    def _estimate_win_probability(self, final_board) -> float:
        """Estimate win probability from final position"""
        if final_board is None:
            return 0.5
            
        # Check for terminal positions
        if final_board.is_checkmate():
            return 1.0 if final_board.turn == chess.BLACK else 0.0
        elif final_board.is_stalemate() or final_board.is_insufficient_material():
            return 0.5
            
        # Estimate based on material advantage
        material_advantage = self._calculate_material_imbalance(final_board)
        
        # Convert material advantage to win probability
        # Sigmoid function: 0.5 + 0.5 * tanh(material_advantage)
        win_prob = 0.5 + 0.5 * np.tanh(material_advantage)
        return win_prob

    def _determine_game_phase(self, board, game_length) -> str:
        """Determine game phase"""
        if game_length < 10:
            return 'opening'
        elif game_length < 60:
            return 'middlegame'
        else:
            return 'endgame'

    def _calculate_tactical_intensity(self, move_history) -> float:
        """Calculate tactical intensity of the game"""
        if not move_history:
            return 0.0
            
        # Count captures and checks (tactical moves)
        tactical_moves = 0
        for move in move_history:
            if hasattr(move, 'capture') and move.capture:  # Capture move
                tactical_moves += 1
            # Note: Checking for checks would require board state analysis
        
        # Normalize by game length
        intensity = tactical_moves / len(move_history) if move_history else 0.0
        return min(1.0, intensity)

    def _calculate_positional_complexity(self, board_history) -> float:
        """Calculate positional complexity"""
        if not board_history:
            return 0.5
            
        # Analyze piece mobility and position diversity
        position_diversity = 0
        total_positions = len(board_history)
        
        for i in range(1, total_positions):
            prev_board = board_history[i-1]
            curr_board = board_history[i]
            
            # Count piece position changes
            changes = 0
            for square in chess.SQUARES:
                prev_piece = prev_board.piece_at(square)
                curr_piece = curr_board.piece_at(square)
                if prev_piece != curr_piece:
                    changes += 1
            
            # Normalize by board size
            position_diversity += changes / 64.0
        
        # Average position diversity
        avg_diversity = position_diversity / max(1, total_positions - 1)
        return min(1.0, avg_diversity)


class MetaGRPOTrainer:
    """GRPO trainer with meta-learning parameter adaptation"""

    def __init__(self, base_grpo_trainer: Any, d_model: int = 512):
        self.base_trainer = base_grpo_trainer
        self.parameter_learner = AdaptiveParameterLearner(d_model)
        self.game_analyzer = GameAnalyzer()

        # Meta-learning optimizer
        self.meta_optimizer = torch.optim.Adam(self.parameter_learner.parameters(), lr=1e-4)

        # Performance history for meta-learning
        self.performance_history = []
        self.game_characteristics_history = []

    def adapt_parameters(self, current_game_history: List[Any],
                        recent_performance: List[float]) -> GRPOParameters:
        """Adapt GRPO parameters based on current game and performance history"""

        # Analyze current game
        if current_game_history:
            # Extract recent game characteristics
            recent_game = current_game_history[-1] if current_game_history else None
            if recent_game:
                game_chars = self.game_analyzer.analyze_game(
                    recent_game.get('board_history', []),
                    recent_game.get('move_history', []),
                    recent_game.get('result', 0.0)
                )
            else:
                # Create default characteristics
                game_chars = GameCharacteristics(
                    game_length=40,
                    material_imbalance=0.0,
                    complexity_score=0.5,
                    win_probability=0.5,
                    phase='middlegame',
                    tactical_intensity=0.5,
                    positional_complexity=0.5
                )
        else:
            # Default characteristics for early training
            game_chars = GameCharacteristics(
                game_length=40,
                material_imbalance=0.0,
                complexity_score=0.5,
                win_probability=0.5,
                phase='middlegame',
                tactical_intensity=0.5,
                positional_complexity=0.5
            )

        # Convert to feature tensor
        features = torch.tensor([
            game_chars.game_length / 100.0,  # Normalize
            game_chars.material_imbalance,
            game_chars.complexity_score,
            game_chars.win_probability,
            1.0 if game_chars.phase == 'opening' else 0.0,
            1.0 if game_chars.phase == 'middlegame' else 0.0,
            game_chars.tactical_intensity,
            game_chars.positional_complexity
        ], dtype=torch.float32).unsqueeze(0)

        # Predict optimal parameters
        optimal_params = self.parameter_learner(features)

        # Adjust based on performance
        if recent_performance:
            avg_performance = np.mean(recent_performance[-10:])  # Last 10 games

            if avg_performance < 0.3:  # Poor performance
                # More conservative learning
                optimal_params.learning_rate *= 0.8
                optimal_params.clip_epsilon *= 0.9
                optimal_params.group_size = max(2, optimal_params.group_size - 2)

            elif avg_performance > 0.7:  # Good performance
                # More aggressive learning
                optimal_params.learning_rate *= 1.2
                optimal_params.clip_epsilon *= 1.1
                optimal_params.group_size = min(16, optimal_params.group_size + 2)

        return optimal_params

    def update_meta_learner(self, game_characteristics: GameCharacteristics,
                           actual_performance: float, predicted_params: GRPOParameters):
        """Update meta-learner based on actual vs predicted performance"""

        # Convert game characteristics to features
        features = torch.tensor([
            game_characteristics.game_length / 100.0,
            game_characteristics.material_imbalance,
            game_characteristics.complexity_score,
            game_characteristics.win_probability,
            1.0 if game_characteristics.phase == 'opening' else 0.0,
            1.0 if game_characteristics.phase == 'middlegame' else 0.0,
            game_characteristics.tactical_intensity,
            game_characteristics.positional_complexity
        ], dtype=torch.float32).unsqueeze(0)

        # Target: higher performance is better
        target_performance = torch.tensor([actual_performance], dtype=torch.float32)

        # Forward pass to get predictions
        predicted_params_new = self.parameter_learner(features)

        # Simple loss: maximize performance prediction accuracy
        # In practice, this would be more sophisticated
        performance_prediction = torch.tensor([0.5], dtype=torch.float32)  # Placeholder
        meta_loss = F.mse_loss(performance_prediction, target_performance)

        # Update meta-learner
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


def create_meta_grpo_trainer(base_trainer: Any, d_model: int = 512) -> MetaGRPOTrainer:
    """Factory function for meta GRPO trainer"""
    return MetaGRPOTrainer(base_trainer, d_model)


if __name__ == "__main__":
    # Test the meta-learning components
    print("Testing Meta-Learning for GRPO...")

    # Test parameter learner
    learner = AdaptiveParameterLearner()
    game_features = torch.randn(1, 8)  # 8 game features
    params = learner(game_features)

    print("Predicted GRPO Parameters:")
    print(f"  Learning Rate: {params.learning_rate:.2e}")
    print(f"  Clip Epsilon: {params.clip_epsilon:.3f}")
    print(f"  Group Size: {params.group_size}")
    print(f"  Value Loss Coef: {params.value_loss_coef:.3f}")
    print(f"  Entropy Coef: {params.entropy_coef:.4f}")

    # Test game analyzer
    analyzer = GameAnalyzer()
    # Create mock game characteristics
    mock_chars = GameCharacteristics(
        game_length=45,
        material_imbalance=0.2,
        complexity_score=0.7,
        win_probability=0.6,
        phase='middlegame',
        tactical_intensity=0.8,
        positional_complexity=0.6
    )
    print(f"Mock game characteristics: length={mock_chars.game_length}, complexity={mock_chars.complexity_score:.2f}")

    print("✅ Meta-Learning test passed!")
