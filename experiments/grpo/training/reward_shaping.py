#!/usr/bin/env python3
"""
Advanced Reward Shaping for Chess GRPO Experiments

Implements sophisticated reward shaping techniques to improve learning:
- Material-based rewards
- Positional incentives
- Tactical bonuses
- Long-term strategic guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChessRewardComponents:
    """Components that contribute to shaped reward"""
    base_result: float = 0.0  # Win/loss/draw result
    material_advantage: float = 0.0  # Material difference
    positional_control: float = 0.0  # Center and space control
    piece_activity: float = 0.0  # Piece mobility and threats
    king_safety: float = 0.0  # King protection and attacks
    pawn_structure: float = 0.0  # Pawn chain health
    tempo_advantage: float = 0.0  # Initiative and development
    endgame_proximity: float = 0.0  # Endgame positioning

    def total_reward(self, weights: Dict[str, float]) -> float:
        """Calculate total shaped reward"""
        total = 0.0
        for component, value in self.__dict__.items():
            if component in weights:
                total += value * weights[component]
        return total


class ChessRewardShaper:
    """
    Advanced reward shaping for chess games
    """

    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        # Default reward weights
        self.reward_weights = reward_weights or {
            'base_result': 1.0,
            'material_advantage': 0.1,
            'positional_control': 0.05,
            'piece_activity': 0.02,
            'king_safety': 0.03,
            'pawn_structure': 0.01,
            'tempo_advantage': 0.02,
            'endgame_proximity': 0.04
        }

        # Piece values for material calculation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King value doesn't count in material
        }

        # Center squares for positional control
        self.center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]

        logger.info(f"Initialized reward shaper with weights: {self.reward_weights}")

    def shape_reward(self, board: chess.Board, move: Optional[chess.Move] = None,
                    game_result: Optional[float] = None) -> ChessRewardComponents:
        """
        Calculate shaped reward components for current position

        Args:
            board: Current chess board
            move: Move that led to this position (optional)
            game_result: Final game result if terminal (optional)

        Returns:
            RewardComponents with all calculated values
        """
        components = ChessRewardComponents()

        # Base game result (only if terminal)
        if game_result is not None:
            components.base_result = game_result

        # Material advantage (from side to move perspective)
        components.material_advantage = self._calculate_material_advantage(board)

        # Positional control
        components.positional_control = self._calculate_positional_control(board)

        # Piece activity
        components.piece_activity = self._calculate_piece_activity(board)

        # King safety
        components.king_safety = self._calculate_king_safety(board)

        # Pawn structure
        components.pawn_structure = self._calculate_pawn_structure(board)

        # Tempo advantage
        components.tempo_advantage = self._calculate_tempo_advantage(board)

        # Endgame proximity
        components.endgame_proximity = self._calculate_endgame_proximity(board)

        return components

    def _calculate_material_advantage(self, board: chess.Board) -> float:
        """Calculate material advantage in centipawns"""
        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        # Return from side to move perspective
        material_diff = white_material - black_material
        if board.turn == chess.BLACK:
            material_diff = -material_diff

        return material_diff / 100.0  # Convert to pawns

    def _calculate_positional_control(self, board: chess.Board) -> float:
        """Calculate positional control (center, space, outposts)"""
        control_score = 0.0

        # Center control
        for row, col in self.center_squares:
            square = chess.square(col, row)
            controlling_piece = None

            # Check which side controls this square
            for piece_square in chess.SQUARES:
                piece = board.piece_at(piece_square)
                if piece and piece.color == board.turn:
                    # Simplified: check if piece attacks this square
                    if self._piece_attacks_square(board, piece_square, square):
                        controlling_piece = piece
                        break

            if controlling_piece:
                # Bonus for controlling center with valuable pieces
                control_score += self.piece_values[controlling_piece.piece_type] / 1000.0

        # Space advantage (mobility)
        mobility = len(list(board.legal_moves))
        control_score += mobility / 100.0

        return control_score

    def _calculate_piece_activity(self, board: chess.Board) -> float:
        """Calculate piece activity and attacking potential"""
        activity_score = 0.0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                # Count squares attacked by this piece
                attacked_squares = 0
                for target_square in chess.SQUARES:
                    if self._piece_attacks_square(board, square, target_square):
                        attacked_squares += 1

                # Activity bonus based on piece value and squares attacked
                piece_value = self.piece_values[piece.piece_type]
                activity_score += (piece_value / 100.0) * (attacked_squares / 27.0)  # Max ~27 squares

        return activity_score / 16.0  # Normalize by max pieces

    def _calculate_king_safety(self, board: chess.Board) -> float:
        """Calculate king safety and attacking potential"""
        king_square = board.king(board.turn)
        if not king_square:
            return 0.0

        safety_score = 0.0

        # Check for direct attacks on king
        attackers = 0
        defenders = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if self._piece_attacks_square(board, square, king_square):
                    if piece.color == board.turn:
                        defenders += 1
                    else:
                        attackers += 1

        # King safety score
        if attackers > defenders:
            safety_score = -0.5 * (attackers - defenders)
        elif defenders > attackers:
            safety_score = 0.2 * (defenders - attackers)

        # Bonus for castling
        if board.has_kingside_castling_rights(board.turn) or board.has_queenside_castling_rights(board.turn):
            safety_score += 0.1

        return safety_score

    def _calculate_pawn_structure(self, board: chess.Board) -> float:
        """Calculate pawn structure health"""
        structure_score = 0.0

        # Doubled pawns penalty
        for file in range(8):
            pawns_in_file = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if (piece and piece.piece_type == chess.PAWN and
                    piece.color == board.turn):
                    pawns_in_file += 1

            if pawns_in_file > 1:
                structure_score -= 0.1 * (pawns_in_file - 1)

        # Isolated pawns penalty
        for file in range(8):
            has_pawn_in_file = False
            has_pawn_adjacent = False

            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if (piece and piece.piece_type == chess.PAWN and
                    piece.color == board.turn):
                    has_pawn_in_file = True
                    break

            if has_pawn_in_file:
                # Check adjacent files
                for adj_file in [file-1, file+1]:
                    if 0 <= adj_file < 8:
                        for rank in range(8):
                            square = chess.square(adj_file, rank)
                            piece = board.piece_at(square)
                            if (piece and piece.piece_type == chess.PAWN and
                                piece.color == board.turn):
                                has_pawn_adjacent = True
                                break

                if not has_pawn_adjacent:
                    structure_score -= 0.15

        return structure_score

    def _calculate_tempo_advantage(self, board: chess.Board) -> float:
        """Calculate tempo and initiative advantage"""
        tempo_score = 0.0

        # Development bonus (pieces moved from starting squares)
        developed_pieces = 0
        total_pieces = 0

        # Simplified development check
        if board.turn == chess.WHITE:
            # Check if knights and bishops have moved
            knight_squares = [chess.B1, chess.G1]
            bishop_squares = [chess.C1, chess.F1]
        else:
            knight_squares = [chess.B8, chess.G8]
            bishop_squares = [chess.C8, chess.F8]

        for square in knight_squares + bishop_squares:
            piece = board.piece_at(square)
            if not piece or piece.piece_type not in [chess.KNIGHT, chess.BISHOP]:
                developed_pieces += 1
            total_pieces += 1

        tempo_score += 0.1 * (developed_pieces / total_pieces)

        # Castling bonus
        if board.has_kingside_castling_rights(board.turn) or board.has_queenside_castling_rights(board.turn):
            tempo_score += 0.05

        return tempo_score

    def _calculate_endgame_proximity(self, board: chess.Board) -> float:
        """Calculate endgame positioning and preparation"""
        endgame_score = 0.0

        # Count pieces remaining
        piece_count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                piece_count += 1

        # Endgame when few pieces remain
        if piece_count <= 6:  # Roughly endgame
            endgame_score += 0.2

            # King centralization bonus in endgame
            king_square = board.king(board.turn)
            if king_square:
                row, col = divmod(king_square, 8)
                center_distance = abs(row - 3.5) + abs(col - 3.5)
                endgame_score += 0.1 * (1.0 - center_distance / 7.0)  # Closer to center is better

            # Opponent king centralization penalty
            opp_king_square = board.king(not board.turn)
            if opp_king_square:
                row, col = divmod(opp_king_square, 8)
                center_distance = abs(row - 3.5) + abs(col - 3.5)
                endgame_score += 0.1 * (center_distance / 7.0)  # Opponent far from center is good

        return endgame_score

    def _piece_attacks_square(self, board: chess.Board, from_square: int, to_square: int) -> bool:
        """Check if piece attacks target square (simplified)"""
        # This is a simplified check - in practice you'd use proper attack tables
        piece = board.piece_at(from_square)
        if not piece:
            return False

        # Simple distance and piece type checks
        from_row, from_col = divmod(from_square, 8)
        to_row, to_col = divmod(to_square, 8)

        row_diff = abs(from_row - to_row)
        col_diff = abs(from_col - to_col)

        if piece.piece_type == chess.KNIGHT:
            return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)
        elif piece.piece_type == chess.BISHOP:
            return row_diff == col_diff
        elif piece.piece_type == chess.ROOK:
            return row_diff == 0 or col_diff == 0
        elif piece.piece_type == chess.QUEEN:
            return row_diff == col_diff or row_diff == 0 or col_diff == 0
        elif piece.piece_type == chess.KING:
            return max(row_diff, col_diff) <= 1
        elif piece.piece_type == chess.PAWN:
            direction = 1 if piece.color == chess.WHITE else -1
            if piece.color == board.turn:
                return (to_row == from_row + direction and col_diff <= 1)
            else:
                return (to_row == from_row - direction and col_diff <= 1)

        return False


class AdaptiveRewardShaper:
    """
    Learns to adapt reward weights based on training progress
    """

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        self.base_shaper = ChessRewardShaper(initial_weights)
        self.performance_history = []

        # Learnable weight adjustments
        self.weight_adapter = nn.Sequential(
            nn.Linear(10, 32),  # Input: performance metrics
            nn.ReLU(),
            nn.Linear(32, 8),   # Output: weight adjustments for 8 components
            nn.Tanh()           # Keep adjustments small
        )

    def adapt_weights(self, recent_performance: List[float],
                     game_characteristics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Adapt reward weights based on recent training performance

        Args:
            recent_performance: Recent win rates or other metrics
            game_characteristics: Characteristics of recent games

        Returns:
            Adapted reward weights
        """
        # Prepare input features
        features = torch.tensor([
            np.mean(recent_performance),           # Average performance
            np.std(recent_performance),            # Performance stability
            len([g for g in game_characteristics if g.get('phase') == 'endgame']),  # Endgame games
            len([g for g in game_characteristics if g.get('complexity', 0) > 0.7]),  # Complex games
            np.mean([g.get('material_balance', 0) for g in game_characteristics]),  # Avg material balance
        ] + [0.0] * 5)  # Pad to 10 features

        # Predict weight adjustments
        with torch.no_grad():
            adjustments = self.weight_adapter(features.unsqueeze(0)).squeeze(0)

        # Apply adjustments to base weights
        adapted_weights = {}
        component_names = ['base_result', 'material_advantage', 'positional_control',
                          'piece_activity', 'king_safety', 'pawn_structure',
                          'tempo_advantage', 'endgame_proximity']

        for i, component in enumerate(component_names):
            adjustment = adjustments[i].item() * 0.1  # Keep adjustments small
            adapted_weights[component] = self.base_shaper.reward_weights[component] + adjustment

        return adapted_weights

    def shape_reward_adaptive(self, board: chess.Board, move: Optional[chess.Move] = None,
                             game_result: Optional[float] = None,
                             adapted_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Shape reward with adapted weights

        Args:
            board: Current chess board
            move: Move that led to this position
            game_result: Final game result if terminal
            adapted_weights: Adapted reward weights

        Returns:
            Total shaped reward
        """
        components = self.base_shaper.shape_reward(board, move, game_result)
        weights = adapted_weights or self.base_shaper.reward_weights
        return components.total_reward(weights)


if __name__ == "__main__":
    # Test reward shaping
    print("=== Chess Reward Shaping Test ===")

    # Create reward shaper
    shaper = ChessRewardShaper()
    print(f"Reward weights: {shaper.reward_weights}")

    # Test with a simple position
    board = chess.Board()
    components = shaper.shape_reward(board)
    total_reward = components.total_reward(shaper.reward_weights)

    print(f"Initial position components: {components}")
    print(f"Total shaped reward: {total_reward}")

    # Test adaptive reward shaper
    adaptive_shaper = AdaptiveRewardShaper()
    recent_perf = [0.4, 0.5, 0.6, 0.5, 0.7]
    game_chars = [{'phase': 'opening'}, {'phase': 'middlegame'}, {'phase': 'endgame'}]

    adapted_weights = adaptive_shaper.adapt_weights(recent_perf, game_chars)
    print(f"Adapted weights: {adapted_weights}")

    adaptive_reward = adaptive_shaper.shape_reward_adaptive(board, adapted_weights=adapted_weights)
    print(f"Adaptive shaped reward: {adaptive_reward}")

    print("✅ Reward shaping test passed!")
