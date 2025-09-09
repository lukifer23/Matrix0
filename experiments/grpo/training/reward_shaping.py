#!/usr/bin/env python3
"""
Advanced Reward Shaping for Chess GRPO

Chess-specific reward engineering to enhance learning from sparse rewards.
Provides domain knowledge to accelerate and stabilize training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ChessRewardComponents:
    """Components of shaped chess reward"""
    material_advantage: float = 0.0
    positional_control: float = 0.0
    king_safety: float = 0.0
    piece_activity: float = 0.0
    pawn_structure: float = 0.0
    center_control: float = 0.0
    development: float = 0.0
    tempo_advantage: float = 0.0

    def total_reward(self, weights: Dict[str, float]) -> float:
        """Compute weighted total reward"""
        total = 0.0
        total += weights.get('material_advantage', 0.1) * self.material_advantage
        total += weights.get('positional_control', 0.05) * self.positional_control
        total += weights.get('king_safety', 0.03) * self.king_safety
        total += weights.get('piece_activity', 0.02) * self.piece_activity
        total += weights.get('pawn_structure', 0.01) * self.pawn_structure
        total += weights.get('center_control', 0.02) * self.center_control
        total += weights.get('development', 0.01) * self.development
        total += weights.get('tempo_advantage', 0.01) * self.tempo_advantage
        return total


class ChessRewardShaper:
    """Advanced reward shaping for chess positions"""

    def __init__(self):
        # Material values (centipawns)
        self.material_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King has no material value in counting
        }

        # Center squares for control evaluation
        self.center_squares = [chess.D4, chess.E4, chess.D5, chess.E5,
                              chess.C3, chess.F3, chess.C6, chess.F6]

        # Extended center for positional control
        self.extended_center = [chess.C4, chess.D4, chess.E4, chess.F4,
                               chess.C5, chess.D5, chess.E5, chess.F5,
                               chess.C3, chess.F3, chess.C6, chess.F6]

    def shape_reward(self, board: chess.Board, move: Optional[chess.Move] = None,
                    game_result: float = 0.0) -> ChessRewardComponents:
        """Compute shaped reward components for a position"""
        components = ChessRewardComponents()

        # Base game result
        components.game_result = game_result

        # Material advantage
        components.material_advantage = self._calculate_material_advantage(board)

        # Positional control
        components.positional_control = self._calculate_positional_control(board)

        # King safety
        components.king_safety = self._calculate_king_safety(board)

        # Piece activity
        components.piece_activity = self._calculate_piece_activity(board, move)

        # Pawn structure
        components.pawn_structure = self._calculate_pawn_structure(board)

        # Center control
        components.center_control = self._calculate_center_control(board)

        # Development
        components.development = self._calculate_development(board)

        # Tempo advantage
        components.tempo_advantage = self._calculate_tempo_advantage(board)

        return components

    def _calculate_material_advantage(self, board: chess.Board) -> float:
        """Calculate material advantage in centipawns"""
        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.material_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        # Return advantage from white's perspective
        advantage = white_material - black_material

        # Normalize to [-1, 1] range (assuming max material ~4000 centipawns)
        return advantage / 4000.0

    def _calculate_positional_control(self, board: chess.Board) -> float:
        """Calculate positional control advantage"""
        white_control = 0
        black_control = 0

        for square in self.extended_center:
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))

            # Weight center squares more heavily
            weight = 2.0 if square in self.center_squares else 1.0

            white_control += white_attackers * weight
            black_control += black_attackers * weight

        # Return advantage normalized
        advantage = white_control - black_control
        return advantage / 20.0  # Normalize by reasonable maximum

    def _calculate_king_safety(self, board: chess.Board) -> float:
        """Calculate king safety advantage"""
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        if white_king_square is None or black_king_square is None:
            return 0.0

        # Count attackers near each king
        white_attackers = 0
        black_attackers = 0

        # Check squares around king
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue

                white_square = white_king_square + rank_offset * 8 + file_offset
                black_square = black_king_square + rank_offset * 8 + file_offset

                if 0 <= white_square < 64:
                    white_attackers += len(board.attackers(chess.BLACK, white_square))
                if 0 <= black_square < 64:
                    black_attackers += len(board.attackers(chess.WHITE, black_square))

        # Higher attackers around opponent king is better
        advantage = black_attackers - white_attackers
        return advantage / 8.0  # Normalize

    def _calculate_piece_activity(self, board: chess.Board, move: Optional[chess.Move]) -> float:
        """Calculate piece activity and mobility"""
        if board is None:
            return 0.0

        # Count legal moves for both sides
        white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0

        # Switch turn to count black moves
        board_copy = board.copy()
        board_copy.turn = chess.BLACK
        black_moves = len(list(board_copy.legal_moves))

        # Calculate piece activity based on piece mobility
        white_activity = 0
        black_activity = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Count squares this piece can move to
                piece_moves = [m for m in board.legal_moves if m.from_square == square]
                activity = len(piece_moves)
                
                if piece.color == chess.WHITE:
                    white_activity += activity
                else:
                    black_activity += activity

        # Return advantage normalized
        advantage = (white_moves + white_activity) - (black_moves + black_activity)
        return advantage / 100.0  # Normalize by reasonable maximum

    def _calculate_pawn_structure(self, board: chess.Board) -> float:
        """Calculate pawn structure advantage"""
        if board is None:
            return 0.0
            
        white_pawn_score = 0
        black_pawn_score = 0

        # Evaluate pawn chains, isolated pawns, etc.
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                row, col = divmod(square, 8)

                # Center pawns are better
                center_bonus = 1.0 if 2 <= col <= 5 else 0.5

                # Advanced pawns are better
                advancement_bonus = row / 7.0 if piece.color == chess.WHITE else (6 - row) / 7.0

                # Pawn chain bonus (pawns supporting each other)
                chain_bonus = 0
                if piece.color == chess.WHITE:
                    # Check for supporting pawns
                    for offset in [-1, 1]:
                        if 0 <= col + offset < 8 and row > 0:
                            support_square = chess.square(col + offset, row - 1)
                            support_piece = board.piece_at(support_square)
                            if support_piece and support_piece.piece_type == chess.PAWN and support_piece.color == chess.WHITE:
                                chain_bonus += 0.2
                else:
                    # Check for supporting pawns
                    for offset in [-1, 1]:
                        if 0 <= col + offset < 8 and row < 7:
                            support_square = chess.square(col + offset, row + 1)
                            support_piece = board.piece_at(support_square)
                            if support_piece and support_piece.piece_type == chess.PAWN and support_piece.color == chess.BLACK:
                                chain_bonus += 0.2

                score = center_bonus + advancement_bonus + chain_bonus

                if piece.color == chess.WHITE:
                    white_pawn_score += score
                else:
                    black_pawn_score += score

        advantage = white_pawn_score - black_pawn_score
        return advantage / 16.0  # Normalize by max pawns

    def _calculate_center_control(self, board: chess.Board) -> float:
        """Calculate center control advantage"""
        white_control = 0
        black_control = 0

        for square in self.center_squares:
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))

            white_control += white_attackers
            black_control += black_attackers

        advantage = white_control - black_control
        return advantage / 8.0  # Normalize

    def _calculate_development(self, board: chess.Board) -> float:
        """Calculate development advantage"""
        if board is None:
            return 0.0
            
        white_developed = 0
        black_developed = 0

        # Check if pieces have moved from starting squares
        # Knights
        white_knight_squares = [chess.B1, chess.G1]
        black_knight_squares = [chess.B8, chess.G8]
        
        for square in white_knight_squares:
            piece = board.piece_at(square)
            if not piece or piece.piece_type != chess.KNIGHT or piece.color != chess.WHITE:
                white_developed += 1

        for square in black_knight_squares:
            piece = board.piece_at(square)
            if not piece or piece.piece_type != chess.KNIGHT or piece.color != chess.BLACK:
                black_developed += 1

        # Bishops
        white_bishop_squares = [chess.C1, chess.F1]
        black_bishop_squares = [chess.C8, chess.F8]
        
        for square in white_bishop_squares:
            piece = board.piece_at(square)
            if not piece or piece.piece_type != chess.BISHOP or piece.color != chess.WHITE:
                white_developed += 1

        for square in black_bishop_squares:
            piece = board.piece_at(square)
            if not piece or piece.piece_type != chess.BISHOP or piece.color != chess.BLACK:
                black_developed += 1

        advantage = white_developed - black_developed
        return advantage / 8.0  # Normalize by total pieces checked

    def _calculate_tempo_advantage(self, board: chess.Board) -> float:
        """Calculate tempo advantage (side to move)"""
        return 1.0 if board.turn == chess.WHITE else -1.0


class AdaptiveRewardShaper:
    """Adaptive reward shaping that learns optimal weights"""

    def __init__(self, d_model: int = 512):
        self.reward_shaper = ChessRewardShaper()

        # Learnable weight adaptation
        self.weight_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 8),  # 8 reward components
            nn.Sigmoid()  # Output weights between 0 and 1
        )

        # Component names for indexing
        self.component_names = [
            'material_advantage', 'positional_control', 'king_safety',
            'piece_activity', 'pawn_structure', 'center_control',
            'development', 'tempo_advantage'
        ]

    def adapt_weights(self, board_embedding: torch.Tensor,
                     performance_history: List[float]) -> Dict[str, float]:
        """Adapt reward weights based on board state and performance"""
        # Use board embedding to predict optimal weights
        weights_tensor = self.weight_adapter(board_embedding.mean(dim=1))

        # Convert to dictionary
        weights = {}
        for i, name in enumerate(self.component_names):
            weights[name] = weights_tensor[0, i].item()

        # Boost weights based on recent performance
        if performance_history:
            recent_avg = np.mean(performance_history[-10:])  # Last 10 games
            if recent_avg < 0.3:  # Poor performance
                # Boost material and king safety
                weights['material_advantage'] *= 1.5
                weights['king_safety'] *= 1.5
            elif recent_avg > 0.7:  # Good performance
                # Boost positional and activity rewards
                weights['positional_control'] *= 1.5
                weights['piece_activity'] *= 1.5

        return weights

    def shape_reward_adaptive(self, board: chess.Board, move: Optional[chess.Move] = None,
                             game_result: float = 0.0, board_embedding: Optional[torch.Tensor] = None,
                             performance_history: Optional[List[float]] = None) -> float:
        """Shape reward with adaptive weights"""
        # Get base components
        components = self.reward_shaper.shape_reward(board, move, game_result)

        # Get adaptive weights
        if board_embedding is not None:
            weights = self.adapt_weights(board_embedding, performance_history or [])
        else:
            # Default weights
            weights = {name: 0.1 for name in self.component_names}

        # Compute shaped reward
        shaped_reward = components.total_reward(weights)
        shaped_reward += game_result  # Always include base game result

        return shaped_reward


def create_chess_reward_shaper() -> ChessRewardShaper:
    """Factory function for basic reward shaper"""
    return ChessRewardShaper()


def create_adaptive_reward_shaper(d_model: int = 512) -> AdaptiveRewardShaper:
    """Factory function for adaptive reward shaper"""
    return AdaptiveRewardShaper(d_model)


if __name__ == "__main__":
    # Test the reward shaper
    print("Testing Chess Reward Shaper...")

    shaper = ChessRewardShaper()
    board = chess.Board()

    # Test reward shaping
    components = shaper.shape_reward(board)
    print(f"Material advantage: {components.material_advantage:.4f}")
    print(f"Positional control: {components.positional_control:.4f}")
    print(f"King safety: {components.king_safety:.4f}")

    # Test with default weights
    weights = {'material_advantage': 0.1, 'positional_control': 0.05,
              'king_safety': 0.03, 'piece_activity': 0.02}
    total_reward = components.total_reward(weights)
    print(f"Total shaped reward: {total_reward:.4f}")

    # Test adaptive shaper
    adaptive_shaper = AdaptiveRewardShaper()
    board_embedding = torch.randn(1, 64, 512)
    adaptive_weights = adaptive_shaper.adapt_weights(board_embedding, [0.5, 0.6, 0.4])
    print(f"Adaptive weights sample: {dict(list(adaptive_weights.items())[:3])}")

    print("✅ Chess Reward Shaper test passed!")
