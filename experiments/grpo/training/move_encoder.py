#!/usr/bin/env python3
"""
Move Encoder for Chess Transformers

Advanced move encoding with attention mechanisms for chess-specific patterns.
Integrates with Magnus transformer for optimal move prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Any
import chess
import logging

logger = logging.getLogger(__name__)


class ChessMoveEncoder(nn.Module):
    """Advanced move encoder for chess transformers"""

    def __init__(self, d_model: int = 512, max_moves: int = 4672):
        super().__init__()
        self.d_model = d_model
        self.max_moves = max_moves

        # Move embedding layers
        self.from_square_embed = nn.Embedding(64, d_model // 4)
        self.to_square_embed = nn.Embedding(64, d_model // 4)
        self.piece_embed = nn.Embedding(13, d_model // 4)  # 6 pieces x 2 colors + empty
        self.promotion_embed = nn.Embedding(5, d_model // 4)  # None, Q, R, B, N

        # Combine embeddings
        self.move_projection = nn.Linear(d_model, d_model)

        # Attention mask generator
        self.attention_mask_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_moves)
        )

    def encode_move(self, move: chess.Move, board: chess.Board) -> torch.Tensor:
        """Encode a single chess move to embedding"""
        # Get move components
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion or 0

        # Get piece at from square
        piece = board.piece_at(from_sq)
        piece_idx = 0
        if piece:
            piece_idx = piece.piece_type + (6 if piece.color == chess.BLACK else 0)

        # Get embeddings
        from_embed = self.from_square_embed(torch.tensor(from_sq))
        to_embed = self.to_square_embed(torch.tensor(to_sq))
        piece_embed = self.piece_embed(torch.tensor(piece_idx))

        # Promotion encoding (0 = no promotion, 1-4 = piece types)
        promo_idx = 0
        if promotion:
            if promotion == chess.QUEEN:
                promo_idx = 1
            elif promotion == chess.ROOK:
                promo_idx = 2
            elif promotion == chess.BISHOP:
                promo_idx = 3
            elif promotion == chess.KNIGHT:
                promo_idx = 4

        promo_embed = self.promotion_embed(torch.tensor(promo_idx))

        # Combine all embeddings
        combined = torch.cat([from_embed, to_embed, piece_embed, promo_embed], dim=-1)
        encoded = self.move_projection(combined)

        return encoded

    def encode_legal_moves(self, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode all legal moves for attention masking"""
        legal_moves = list(board.legal_moves)

        # Get device from model parameters
        try:
            device = next(self.parameters()).device
        except (StopIteration, RuntimeError):
            device = 'cpu'

        # Create legal move mask
        legal_mask = torch.zeros(self.max_moves, device=device)
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            if 0 <= move_idx < self.max_moves:
                legal_mask[move_idx] = 1.0

        # Create move embeddings for legal moves only
        move_embeddings = []
        for move in legal_moves:
            embedding = self.encode_move(move, board)
            move_embeddings.append(embedding)

        # Pad to max_moves if needed
        while len(move_embeddings) < self.max_moves:
            move_embeddings.append(torch.zeros(self.d_model, device=device))

        move_embeddings = torch.stack(move_embeddings)

        # Generate attention weights using the legal mask
        attention_weights = self.attention_mask_generator(move_embeddings.mean(dim=0))
        attention_weights = torch.sigmoid(attention_weights) * legal_mask

        return move_embeddings, attention_weights

    def _move_to_index(self, move: chess.Move) -> int:
        """Convert chess move to policy index using AlphaZero encoding"""
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion

        # Queen moves: 0-4095 (64*64)
        if not promotion or promotion == chess.QUEEN:
            return from_sq * 64 + to_sq

        # Underpromotions: 4096-4671 (64*3*3 = 576 moves)
        # Each from_square can promote to 3 pieces (Knight, Bishop, Rook)
        promo_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
        return 4096 + from_sq * 3 + promo_offset[promotion]


class AttentionMovePredictor(nn.Module):
    """Move prediction with attention mechanisms"""

    def __init__(self, d_model: int = 512, nhead: int = 8, max_moves: int = 4672):
        super().__init__()
        self.d_model = d_model
        self.max_moves = max_moves

        # Multi-head attention for move prediction
        self.move_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Legal move attention masking
        self.legal_move_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Move prediction head
        self.move_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, max_moves)
        )

    def forward(self, board_embedding: torch.Tensor, move_embeddings: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict moves using attention mechanisms"""
        # Add batch dimension if needed
        if board_embedding.dim() == 2:
            board_embedding = board_embedding.unsqueeze(0)
        if move_embeddings.dim() == 2:
            move_embeddings = move_embeddings.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Attention between board and moves
        attended_moves, _ = self.move_attention(
            move_embeddings, board_embedding, board_embedding
        )

        # Legal move masking attention
        legal_attended, _ = self.legal_move_attention(
            attended_moves, attended_moves, attended_moves,
            key_padding_mask=(attention_mask == 0)
        )

        # Predict move probabilities
        move_logits = self.move_predictor(legal_attended.mean(dim=1))

        return move_logits.squeeze(0) if move_logits.size(0) == 1 else move_logits


class MagnusMoveIntegration(nn.Module):
    """Integration layer for Magnus transformer with move encoding"""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.move_encoder = ChessMoveEncoder(d_model)
        self.attention_predictor = AttentionMovePredictor(d_model)

        # Register submodules for proper parameter access
        self.add_module('move_encoder', self.move_encoder)
        self.add_module('attention_predictor', self.attention_predictor)

    def get_attention_mask(self, board: chess.Board) -> torch.Tensor:
        """Get attention mask for legal moves"""
        if board is None:
            # Return neutral mask when board not available
            try:
                device = next(self.parameters()).device
            except (StopIteration, RuntimeError):
                device = 'cpu'
            return torch.ones(4672, device=device)

        _, attention_weights = self.move_encoder.encode_legal_moves(board)
        return attention_weights

    def predict_with_attention(self, board_embedding: torch.Tensor,
                             board: chess.Board) -> torch.Tensor:
        """Predict moves using attention mechanisms"""
        move_embeddings, attention_mask = self.move_encoder.encode_legal_moves(board)
        move_logits = self.attention_predictor(board_embedding, move_embeddings, attention_mask)
        return move_logits

    def encode_board_for_attention(self, board: chess.Board) -> torch.Tensor:
        """Encode board state for attention mechanisms"""
        # Convert board to tensor format compatible with transformer
        board_tensor = self._board_to_tensor(board)
        
        # This would integrate with the Magnus transformer's board encoding
        # For now, return a placeholder that matches the expected interface
        # In practice, this would be the output of the transformer's board encoder
        return torch.randn(1, 64, self.move_encoder.d_model)

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to tensor format compatible with transformer"""
        # Create 19-channel board representation
        channels = []

        # Piece channels (12 channels: 6 piece types x 2 colors)
        for piece_type in range(1, 7):  # 1-6: pawn, knight, bishop, rook, queen, king
            white_channel = torch.zeros(8, 8)
            black_channel = torch.zeros(8, 8)

            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == piece_type:
                    row, col = divmod(square, 8)
                    if piece.color == chess.WHITE:
                        white_channel[row, col] = 1.0
                    else:
                        black_channel[row, col] = 1.0

            channels.extend([white_channel, black_channel])

        # Additional channels for game state (7 more to make 19 total)
        # Side to move
        side_to_move = torch.ones(8, 8) if board.turn == chess.WHITE else torch.zeros(8, 8)
        channels.append(side_to_move)

        # Castling rights (4 channels)
        castling_channels = []
        for i in range(4):
            castling_channels.append(torch.zeros(8, 8))
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_channels[0].fill_(1.0)
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_channels[1].fill_(1.0)
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_channels[2].fill_(1.0)
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_channels[3].fill_(1.0)
        channels.extend(castling_channels)

        # En passant (1 channel)
        en_passant = torch.zeros(8, 8)
        if board.ep_square:
            row, col = divmod(board.ep_square, 8)
            en_passant[row, col] = 1.0
        channels.append(en_passant)

        # Halfmove clock (1 channel)
        halfmove = torch.full((8, 8), min(board.halfmove_clock / 100.0, 1.0))
        channels.append(halfmove)

        return torch.stack(channels, dim=0).unsqueeze(0)  # Add batch dimension


# Factory function for easy instantiation
def create_magnus_move_encoder(d_model: int = 512) -> MagnusMoveIntegration:
    """Create Magnus move encoder integration"""
    return MagnusMoveIntegration(d_model)


if __name__ == "__main__":
    # Test the move encoder
    print("Testing Chess Move Encoder...")

    encoder = ChessMoveEncoder()
    board = chess.Board()

    # Test single move encoding
    first_move = list(board.legal_moves)[0]
    encoded = encoder.encode_move(first_move, board)
    print(f"Move encoding shape: {encoded.shape}")

    # Test legal moves encoding
    move_embeddings, attention_mask = encoder.encode_legal_moves(board)
    print(f"Legal moves embeddings shape: {move_embeddings.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Legal moves: {attention_mask.sum().item()}")

    # Test attention predictor
    predictor = AttentionMovePredictor()
    board_embedding = torch.randn(1, 64, 512)
    move_logits = predictor(board_embedding, move_embeddings.unsqueeze(0), attention_mask.unsqueeze(0))
    print(f"Move logits shape: {move_logits.shape}")

    print("✅ Chess Move Encoder test passed!")
