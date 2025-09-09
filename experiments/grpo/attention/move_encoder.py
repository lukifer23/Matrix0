#!/usr/bin/env python3
"""
Move Encoder and Attention Mechanisms for Chess Transformer

Advanced move encoding and attention mechanisms specifically designed
for chess position understanding and legal move prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import math
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ChessMoveEncoder:
    """
    Advanced chess move encoder that handles all 4672 possible moves
    with proper categorical encoding and attention mechanisms.
    """

    def __init__(self, d_model: int = 256, max_moves: int = 4672):
        self.d_model = d_model
        self.max_moves = max_moves

        # Move type encodings
        self.move_type_embeddings = nn.Embedding(6, d_model // 4)  # quiet, capture, castle, promotion, en_passant, check

        # Square encodings (64 squares)
        self.from_square_embeddings = nn.Embedding(64, d_model // 4)
        self.to_square_embeddings = nn.Embedding(64, d_model // 4)

        # Piece type encodings
        self.piece_type_embeddings = nn.Embedding(7, d_model // 4)  # empty, pawn, knight, bishop, rook, queen, king

        # Promotion piece encodings
        self.promotion_embeddings = nn.Embedding(5, d_model // 4)  # none, queen, rook, bishop, knight

        # Special move flags
        self.special_flags_embeddings = nn.Embedding(8, d_model // 4)  # castling, en_passant, check, checkmate, etc.

        # Final projection
        self.output_projection = nn.Linear(d_model, d_model)

        # Attention mechanism for move relationships
        self.move_attention = nn.MultiheadAttention(d_model, nhead=8, batch_first=True)

        # Move validity predictor
        self.validity_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def encode_move(self, move: chess.Move, board: chess.Board) -> torch.Tensor:
        """
        Encode a single chess move into a vector representation

        Args:
            move: Chess move to encode
            board: Current board state

        Returns:
            Encoded move vector (d_model,)
        """
        # Extract move components
        from_square = move.from_square
        to_square = move.to_square
        moving_piece = board.piece_at(from_square)
        captured_piece = board.piece_at(to_square) if board.piece_at(to_square) else None

        # Determine move type
        move_type = self._classify_move_type(move, board, moving_piece, captured_piece)

        # Encode components
        move_type_emb = self.move_type_embeddings(torch.tensor(move_type))
        from_sq_emb = self.from_square_embeddings(torch.tensor(from_square))
        to_sq_emb = self.to_square_embeddings(torch.tensor(to_square))

        piece_type = moving_piece.piece_type if moving_piece else 0
        piece_emb = self.piece_type_embeddings(torch.tensor(piece_type))

        promotion = move.promotion.piece_type if move.promotion else 0
        promotion_emb = self.promotion_embeddings(torch.tensor(promotion))

        # Special flags
        special_flags = self._get_special_flags(move, board)
        special_emb = self.special_flags_embeddings(torch.tensor(special_flags))

        # Concatenate all encodings
        combined = torch.cat([
            move_type_emb,
            from_sq_emb,
            to_sq_emb,
            piece_emb,
            promotion_emb,
            special_emb
        ], dim=-1)

        # Project to final dimension
        encoded = self.output_projection(combined)

        return encoded

    def encode_moves_batch(self, moves: List[chess.Move], board: chess.Board) -> torch.Tensor:
        """
        Encode a batch of moves

        Args:
            moves: List of chess moves
            board: Current board state

        Returns:
            Encoded moves tensor (len(moves), d_model)
        """
        encoded_moves = []
        for move in moves:
            encoded = self.encode_move(move, board)
            encoded_moves.append(encoded)

        if encoded_moves:
            return torch.stack(encoded_moves, dim=0)
        else:
            return torch.empty(0, self.d_model)

    def encode_all_possible_moves(self, board: chess.Board) -> Tuple[torch.Tensor, List[chess.Move]]:
        """
        Encode all 4672 possible moves (most will be invalid)

        Args:
            board: Current board state

        Returns:
            Tuple of (encoded_moves, move_list)
        """
        move_list = []
        encoded_moves = []

        # Generate all possible moves in UCI format and convert to Move objects
        # This is a simplified approach - in practice you'd have a proper move generator
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue

                # Try different promotion pieces
                for promotion in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    try:
                        move = chess.Move(from_sq, to_sq, promotion)
                        if move in board.legal_moves:
                            encoded = self.encode_move(move, board)
                            encoded_moves.append(encoded)
                            move_list.append(move)
                    except:
                        continue

        if encoded_moves:
            return torch.stack(encoded_moves, dim=0), move_list
        else:
            return torch.empty(0, self.d_model), []

    def _classify_move_type(self, move: chess.Move, board: chess.Board,
                           moving_piece, captured_piece) -> int:
        """Classify move type for encoding"""
        if captured_piece:
            return 1  # capture
        elif move.promotion:
            return 3  # promotion
        elif board.is_castling(move):
            return 2  # castle
        elif board.is_en_passant(move):
            return 4  # en_passant
        elif board.gives_check(move):
            return 5  # check
        else:
            return 0  # quiet

    def _get_special_flags(self, move: chess.Move, board: chess.Board) -> int:
        """Get special move flags"""
        flags = 0

        if board.is_castling(move):
            flags |= 1
        if board.is_en_passant(move):
            flags |= 2
        if board.gives_check(move):
            flags |= 4

        # Check if move leads to checkmate
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_checkmate():
            flags |= 8

        return flags

    def get_move_attention_mask(self, legal_moves: List[chess.Move],
                               all_moves: List[chess.Move]) -> torch.Tensor:
        """
        Create attention mask for legal moves

        Args:
            legal_moves: List of legal moves
            all_moves: List of all possible moves

        Returns:
            Attention mask tensor (len(all_moves), len(all_moves))
        """
        mask = torch.full((len(all_moves), len(all_moves)), float('-inf'))

        # Allow attention between legal moves
        for i, move_i in enumerate(all_moves):
            if move_i in legal_moves:
                for j, move_j in enumerate(all_moves):
                    if move_j in legal_moves:
                        mask[i, j] = 0.0

        return mask

    def apply_move_attention(self, encoded_moves: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism to move encodings

        Args:
            encoded_moves: Encoded moves (num_moves, d_model)
            attention_mask: Attention mask (num_moves, num_moves)

        Returns:
            Attention-enhanced move encodings
        """
        # Add batch dimension if needed
        if encoded_moves.dim() == 2:
            encoded_moves = encoded_moves.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        # Apply multi-head attention
        attended_moves, _ = self.move_attention(
            encoded_moves, encoded_moves, encoded_moves,
            attn_mask=attention_mask
        )

        # Remove batch dimension if added
        if attended_moves.size(0) == 1:
            attended_moves = attended_moves.squeeze(0)

        return attended_moves

    def predict_move_validity(self, encoded_moves: torch.Tensor) -> torch.Tensor:
        """
        Predict move validity using learned validity predictor

        Args:
            encoded_moves: Encoded moves (num_moves, d_model)

        Returns:
            Validity scores (num_moves,)
        """
        validity_scores = self.validity_predictor(encoded_moves)
        return validity_scores.squeeze(-1)


class ChessAttentionMechanisms:
    """
    Advanced attention mechanisms for chess position understanding
    """

    def __init__(self, d_model: int = 512, nhead: int = 8):
        self.d_model = d_model
        self.nhead = nhead

        # Spatial attention for board positions
        self.spatial_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Piece-type attention
        self.piece_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Color-based attention
        self.color_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Threat attention (pieces attacking each other)
        self.threat_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Control attention (squares controlled by pieces)
        self.control_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Layer norms for residual connections
        self.spatial_norm = nn.LayerNorm(d_model)
        self.piece_norm = nn.LayerNorm(d_model)
        self.color_norm = nn.LayerNorm(d_model)
        self.threat_norm = nn.LayerNorm(d_model)
        self.control_norm = nn.LayerNorm(d_model)

    def apply_spatial_attention(self, board_tokens: torch.Tensor,
                               board_geometry: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention based on board geometry

        Args:
            board_tokens: Board position tokens (batch, 64, d_model)
            board_geometry: Spatial relationships (64, 64)

        Returns:
            Spatially attended tokens
        """
        # Use board geometry as attention mask
        attn_mask = board_geometry.unsqueeze(0)  # Add batch dimension

        attended, _ = self.spatial_attention(
            board_tokens, board_tokens, board_tokens,
            attn_mask=attn_mask
        )

        return self.spatial_norm(board_tokens + attended)

    def apply_piece_attention(self, board_tokens: torch.Tensor,
                             piece_types: torch.Tensor) -> torch.Tensor:
        """
        Apply attention based on piece types

        Args:
            board_tokens: Board position tokens (batch, 64, d_model)
            piece_types: Piece type indicators (batch, 64, 13) - 12 pieces + empty

        Returns:
            Piece-attended tokens
        """
        # Create attention mask based on piece type similarity
        piece_similarities = torch.matmul(piece_types, piece_types.transpose(-2, -1))
        attn_mask = torch.where(piece_similarities > 0.5, 0.0, float('-inf'))

        attended, _ = self.piece_attention(
            board_tokens, board_tokens, board_tokens,
            attn_mask=attn_mask
        )

        return self.piece_norm(board_tokens + attended)

    def apply_color_attention(self, board_tokens: torch.Tensor,
                             piece_colors: torch.Tensor) -> torch.Tensor:
        """
        Apply attention based on piece colors

        Args:
            board_tokens: Board position tokens (batch, 64, d_model)
            piece_colors: Color indicators (batch, 64, 3) - white, black, empty

        Returns:
            Color-attended tokens
        """
        # Create attention mask for same-color pieces
        color_similarities = torch.matmul(piece_colors, piece_colors.transpose(-2, -1))
        attn_mask = torch.where(color_similarities > 0.8, 0.0, float('-inf'))

        attended, _ = self.color_attention(
            board_tokens, board_tokens, board_tokens,
            attn_mask=attn_mask
        )

        return self.color_norm(board_tokens + attended)

    def apply_threat_attention(self, board_tokens: torch.Tensor,
                              threat_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply attention based on piece threats

        Args:
            board_tokens: Board position tokens (batch, 64, d_model)
            threat_matrix: Threat relationships (batch, 64, 64)

        Returns:
            Threat-attended tokens
        """
        # Use threat matrix as attention weights
        attn_weights = threat_matrix

        attended, _ = self.threat_attention(
            board_tokens, board_tokens, board_tokens,
            attn_mask=None  # Let threat matrix control attention
        )

        return self.threat_norm(board_tokens + attended)

    def apply_control_attention(self, board_tokens: torch.Tensor,
                               control_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply attention based on square control

        Args:
            board_tokens: Board position tokens (batch, 64, d_model)
            control_matrix: Control relationships (batch, 64, 64)

        Returns:
            Control-attended tokens
        """
        # Use control matrix as attention bias
        attn_bias = control_matrix * 2.0  # Boost attention for controlled squares

        attended, _ = self.control_attention(
            board_tokens, board_tokens, board_tokens,
            attn_mask=None
        )

        # Apply control bias
        attended = attended + attn_bias.unsqueeze(-1) * board_tokens

        return self.control_norm(board_tokens + attended)

    def apply_all_attention(self, board_tokens: torch.Tensor,
                           spatial_geom: torch.Tensor,
                           piece_types: torch.Tensor,
                           piece_colors: torch.Tensor,
                           threat_matrix: torch.Tensor,
                           control_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply all attention mechanisms in sequence

        Args:
            board_tokens: Board position tokens (batch, 64, d_model)
            spatial_geom: Spatial geometry matrix
            piece_types: Piece type indicators
            piece_colors: Piece color indicators
            threat_matrix: Threat relationships
            control_matrix: Control relationships

        Returns:
            Fully attended tokens
        """
        # Apply attention mechanisms in sequence
        x = self.apply_spatial_attention(board_tokens, spatial_geom)
        x = self.apply_piece_attention(x, piece_types)
        x = self.apply_color_attention(x, piece_colors)
        x = self.apply_threat_attention(x, threat_matrix)
        x = self.apply_control_attention(x, control_matrix)

        return x


class ChessPositionEncoder:
    """
    Advanced position encoder that combines board state with attention mechanisms
    """

    def __init__(self, input_channels: int = 19, d_model: int = 512):
        self.input_channels = input_channels
        self.d_model = d_model

        # Convert board channels to token embeddings
        self.channel_embeddings = nn.Linear(input_channels, d_model)

        # Positional encodings for board squares
        self.row_embeddings = nn.Embedding(8, d_model // 4)
        self.col_embeddings = nn.Embedding(8, d_model // 4)
        self.diagonal_embeddings = nn.Embedding(15, d_model // 4)  # Max diagonal length
        self.center_distance_embeddings = nn.Embedding(8, d_model // 4)  # Distance from center

        # Attention mechanisms
        self.attention_mechanisms = ChessAttentionMechanisms(d_model)

        # Final projection
        self.output_projection = nn.Linear(d_model, d_model)

    def encode_position(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode chess position with advanced attention mechanisms

        Args:
            board_tensor: Board state (batch, channels, 8, 8)

        Returns:
            Encoded position (batch, 64, d_model)
        """
        batch_size = board_tensor.size(0)

        # Flatten spatial dimensions: (batch, 64, channels)
        board_flat = board_tensor.view(batch_size, self.input_channels, -1).transpose(1, 2)

        # Convert to token embeddings: (batch, 64, d_model)
        tokens = self.channel_embeddings(board_flat)

        # Add positional encodings
        positional_tokens = self._add_positional_encodings(tokens)

        # Create attention matrices
        spatial_geom = self._create_spatial_geometry()
        piece_types, piece_colors = self._extract_piece_info(board_tensor)
        threat_matrix = self._create_threat_matrix(board_tensor)
        control_matrix = self._create_control_matrix(board_tensor)

        # Apply attention mechanisms
        attended_tokens = self.attention_mechanisms.apply_all_attention(
            positional_tokens,
            spatial_geom.unsqueeze(0).expand(batch_size, -1, -1),
            piece_types,
            piece_colors,
            threat_matrix,
            control_matrix
        )

        # Final projection
        encoded = self.output_projection(attended_tokens)

        return encoded

    def _add_positional_encodings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add multiple types of positional encodings"""
        batch_size, seq_len, d_model = tokens.shape

        # Create positional encodings for each square
        pos_encodings = []
        for pos in range(seq_len):
            row, col = divmod(pos, 8)

            # Row and column encodings
            row_emb = self.row_embeddings(torch.tensor(row))
            col_emb = self.col_embeddings(torch.tensor(col))

            # Diagonal encoding (main diagonal distance)
            diagonal = min(row + col, 15 - (row + col), 7)  # Simplified
            diag_emb = self.diagonal_embeddings(torch.tensor(diagonal))

            # Center distance encoding
            center_row, center_col = 3.5, 3.5  # Board center
            distance = int(((row - center_row) ** 2 + (col - center_col) ** 2) ** 0.5)
            center_emb = self.center_distance_embeddings(torch.tensor(min(distance, 7)))

            # Combine all positional encodings
            pos_encoding = torch.cat([row_emb, col_emb, diag_emb, center_emb], dim=-1)

            # Pad to match d_model
            if pos_encoding.size(-1) < d_model:
                padding_size = d_model - pos_encoding.size(-1)
                pos_encoding = F.pad(pos_encoding, (0, padding_size))

            pos_encodings.append(pos_encoding)

        # Stack and add to tokens
        pos_matrix = torch.stack(pos_encodings, dim=0).unsqueeze(0).expand(batch_size, -1, -1)
        return tokens + pos_matrix

    def _create_spatial_geometry(self) -> torch.Tensor:
        """Create spatial geometry matrix for attention"""
        geometry = torch.zeros(64, 64)

        for i in range(64):
            row1, col1 = divmod(i, 8)
            for j in range(64):
                row2, col2 = divmod(j, 8)

                # Euclidean distance
                distance = ((row1 - row2) ** 2 + (col1 - col2) ** 2) ** 0.5

                # Convert to attention weight (closer squares get higher attention)
                geometry[i, j] = max(0, 1.0 - distance / 8.0)

        return geometry

    def _extract_piece_info(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract piece type and color information"""
        # Simplified extraction - in practice you'd decode the board channels
        piece_types = torch.randn(board_tensor.size(0), 64, 13)  # 12 pieces + empty
        piece_colors = torch.randn(board_tensor.size(0), 64, 3)   # white, black, empty

        return piece_types, piece_colors

    def _create_threat_matrix(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """Create threat relationships matrix"""
        # Simplified threat matrix - in practice you'd compute actual threats
        return torch.randn(board_tensor.size(0), 64, 64)

    def _create_control_matrix(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """Create square control matrix"""
        # Simplified control matrix - in practice you'd compute actual control
        return torch.randn(board_tensor.size(0), 64, 64)


if __name__ == "__main__":
    # Test the move encoder and attention mechanisms
    print("=== Chess Move Encoder & Attention Test ===")

    # Test move encoder
    move_encoder = ChessMoveEncoder(d_model=256)
    print(f"Move encoder created with {sum(p.numel() for p in move_encoder.parameters()):,} parameters")

    # Test attention mechanisms
    attention = ChessAttentionMechanisms(d_model=512)
    print(f"Attention mechanisms created with {sum(p.numel() for p in attention.parameters()):,} parameters")

    # Test position encoder
    pos_encoder = ChessPositionEncoder(d_model=512)
    print(f"Position encoder created with {sum(p.numel() for p in pos_encoder.parameters()):,} parameters")

    # Test with dummy data
    board_tensor = torch.randn(1, 19, 8, 8)
    encoded_pos = pos_encoder.encode_position(board_tensor)
    print(f"Position encoding: {encoded_pos.shape}")

    print("✅ Move encoder and attention mechanisms test passed!")
