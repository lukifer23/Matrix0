"""
SSL Algorithms for Matrix0 Chess Engine
Implements advanced self-supervised learning targets for chess positions.

SSL (Self-Supervised Learning) vs SSRL (Self-Supervised Representation Learning):

SSL Tasks (Chess-Specific):
- Piece recognition: Identify pieces and empty squares
- Threat detection: Identify squares under attack
- Pin detection: Identify pinned pieces
- Fork detection: Identify fork opportunities
- Square control: Identify who controls each square

SSRL Tasks (General):
- Position prediction: Predict board positions
- Material balance: Predict material counts
- Rotation invariance: Learn rotation-invariant features

This module focuses on SSL tasks that are specific to chess rules and tactics.
"""

from __future__ import annotations

import torch
import numpy as np
import chess
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ChessSSLAlgorithms:
    """Advanced SSL algorithms for chess position analysis."""

    def __init__(self):
        # Pre-compute piece movement patterns for efficiency
        self._init_piece_movements()

    def _init_piece_movements(self):
        """Initialize piece movement patterns for vectorized operations."""
        # These will be used for efficient threat detection
        self.directions = {
            'rook': [(-1, 0), (1, 0), (0, -1), (0, 1)],
            'bishop': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            'queen': [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
            'knight': [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        }

    def detect_threats_batch(self, board_states: torch.Tensor) -> torch.Tensor:
        """
        Detect piece threats using simplified vectorized operations.

        Args:
            board_states: (B, 19, 8, 8) tensor of board states

        Returns:
            threat_targets: (B, 8, 8) tensor where 1.0 = square under threat
        """
        batch_size = board_states.size(0)
        device = board_states.device

        # Initialize threat map
        threat_map = torch.zeros(batch_size, 8, 8, device=device, dtype=torch.float32)

        # Extract piece positions (planes 0-11)
        pieces = board_states[:, :12, :, :]  # (B, 12, 8, 8)

        # For now, implement a simplified threat detection
        # This focuses on basic patterns that can be detected with simple operations

        # 1. Detect knight threats (L-shapes) - simplified
        knights = torch.zeros_like(threat_map)
        knights += pieces[:, 1, :, :]  # White knights
        knights += pieces[:, 7, :, :]  # Black knights

        # Simple knight threat detection - just mark squares near knights
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            if abs(dr) < 8 and abs(dc) < 8:
                shifted = torch.roll(knights, shifts=(dr, dc), dims=(1, 2))
                if dr > 0:
                    shifted[:, :dr, :] = 0
                elif dr < 0:
                    shifted[:, dr:, :] = 0
                if dc > 0:
                    shifted[:, :, :dc] = 0
                elif dc < 0:
                    shifted[:, :, dc:] = 0
                threat_map += shifted

        # 2. Detect sliding piece threats (rooks, bishops, queens) - very simplified
        sliding_pieces = torch.zeros_like(threat_map)
        sliding_pieces += pieces[:, 3, :, :] + pieces[:, 4, :, :]  # White rooks and queens
        sliding_pieces += pieces[:, 9, :, :] + pieces[:, 10, :, :]  # Black rooks and queens

        # Simple sliding piece detection - mark adjacent squares
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            if abs(dr) < 8 and abs(dc) < 8:
                shifted = torch.roll(sliding_pieces, shifts=(dr, dc), dims=(1, 2))
                if dr > 0:
                    shifted[:, :dr, :] = 0
                elif dr < 0:
                    shifted[:, dr:, :] = 0
                if dc > 0:
                    shifted[:, :, :dc] = 0
                elif dc < 0:
                    shifted[:, :, dc:] = 0
                threat_map += shifted

        # 3. Detect king threats (adjacent squares)
        kings = torch.zeros_like(threat_map)
        kings += pieces[:, 5, :, :]  # White king
        kings += pieces[:, 11, :, :]  # Black king

        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            if abs(dr) < 8 and abs(dc) < 8:
                shifted = torch.roll(kings, shifts=(dr, dc), dims=(1, 2))
                if dr > 0:
                    shifted[:, :dr, :] = 0
                elif dr < 0:
                    shifted[:, dr:, :] = 0
                if dc > 0:
                    shifted[:, :, :dc] = 0
                elif dc < 0:
                    shifted[:, :, dc:] = 0
                threat_map += shifted

        return torch.clamp(threat_map, 0.0, 1.0)  # Ensure values are 0 or 1

    def _calculate_piece_threats(self, piece_positions: torch.Tensor, piece_type: str,
                                is_white: bool, device: torch.device) -> torch.Tensor:
        """
        Calculate threats for a specific piece type using vectorized operations.

        Args:
            piece_positions: (B, 8, 8) tensor of piece positions
            piece_type: Type of piece ('pawn', 'knight', 'bishop', 'rook', 'queen', 'king')
            is_white: Whether this is a white piece
            device: Torch device

        Returns:
            threat_map: (B, 8, 8) tensor of threatened squares
        """
        batch_size = piece_positions.size(0)
        threat_map = torch.zeros_like(piece_positions)

        if piece_type == 'pawn':
            # Pawn threats are diagonal - corrected implementation
            if is_white:
                # White pawns attack diagonally forward (increasing row number)
                # From (r,c) attacks (r+1,c-1) and (r+1,c+1)
                # Shift pawns down and left/right to get attack positions
                threat_map[:, 1:, :-1] += piece_positions[:, :-1, 1:]   # Attack left (from r-1,c+1 to r,c)
                threat_map[:, 1:, 1:] += piece_positions[:, :-1, :-1]   # Attack right (from r-1,c-1 to r,c)
            else:
                # Black pawns attack diagonally backward (decreasing row number)
                # From (r,c) attacks (r-1,c-1) and (r-1,c+1)
                # Shift pawns up and left/right to get attack positions
                threat_map[:, :-1, :-1] += piece_positions[:, 1:, 1:]   # Attack left (from r+1,c+1 to r,c)
                threat_map[:, :-1, 1:] += piece_positions[:, 1:, :-1]   # Attack right (from r+1,c-1 to r,c)

        elif piece_type == 'knight':
            # Knight moves: L-shaped patterns
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                           (1, -2), (1, 2), (2, -1), (2, 1)]

            for dr, dc in knight_moves:
                # Use torch.roll with proper padding
                shifted = torch.roll(piece_positions, shifts=(dr, dc), dims=(1, 2))

                # Clear pieces that moved off the board
                if dr > 0:
                    shifted[:, :dr, :] = 0
                elif dr < 0:
                    shifted[:, dr:, :] = 0
                if dc > 0:
                    shifted[:, :, :dc] = 0
                elif dc < 0:
                    shifted[:, :, dc:] = 0

                threat_map += shifted

        elif piece_type in ['bishop', 'rook', 'queen']:
            # Sliding pieces: check in all directions
            directions = self.directions[piece_type]

            for dr, dc in directions:
                # For each direction, propagate threats along the line
                for step in range(1, 8):
                    # Calculate shift for this step
                    shift_r = dr * step
                    shift_c = dc * step

                    # Skip if shift is too large
                    if abs(shift_r) >= 8 or abs(shift_c) >= 8:
                        continue

                    # Shift piece positions
                    shifted = torch.roll(piece_positions, shifts=(shift_r, shift_c), dims=(1, 2))

                    # Clear pieces that moved off the board
                    if shift_r > 0:
                        shifted[:, :shift_r, :] = 0
                    elif shift_r < 0:
                        shifted[:, shift_r:, :] = 0
                    if shift_c > 0:
                        shifted[:, :, :shift_c] = 0
                    elif shift_c < 0:
                        shifted[:, :, shift_c:] = 0

                    # Add to threat map
                    threat_map += shifted

        elif piece_type == 'king':
            # King attacks adjacent squares
            king_moves = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),           (0, 1),
                         (1, -1),  (1, 0),  (1, 1)]

            for dr, dc in king_moves:
                # Skip invalid moves
                if abs(dr) >= 8 or abs(dc) >= 8:
                    continue

                shifted = torch.roll(piece_positions, shifts=(dr, dc), dims=(1, 2))

                # Clear boundary pieces
                if dr > 0:
                    shifted[:, :dr, :] = 0
                elif dr < 0:
                    shifted[:, dr:, :] = 0
                if dc > 0:
                    shifted[:, :, :dc] = 0
                elif dc < 0:
                    shifted[:, :, dc:] = 0

                threat_map += shifted

        return threat_map

    def detect_pins_batch(self, board_states: torch.Tensor) -> torch.Tensor:
        """
        Detect pinned pieces using vectorized operations.

        A pinned piece is one that cannot move because doing so would expose
        the king to check.

        Args:
            board_states: (B, 19, 8, 8) tensor of board states

        Returns:
            pin_targets: (B, 8, 8) tensor where 1.0 = pinned piece
        """
        batch_size = board_states.size(0)
        device = board_states.device

        pin_map = torch.zeros(batch_size, 8, 8, device=device, dtype=torch.float32)

        # Extract relevant planes
        pieces = board_states[:, :12, :, :]  # All pieces
        side_to_move = board_states[:, 12:13, 0, 0]  # Side to move (plane 12)

        # This is a simplified pin detection that identifies potential pins
        # In a full implementation, this would require more complex analysis

        # For now, implement a basic heuristic:
        # Pieces on the same line as the king and an enemy sliding piece
        white_kings = pieces[:, 5, :, :]  # White king
        black_kings = pieces[:, 11, :, :]  # Black king

        # Identify potential pinning situations (simplified)
        for b in range(min(batch_size, 16)):  # Limit batch size for memory
            stm = side_to_move[b].item()

            # Get king position
            if stm > 0:  # White to move
                king_pos = white_kings[b].nonzero(as_tuple=True)
                if len(king_pos[0]) == 0:
                    continue
                kr, kc = king_pos[0][0].item(), king_pos[1][0].item()
                own_pieces = pieces[b, :6, :, :]  # White pieces
                enemy_pieces = pieces[b, 6:12, :, :]  # Black pieces
            else:  # Black to move
                king_pos = black_kings[b].nonzero(as_tuple=True)
                if len(king_pos[0]) == 0:
                    continue
                kr, kc = king_pos[0][0].item(), king_pos[1][0].item()
                own_pieces = pieces[b, 6:12, :, :]  # Black pieces
                enemy_pieces = pieces[b, :6, :, :]  # White pieces

            # Check each direction from the king for potential pins
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]

            for dr, dc in directions:
                # Look for enemy sliding pieces in this direction
                enemy_sliders = enemy_pieces[2:5, :, :].sum(dim=0)  # Bishops, rooks, queens

                # Check if there's an enemy slider in this direction
                found_slider = False
                slider_pos = None

                r, c = kr + dr, kc + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    if enemy_sliders[r, c] > 0:
                        found_slider = True
                        slider_pos = (r, c)
                        break
                    elif own_pieces[:, r, c].sum() > 0:
                        # Hit own piece, no pin possible in this direction
                        break
                    r += dr
                    c += dc

                if found_slider:
                    # Check for own piece between king and slider
                    r, c = kr + dr, kc + dc
                    pinned_piece_pos = None

                    while (r, c) != slider_pos:
                        if own_pieces[:, r, c].sum() > 0:
                            if pinned_piece_pos is not None:
                                # Multiple pieces between king and slider
                                break
                            pinned_piece_pos = (r, c)
                        r += dr
                        c += dc

                    if pinned_piece_pos:
                        pin_map[b, pinned_piece_pos[0], pinned_piece_pos[1]] = 1.0

        return pin_map

    def detect_forks_batch(self, board_states: torch.Tensor) -> torch.Tensor:
        """
        Detect fork opportunities using improved vectorized operations.

        A fork is when one piece attacks multiple enemy pieces simultaneously.
        This implementation focuses on tactical patterns that commonly lead to forks.

        Args:
            board_states: (B, 19, 8, 8) tensor of board states

        Returns:
            fork_targets: (B, 8, 8) tensor where 1.0 = square with fork opportunity
        """
        batch_size = board_states.size(0)
        device = board_states.device

        fork_map = torch.zeros(batch_size, 8, 8, device=device, dtype=torch.float32)

        # Extract piece positions
        pieces = board_states[:, :12, :, :]
        side_to_move = board_states[:, 12:13, 0, 0]

        for b in range(min(batch_size, 32)):  # Increased limit for better detection
            stm = side_to_move[b].item()

            # Determine which pieces belong to current player and enemy
            if stm > 0:  # White to move
                own_pieces = pieces[b, :6, :, :]
                enemy_pieces = pieces[b, 6:12, :, :]
                own_knights = pieces[b, 1, :, :]  # White knights
                own_bishops = pieces[b, 2, :, :]  # White bishops
                own_rooks = pieces[b, 3, :, :]    # White rooks
                own_queens = pieces[b, 4, :, :]   # White queens
            else:  # Black to move
                own_pieces = pieces[b, 6:12, :, :]
                enemy_pieces = pieces[b, :6, :, :]
                own_knights = pieces[b, 7, :, :]  # Black knights
                own_bishops = pieces[b, 8, :, :]  # Black bishops
                own_rooks = pieces[b, 9, :, :]    # Black rooks
                own_queens = pieces[b, 10, :, :]  # Black queens

            # Focus on tactical pieces that commonly create forks: knights, bishops, rooks, queens
            tactical_pieces = torch.zeros_like(own_knights)
            tactical_pieces += own_knights + own_bishops + own_rooks + own_queens

            # Check each square for fork potential
            for r in range(8):
                for c in range(8):
                    if tactical_pieces[r, c] > 0:  # Own tactical piece at this square
                        # Count enemy pieces that would be attacked from this position
                        attacked_count = 0

                        # Check knight attacks (L-shapes)
                        if own_knights[r, c] > 0:
                            for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                                         (1, -2), (1, 2), (2, -1), (2, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < 8 and 0 <= nc < 8:
                                    if enemy_pieces[:, nr, nc].sum() > 0:
                                        attacked_count += 1

                        # Check sliding piece attacks
                        elif own_bishops[r, c] > 0 or own_rooks[r, c] > 0 or own_queens[r, c] > 0:
                            # Check all 8 directions for sliding pieces
                            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                                        (-1, -1), (-1, 1), (1, -1), (1, 1)]

                            for dr, dc in directions:
                                for step in range(1, 8):
                                    nr, nc = r + dr * step, c + dc * step
                                    if not (0 <= nr < 8 and 0 <= nc < 8):
                                        break

                                    if enemy_pieces[:, nr, nc].sum() > 0:
                                        attacked_count += 1
                                        break  # Can only attack one piece per direction
                                    elif own_pieces[:, nr, nc].sum() > 0:
                                        break  # Blocked by own piece

                        # If this piece attacks 2 or more enemy pieces, it's a fork
                        if attacked_count >= 2:
                            fork_map[b, r, c] = 1.0

        return fork_map

    def calculate_square_control_batch(self, board_states: torch.Tensor) -> torch.Tensor:
        """
        Calculate square control using piece attack patterns.

        Square control indicates which player controls each square based on piece attacks.

        Args:
            board_states: (B, 19, 8, 8) tensor of board states

        Returns:
            control_targets: (B, 8, 8) tensor where 1.0 = white controls, -1.0 = black controls, 0.0 = contested/neutral
        """
        batch_size = board_states.size(0)
        device = board_states.device

        control_map = torch.zeros(batch_size, 8, 8, device=device, dtype=torch.float32)

        # Extract piece positions
        pieces = board_states[:, :12, :, :]

        for b in range(min(batch_size, 32)):  # Limit for memory efficiency
            # Initialize attack counts for each square
            white_attacks = torch.zeros(8, 8, device=device)
            black_attacks = torch.zeros(8, 8, device=device)

            # Check each piece for its attacks
            for r in range(8):
                for c in range(8):
                    # White pieces (planes 0-5)
                    if pieces[b, 0, r, c] > 0:  # White pawn
                        # Pawns attack diagonally forward
                        if r < 7:
                            if c > 0: white_attacks[r+1, c-1] += 1  # Left diagonal
                            if c < 7: white_attacks[r+1, c+1] += 1  # Right diagonal

                    elif pieces[b, 1, r, c] > 0:  # White knight
                        # Knight attacks (L-shapes)
                        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                white_attacks[nr, nc] += 1

                    elif pieces[b, 2, r, c] > 0 or pieces[b, 4, r, c] > 0:  # White bishop or queen (diagonal)
                        # Diagonal attacks
                        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            for step in range(1, 8):
                                nr, nc = r + dr * step, c + dc * step
                                if not (0 <= nr < 8 and 0 <= nc < 8):
                                    break
                                white_attacks[nr, nc] += 1
                                if pieces[b, :, nr, nc].sum() > 0:  # Any piece blocks
                                    break

                    elif pieces[b, 3, r, c] > 0 or pieces[b, 4, r, c] > 0:  # White rook or queen (orthogonal)
                        # Orthogonal attacks
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            for step in range(1, 8):
                                nr, nc = r + dr * step, c + dc * step
                                if not (0 <= nr < 8 and 0 <= nc < 8):
                                    break
                                white_attacks[nr, nc] += 1
                                if pieces[b, :, nr, nc].sum() > 0:  # Any piece blocks
                                    break

                    elif pieces[b, 5, r, c] > 0:  # White king
                        # King attacks adjacent squares
                        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                white_attacks[nr, nc] += 1

                    # Black pieces (planes 6-11)
                    elif pieces[b, 6, r, c] > 0:  # Black pawn
                        # Pawns attack diagonally backward
                        if r > 0:
                            if c > 0: black_attacks[r-1, c-1] += 1  # Left diagonal
                            if c < 7: black_attacks[r-1, c+1] += 1  # Right diagonal

                    elif pieces[b, 7, r, c] > 0:  # Black knight
                        # Knight attacks (L-shapes)
                        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                black_attacks[nr, nc] += 1

                    elif pieces[b, 8, r, c] > 0 or pieces[b, 10, r, c] > 0:  # Black bishop or queen (diagonal)
                        # Diagonal attacks
                        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            for step in range(1, 8):
                                nr, nc = r + dr * step, c + dc * step
                                if not (0 <= nr < 8 and 0 <= nc < 8):
                                    break
                                black_attacks[nr, nc] += 1
                                if pieces[b, :, nr, nc].sum() > 0:  # Any piece blocks
                                    break

                    elif pieces[b, 9, r, c] > 0 or pieces[b, 10, r, c] > 0:  # Black rook or queen (orthogonal)
                        # Orthogonal attacks
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            for step in range(1, 8):
                                nr, nc = r + dr * step, c + dc * step
                                if not (0 <= nr < 8 and 0 <= nc < 8):
                                    break
                                black_attacks[nr, nc] += 1
                                if pieces[b, :, nr, nc].sum() > 0:  # Any piece blocks
                                    break

                    elif pieces[b, 11, r, c] > 0:  # Black king
                        # King attacks adjacent squares
                        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                black_attacks[nr, nc] += 1

            # Determine control based on attack difference
            attack_diff = white_attacks - black_attacks

            # Set control values
            control_map[b, :, :] = torch.where(
                attack_diff > 0, 1.0,  # White controls
                torch.where(attack_diff < 0, -1.0, 0.0)  # Black controls or contested
            )

        return control_map

    def create_enhanced_ssl_targets(self, board_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create enhanced SSL targets for multiple tasks.

        Args:
            board_states: (B, 19, 8, 8) tensor of board states

        Returns:
            targets: Dictionary of SSL targets for different tasks
        """
        targets = {}

        # Piece recognition (original task)
        targets['piece'] = self._create_piece_targets(board_states)

        # Threat detection
        targets['threat'] = self.detect_threats_batch(board_states)

        # Pin detection
        targets['pin'] = self.detect_pins_batch(board_states)

        # Fork detection
        targets['fork'] = self.detect_forks_batch(board_states)

        # Square control
        targets['control'] = self.calculate_square_control_batch(board_states)

        return targets

    def _create_piece_targets(self, board_states: torch.Tensor) -> torch.Tensor:
        """Create piece recognition targets (planes 0-12 for 12 piece types + empty)."""
        batch_size = board_states.size(0)
        device = board_states.device

        # Initialize targets (13 classes: 12 pieces + empty)
        targets = torch.zeros(batch_size, 13, 8, 8, device=device, dtype=torch.long)

        # Planes 0-11: piece positions
        pieces = board_states[:, :12, :, :]  # (B, 12, 8, 8)

        # For each piece type, set the target
        for piece_idx in range(12):
            mask = pieces[:, piece_idx, :, :] > 0
            targets[:, piece_idx, :, :][mask] = 1

        # Plane 12: empty squares (where no pieces exist)
        empty_mask = pieces.sum(dim=1) == 0
        targets[:, 12, :, :][empty_mask] = 1

        return targets


# Global instance for reuse
_ssl_algorithms = ChessSSLAlgorithms()


def get_ssl_algorithms() -> ChessSSLAlgorithms:
    """Get the global SSL algorithms instance."""
    return _ssl_algorithms
