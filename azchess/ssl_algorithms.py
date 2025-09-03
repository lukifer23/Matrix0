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

import logging
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

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
        Detect opponent threats with full ray-cast attacks and proper blocking.

        Returns a binary map for the side-to-move indicating if a square is
        attacked by the opponent. This uses vectorized shifts and blocking-aware
        ray accumulation similar to square control, but keeps both sides'
        attack maps separate and then selects the opponent’s.

        Args:
            board_states: (B, 19, 8, 8)
        Returns:
            (B, 8, 8) float32 in {0.0, 1.0}
        """
        device = board_states.device
        B = board_states.size(0)
        pieces = board_states[:, :12, :, :].to(torch.float32)  # (B,12,8,8)
        occ = (pieces.sum(dim=1) > 0)

        def _shift(mask: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
            s = torch.roll(mask, shifts=(dr, dc), dims=(1, 2))
            if dr > 0:
                s[:, :dr, :] = 0
            elif dr < 0:
                s[:, dr:, :] = 0
            if dc > 0:
                s[:, :, :dc] = 0
            elif dc < 0:
                s[:, :, dc:] = 0
            return s

        def _accumulate_rays(src: torch.Tensor, directions) -> torch.Tensor:
            attacks = torch.zeros(B, 8, 8, device=device, dtype=torch.float32)
            frontier = None
            for dr, dc in directions:
                frontier = src.clone().to(torch.float32)
                for _ in range(1, 8):
                    frontier = _shift(frontier, dr, dc)
                    attacks += frontier
                    # Stop beyond any occupied square (block further propagation)
                    frontier = frontier * (~occ).to(torch.float32)
            return attacks

        # White pieces
        wp = pieces[:, 0]
        wn = pieces[:, 1]
        wb = pieces[:, 2]
        wr = pieces[:, 3]
        wq = pieces[:, 4]
        wk = pieces[:, 5]

        # Black pieces
        bp = pieces[:, 6]
        bn = pieces[:, 7]
        bb = pieces[:, 8]
        br = pieces[:, 9]
        bq = pieces[:, 10]
        bk = pieces[:, 11]

        # Directions
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ortho_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # White attacks
        white_att = torch.zeros(B, 8, 8, device=device, dtype=torch.float32)
        white_att += _shift(wp, +1, -1)
        white_att += _shift(wp, +1, +1)
        for dr, dc in knight_moves:
            white_att += _shift(wn, dr, dc)
        for dr, dc in king_moves:
            white_att += _shift(wk, dr, dc)
        white_att += _accumulate_rays(wb + wq, diag_dirs)
        white_att += _accumulate_rays(wr + wq, ortho_dirs)

        # Black attacks
        black_att = torch.zeros(B, 8, 8, device=device, dtype=torch.float32)
        black_att += _shift(bp, -1, -1)
        black_att += _shift(bp, -1, +1)
        for dr, dc in knight_moves:
            black_att += _shift(bn, dr, dc)
        for dr, dc in king_moves:
            black_att += _shift(bk, dr, dc)
        black_att += _accumulate_rays(bb + bq, diag_dirs)
        black_att += _accumulate_rays(br + bq, ortho_dirs)

        # Pick opponent threats based on side to move
        stm_white = (board_states[:, 12, 0, 0] > 0.5).to(torch.float32).view(B, 1, 1)
        threat_map = white_att * (1.0 - stm_white) + black_att * stm_white
        return torch.clamp(threat_map, 0.0, 1.0)

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
        Vectorized pin detection using first-two-blockers logic along rays.

        A piece is pinned if it is the first blocker from the king along a ray
        and the next blocker beyond it is an enemy slider aligned to that ray.
        Evaluated for the side-to-move.
        """
        device = board_states.device
        B = board_states.size(0)
        pieces = board_states[:, :12, :, :].to(torch.float32)
        occ = (pieces.sum(dim=1) > 0)

        stm_white = (board_states[:, 12, 0, 0] > 0.5).to(torch.float32).view(B, 1, 1)
        own_any = (pieces[:, :6].sum(dim=1) * stm_white + pieces[:, 6:12].sum(dim=1) * (1.0 - stm_white)) > 0
        king_plane = pieces[:, 5] * stm_white + pieces[:, 11] * (1.0 - stm_white)
        enemy_rook = pieces[:, 9] * stm_white + pieces[:, 3] * (1.0 - stm_white)
        enemy_bishop = pieces[:, 8] * stm_white + pieces[:, 2] * (1.0 - stm_white)
        enemy_queen = pieces[:, 10] * stm_white + pieces[:, 4] * (1.0 - stm_white)

        def _shift(mask: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
            s = torch.roll(mask, shifts=(dr, dc), dims=(1, 2))
            if dr > 0:
                s[:, :dr, :] = 0
            elif dr < 0:
                s[:, dr:, :] = 0
            if dc > 0:
                s[:, :, :dc] = 0
            elif dc < 0:
                s[:, :, dc:] = 0
            return s

        def _ray_stack(origin: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
            rays = []
            frontier = origin
            for _ in range(1, 8):
                frontier = _shift(frontier, dr, dc)
                rays.append(frontier)
            return torch.stack(rays, dim=1)  # (B,7,8,8)

        diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ortho_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        pin_map = torch.zeros(B, 8, 8, device=device, dtype=torch.float32)

        for dirs, slider_mask in [
            (diag_dirs, (enemy_bishop + enemy_queen) > 0),
            (ortho_dirs, (enemy_rook + enemy_queen) > 0),
        ]:
            for dr, dc in dirs:
                ray = _ray_stack(king_plane, dr, dc)  # (B,7,8,8)
                occ_steps = (ray * occ.unsqueeze(1)).flatten(2).sum(dim=2) > 0
                own_steps = (ray * own_any.unsqueeze(1)).flatten(2).sum(dim=2) > 0
                slider_steps = (ray * slider_mask.unsqueeze(1)).flatten(2).sum(dim=2) > 0

                occ_cum = torch.cumsum(occ_steps.int(), dim=1)
                first_occ = occ_steps & (occ_cum == 1)
                own_first = first_occ & own_steps

                after_first = (torch.cumsum(first_occ.int(), dim=1) > 0) & (~first_occ)
                occ_after = occ_steps & after_first
                occ_after_cum = torch.cumsum(occ_after.int(), dim=1)
                second_occ = occ_after & (occ_after_cum == 1)

                valid_slider_after = (second_occ & slider_steps).any(dim=1, keepdim=True)
                has_own_first = own_first.any(dim=1, keepdim=True)
                gate = (valid_slider_after & has_own_first).to(torch.float32).view(B, 1, 1)

                own_first_board = (ray * own_first.view(B, 7, 1, 1).to(ray.dtype)).sum(dim=1)
                pin_map = pin_map + (own_first_board * gate)

        return torch.clamp(pin_map, 0.0, 1.0)

    def detect_forks_batch(self, board_states: torch.Tensor) -> torch.Tensor:
        """
        Vectorized fork detection for the side-to-move.

        Marks own tactical piece squares that attack two or more enemy pieces,
        respecting blockers for sliding pieces.
        """
        device = board_states.device
        B = board_states.size(0)
        pieces = board_states[:, :12, :, :].to(torch.float32)
        occ = (pieces.sum(dim=1) > 0)
        stm_white = (board_states[:, 12, 0, 0] > 0.5).to(torch.float32).view(B, 1, 1)

        own_knights = pieces[:, 1] * stm_white + pieces[:, 7] * (1.0 - stm_white)
        own_bishops = pieces[:, 2] * stm_white + pieces[:, 8] * (1.0 - stm_white)
        own_rooks = pieces[:, 3] * stm_white + pieces[:, 9] * (1.0 - stm_white)
        own_queens = pieces[:, 4] * stm_white + pieces[:, 10] * (1.0 - stm_white)
        own_kings = pieces[:, 5] * stm_white + pieces[:, 11] * (1.0 - stm_white)
        own_tactical = (own_knights + own_bishops + own_rooks + own_queens + own_kings) > 0

        enemy_any = (pieces[:, 6:12].sum(dim=1) * stm_white + pieces[:, :6].sum(dim=1) * (1.0 - stm_white)) > 0

        def _shift(mask: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
            s = torch.roll(mask, shifts=(dr, dc), dims=(1, 2))
            if dr > 0:
                s[:, :dr, :] = 0
            elif dr < 0:
                s[:, dr:, :] = 0
            if dc > 0:
                s[:, :, :dc] = 0
            elif dc < 0:
                s[:, :, dc:] = 0
            return s

        count_map = torch.zeros(B, 8, 8, device=device, dtype=torch.float32)

        # Knights
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            enemy_back = _shift(enemy_any.to(torch.float32), -dr, -dc) > 0
            hits = (own_knights > 0) & enemy_back
            count_map += hits.to(torch.float32)

        # Kings
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            enemy_back = _shift(enemy_any.to(torch.float32), -dr, -dc) > 0
            hits = (own_kings > 0) & enemy_back
            count_map += hits.to(torch.float32)

        # Sliding pieces
        diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ortho_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def _accumulate_sliding_hits(origins: torch.Tensor, directions) -> None:
            nonlocal count_map
            origins_b = origins > 0
            for dr, dc in directions:
                for s in range(1, 8):
                    enemy_back = _shift(enemy_any.to(torch.float32), -dr * s, -dc * s) > 0
                    if s == 1:
                        blocked = torch.zeros_like(enemy_back)
                    else:
                        blocked = torch.zeros_like(enemy_back)
                        for t in range(1, s):
                            blocked |= _shift(occ.to(torch.float32), -dr * t, -dc * t) > 0
                    hits = origins_b & enemy_back & (~blocked)
                    count_map += hits.to(torch.float32)

        _accumulate_sliding_hits(own_bishops, diag_dirs)
        _accumulate_sliding_hits(own_rooks, ortho_dirs)
        _accumulate_sliding_hits(own_queens, diag_dirs)
        _accumulate_sliding_hits(own_queens, ortho_dirs)

        fork_map = ((count_map >= 2.0) & own_tactical).to(torch.float32)
        return fork_map

    def calculate_square_control_batch(self, board_states: torch.Tensor) -> torch.Tensor:
        """
        Vectorized square control using piece attack patterns with blocking-aware ray casts.

        Returns a float map in {-1.0, 0.0, 1.0}: -1=black controls, 1=white controls, 0=neutral/contested.
        """
        device = board_states.device
        B = board_states.size(0)
        pieces = board_states[:, :12, :, :]  # (B,12,8,8)
        occ = (pieces.sum(dim=1) > 0)

        def _shift(mask: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
            s = torch.roll(mask, shifts=(dr, dc), dims=(1, 2))
            if dr > 0:
                s[:, :dr, :] = 0
            elif dr < 0:
                s[:, dr:, :] = 0
            if dc > 0:
                s[:, :, :dc] = 0
            elif dc < 0:
                s[:, :, dc:] = 0
            return s

        white_att = torch.zeros(B, 8, 8, device=device, dtype=torch.float32)
        black_att = torch.zeros(B, 8, 8, device=device, dtype=torch.float32)

        # Pawns
        wp = pieces[:, 0]
        bp = pieces[:, 6]
        white_att += _shift(wp, +1, -1)
        white_att += _shift(wp, +1, +1)
        black_att += _shift(bp, -1, -1)
        black_att += _shift(bp, -1, +1)

        # Knights
        wn = pieces[:, 1]
        bn = pieces[:, 7]
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            white_att += _shift(wn, dr, dc)
            black_att += _shift(bn, dr, dc)

        # Kings
        wk = pieces[:, 5]
        bk = pieces[:, 11]
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            white_att += _shift(wk, dr, dc)
            black_att += _shift(bk, dr, dc)

        # Sliding pieces with blocking-aware rays
        wb = pieces[:, 2]
        wr = pieces[:, 3]
        wq = pieces[:, 4]
        bb = pieces[:, 8]
        br = pieces[:, 9]
        bq = pieces[:, 10]

        diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ortho_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def _accumulate_rays(src: torch.Tensor, directions) -> torch.Tensor:
            attacks = torch.zeros(B, 8, 8, device=device, dtype=torch.float32)
            for dr, dc in directions:
                frontier = src.clone().to(torch.float32)
                for _ in range(1, 8):
                    frontier = _shift(frontier, dr, dc)
                    attacks += frontier
                    # Stop propagation beyond any occupied square
                    frontier = frontier * (~occ).to(torch.float32)
            return attacks

        white_att += _accumulate_rays(wb + wq, diag_dirs)
        white_att += _accumulate_rays(wr + wq, ortho_dirs)
        black_att += _accumulate_rays(bb + bq, diag_dirs)
        black_att += _accumulate_rays(br + bq, ortho_dirs)

        diff = white_att - black_att
        control = torch.sign(diff).to(torch.float32)
        return control

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
