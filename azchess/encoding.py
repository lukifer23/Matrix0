from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess


# --- Board planes encoding (19 planes) ---

def encode_board(board: chess.Board, planes: int = 19) -> np.ndarray:
    """
    Encode board into planes [planes, 8, 8]. Order (total 19):
      0..11: piece planes (white then black; P,N,B,R,Q,K)
      12: side to move (1.0 white, 0.0 black)
      13..16: castling rights (W-K, W-Q, B-K, B-Q)
      17..18: normalized halfmove and fullmove counters
    """
    P = []
    for color in (chess.WHITE, chess.BLACK):
        for piece in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            P.append(_bitboard_to_plane(board.pieces(piece, color)))

    P.append(np.full((8, 8), 1.0 if board.turn == chess.WHITE else 0.0, dtype=np.float32))
    P.append(np.full((8, 8), 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0, dtype=np.float32))
    P.append(np.full((8, 8), 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0, dtype=np.float32))
    P.append(np.full((8, 8), 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0, dtype=np.float32))
    P.append(np.full((8, 8), 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0, dtype=np.float32))

    halfmove = min(board.halfmove_clock, 99) / 99.0
    fullmove = min(board.fullmove_number, 199) / 199.0
    P.append(np.full((8, 8), halfmove, dtype=np.float32))
    P.append(np.full((8, 8), fullmove, dtype=np.float32))

    if len(P) != planes:
        raise ValueError(f"Expected {planes} planes, got {len(P)}")
    return np.stack(P, axis=0).astype(np.float32)


def _bitboard_to_plane(bb: chess.Bitboard) -> np.ndarray:
    plane = np.zeros((8, 8), dtype=np.float32)
    for sq in bb:
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        plane[r, c] = 1.0
    return plane


# --- 4672-action mapping ---

POLICY_SHAPE = (8, 8, 73)

RAY_DIRS: Tuple[Tuple[int, int], ...] = (
    (1, 0),   # north (toward higher rank)
    (-1, 0),  # south
    (0, 1),   # east
    (0, -1),  # west
    (1, 1),   # northeast
    (1, -1),  # northwest
    (-1, 1),  # southeast
    (-1, -1), # southwest
)
KNIGHT_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)
)
UNDERPROMO_PIECES: Tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)  # order matters


def _is_on_board(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _ray_offset(dr: int, df: int, steps: int) -> Optional[int]:
    try:
        d = RAY_DIRS.index((dr, df))
    except ValueError:
        return None
    if not (1 <= steps <= 7):
        return None
    return d * 7 + (steps - 1)  # 0..55


def _knight_offset(dr: int, df: int) -> Optional[int]:
    try:
        return KNIGHT_DELTAS.index((dr, df))  # 0..7
    except ValueError:
        return None


def _underpromo_offset(board: chess.Board, dr: int, df: int, promo: Optional[int]) -> Optional[int]:
    if promo not in UNDERPROMO_PIECES:
        return None
    # Directions relative to side to move
    if board.turn == chess.WHITE:
        dirs = ((1, 0), (1, -1), (1, 1))
    else:
        dirs = ((-1, 0), (-1, 1), (-1, -1))
    try:
        dir_idx = dirs.index((dr, df))  # 0..2
    except ValueError:
        return None
    piece_idx = UNDERPROMO_PIECES.index(promo)  # 0..2
    return piece_idx * 3 + dir_idx  # 0..8


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """
    Map a python-chess Move to [0, 4671] using fixed 64×73 layout.
    Queen promotions are encoded via ray moves; underpromotions occupy the last 9 slots.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)
    tr, tf = chess.square_rank(to_sq), chess.square_file(to_sq)
    dr, df = tr - fr, tf - ff

    # Knight
    k = _knight_offset(dr, df)
    if k is not None:
        return from_sq * 73 + (56 + k)

    # Underpromotions (N,B,R) in 3 forward directions
    if move.promotion in UNDERPROMO_PIECES:
        u = _underpromo_offset(board, dr, df, move.promotion)
        if u is not None:
            return from_sq * 73 + (64 + u)

    # Rays (includes castling, en passant, queen promotions as regular moves)
    if dr == 0 or df == 0 or abs(dr) == abs(df):
        step = max(abs(dr), abs(df))
        sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
        sdf = 0 if df == 0 else (1 if df > 0 else -1)
        ro = _ray_offset(sdr, sdf, step)
        if ro is not None:
            return from_sq * 73 + ro

    # Fallback: best-effort map to first legal move from the square
    for m in board.legal_moves:
        if m.from_square == from_sq:
            return move_to_index(board, m)
    return 0


@dataclass
class MoveEncoder:
    """Helper with encode/decode utilities and masks."""
    
    def __post_init__(self):
        self._cache: Dict[Tuple[str, chess.Move], int] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def encode_move(self, board: chess.Board, move: chess.Move) -> int:
        cache_key = (board.fen(), move)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        result = move_to_index(board, move)
        self._cache[cache_key] = result
        return result

    def decode_move(self, board: chess.Board, action_idx: int) -> chess.Move:
        if not (0 <= action_idx < 4672):
            raise ValueError("action_idx out of range")
        from_sq = action_idx // 73
        off = action_idx % 73
        fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)

        def mk(dr: int, df: int, steps: int = 1, promo: Optional[int] = None) -> chess.Move:
            tr = fr + dr * steps
            tf = ff + df * steps
            if not _is_on_board(tr, tf):
                return chess.Move.null()
            to_sq = chess.square(tf, tr)
            # Auto queen promotion for pawn reaching last rank on ray move
            p = promo
            if promo is None:
                piece = board.piece_at(from_sq)
                if piece and piece.piece_type == chess.PAWN and (tr == 0 or tr == 7):
                    p = chess.QUEEN
            return chess.Move(from_sq, to_sq, p)

        if off < 56:
            d = off // 7
            s = (off % 7) + 1
            dr, df = RAY_DIRS[d]
            mv = mk(dr, df, s)
        elif off < 64:
            d = off - 56
            dr, df = KNIGHT_DELTAS[d]
            mv = mk(dr, df)
        else:
            u = off - 64
            piece_idx = u // 3
            dir_idx = u % 3
            if board.turn == chess.WHITE:
                dirs = ((1, 0), (1, -1), (1, 1))
            else:
                dirs = ((-1, 0), (-1, 1), (-1, -1))
            dr, df = dirs[dir_idx]
            promo = UNDERPROMO_PIECES[piece_idx]
            mv = mk(dr, df, 1, promo)

        if mv in board.legal_moves:
            return mv
        # If illegal, try to find a legal move with same destination
        for lm in board.legal_moves:
            if lm.from_square == from_sq and lm.to_square == mv.to_square:
                return lm
        # Fallback: any legal move from from_sq
        for lm in board.legal_moves:
            if lm.from_square == from_sq:
                return lm
        return chess.Move.null()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total_requests, 1)
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }

    def get_legal_actions(self, board: chess.Board) -> np.ndarray:
        mask = np.zeros(4672, dtype=bool)
        for m in board.legal_moves:
            mask[move_to_index(board, m)] = True
        return mask

    def validate_encoding(self, board: chess.Board) -> bool:
        try:
            for m in board.legal_moves:
                a = self.encode_move(board, m)
                m2 = self.decode_move(board, a)
                if m.from_square != m2.from_square or m.to_square != m2.to_square:
                    return False
        except Exception:
            return False
        return True

    def get_action_statistics(self, board: chess.Board) -> Dict[str, int | float]:
        mask = self.get_legal_actions(board)
        return {
            "total_actions": 4672,
            "legal_actions": int(mask.sum()),
            "illegal_actions": int((~mask).sum()),
            "legal_ratio": float(mask.sum() / 4672.0),
        }


# Global instance for convenience
move_encoder = MoveEncoder()


def action_map() -> Tuple[Dict[chess.Move, int], List[Tuple[int, int, Optional[int]]]]:
    """Legacy helper: return empty dict and reverse list of (from,to,promo)."""
    rev: List[Tuple[int, int, Optional[int]]] = []
    for from_sq in range(64):
        fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)
        # Rays
        for d in range(8):
            dr, df = RAY_DIRS[d]
            for s in range(1, 8):
                tr = fr + dr * s
                tf = ff + df * s
                if _is_on_board(tr, tf):
                    rev.append((from_sq, chess.square(tf, tr), None))
                else:
                    rev.append((from_sq, from_sq, None))
        # Knights
        for dr, df in KNIGHT_DELTAS:
            tr = fr + dr
            tf = ff + df
            if _is_on_board(tr, tf):
                rev.append((from_sq, chess.square(tf, tr), None))
            else:
                rev.append((from_sq, from_sq, None))
        # Underpromotions (order: N,B,R) × 3 dirs; leave to-square as from (direction is side-dependent)
        for _ in range(9):
            rev.append((from_sq, from_sq, None))
    assert len(rev) == 4672
    return {}, rev


def build_horizontal_flip_permutation() -> np.ndarray:
    """Permutation indices (length 73) to horizontally mirror the 73 move-type channels.

    Ordering (per from-square):
      - 56 ray channels in blocks of 7 for directions: N,S,E,W,NE,NW,SE,SW
      - 8 knight channels
      - 9 underpromotion channels (3 pieces × 3 directions, side-relative)
    """
    perm = list(range(73))
    # Swap E<->W; NE<->NW; SE<->SW for each step (0..6)
    for step in range(7):
        base = step
        # E (2) <-> W (3)
        i_e = 2 * 7 + base
        i_w = 3 * 7 + base
        perm[i_e], perm[i_w] = perm[i_w], perm[i_e]
        # NE (4) <-> NW (5)
        i_ne = 4 * 7 + base
        i_nw = 5 * 7 + base
        perm[i_ne], perm[i_nw] = perm[i_nw], perm[i_ne]
        # SE (6) <-> SW (7)
        i_se = 6 * 7 + base
        i_sw = 7 * 7 + base
        perm[i_se], perm[i_sw] = perm[i_sw], perm[i_se]
    # Knights swap mirrored pairs: (56<->57), (58<->59), (60<->61), (62<->63)
    for off in (0, 2, 4, 6):
        a = 56 + off
        b = 56 + off + 1
        perm[a], perm[b] = perm[b], perm[a]
    # Underpromotions: swap right/left capture per piece block: (65<->66), (68<->69), (71<->72)
    for piece_block in (64, 67, 70):
        a = piece_block + 1
        b = piece_block + 2
        perm[a], perm[b] = perm[b], perm[a]
    # Ensure the returned array is contiguous
    return np.ascontiguousarray(perm, dtype=np.int64)


def build_rotate180_permutation() -> np.ndarray:
    """Permutation indices (length 73) to rotate the 73 move-type channels by 180 degrees.

    Mapping:
      - Rays: N<->S, E<->W, NE<->SW, NW<->SE (per step)
      - Knights: (-2,-1)<->(2,1), (-2,1)<->(2,-1), (-1,-2)<->(1,2), (-1,2)<->(1,-2)
      - Underpromotions: forward stays forward; left<->right per piece block
    """
    perm = list(range(73))
    # Rays swap pairs for each step
    for step in range(7):
        base = step
        # N (0) <-> S (1)
        i_n = 0 * 7 + base
        i_s = 1 * 7 + base
        perm[i_n], perm[i_s] = perm[i_s], perm[i_n]
        # E (2) <-> W (3)
        i_e = 2 * 7 + base
        i_w = 3 * 7 + base
        perm[i_e], perm[i_w] = perm[i_w], perm[i_e]
        # NE (4) <-> SW (7)
        i_ne = 4 * 7 + base
        i_sw = 7 * 7 + base
        perm[i_ne], perm[i_sw] = perm[i_sw], perm[i_ne]
        # NW (5) <-> SE (6)
        i_nw = 5 * 7 + base
        i_se = 6 * 7 + base
        perm[i_nw], perm[i_se] = perm[i_se], perm[i_nw]
    # Knights swap as 180-rotated pairs
    pairs = ((56, 63), (57, 62), (58, 61), (59, 60))
    for a, b in pairs:
        perm[a], perm[b] = perm[b], perm[a]
    # Underpromotions: forward unchanged; swap left/right per piece block
    for piece_block in (64, 67, 70):
        a = piece_block + 1  # left
        b = piece_block + 2  # right
        perm[a], perm[b] = perm[b], perm[a]
    # Ensure the returned array is contiguous
    return np.ascontiguousarray(perm, dtype=np.int64)
