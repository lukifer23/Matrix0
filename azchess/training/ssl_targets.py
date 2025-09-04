from __future__ import annotations

"""
Lightweight SSL target generators for tactical curriculum tasks.

These functions create per-square targets for:
  - piece: 13-class per square (empty + 12 pieces)
  - control: squares controlled by side to move (binary per square)
  - threat: own pieces under attack by opponent (binary per square)
  - pin: own pinned pieces (binary per square)
  - fork: own pieces attacked by 2+ opponent pieces (binary per square)
  - pawn_structure: 8-channels of simple pawn structure features
  - king_safety: 3-class safety in king zone (per square; one-hot 3 channels)

Outputs are numpy float32 arrays with shapes compatible with SSL heads
in azchess.model.resnet.PolicyValueNet.
"""

from typing import Dict, Tuple, List
import numpy as np
import chess


PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _plane() -> np.ndarray:
    return np.zeros((8, 8), dtype=np.float32)


def piece_targets(board: chess.Board) -> np.ndarray:
    """(13,8,8): one-hot piece map, 12 pieces + 1 empty as class 0 last channel.
    Channel order matches model's piece SSL head expectation.
    """
    planes: List[np.ndarray] = []
    occ = np.zeros((8, 8), dtype=np.float32)
    for color in (chess.WHITE, chess.BLACK):
        for pt in PIECE_TYPES:
            pl = _plane()
            for sq in board.pieces(pt, color):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                pl[r, c] = 1.0
                occ[r, c] = 1.0
            planes.append(pl)
    empty = (1.0 - occ).clip(0.0, 1.0)
    planes.append(empty)
    return np.stack(planes, axis=0).astype(np.float32)


def control_targets(board: chess.Board) -> np.ndarray:
    """(1,8,8): squares controlled by side-to-move (binary)."""
    side = board.turn
    ctrl = _plane()
    for sq in chess.SQUARES:
        tr = 7 - chess.square_rank(sq)
        tc = chess.square_file(sq)
        if board.is_attacked_by(side, sq):
            ctrl[tr, tc] = 1.0
    return ctrl[np.newaxis, ...]


def threat_targets(board: chess.Board) -> np.ndarray:
    """(1,8,8): own pieces currently under attack by opponent."""
    side = board.turn
    opp = not side
    tgt = _plane()
    for sq in chess.SQUARES:
        if board.piece_at(sq) is not None and board.piece_at(sq).color == side:
            if board.is_attacked_by(opp, sq):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                tgt[r, c] = 1.0
    return tgt[np.newaxis, ...]


def pin_targets(board: chess.Board) -> np.ndarray:
    """(1,8,8): own pieces pinned to own king by opponent (simple heuristic)."""
    side = board.turn
    king_sq = board.king(side)
    tgt = _plane()
    if king_sq is None:
        return tgt[np.newaxis, ...]
    # A piece is pinned if moving it would expose king to attack
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == side and piece.piece_type != chess.KING:
            if board.is_pinned(side, sq):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                tgt[r, c] = 1.0
    return tgt[np.newaxis, ...]


def fork_targets(board: chess.Board) -> np.ndarray:
    """(1,8,8): own pieces attacked by >=2 enemy pieces (approximate fork pressure)."""
    side = board.turn
    opp = not side
    tgt = _plane()
    # Build attack counts from opponent
    attack_counts = np.zeros((8, 8), dtype=np.int32)
    for sq in chess.SQUARES:
        if board.is_attacked_by(opp, sq):
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            attack_counts[r, c] += 1
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == side:
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            if attack_counts[r, c] >= 2:
                tgt[r, c] = 1.0
    return tgt[np.newaxis, ...]


def pawn_structure_targets(board: chess.Board) -> np.ndarray:
    """(8,8,8): simple pawn structure planes.
    Channels:
      0 white pawns, 1 black pawns,
      2 isolated pawns (own), 3 doubled pawns (own),
      4 passed pawns (own), 5 king shield (own 3x3 zone occupancy by pawns),
      6 open files (no pawns), 7 semi-open files for side to move.
    """
    planes = [ _plane() for _ in range(8) ]
    # Basic pawn maps
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        planes[0][r, c] = 1.0
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        planes[1][r, c] = 1.0
    # Files info
    file_wp = planes[0].sum(axis=0)
    file_bp = planes[1].sum(axis=0)
    # Isolated: no friendly pawns on adjacent files
    side = board.turn
    own = planes[0] if side == chess.WHITE else planes[1]
    opp = planes[1] if side == chess.WHITE else planes[0]
    for r in range(8):
        for c in range(8):
            if own[r, c] > 0.5:
                left = own[r, max(0, c-1):c]
                right = own[r, c+1:min(8, c+2)]
                if left.sum() + right.sum() == 0:
                    planes[2][r, c] = 1.0
    # Doubled: >1 pawn of same color on file
    for c in range(8):
        if file_wp[c] >= 2:
            planes[3][:, c] = np.maximum(planes[3][:, c], planes[0][:, c])
        if file_bp[c] >= 2:
            planes[3][:, c] = np.maximum(planes[3][:, c], planes[1][:, c])
    # Passed pawn: no opposing pawns ahead on same/adjacent files
    for sq in board.pieces(chess.PAWN, side):
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        passed = True
        if side == chess.WHITE:
            rows = slice(0, r)
        else:
            rows = slice(r+1, 8)
        cols = slice(max(0, c-1), min(8, c+2))
        if (opp[rows, cols] > 0.5).any():
            passed = False
        if passed:
            planes[4][r, c] = 1.0
    # King shield: pawns in 3x3 around own king
    ks = board.king(side)
    if ks is not None:
        kr = 7 - chess.square_rank(ks)
        kc = chess.square_file(ks)
        for r in range(max(0, kr-1), min(8, kr+2)):
            for c in range(max(0, kc-1), min(8, kc+2)):
                planes[5][r, c] = own[r, c]
    # Open files (no pawns at all)
    for c in range(8):
        if file_wp[c] + file_bp[c] == 0:
            planes[6][:, c] = 1.0
    # Semi-open for side to move (no own pawns on file)
    own_file = file_wp if side == chess.WHITE else file_bp
    for c in range(8):
        if own_file[c] == 0:
            planes[7][:, c] = 1.0
    return np.stack(planes, axis=0).astype(np.float32)


def king_safety_targets(board: chess.Board) -> np.ndarray:
    """(3,8,8): king zone safety one-hot: 0 safe, 1 under attack, 2 heavy attack (>=2 attackers).
    Values are only populated in squares around own king; elsewhere all zeros.
    """
    side = board.turn
    ks = board.king(side)
    planes = [ _plane() for _ in range(3) ]
    if ks is None:
        return np.stack(planes, axis=0)
    kr = 7 - chess.square_rank(ks)
    kc = chess.square_file(ks)
    for r in range(max(0, kr-1), min(8, kr+2)):
        for c in range(max(0, kc-1), min(8, kc+2)):
            sq = chess.square(c, 7 - r)
            attackers = len(list(board.attackers(not side, sq)))
            if attackers == 0:
                planes[0][r, c] = 1.0
            elif attackers == 1:
                planes[1][r, c] = 1.0
            else:
                planes[2][r, c] = 1.0
    return np.stack(planes, axis=0).astype(np.float32)


def generate_ssl_targets(board: chess.Board) -> Dict[str, np.ndarray]:
    """Generate a dictionary of SSL targets for a single board position."""
    return {
        'piece': piece_targets(board),
        'control': control_targets(board),
        'threat': threat_targets(board),
        'pin': pin_targets(board),
        'fork': fork_targets(board),
        'pawn_structure': pawn_structure_targets(board),
        'king_safety': king_safety_targets(board),
    }
def decode_board_from_planes(planes: np.ndarray) -> chess.Board:
    """Reconstruct a chess.Board from 19-plane encoding.

    planes shape: (19,8,8). Order:
      0..11: piece planes (white then black; P,N,B,R,Q,K)
      12: side to move
      13..16: castling rights (W-K, W-Q, B-K, B-Q)
      17..18: halfmove/fullmove (normalized)
    En passant is not reconstructed (set to '-')
    """
    board = chess.Board(None)  # empty board
    # Place pieces
    idx = 0
    colors = (chess.WHITE, chess.BLACK)
    piece_types = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
    for color in colors:
        for pt in piece_types:
            pl = planes[idx]
            idx += 1
            for r in range(8):
                for c in range(8):
                    if pl[r, c] > 0.5:
                        sq = chess.square(c, 7 - r)
                        board.set_piece_at(sq, chess.Piece(pt, color))
    # Side to move
    board.turn = bool(planes[12, 0, 0] > 0.5)
    # Castling rights
    rights = 0
    if planes[13, 0, 0] > 0.5:
        rights |= chess.BB_H1  # WK castling uses rook/king squares bitboard later
        board.castling_rights |= chess.CASTLING_WHITE_KINGSIDE
    if planes[14, 0, 0] > 0.5:
        board.castling_rights |= chess.CASTLING_WHITE_QUEENSIDE
    if planes[15, 0, 0] > 0.5:
        board.castling_rights |= chess.CASTLING_BLACK_KINGSIDE
    if planes[16, 0, 0] > 0.5:
        board.castling_rights |= chess.CASTLING_BLACK_QUEENSIDE
    # Move counters (approximate from normalized planes)
    try:
        board.halfmove_clock = int(float(planes[17, 0, 0]) * 99)
        board.fullmove_number = max(1, int(float(planes[18, 0, 0]) * 199))
    except Exception:
        pass
    return board


def generate_ssl_targets_from_states(states: np.ndarray) -> Dict[str, np.ndarray]:
    """Batch wrapper: generate SSL targets for a batch of encoded board states.

    Args:
        states: (B,19,8,8) numpy float32 array

    Returns:
        Dict of task -> (B,C,H,W) numpy arrays
    """
    B = states.shape[0]
    # Gather per-sample dicts then stack by key
    out_per_key: Dict[str, List[np.ndarray]] = {}
    for b in range(B):
        board = decode_board_from_planes(states[b])
        targets = generate_ssl_targets(board)
        for k, v in targets.items():
            out_per_key.setdefault(k, []).append(v)
    # Stack along batch dimension
    batched: Dict[str, np.ndarray] = {}
    for k, lst in out_per_key.items():
        batched[k] = np.stack(lst, axis=0).astype(np.float32)
    return batched
