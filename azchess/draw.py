from __future__ import annotations

from typing import Sequence, Dict, Any
import chess


def should_adjudicate_draw(board: chess.Board, moves: Sequence[chess.Move], cfg: Dict[str, Any]) -> bool:
    """Enhanced draw adjudication with performance optimizations and better heuristics.

    Parameters
    ----------
    board: chess.Board
        The current board state.
    moves: Sequence[chess.Move]
        Sequence of moves played so far.
    cfg: Dict[str, Any]
        Configuration dictionary controlling the heuristics. Expected keys:
        - ``enabled`` (bool): activate heuristic checks.
        - ``min_plies`` (int): minimum plies before heuristics apply.
        - ``window`` (int): size of the repetition window.
        - ``min_unique`` (int): minimum unique moves within the window.
        - ``halfmove_cap`` (int): optional cap for the halfmove clock.
        - ``material_draw_threshold`` (int): material count threshold for draw.
        - ``stalemate_draw`` (bool): treat stalemate as draw.

    The function always checks standard draw rules (threefold repetition,
    fifty-move rule, insufficient material). When ``enabled`` it additionally
    applies heuristic early draw adjudication to avoid marathon games.
    """
    # Standard draw conditions - optimized order for performance
    if board.is_insufficient_material():
        return True
    if board.can_claim_fifty_moves():
        return True
    if board.is_repetition(3) or board.can_claim_threefold_repetition():
        return True

    # Check for stalemate if configured
    if bool(cfg.get("stalemate_draw", True)) and board.is_stalemate():
        return True

    if not bool(cfg.get("enabled", False)):
        return False

    min_plies = int(cfg.get("min_plies", 30))  # Default to 15 moves
    if len(moves) < min_plies:
        return False

    # Enhanced repetition detection with sliding window
    window = int(cfg.get("window", 12))  # Last 6 moves
    min_unique = int(cfg.get("min_unique", 3))
    if window > 0 and min_unique > 0 and len(moves) >= window:
        recent = moves[-window:]
        unique_moves = len(set(str(m) for m in recent))
        if unique_moves < min_unique:
            return True

    # Halfmove clock cap
    halfmove_cap = int(cfg.get("halfmove_cap", 50))
    if halfmove_cap and board.halfmove_clock >= halfmove_cap:
        return True

    # Material-based draw detection
    material_threshold = int(cfg.get("material_draw_threshold", 10))
    if material_threshold > 0:
        white_material = sum([
            len(board.pieces(chess.PAWN, chess.WHITE)),
            len(board.pieces(chess.KNIGHT, chess.WHITE)) * 3,
            len(board.pieces(chess.BISHOP, chess.WHITE)) * 3,
            len(board.pieces(chess.ROOK, chess.WHITE)) * 5,
            len(board.pieces(chess.QUEEN, chess.WHITE)) * 9
        ])
        black_material = sum([
            len(board.pieces(chess.PAWN, chess.BLACK)),
            len(board.pieces(chess.KNIGHT, chess.BLACK)) * 3,
            len(board.pieces(chess.BISHOP, chess.BLACK)) * 3,
            len(board.pieces(chess.ROOK, chess.BLACK)) * 5,
            len(board.pieces(chess.QUEEN, chess.BLACK)) * 9
        ])
        if white_material + black_material <= material_threshold:
            return True

    return False
