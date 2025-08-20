from __future__ import annotations

from typing import Sequence, Dict, Any
import chess


def should_adjudicate_draw(board: chess.Board, moves: Sequence[chess.Move], cfg: Dict[str, Any]) -> bool:
    """Determine if the current game should be adjudicated as a draw.

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

    The function always checks standard draw rules (threefold repetition,
    fifty-move rule, insufficient material). When ``enabled`` it additionally
    applies heuristic early draw adjudication to avoid marathon games.
    """
    # Standard draw conditions
    if board.is_repetition(3) or board.can_claim_threefold_repetition():
        return True
    if board.can_claim_fifty_moves():
        return True
    if board.is_insufficient_material():
        return True

    if not bool(cfg.get("enabled", False)):
        return False

    min_plies = int(cfg.get("min_plies", 0))
    if len(moves) < min_plies:
        return False

    window = int(cfg.get("window", 0))
    min_unique = int(cfg.get("min_unique", 0))
    if window > 0 and min_unique > 0:
        recent = moves[-window:]
        if len(set(recent)) < min_unique:
            return True

    halfmove_cap = int(cfg.get("halfmove_cap", 0))
    if halfmove_cap and board.halfmove_clock >= halfmove_cap:
        return True

    return False
