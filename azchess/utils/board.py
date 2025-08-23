from __future__ import annotations

import chess
import numpy as np


def random_board(
    plies: int | None = None,
    *,
    max_plies: int = 40,
    seed: int | None = None,
) -> chess.Board:
    """Generate a random chess board by playing random plies.

    If ``plies`` is provided, exactly that many plies are played. Otherwise a
    random number of plies between 1 and ``max_plies`` (inclusive) is used.

    Args:
        plies: Exact number of plies to play, or ``None`` to choose randomly.
        max_plies: Upper bound for random plies when ``plies`` is ``None``.
        seed: Optional RNG seed for determinism.

    Returns:
        A :class:`chess.Board` after the random plies have been played.
    """

    rng = np.random.default_rng(seed)
    b = chess.Board()
    n = plies if plies is not None else int(rng.integers(1, max_plies + 1))
    for _ in range(n):
        if b.is_game_over():
            break
        moves = list(b.legal_moves)
        if not moves:
            break
        mv = moves[int(rng.integers(0, len(moves)))]
        b.push(mv)
    return b
