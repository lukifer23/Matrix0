import chess
import numpy as np

from azchess.encoding import move_to_index


def random_board(plies: int = 20, seed: int = 0) -> chess.Board:
    rng = np.random.default_rng(seed)
    b = chess.Board()
    for _ in range(plies):
        if b.is_game_over():
            break
        moves = list(b.legal_moves)
        if not moves:
            break
        mv = moves[int(rng.integers(0, len(moves)))]
        b.push(mv)
    return b


def test_random_uniqueness():
    # Ensure legal moves map to unique indices on random boards
    for s in range(5):
        b = random_board(plies=10 + s, seed=1234 + s)
        legals = list(b.legal_moves)
        idxs = [move_to_index(b, m) for m in legals]
        assert len(set(idxs)) == len(idxs)
        for i in idxs:
            assert 0 <= i < 64 * 73

