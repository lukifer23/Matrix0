import chess
import numpy as np

from azchess.encoding import (POLICY_SHAPE, build_horizontal_flip_permutation,
                              move_to_index)
from azchess.utils.board import random_board


def test_random_move_indices_unique():
    """Ensure legal moves map to unique indices on random boards."""
    for s in range(5):
        b = random_board(plies=10 + s)
        legals = list(b.legal_moves)
        idxs = [move_to_index(b, m) for m in legals]
        assert len(set(idxs)) == len(idxs)
        for i in idxs:
            assert 0 <= i < 64 * 73


def test_horizontal_flip_permutation_involution():
    """Applying the permutation twice should return identity."""
    perm = build_horizontal_flip_permutation()
    assert perm.shape[0] == POLICY_SHAPE[2]
    pi = np.arange(POLICY_SHAPE[2], dtype=np.int32)
    pi1 = pi[perm]
    pi2 = pi1[perm]
    assert np.array_equal(pi, pi2)

