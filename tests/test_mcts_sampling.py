import math
import numpy as np
import chess
import torch
import pytest

from azchess.mcts import MCTS, MCTSConfig

def test_fallback_sampling_matches_torch(monkeypatch):
    mcts = MCTS(model=None, config=MCTSConfig(), device="cpu")
    board = chess.Board()
    legal_moves = list(board.legal_moves)[:2]

    policy = torch.zeros(4672)
    for move in legal_moves:
        idx = mcts._move_to_index(move)
        policy[idx] = 1.0

    torch.manual_seed(0)
    np.random.seed(0)
    move_torch, log_prob_torch = mcts._sample_move_from_policy(policy, legal_moves)

    def fail_multinomial(*args, **kwargs):
        raise RuntimeError("forced error")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)

    torch.manual_seed(0)
    np.random.seed(0)
    move_np, log_prob_np = mcts._sample_move_from_policy(policy, legal_moves)

    assert move_np in legal_moves
    assert math.isclose(log_prob_np, log_prob_torch)
    assert math.isclose(log_prob_np, math.log(0.5))
