import hashlib
from unittest.mock import patch

import chess
import torch

from experiments.grpo.mcts.mcts_integration import MCTS, MCTSConfig


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        policy = torch.zeros((batch, 4672), dtype=torch.float32)
        value = torch.zeros((batch, 1), dtype=torch.float32)
        return policy, value


def _uid(board: chess.Board) -> int:
    return int(hashlib.md5(board.fen().encode()).hexdigest(), 16) % 1000


def test_search_batch_preserves_input_order():
    model = DummyModel()
    cfg = MCTSConfig(num_simulations=0, batch_size=2)
    mcts = MCTS(model, cfg)

    board1 = chess.Board()
    board2 = chess.Board()
    board2.push_san("e4")
    boards = [board1, board2]

    def fake_run(self, root):
        uid = _uid(root.board)
        policy = torch.full((4672,), float(uid))
        value = float(uid)
        return policy, value

    with patch.object(MCTS, "_run_simulations_for_root", fake_run):
        results = mcts.search_batch(boards)

    assert len(results) == len(boards)
    for (policy, value), board in zip(results, boards):
        uid = _uid(board)
        assert policy[0].item() == float(uid)
        assert value == float(uid)
