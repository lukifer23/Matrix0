import importlib
import sys
from unittest.mock import patch

import chess
import torch


def _get_mcts_module_without_psutil():
    with patch.dict(sys.modules, {'psutil': None}):
        if 'azchess.mcts' in sys.modules:
            del sys.modules['azchess.mcts']
        module = importlib.import_module('azchess.mcts')
        importlib.reload(module)
        return module


class DummyModel(torch.nn.Module):
    def forward(self, x, return_ssl=False):
        batch = x.shape[0]
        p = torch.zeros((batch, 4672), dtype=torch.float32)
        v = torch.zeros((batch, 1), dtype=torch.float32)
        return p, v


def test_mcts_runs_without_psutil():
    mcts_mod = _get_mcts_module_without_psutil()
    assert not mcts_mod.psutil_available

    model = DummyModel()
    cfg = mcts_mod.MCTSConfig(num_simulations=1, batch_size=1)
    mcts = mcts_mod.MCTS(model, cfg)
    board = chess.Board()
    moves, policy, value = mcts.run(board, num_simulations=1)

    assert isinstance(moves, dict)
    assert policy.shape == (4672,)
    assert isinstance(value, float)
