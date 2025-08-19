import numpy as np
import chess
import torch
import pytest

from azchess.mcts import MCTS, MCTSConfig, Node


class DummyModel(torch.nn.Module):
    def forward(self, x, return_ssl=False):
        batch = x.shape[0]
        p = torch.zeros((batch, 4672), dtype=torch.float32)
        v = torch.zeros((batch, 1), dtype=torch.float32)
        return p, v


def test_ucb_selection():
    cfg = MCTSConfig(cpuct=1.0, selection_jitter=0.0, dirichlet_frac=0.0, batch_size=1)
    mcts = MCTS(DummyModel(), cfg=cfg, device="cpu")
    board = chess.Board()

    move1 = chess.Move.from_uci('e2e4')
    move2 = chess.Move.from_uci('d2d4')
    root = Node()
    root.n = 3
    root.expanded = True
    child1 = Node(prior=0.5, move=move1, parent=root)
    child1.n = 1
    child1.w = 0.5
    child1.q = 0.5
    child2 = Node(prior=0.5, move=move2, parent=root)
    child2.n = 2
    child2.w = 0.6
    child2.q = 0.3
    root.children = {move1: child1, move2: child2}

    node, path, _ = mcts._select(board.copy(), root)
    assert node is child1
    assert path[-1] is child1


def test_batch_inference(monkeypatch):
    cfg = MCTSConfig(num_simulations=4, cpuct=1.0, dirichlet_frac=0.0, selection_jitter=0.0, batch_size=2, fpu=0.0, parent_q_init=False)
    mcts = MCTS(DummyModel(), cfg=cfg, device="cpu")
    board = chess.Board()

    def fake_infer(self, b):
        return np.zeros(4672, dtype=np.float32), 0.0

    batch_sizes = []

    def fake_infer_batch(self, boards):
        batch_sizes.append(len(boards))
        p_list = [np.zeros(4672, dtype=np.float32) for _ in boards]
        v_list = [0.0 for _ in boards]
        return p_list, v_list

    monkeypatch.setattr(MCTS, "_infer", fake_infer)
    monkeypatch.setattr(MCTS, "_infer_batch", fake_infer_batch)

    mcts.run(board, num_simulations=4)
    assert any(size > 1 for size in batch_sizes)
