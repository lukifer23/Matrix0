import chess
import numpy as np
import pytest
import torch

from azchess.mcts import MCTS, MCTSConfig, Node, move_to_index


class DummyBackend:
    def __init__(self, logits: np.ndarray, value: float):
        self.logits = logits.astype(np.float32, copy=False)
        self.value = float(value)

    def infer_np(self, arr_batch: np.ndarray):
        batch_size = arr_batch.shape[0]
        logits_batch = np.repeat(self.logits[None, :], batch_size, axis=0)
        values = np.full((batch_size,), self.value, dtype=np.float32)
        return logits_batch, values


def _assert_child_priors(board: chess.Board, node: Node, logits: np.ndarray) -> None:
    legal_moves = list(board.legal_moves)
    indices = [move_to_index(board, mv) for mv in legal_moves]
    legal_logits = logits[indices]
    expected = torch.softmax(torch.from_numpy(legal_logits), dim=-1).numpy()

    for i, move in enumerate(legal_moves):
        child = node.children[move]
        assert child.prior == pytest.approx(float(expected[i]), rel=1e-6, abs=1e-6)


def test_mcts_priors_match_logits_with_global_softmax():
    board = chess.Board()
    logits = np.linspace(-3.0, 4.0, 4672, dtype=np.float32)
    backend = DummyBackend(logits, value=0.25)

    cfg = MCTSConfig(num_simulations=0)
    cfg.enable_entropy_noise = False
    mcts = MCTS(cfg, None, device="cpu", inference_backend=backend)

    policy_logits, value = mcts._infer(board)
    assert np.allclose(policy_logits, logits)
    assert value == pytest.approx(0.25, rel=1e-6)

    node = Node()
    node._expand(
        board,
        policy_logits,
        encoder=mcts._enc,
        legal_only=False,
        allow_noise=False,
    )

    _assert_child_priors(board, node, logits)


def test_mcts_priors_match_logits_with_legal_softmax():
    board = chess.Board()
    logits = np.linspace(2.0, -2.0, 4672, dtype=np.float32)
    backend = DummyBackend(logits, value=-0.1)

    cfg = MCTSConfig(num_simulations=0, legal_softmax=True)
    cfg.enable_entropy_noise = False
    mcts = MCTS(cfg, None, device="cpu", inference_backend=backend)

    policy_logits, value = mcts._infer(board)
    assert np.allclose(policy_logits, logits)
    assert value == pytest.approx(-0.1, rel=1e-6)

    node = Node()
    node._expand(
        board,
        policy_logits,
        encoder=mcts._enc,
        legal_only=True,
        allow_noise=False,
    )

    _assert_child_priors(board, node, logits)
