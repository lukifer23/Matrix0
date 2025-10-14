import chess
import numpy as np
import pytest

from azchess.mcts import MCTS, MCTSConfig


class ConstantBackend:
    """Inference backend that returns uniform logits and a fixed value."""

    def __init__(self, value: float):
        self.logits = np.zeros((4672,), dtype=np.float32)
        self.value = float(value)

    def infer_np(self, batch: np.ndarray):
        batch_size = batch.shape[0]
        logits = np.repeat(self.logits[None, :], batch_size, axis=0)
        values = np.full((batch_size,), self.value, dtype=np.float32)
        return logits, values


@pytest.mark.parametrize("fixed_value", [-0.35, 0.42])
def test_parallel_batched_backprop_restores_network_value(fixed_value: float):
    backend = ConstantBackend(value=fixed_value)
    cfg = MCTSConfig(
        num_simulations=16,
        dirichlet_frac=0.0,
        enable_entropy_noise=False,
        batch_size=4,
    )

    # Force threaded execution to exercise the batched parallel path
    mcts = MCTS(cfg, None, device="cpu", inference_backend=backend, num_threads=2)

    board = chess.Board()
    visits, _, root_q = mcts.run(board)

    # Root value should reflect the network evaluation, not default to zero
    assert root_q == pytest.approx(fixed_value, rel=1e-4, abs=1e-4)
    assert mcts._last_root is not None
    assert mcts._last_root.q == pytest.approx(fixed_value, rel=1e-4, abs=1e-4)

    # Ensure simulations actually updated visit statistics
    total_visits = sum(visits.values())
    assert total_visits == cfg.num_simulations

