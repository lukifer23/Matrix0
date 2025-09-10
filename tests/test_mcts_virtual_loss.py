import chess
import torch
import logging
from concurrent.futures import ThreadPoolExecutor

from experiments.grpo.mcts.mcts_integration import MCTS, MCTSConfig, MCTSNode

logging.raiseExceptions = False
root_logger = logging.getLogger()
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        p = torch.ones((batch, 4672), dtype=torch.float32)
        v = torch.zeros((batch, 1), dtype=torch.float32)
        return p, v


def test_virtual_loss_reduces_leaf_collisions():
    model = DummyModel()
    cfg = MCTSConfig(num_simulations=1, batch_size=1)
    mcts = MCTS(model, cfg)
    board = chess.Board()

    # Prepare root with only two children to force contention
    root = MCTSNode(board=board.copy())
    policy_logits, _ = mcts._evaluate_position(board)
    legal_moves = list(board.legal_moves)[:2]
    root.expand(legal_moves, policy_logits, cfg)

    # Run two simulations in parallel; without virtual loss they would collide
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(mcts._simulate, root) for _ in range(2)]
        for f in futures:
            f.result()

    visits = [child.visit_count for child in root.children.values()]
    assert sorted(visits) == [1, 1], f"Unexpected visit counts: {visits}"
