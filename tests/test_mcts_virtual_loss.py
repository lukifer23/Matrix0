import chess
import torch
import logging
from concurrent.futures import ThreadPoolExecutor

from azchess.mcts import MCTS, MCTSConfig, Node

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
    root = Node()
    policy_logits, _ = mcts._infer(board)
    legal_moves = list(board.legal_moves)[:2]
    # Create a modified policy that only has non-zero values for the first 2 moves
    modified_logits = np.zeros_like(policy_logits)
    for i, move in enumerate(legal_moves):
        # Find the policy index for this move and set it to a high value
        modified_logits[i] = 10.0  # Give high prior to first 2 moves
    root._expand(board, modified_logits)

    # Run two simulations in parallel; without virtual loss they would collide
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(mcts._run_simulation, board, root) for _ in range(2)]
        for f in futures:
            f.result()

    visits = [child.n for child in root.children.values()]
    assert sorted(visits) == [1, 1], f"Unexpected visit counts: {visits}"
