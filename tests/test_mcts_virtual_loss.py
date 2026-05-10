import chess
import numpy as np
import torch
import logging

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
    cfg = MCTSConfig(num_simulations=1, batch_size=1, virtual_loss=1.0, selection_jitter=0.0)
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
        modified_logits[mcts._move_to_index(move, board)] = 10.0  # Give high prior to first 2 moves
    root._expand(board, modified_logits)

    inflight = {}
    first_node, _, first_board = mcts._select(board.copy(), root, inflight_counts=inflight)
    second_node, _, second_board = mcts._select(board.copy(), root, inflight_counts=inflight)

    assert first_node.move in legal_moves
    assert second_node.move in legal_moves
    assert first_node.move != second_node.move
    assert first_board.peek() == first_node.move
    assert second_board.peek() == second_node.move


def test_zero_selection_jitter_does_not_call_random(monkeypatch):
    model = DummyModel()
    cfg = MCTSConfig(num_simulations=1, batch_size=1, virtual_loss=0.0, selection_jitter=0.0)
    mcts = MCTS(model, cfg)
    board = chess.Board()
    root = Node()

    legal_moves = list(board.legal_moves)[:2]
    root.children = {
        legal_moves[0]: Node(prior=0.7, move=legal_moves[0], parent=root),
        legal_moves[1]: Node(prior=0.3, move=legal_moves[1], parent=root),
    }
    root.expanded = True
    root.n = 1

    def fail_random():
        raise AssertionError("selection_jitter=0.0 should not call random.random")

    monkeypatch.setattr("azchess.mcts.random.random", fail_random)

    selected, path, selected_board = mcts._select(board.copy(), root)

    assert selected.move == legal_moves[0]
    assert path[-1] is selected
    assert selected_board.peek() == legal_moves[0]
