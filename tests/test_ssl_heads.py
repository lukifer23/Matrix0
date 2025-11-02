import chess
import numpy as np
import torch

from azchess.model.resnet import NetConfig, PolicyValueNet
from azchess.ssl_algorithms import ChessSSLAlgorithms
from azchess.encoding import encode_board


def _make_batch(batch_size: int = 2) -> torch.Tensor:
    sequences = [
        [],
        ["e4", "e5", "Nf3", "Nc6"],
        ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6"],
    ]

    boards = []
    for seq in sequences:
        board = chess.Board()
        for san in seq:
            board.push_san(san)
        boards.append(board)

    if batch_size > len(boards):
        boards.extend(boards[: batch_size - len(boards)])

    tensors = [encode_board(b) for b in boards[:batch_size]]
    return torch.from_numpy(np.stack(tensors)).float()


def test_enhanced_ssl_heads_outputs_and_loss():
    torch.manual_seed(0)
    batch = _make_batch(batch_size=3)

    cfg = NetConfig(
        channels=32,
        blocks=2,
        attention=False,
        chess_features=False,
        self_supervised=True,
        ssl_tasks=["piece", "threat", "pin", "fork", "control"],
    )

    model = PolicyValueNet(cfg)
    model.eval()

    # Forward pass to obtain SSL head outputs and shared features
    _, _, ssl_outputs, feats = model.forward_with_features(batch, return_ssl=True)

    expected_shapes = {
        "piece": (batch.size(0), 13, 8, 8),
        "threat": (batch.size(0), 1, 8, 8),
        "pin": (batch.size(0), 1, 8, 8),
        "fork": (batch.size(0), 1, 8, 8),
        "control": (batch.size(0), 3, 8, 8),
    }

    assert set(ssl_outputs.keys()) == set(expected_shapes.keys())
    for task, expected_shape in expected_shapes.items():
        output = ssl_outputs[task]
        assert output.shape == expected_shape
        assert output.dtype == batch.dtype

    ssl_alg = ChessSSLAlgorithms()
    targets = ssl_alg.create_enhanced_ssl_targets(batch)

    loss = model.get_enhanced_ssl_loss(batch, targets, feats=feats)
    assert torch.isfinite(loss).item()
    assert loss.dtype == torch.float32

    stats_keys = {k for k in model._ssl_loss_stats if k.startswith("task:")}
    assert stats_keys == {f"task:{name}" for name in expected_shapes.keys()}

    # Regression: disabling tasks should immediately reflect in loss tracking
    cfg_subset = NetConfig(
        channels=32,
        blocks=2,
        attention=False,
        chess_features=False,
        self_supervised=True,
        ssl_tasks=["piece", "threat"],
    )

    model_subset = PolicyValueNet(cfg_subset)
    model_subset.eval()

    _, _, _, feats_subset = model_subset.forward_with_features(batch, return_ssl=True)
    model_subset.get_enhanced_ssl_loss(batch, targets, feats=feats_subset)

    subset_stats_keys = {k for k in model_subset._ssl_loss_stats if k.startswith("task:")}
    assert subset_stats_keys == {"task:piece", "task:threat"}
