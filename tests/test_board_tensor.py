import chess
import torch

from experiments.grpo.utils.board_encoding import board_to_tensor


def test_board_to_tensor_consistency():
    """Board encoding should match expected channel semantics."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    tensor = board_to_tensor(board)

    assert tensor.shape == (1, 19, 8, 8)

    # White pawn on e4
    assert tensor[0, 0, 3, 4] == 1.0
    # Black pawn on a7
    assert tensor[0, 1, 6, 0] == 1.0

    # Side to move: black
    assert torch.equal(tensor[0, 12], torch.zeros(8, 8))

    # All castling rights available
    for idx in range(13, 17):
        assert torch.equal(tensor[0, idx], torch.ones(8, 8))

    # En passant square at e3
    assert tensor[0, 17, 2, 4] == 1.0

    # Halfmove clock zero
    assert torch.equal(tensor[0, 18], torch.zeros(8, 8))

