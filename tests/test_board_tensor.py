import chess
import torch

from azchess.encoding import encode_board


def test_board_to_tensor_consistency():
    """Board encoding should match expected channel semantics."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    tensor = encode_board(board)

    assert tensor.shape == (19, 8, 8)

    # White pawn on e4
    assert tensor[0, 4, 4] == 1.0
    # Black pawn on a7
    assert tensor[6, 1, 0] == 1.0

    # Side to move: black
    assert torch.equal(torch.from_numpy(tensor[12]), torch.zeros(8, 8))

    # All castling rights available
    for idx in range(13, 17):
        assert torch.equal(torch.from_numpy(tensor[idx]), torch.ones(8, 8))

    # Halfmove clock normalized (0/99 = 0.0)
    assert abs(tensor[17].mean() - 0.0) < 1e-6

    # Fullmove clock normalized (1/199 ≈ 0.005)
    assert abs(tensor[18].mean() - 0.005025) < 1e-6

