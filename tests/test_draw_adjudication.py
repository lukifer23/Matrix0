import chess

from azchess.draw import should_adjudicate_draw


def _threefold_board() -> tuple[chess.Board, list[chess.Move]]:
    board = chess.Board()
    moves: list[chess.Move] = []
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        move = chess.Move.from_uci(uci)
        board.push(move)
        moves.append(move)
    return board, moves


def test_repetition_claim_can_be_delayed_by_min_plies():
    board, moves = _threefold_board()

    assert should_adjudicate_draw(board, moves, {"claim_min_plies": 0}) is True
    assert should_adjudicate_draw(board, moves, {"claim_min_plies": 120}) is False


def test_repetition_claim_can_be_disabled():
    board, moves = _threefold_board()

    assert should_adjudicate_draw(board, moves, {"claim_repetition": False}) is False


def test_halfmove_cap_still_applies_when_heuristics_enabled():
    board = chess.Board()
    board.halfmove_clock = 100

    assert should_adjudicate_draw(
        board,
        [chess.Move.from_uci("g1f3")] * 120,
        {
            "enabled": True,
            "claim_repetition": False,
            "claim_fifty_moves": False,
            "halfmove_cap": 100,
            "min_plies": 120,
            "window": 0,
            "min_unique": 0,
            "material_draw_threshold": 0,
        },
    ) is True
