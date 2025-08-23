import chess
import pytest

from webui import server
from webui.server import NewGameRequest, HTTPException


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    monkeypatch.setattr(server, "_load_matrix0", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_jsonl_write", lambda *args, **kwargs: None)
    server.GAMES.clear()


def test_new_game_valid_fen():
    req = NewGameRequest(fen=chess.STARTING_FEN)
    resp = server.new_game(req)
    assert resp["fen"] == chess.STARTING_FEN


def test_new_game_invalid_fen():
    req = NewGameRequest(fen="not a fen")
    with pytest.raises(HTTPException) as exc:
        server.new_game(req)
    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid FEN"
