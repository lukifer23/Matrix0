import chess
import pytest

from webui import server
from webui.server import HTTPException, NewGameRequest


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


def test_system_metrics_endpoint(monkeypatch):
    sample_memory = {"device": "cpu", "memory_gb": 12.5, "available": True}
    monkeypatch.setattr(server.monitor, "get_memory_usage", lambda device='auto': sample_memory)
    monkeypatch.setattr(server, "training_status", lambda: {"is_training": True, "progress": 47.5})
    monkeypatch.setattr(server, "ssl_status", lambda: {"enabled": True, "tasks": ["task1", "task2"]})
    monkeypatch.setattr(server, "list_tournaments", lambda: {"total_active": 3, "total_completed": 1})

    server.GAMES["gid"] = server.GameState(
        game_id="gid",
        created_ts=0.0,
        board=chess.Board(),
        white="human",
        black="matrix0",
        moves=[],
        engine_tc_ms=100,
    )

    metrics = server.system_metrics()

    assert metrics["memory"] == sample_memory
    assert metrics["active_games"] == 1
    assert metrics["training"]["is_training"] is True
    assert metrics["training"]["progress"] == 47.5
    assert metrics["ssl"]["enabled"] is True
    assert metrics["ssl"]["task_count"] == 2
    assert metrics["tournaments"]["active"] == 3
    assert metrics["tournaments"]["completed"] == 1
    assert isinstance(metrics["timestamp"], float)

    server.GAMES.clear()
