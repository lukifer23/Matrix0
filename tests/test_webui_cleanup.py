import psutil
from fastapi.testclient import TestClient

from webui import server

# helper to play fool's mate
FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]


def _play_game(client):
    resp = client.post("/new", json={"white": "human", "black": "human"})
    gid = resp.json()["game_id"]
    for move in FOOLS_MATE:
        client.post("/move", json={"game_id": gid, "uci": move})
    return gid


def test_game_cleanup_after_completion(monkeypatch):
    monkeypatch.setattr(server, "_load_matrix0", lambda *a, **k: None)
    # Remove static mount to allow API access in tests
    server.app.router.routes = [r for r in server.app.router.routes if getattr(r, "path", "")]
    client = TestClient(server.app)
    gid = _play_game(client)
    assert gid not in server.GAMES


def test_admin_purge_stale(monkeypatch):
    monkeypatch.setattr(server, "_load_matrix0", lambda *a, **k: None)
    server.app.router.routes = [r for r in server.app.router.routes if getattr(r, "path", "")]
    client = TestClient(server.app)
    resp = client.post("/new", json={"white": "human", "black": "human"})
    gid = resp.json()["game_id"]
    # Make the game stale
    server.GAMES[gid].created_ts -= server.GAME_TTL_SEC + 1
    out = client.post("/admin/purge")
    assert out.json()["removed"] >= 1
    assert gid not in server.GAMES


def test_memory_stabilizes(monkeypatch):
    monkeypatch.setattr(server, "_load_matrix0", lambda *a, **k: None)
    server.app.router.routes = [r for r in server.app.router.routes if getattr(r, "path", "")]
    client = TestClient(server.app)
    proc = psutil.Process()
    rss_start = proc.memory_info().rss
    for _ in range(3):
        gid = _play_game(client)
        assert gid not in server.GAMES
    rss_end = proc.memory_info().rss
    assert rss_end - rss_start < 20 * 1024 * 1024  # <20MB increase
    assert len(server.GAMES) == 0

