import asyncio
import math
import sys
import types
import pytest

from benchmarks.tournament import (
    Tournament,
    TournamentConfig,
    TournamentFormat,
    parse_time_control,
)


class FakeStockfishEngine:
    """Minimal Stockfish stub capturing play limits."""

    def __init__(self):
        self.play_limits = []

    def play(self, board, limit, info=None):  # pragma: no cover - signature parity
        self.play_limits.append(limit)
        move = next(iter(board.legal_moves))
        return types.SimpleNamespace(move=move)

    def quit(self):  # pragma: no cover - simple stub
        pass


@pytest.fixture(autouse=True)
def stub_azchess_modules(monkeypatch):
    """Provide lightweight stand-ins for azchess dependencies."""

    azchess_module = types.ModuleType("azchess")
    azchess_module.__path__ = []  # Mark as package for submodule imports
    monkeypatch.setitem(sys.modules, "azchess", azchess_module)

    config_module = types.ModuleType("azchess.config")

    class DummyConfig:
        @staticmethod
        def load(path):  # pragma: no cover - configuration stub
            return DummyConfig()

        def training(self):
            return {"device": "cpu"}

        def model(self):
            return {}

        def eval(self):
            return {"num_simulations": 1, "cpuct": 1.0}

    config_module.Config = DummyConfig
    config_module.select_device = lambda *args, **kwargs: "cpu"
    monkeypatch.setitem(sys.modules, "azchess.config", config_module)

    model_module = types.ModuleType("azchess.model")

    class DummyPolicyValueNet:
        @staticmethod
        def from_config(cfg):  # pragma: no cover - not used in tests
            return DummyPolicyValueNet()

    model_module.PolicyValueNet = DummyPolicyValueNet
    monkeypatch.setitem(sys.modules, "azchess.model", model_module)

    mcts_module = types.ModuleType("azchess.mcts")

    class DummyMCTSConfig:
        def __init__(self, *args, **kwargs):  # pragma: no cover - simple init
            pass

    class DummyMCTS:
        def __init__(self, *args, **kwargs):  # pragma: no cover - simple init
            pass

        def run(self, board):  # pragma: no cover - matrix0 not exercised
            return {}, {}, 0.0

    mcts_module.MCTSConfig = DummyMCTSConfig
    mcts_module.MCTS = DummyMCTS
    monkeypatch.setitem(sys.modules, "azchess.mcts", mcts_module)

    encoding_module = types.ModuleType("azchess.encoding")
    encoding_module.encode_board = lambda board: None
    monkeypatch.setitem(sys.modules, "azchess.encoding", encoding_module)

    yield

    for name in [
        "azchess",
        "azchess.config",
        "azchess.model",
        "azchess.mcts",
        "azchess.encoding",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize(
    "raw, expected_time, expected_increment",
    [
        ("30+0.3", 30.0, 0.3),
        ("100ms", 0.1, 0.0),
        ("60s", 60.0, 0.0),
        (45, 45.0, 0.0),
    ],
)
def test_parse_time_control_variants(raw, expected_time, expected_increment):
    parsed = parse_time_control(raw)
    assert parsed.original == str(raw)
    assert math.isclose(parsed.time_seconds, expected_time)
    assert math.isclose(parsed.increment_seconds, expected_increment)

    limit_kwargs = parsed.limit_kwargs()
    if expected_time is not None:
        assert math.isclose(limit_kwargs.get("time", 0.0), expected_time)
    if expected_increment:
        assert math.isclose(limit_kwargs.get("white_inc", 0.0), expected_increment)
        assert math.isclose(limit_kwargs.get("black_inc", 0.0), expected_increment)
    else:
        assert "white_inc" not in limit_kwargs
        assert "black_inc" not in limit_kwargs


@pytest.mark.parametrize(
    "time_control, expected_time, expected_increment",
    [
        ("30+0.3", 30.0, 0.3),
        ("100ms", 0.1, 0.0),
        ("60s", 60.0, 0.0),
        (20, 20.0, 0.0),
    ],
)
def test_stockfish_game_uses_parsed_time_control(
    time_control, expected_time, expected_increment, monkeypatch
):
    fake_engine = FakeStockfishEngine()
    monkeypatch.setattr(
        "chess.engine.SimpleEngine.popen_uci", lambda _: fake_engine
    )

    config = TournamentConfig(
        name="tc",
        format=TournamentFormat.ROUND_ROBIN,
        engines=["stockfish", "random"],
        num_games_per_pairing=1,
        time_control=time_control,
        max_moves=1,
        concurrency=1,
        random_openings=False,
        save_pgns=False,
        calculate_ratings=False,
    )

    tournament = Tournament(config)
    result = asyncio.run(tournament._play_game("stockfish", "random"))

    assert fake_engine.play_limits, "Stockfish engine did not receive a move limit"
    limit = fake_engine.play_limits[0]
    assert math.isclose(limit.time or 0.0, expected_time)

    if expected_increment:
        assert math.isclose(limit.white_inc or 0.0, expected_increment)
        assert math.isclose(limit.black_inc or 0.0, expected_increment)
    else:
        assert (limit.white_inc or 0.0) == 0.0
        assert (limit.black_inc or 0.0) == 0.0

    metadata = result.metadata.get("time_control")
    assert metadata["original"] == str(time_control)
    assert math.isclose(metadata["time_seconds"], expected_time)
    assert math.isclose(metadata["increment_seconds"], expected_increment)

