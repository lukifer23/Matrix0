import gc
import logging
import os
from pathlib import Path

import chess
import psutil
import pytest

from azchess.selfplay import internal as sp_internal


def _get_handle_count(proc: psutil.Process) -> int:
    if hasattr(proc, "num_fds"):
        return proc.num_fds()
    if hasattr(proc, "num_handles"):
        return proc.num_handles()  # pragma: no cover - Windows fallback
    pytest.skip("Process handle count not supported on this platform")


class DummyTablebase:
    def __init__(self, fd_path: Path):
        self._fd = os.open(str(fd_path), os.O_RDONLY)
        self.closed = False

    def probe_wdl(self, board: chess.Board):  # pragma: no cover - exercised via worker
        return None

    def close(self) -> None:
        if not self.closed:
            os.close(self._fd)
            self.closed = True


def test_selfplay_worker_closes_tablebase(monkeypatch, tmp_path):
    proc = psutil.Process()
    baseline = _get_handle_count(proc)

    fd_source = Path(__file__)
    tb_instances: list[DummyTablebase] = []

    def fake_open_tablebase(path):
        tb = DummyTablebase(fd_source)
        tb_instances.append(tb)
        return tb

    def dummy_setup_logging(log_dir: str = "logs", level: int = logging.INFO, name: str | None = None):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(level)
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    monkeypatch.setattr(chess.syzygy, "open_tablebase", fake_open_tablebase)
    monkeypatch.setattr(sp_internal, "setup_logging", dummy_setup_logging)

    tb_dir = tmp_path / "tb"
    tb_dir.mkdir()
    data_dir = tmp_path / "data"

    cfg = {
        "device": "cpu",
        "seed": 0,
        "data_dir": str(data_dir),
        "model": {
            "planes": 19,
            "channels": 8,
            "blocks": 1,
            "attention": False,
            "attention_heads": 1,
            "se": False,
            "chess_features": False,
            "self_supervised": False,
            "policy_size": 4672,
        },
        "selfplay": {
            "num_simulations": 1,
            "max_game_len": 1,
            "temperature_start": 0.0,
            "temperature_end": 0.0,
            "temperature_moves": 0,
            "resign_threshold": -1.0,
            "min_resign_plies": 0,
            "batch_size": 1,
        },
        "mcts": {
            "num_simulations": 1,
            "cpuct": 1.0,
            "dirichlet_frac": 0.0,
            "selection_jitter": 0.0,
            "batch_size": 1,
            "tt_capacity": 32,
            "tt_cleanup_frequency": 10,
            "tt_memory_limit_mb": 16,
            "num_threads": 1,
            "parallel_simulations": False,
            "tree_parallelism": False,
            "simulation_batch_size": 1,
            "enable_entropy_noise": False,
            "encoder_cache": False,
        },
        "tablebases": {"enabled": True, "path": str(tb_dir), "max_pieces": 7},
        "draw": {},
        "openings": {},
        "presets": {},
    }

    sp_internal.selfplay_worker(0, cfg, ckpt_path=None, games=1, q=None)

    gc.collect()
    after = _get_handle_count(proc)

    assert tb_instances, "tablebase was not initialized"
    closed_flags = [tb.closed for tb in tb_instances]
    assert all(closed_flags)
    assert after == baseline

    for tb in tb_instances:
        if not tb.closed:
            tb.close()
