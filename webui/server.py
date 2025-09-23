from __future__ import annotations

"""
Allow `python webui/server.py` to work when executed directly by ensuring
the repository root (containing the `azchess` package) is on sys.path.
This complements the recommended invocation `python -m webui.server`.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import logging
import subprocess
import threading
import re

import chess
import chess.pgn
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse, Response

from azchess import monitor
from azchess.config import Config, select_device
from azchess.draw import should_adjudicate_draw
from azchess.mcts import MCTS, MCTSConfig
from azchess.model import PolicyValueNet
from azchess.elo import expected_score, update_elo
from azchess.ratings import Glicko2Rating, update_glicko2_player
from benchmarks.tournament import (
    Tournament, TournamentConfig, TournamentFormat,
    TournamentResult, EngineStanding, GameResult,
    create_tournament_config, run_tournament
)

try:
    import chess.engine
    HAVE_ENGINE = True
except Exception:
    HAVE_ENGINE = False


BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
PGN_DIR = LOGS_DIR / "webui_pgn"
WEBUI_LOG = LOGS_DIR / "webui.jsonl"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"


def _jsonl_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _now_ts() -> float:
    return time.time()


@dataclass
class GameState:
    game_id: str
    created_ts: float
    board: chess.Board
    white: str  # "human" | "matrix0" | "stockfish"
    black: str  # "human" | "matrix0" | "stockfish"
    moves: List[str]
    engine_tc_ms: int


class NewGameRequest(BaseModel):
    white: str = "human"
    black: str = "matrix0"
    fen: Optional[str] = None
    engine_tc_ms: int = 100


class MoveRequest(BaseModel):
    game_id: str
    uci: str


class EngineMoveRequest(BaseModel):
    game_id: str
    engine: str = "matrix0"  # or "stockfish"


class EvalRequest(BaseModel):
    game_id: str
    include_stockfish: bool = True


class BatchMatchRequest(BaseModel):
    games: int = 20
    engine_tc_ms: int = 100
    start_white: str = "matrix0"  # which engine starts as white: matrix0|stockfish


class TournamentCreateRequest(BaseModel):
    name: str
    engines: List[str]
    format: str = "round_robin"  # round_robin, single_elimination, swiss, double_round_robin
    num_games_per_pairing: int = 1
    time_control_ms: int = 100
    max_concurrency: int = 2


class TournamentStatusRequest(BaseModel):
    tournament_id: str


class EloCalculationRequest(BaseModel):
    engine_ratings: Dict[str, float]  # engine_name -> current_rating
    game_results: List[Dict[str, Any]]  # list of {white: str, black: str, result: str}


class OrchestratorStartRequest(BaseModel):
    config: str = "config.yaml"
    games: Optional[int] = None
    workers: Optional[int] = None
    sims: Optional[int] = None
    eval_games: Optional[int] = None
    promotion_threshold: Optional[float] = None
    device: Optional[str] = None  # auto/cpu/mps/cuda
    quick_start: bool = False
    no_shared_infer: bool = False
    tui: Optional[str] = None   # bars|table
    continuous: bool = False    # currently ignored; single-cycle only


app = FastAPI(title="Matrix0 WebUI", version="1.0")

# Serve static frontend
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Local logger
logger = logging.getLogger("webui")

# Serve index.html for root path
@app.get("/")
async def read_root():
    try:
        path = STATIC_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(str(path), media_type="text/html")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve index: {e}")


@app.get("/favicon.ico")
async def favicon():
    ico = STATIC_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    return Response(status_code=204)


# Lazy-loaded engines
_matrix0_model = None
_matrix0_mcts = None
_device = None
_cfg: Config | None = None
_stockfish = None
_stockfish_path = (BASE_DIR / "engines" / "bin" / "stockfish")

# Orchestrator process management (simple single-run control)
_ORCH_PROC: Optional[subprocess.Popen] = None
_ORCH_SUP_THREAD: Optional[threading.Thread] = None
_ORCH_STOP_EVENT = threading.Event()
_ORCH_CONTINUOUS: bool = False
_ORCH_CYCLE: int = 0
_ORCH_START_TS: Optional[float] = None
_ORCH_CFG: Dict[str, Any] = {}
_ORCH_LOCK = threading.RLock()
_ORCH_LOG_PATH = LOGS_DIR / "structured.jsonl"


def _compute_target_games(cfg_path: str, overrides: Dict[str, Any]) -> int:
    try:
        cfg = Config.load(cfg_path)
        orch = cfg.orchestrator() or {}
        games_override = overrides.get("games")
        if games_override is not None:
            return int(games_override)
        quick_start = bool(overrides.get("quick_start", False))
        initial_games = int(orch.get("initial_games", 64))
        subsequent_games = int(orch.get("subsequent_games", 64))
        # Mirror orchestrator logic: if no/bare checkpoints → initial_games
        checkpoint_dir = Path("checkpoints")
        ckpts = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
        if quick_start or len(ckpts) <= 1:
            return initial_games
        return subsequent_games
    except Exception:
        return int(overrides.get("games", 64) or 64)


def _read_structured_since(ts_cutoff: float) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    try:
        if not _ORCH_LOG_PATH.exists():
            return events
        # Read last ~10k lines max to bound time; fall back to full if small file
        with _ORCH_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if float(rec.get("ts", 0.0)) >= ts_cutoff:
                    events.append(rec)
    except Exception:
        pass
    return events


def _resolve_workers(cfg_path: str, override_workers: Optional[int]) -> int:
    if override_workers is not None:
        return max(1, int(override_workers))
    try:
        cfg = Config.load(cfg_path)
        workers = int(cfg.selfplay().get("num_workers", 2))
        return max(1, workers)
    except Exception:
        return 2


@app.post("/orchestrator/start")
def start_orchestrator(req: OrchestratorStartRequest):
    global _ORCH_PROC, _ORCH_START_TS, _ORCH_CFG, _ORCH_SUP_THREAD, _ORCH_CONTINUOUS, _ORCH_CYCLE
    with _ORCH_LOCK:
        # Check if already running
        if (_ORCH_PROC is not None and _ORCH_PROC.poll() is None) or (_ORCH_SUP_THREAD is not None and _ORCH_SUP_THREAD.is_alive()):
            raise HTTPException(status_code=409, detail="orchestrator already running")

        # Build command
        cmd = [sys.executable, "-m", "azchess.orchestrator", "--config", req.config]
        overrides: Dict[str, Any] = {}
        if req.games is not None:
            cmd += ["--games", str(int(req.games))]
            overrides["games"] = int(req.games)
        if req.workers is not None:
            cmd += ["--workers", str(int(req.workers))]
            overrides["workers"] = int(req.workers)
        if req.sims is not None:
            cmd += ["--sims", str(int(req.sims))]
            overrides["sims"] = int(req.sims)
        if req.eval_games is not None:
            cmd += ["--eval-games", str(int(req.eval_games))]
            overrides["eval_games"] = int(req.eval_games)
        if req.promotion_threshold is not None:
            cmd += ["--promotion-threshold", str(float(req.promotion_threshold))]
            overrides["promotion_threshold"] = float(req.promotion_threshold)
        if req.device is not None:
            cmd += ["--device", str(req.device)]
            overrides["device"] = str(req.device)
        if req.no_shared_infer:
            cmd += ["--no-shared-infer"]
            overrides["no_shared_infer"] = True
        if req.quick_start:
            cmd += ["--quick-start"]
            overrides["quick_start"] = True
        if req.tui:
            cmd += ["--tui", req.tui]
            overrides["tui"] = req.tui

        # Ensure logs directory
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        orch_out = LOGS_DIR / "orchestrator.out"

        def _spawn_once():
            out_f = open(orch_out, "a", buffering=1)
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(BASE_DIR),
                    env=os.environ.copy(),
                )
            except Exception as e:
                out_f.close()
                raise
            return proc

        # Compute plan for visibility
        workers = _resolve_workers(req.config, req.workers)
        target_games = _compute_target_games(req.config, overrides)
        from math import ceil
        games_per_worker = int(ceil(target_games / max(1, workers)))
        total_games = games_per_worker * workers

        if req.continuous:
            _ORCH_CONTINUOUS = True
            _ORCH_STOP_EVENT.clear()
            _ORCH_CYCLE = 0

            def _supervisor():
                global _ORCH_PROC, _ORCH_START_TS, _ORCH_CONTINUOUS
                nonlocal workers, target_games, games_per_worker, total_games
                while not _ORCH_STOP_EVENT.is_set():
                    try:
                        with _ORCH_LOCK:
                            proc = _spawn_once()
                            _ORCH_PROC = proc
                            _ORCH_START_TS = _now_ts()
                            _ORCH_CYCLE += 1
                            _ORCH_CFG.update({
                                "config": req.config,
                                "overrides": overrides,
                                "workers": workers,
                                "target_games": target_games,
                                "games_per_worker": games_per_worker,
                                "total_games": total_games,
                                "pid": proc.pid,
                                "orch_out": str(orch_out),
                                "continuous": True,
                                "cycle": _ORCH_CYCLE,
                            })
                        proc.wait()
                    except Exception:
                        pass
                    if _ORCH_STOP_EVENT.is_set():
                        break
                    # small backoff between cycles
                    time.sleep(10)
                with _ORCH_LOCK:
                    # ensure cleared
                    _ORCH_CONTINUOUS = False
                    _ORCH_PROC = None

            _ORCH_SUP_THREAD = threading.Thread(target=_supervisor, name="orch-supervisor", daemon=True)
            _ORCH_SUP_THREAD.start()
            _ORCH_START_TS = _now_ts()
            _ORCH_CFG = {
                "config": req.config,
                "overrides": overrides,
                "workers": workers,
                "target_games": target_games,
                "games_per_worker": games_per_worker,
                "total_games": total_games,
                "pid": None,
                "orch_out": str(orch_out),
                "continuous": True,
                "cycle": 0,
            }
            return {"started": True, "continuous": True, "cmd": cmd, "plan": _ORCH_CFG}
        else:
            try:
                proc = _spawn_once()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"failed to start orchestrator: {e}")

            _ORCH_PROC = proc
            _ORCH_START_TS = _now_ts()
            _ORCH_CFG = {
                "config": req.config,
                "overrides": overrides,
                "workers": workers,
                "target_games": target_games,
                "games_per_worker": games_per_worker,
                "total_games": total_games,
                "pid": proc.pid,
                "orch_out": str(orch_out),
                "continuous": False,
                "cycle": 1,
            }
            return {"pid": proc.pid, "started": True, "cmd": cmd, "plan": _ORCH_CFG}


@app.post("/orchestrator/stop")
def stop_orchestrator():
    global _ORCH_PROC, _ORCH_SUP_THREAD, _ORCH_CONTINUOUS
    with _ORCH_LOCK:
        if _ORCH_SUP_THREAD is not None and _ORCH_SUP_THREAD.is_alive():
            _ORCH_STOP_EVENT.set()
        if _ORCH_PROC is None or (_ORCH_PROC and _ORCH_PROC.poll() is not None):
            # Nothing active; if supervisor existed, wait briefly
            if _ORCH_SUP_THREAD is not None:
                try:
                    _ORCH_SUP_THREAD.join(timeout=3)
                except Exception:
                    pass
                _ORCH_SUP_THREAD = None
                _ORCH_CONTINUOUS = False
            return {"stopped": False, "message": "orchestrator not running"}
        try:
            _ORCH_PROC.terminate()
            try:
                _ORCH_PROC.wait(timeout=10)
            except Exception:
                _ORCH_PROC.kill()
        finally:
            _ORCH_PROC = None
            if _ORCH_SUP_THREAD is not None:
                try:
                    _ORCH_SUP_THREAD.join(timeout=3)
                except Exception:
                    pass
                _ORCH_SUP_THREAD = None
                _ORCH_CONTINUOUS = False
        return {"stopped": True}


@app.get("/orchestrator/status")
def orchestrator_status():
    with _ORCH_LOCK:
        running = bool((_ORCH_PROC is not None and _ORCH_PROC.poll() is None) or (_ORCH_SUP_THREAD is not None and _ORCH_SUP_THREAD.is_alive()))
        pid = int(_ORCH_PROC.pid) if running and _ORCH_PROC is not None else None
        started_at = _ORCH_START_TS
        plan = _ORCH_CFG.copy() if _ORCH_CFG else {}

    # Aggregate progress from structured logs since start
    progress = {
        "completed_games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }
    eval_info: Dict[str, Any] = {}
    promo_info: Dict[str, Any] = {}

    inferred = False
    # Allow inference of activity even if not launched by WebUI (external orchestrator)
    recent_ts_cutoff = time.time() - 300  # last 5 minutes
    recent_events = _read_structured_since(recent_ts_cutoff)
    if not running and recent_events:
        # Look for orchestrator/selfplay/training activity in recent logs
        for ev in reversed(recent_events[-200:]):
            msg = str(ev.get("msg", ""))
            if ("TRAINING_HB" in msg) or ("Game" in msg and "completed" in msg) or ("Starting training phase" in msg) or ("Compacting self-play data" in msg) or ("Evaluating candidate" in msg):
                running = True
                inferred = True
                break

    if started_at or inferred:
        try:
            # When inferring, scan full file to avoid missing earlier self-play events
            ts_cutoff = float(started_at) if started_at else 0.0
            events = _read_structured_since(ts_cutoff)
            game_rx = re.compile(r"Game\s+\d+\s+completed: .*?result:\s*([+-]?[0-9.]+)")
            eval_start_rx = re.compile(r"Evaluating candidate.*?vs best")
            eval_done_rx = re.compile(r"Evaluation complete: win_rate=([0-9.]+)")
            promo_rx = re.compile(r"Promoted candidate to best")
            for ev in events:
                msg = str(ev.get("msg", ""))
                m = game_rx.search(msg)
                if m:
                    progress["completed_games"] += 1
                    try:
                        res = float(m.group(1))
                        if res > 0:
                            progress["wins"] += 1
                        elif res < 0:
                            progress["losses"] += 1
                        else:
                            progress["draws"] += 1
                    except Exception:
                        pass
                    continue
                if eval_start_rx.search(msg):
                    eval_info["status"] = "running"
                    eval_info["ts"] = ev.get("ts")
                m2 = eval_done_rx.search(msg)
                if m2:
                    try:
                        eval_info["status"] = "completed"
                        eval_info["win_rate"] = float(m2.group(1))
                        eval_info["ts"] = ev.get("ts")
                    except Exception:
                        pass
                if promo_rx.search(msg):
                    promo_info = {"promoted": True, "ts": ev.get("ts")}
        except Exception:
            pass

    # Compute percent
    # If not launched by WebUI, prefer sp_start from webui.jsonl, else estimate from config
    if True:
        try:
            # Try to derive from recent webui.jsonl events (works for CLI runs too)
            wpath = LOGS_DIR / "webui.jsonl"
            if wpath.exists():
                with wpath.open("r", encoding="utf-8") as f:
                    lines = f.readlines()[-5000:]
                import json as _json
                sp_start = None
                sp_games = []
                saw_activity = False
                for ln in lines:
                    try:
                        ev = _json.loads(ln)
                    except Exception:
                        continue
                    if ev.get("type") == "sp_start":
                        sp_start = ev
                        saw_activity = True
                    elif ev.get("type") == "sp_game":
                        sp_games.append(ev)
                        saw_activity = True
                if sp_start:
                    workers_ev = int(sp_start.get("workers", plan.get("workers", 0) or 0))
                    gpw_ev = int(sp_start.get("games_per_worker", plan.get("games_per_worker", 0) or 0))
                    total_ev = int(sp_start.get("total_target", plan.get("total_games", 0) or 0))
                    if not plan:
                        plan = {}
                    plan.update({
                        "workers": workers_ev or plan.get("workers"),
                        "games_per_worker": gpw_ev or plan.get("games_per_worker"),
                        "total_games": total_ev or plan.get("total_games"),
                        "source": "webui_events",
                    })
                    if sp_games:
                        progress["completed_games"] = len(sp_games)
                        wins = losses = draws = 0
                        for ev in sp_games:
                            try:
                                r = float(ev.get("result", 0.0))
                            except Exception:
                                r = 0.0
                            if r > 0: wins += 1
                            elif r < 0: losses += 1
                            else: draws += 1
                        progress["wins"], progress["losses"], progress["draws"] = wins, losses, draws

                # If we observed activity from webui.jsonl, treat orchestrator as running
                if saw_activity and not running:
                    running = True
                    inferred = True

            # If still no plan, estimate from config
            if not plan:
                cfg_path = "config.yaml"
                workers_cfg = _resolve_workers(cfg_path, None)
                target = _compute_target_games(cfg_path, {})
                from math import ceil
                g_per = int(ceil(target / max(1, workers_cfg)))
                total_est = g_per * workers_cfg
                plan = {
                    "config": cfg_path,
                    "workers": workers_cfg,
                    "target_games": target,
                    "games_per_worker": g_per,
                    "total_games": total_est,
                    "estimated": True,
                }
        except Exception:
            plan = {}

    total_games = int(plan.get("total_games", 0) or 0)
    # Consider CLI runs inferred from webui.jsonl as active
    completed = int(progress["completed_games"]) if (started_at or inferred) else 0
    pct = (completed / total_games * 100.0) if total_games > 0 else 0.0

    # Include training sub-status
    try:
        train = training_status()  # reuse existing endpoint function
    except Exception:
        train = {"is_training": False}

    return {
        "running": running,
        "pid": pid,
        "started_at": started_at,
        "plan": plan,
        "continuous": _ORCH_CONTINUOUS,
        "cycle": _ORCH_CYCLE,
        "selfplay": {
            "completed": completed,
            "total": total_games,
            "progress_pct": pct,
            "wins": progress["wins"],
            "losses": progress["losses"],
            "draws": progress["draws"],
        },
        "evaluation": eval_info,
        "promotion": promo_info,
        "training": train,
        "inferred": inferred,
    }


@app.get("/orchestrator/logs")
def orchestrator_logs(tail: int = 200, kind: str = "structured"):
    tail = max(1, min(int(tail), 2000))
    lines: List[str] = []
    try:
        if kind == "matrix":
            path = LOGS_DIR / "matrix0.log"
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    buf = f.readlines()
                    lines = [ln.rstrip("\n") for ln in buf[-tail:]]
        else:
            path = _ORCH_LOG_PATH
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    buf = f.readlines()
                # Render a compact textual view of structured events
                for ln in buf[-tail:]:
                    try:
                        rec = json.loads(ln)
                        ts = rec.get("ts")
                        lvl = rec.get("level")
                        name = rec.get("name")
                        msg = rec.get("msg")
                        lines.append(f"[{ts}] {lvl} {name}: {msg}")
                    except Exception:
                        lines.append(ln.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read logs: {e}")
    return {"kind": kind, "lines": lines}


@app.get("/orchestrator/stream")
async def orchestrator_stream(request: Request, kind: str = "structured"):
    """Server-Sent Events stream of orchestrator logs.
    kind: structured|matrix
    """
    async def event_gen():
        import asyncio
        path = _ORCH_LOG_PATH if kind != "matrix" else (LOGS_DIR / "matrix0.log")
        pos = 0
        if path.exists():
            try:
                pos = path.stat().st_size
            except Exception:
                pos = 0
        while True:
            if await request.is_disconnected():
                break
            try:
                if path.exists():
                    size = path.stat().st_size
                    if size > pos:
                        with path.open("r", encoding="utf-8") as f:
                            f.seek(pos)
                            chunk = f.read()
                            pos = f.tell()
                        for ln in chunk.splitlines():
                            data = ln
                            if kind != "matrix":
                                try:
                                    rec = json.loads(ln)
                                    data = f"[{rec.get('ts')}] {rec.get('level')} {rec.get('name')}: {rec.get('msg')}"
                                except Exception:
                                    pass
                            yield f"data: {data}\n\n"
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/orchestrator/workers")
def orchestrator_workers():
    """Summarize per-worker status from webui.jsonl sp_heartbeat and sp_game events."""
    path = LOGS_DIR / "webui.jsonl"
    workers: Dict[int, Dict[str, Any]] = {}
    games_per_worker = None
    total_target = None
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-2000:]
            import json as _json
            for ln in lines:
                try:
                    ev = _json.loads(ln)
                except Exception:
                    continue
                et = ev.get("type")
                if et == "sp_start":
                    games_per_worker = int(ev.get("games_per_worker", 0) or 0)
                    total_target = int(ev.get("total_target", 0) or 0)
                elif et == "sp_heartbeat":
                    wid = int(ev.get("worker", -1))
                    if wid < 0:
                        continue
                    w = workers.setdefault(wid, {"done": 0, "avg_ms": 0.0, "avg_sims": 0.0, "moves": 0, "hb_ts": 0.0, "games": 0})
                    w["moves"] = int(ev.get("moves", 0))
                    w["avg_sims"] = float(ev.get("avg_sims", 0.0))
                    w["hb_ts"] = float(ev.get("ts", time.time()))
                elif et == "sp_game":
                    wid = int(ev.get("worker", -1))
                    if wid < 0:
                        continue
                    w = workers.setdefault(wid, {"done": 0, "avg_ms": 0.0, "avg_sims": 0.0, "moves": 0, "hb_ts": 0.0, "games": 0})
                    w["done"] = int(w.get("done", 0)) + 1
                    w["games"] = int(w.get("games", 0)) + 1
                    ms = float(ev.get("avg_ms_per_move", 0.0))
                    sims = float(ev.get("avg_sims", 0.0))
                    # update running averages
                    c = w["games"]
                    w["avg_ms"] = (w.get("avg_ms", 0.0) * (c - 1) + ms) / max(1, c)
                    w["avg_sims"] = (w.get("avg_sims", 0.0) * (c - 1) + sims) / max(1, c)
        # Fill defaults if absent
        if games_per_worker is None:
            plan = orchestrator_status().get("plan", {})
            games_per_worker = int(plan.get("games_per_worker", 0) or 0)
            total_target = int(plan.get("total_games", 0) or 0)
        # Compute heartbeat age
        now = time.time()
        out = []
        for wid in sorted(workers.keys()):
            w = workers[wid]
            hb_age = now - float(w.get("hb_ts", 0.0)) if w.get("hb_ts") else None
            out.append({
                "worker": wid,
                "done": int(w.get("done", 0)),
                "games_per_worker": int(games_per_worker or 0),
                "avg_ms": float(w.get("avg_ms", 0.0)),
                "avg_sims": float(w.get("avg_sims", 0.0)),
                "moves": int(w.get("moves", 0)),
                "hb_age": hb_age,
            })
        return {"workers": out, "games_per_worker": games_per_worker, "total_target": total_target}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize workers: {e}")


def _load_matrix0(cfg_path: str = "config.yaml", ckpt: Optional[str] = None, device_pref: str = "cpu"):
    global _matrix0_model, _matrix0_mcts, _device, _cfg
    if _matrix0_model is not None and _matrix0_mcts is not None:
        return
    _cfg = Config.load(cfg_path)
    cfg = _cfg
    # Default to CPU to avoid interfering with training; allow override via env MATRIX0_WEBUI_DEVICE
    dev_env = os.environ.get("MATRIX0_WEBUI_DEVICE", device_pref)
    _device = select_device(dev_env) if dev_env != "cpu" else "cpu"
    model = PolicyValueNet.from_config(cfg.model())
    model.eval()
    # Load best checkpoint by default
    ckpt_path = ckpt or cfg.engines().get("matrix0", {}).get("checkpoint", str(CHECKPOINTS_DIR / "best.pt"))
    if ckpt_path and Path(ckpt_path).exists():
        import torch
        state = torch.load(ckpt_path, map_location=_device, weights_only=False)
        model.load_state_dict(state.get("model_ema", state.get("model", {})))
    _matrix0_model = model.to(_device)
    e = cfg.eval()
    mcfg_dict = dict(cfg.mcts())
    mcfg_dict.update(
        {
            "num_simulations": int(e.get("num_simulations", mcfg_dict.get("num_simulations", 200))),
            "cpuct": float(e.get("cpuct", mcfg_dict.get("cpuct", 1.5))),
            "dirichlet_alpha": float(e.get("dirichlet_alpha", mcfg_dict.get("dirichlet_alpha", 0.3))),
            "dirichlet_frac": 0.0,
            "tt_capacity": int(e.get("tt_capacity", mcfg_dict.get("tt_capacity", 200000))),
            "selection_jitter": 0.0,
            "tt_cleanup_frequency": int(cfg.mcts().get("tt_cleanup_frequency", 500)),
        }
    )
    _matrix0_mcts = MCTS(_matrix0_model, MCTSConfig.from_dict(mcfg_dict), _device)


def _load_stockfish() -> Optional["chess.engine.SimpleEngine"]:
    global _stockfish
    if not HAVE_ENGINE:
        return None
    if _stockfish is not None:
        return _stockfish
    # Resolve path preference order: config -> env -> PATH -> local default
    cfg_path = None
    try:
        cfg_local = _cfg or Config.load("config.yaml")
        cfg_eng = cfg_local.engines() if cfg_local else {}
        cfg_sf = cfg_eng.get("stockfish", {}) if isinstance(cfg_eng, dict) else {}
        cfg_path = cfg_sf.get("path")
    except Exception:
        cfg_path = None

    env_path = os.environ.get("MATRIX0_STOCKFISH_PATH") or os.environ.get("STOCKFISH_PATH")
    resolved = None
    # Prefer config path if it exists
    for cand in (cfg_path, env_path):
        if cand and Path(cand).exists():
            resolved = cand
            break

    if resolved is None:
        # Try system PATH
        import shutil
        which = shutil.which("stockfish")
        if which:
            resolved = which

    if resolved is None and _stockfish_path.exists():
        resolved = str(_stockfish_path)

    if resolved is None:
        return None
    try:
        _stockfish = chess.engine.SimpleEngine.popen_uci(str(resolved))
        return _stockfish
    except Exception:
        return None


# In-memory games (local only)
GAMES: Dict[str, GameState] = {}

# Remove games after they finish or exceed this TTL (seconds)
GAME_TTL_SEC = 3600

# Tournament management
ACTIVE_TOURNAMENTS: Dict[str, Tournament] = {}
COMPLETED_TOURNAMENTS: Dict[str, Dict[str, Any]] = {}
ENGINE_RATINGS: Dict[str, Dict[str, Any]] = {}  # engine_name -> {"elo": float, "glicko": Glicko2Rating}

# Initialize default ratings
def _init_ratings():
    """Initialize default ratings for common engines."""
    global ENGINE_RATINGS
    default_rating = {
        "elo": 1500.0,
        "glicko": Glicko2Rating(rating=1500.0, rd=350.0, sigma=0.06)
    }

    for engine in ["matrix0", "stockfish", "lc0"]:
        if engine not in ENGINE_RATINGS:
            ENGINE_RATINGS[engine] = default_rating.copy()

_init_ratings()


def _cleanup_games(ttl_sec: int = GAME_TTL_SEC) -> int:
    """Remove finished or stale games.

    Args:
        ttl_sec: Time-to-live for unfinished games. Games older than this are removed.

    Returns:
        Number of games removed.
    """
    now = _now_ts()
    to_del: List[str] = []
    for gid, gs in list(GAMES.items()):
        try:
            is_game_over = gs.board.is_game_over(claim_draw=True)
        except Exception:
            # If there's an error checking game over (e.g., invalid moves in history),
            # assume the game might be corrupted and clean it up
            is_game_over = True

        if is_game_over or now - gs.created_ts > ttl_sec:
            to_del.append(gid)
    for gid in to_del:
        GAMES.pop(gid, None)
    return len(to_del)


@app.on_event("shutdown")
def _cleanup() -> None:
    # Close stockfish if running
    try:
        if _stockfish:
            _stockfish.quit()
    except Exception:
        pass


@app.post("/new")
def new_game(req: NewGameRequest):
    try:
        logger.info(f"New game request: white={req.white}, black={req.black}, tc={req.engine_tc_ms}")
    except Exception:
        pass
    try:
        _load_matrix0()
        try:
            logger.info("Matrix0 loaded successfully")
        except Exception:
            pass
    except Exception as e:
        try:
            logger.error(f"Failed to load Matrix0: {e}")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to load Matrix0: {str(e)}")

    game_id = f"g_{int(_now_ts()*1000)}"
    try:
        logger.info(f"Creating game {game_id}")
    except Exception:
        pass

    try:
        board = chess.Board(req.fen) if req.fen else chess.Board()
        try:
            logger.info(f"Board created with FEN: {board.fen()}")
        except Exception:
            pass
    except ValueError as e:
        try:
            logger.error(f"Invalid FEN: {req.fen}")
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="invalid FEN")
    except Exception as e:
        try:
            logger.error(f"Error creating board: {e}")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Board creation error: {str(e)}")

    gs = GameState(
        game_id=game_id,
        created_ts=_now_ts(),
        board=board,
        white=req.white,
        black=req.black,
        moves=[],
        engine_tc_ms=max(10, int(req.engine_tc_ms)),
    )
    GAMES[game_id] = gs
    try:
        logger.info(f"Game {game_id} created successfully")
    except Exception:
        pass

    _jsonl_write(WEBUI_LOG, {"ts": _now_ts(), "type": "new_game", "game_id": game_id, "white": req.white, "black": req.black, "fen": board.fen(), "tc_ms": gs.engine_tc_ms})
    result = {"game_id": game_id, "fen": board.fen(), "turn": "w" if board.turn == chess.WHITE else "b"}
    try:
        logger.info(f"Returning game result: {result}")
    except Exception:
        pass
    return result


@app.post("/move")
def play_move(req: MoveRequest):
    if req.game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    gs = GAMES[req.game_id]
    try:
        mv = chess.Move.from_uci(req.uci)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid UCI move")
    if mv not in gs.board.legal_moves:
        raise HTTPException(status_code=400, detail="illegal move")
    gs.board.push(mv)
    gs.moves.append(req.uci)
    _jsonl_write(WEBUI_LOG, {"ts": _now_ts(), "type": "human_move", "game_id": gs.game_id, "uci": req.uci, "fen": gs.board.fen()})
    draw_cfg = _cfg.draw() if _cfg else {}
    finished = gs.board.is_game_over(claim_draw=True) or should_adjudicate_draw(gs.board, [chess.Move.from_uci(u) for u in gs.moves], draw_cfg)
    if finished:
        _save_pgn(gs)
        _cleanup_games()
    return {
        "fen": gs.board.fen(),
        "turn": "w" if gs.board.turn == chess.WHITE else "b",
        "game_over": finished,
        "result": gs.board.result(claim_draw=True) if finished else None,
    }


@app.post("/engine-move")
def engine_move(req: EngineMoveRequest):
    if req.game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    gs = GAMES[req.game_id]
    start = time.perf_counter()
    mv = None
    engine = req.engine
    if engine == "matrix0":
        _load_matrix0()
        visits, pi, v = _matrix0_mcts.run(gs.board)

        # Filter visits to only include legal moves and select the best one
        legal_moves = list(gs.board.legal_moves)
        legal_visits = {move: visits.get(move, 0) for move in legal_moves if move in visits}

        if not legal_visits:
            # Fallback: if no legal moves found in visits, use any legal move
            mv = legal_moves[0] if legal_moves else None
        else:
            mv = max(legal_visits.items(), key=lambda kv: kv[1])[0]

        if mv is None:
            raise HTTPException(status_code=500, detail="No legal moves available")

        elapsed = (time.perf_counter() - start) * 1000.0
        gs.board.push(mv)
        gs.moves.append(mv.uci())
        _jsonl_write(WEBUI_LOG, {"ts": _now_ts(), "type": "engine_move", "engine": "matrix0", "uci": mv.uci(), "ms": elapsed, "fen": gs.board.fen(), "game_id": gs.game_id})
    elif engine == "stockfish":
        sf = _load_stockfish()
        if not sf:
            raise HTTPException(status_code=400, detail="stockfish unavailable")
        limit = chess.engine.Limit(time=gs.engine_tc_ms / 1000.0)
        result = sf.play(gs.board, limit)
        if not result.move:
            raise HTTPException(status_code=500, detail="stockfish returned no move")
        mv = result.move
        elapsed = (time.perf_counter() - start) * 1000.0
        gs.board.push(mv)
        gs.moves.append(mv.uci())
        _jsonl_write(WEBUI_LOG, {"ts": _now_ts(), "type": "engine_move", "engine": "stockfish", "uci": mv.uci(), "ms": elapsed, "fen": gs.board.fen(), "game_id": gs.game_id})
    else:
        raise HTTPException(status_code=400, detail="unknown engine")

    draw_cfg = _cfg.draw() if _cfg else {}
    finished = gs.board.is_game_over(claim_draw=True) or should_adjudicate_draw(gs.board, [chess.Move.from_uci(u) for u in gs.moves], draw_cfg)
    if finished:
        _save_pgn(gs)
        _cleanup_games()
    return {
        "uci": mv.uci(),
        "fen": gs.board.fen(),
        "turn": "w" if gs.board.turn == chess.WHITE else "b",
        "game_over": finished,
        "result": gs.board.result(claim_draw=True) if finished else None,
        "ms": float(elapsed),
    }


@app.post("/eval")
def eval_position(req: EvalRequest):
    if req.game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    gs = GAMES[req.game_id]
    _load_matrix0()
    # Matrix0 value
    try:
        import torch

        from azchess.encoding import encode_board
        with torch.no_grad():
            arr = encode_board(gs.board)
            x = torch.from_numpy(arr).unsqueeze(0).to(_device)
            _, v = _matrix0_model(x)
            mat_v = float(v.item())
    except Exception:
        mat_v = None

    sf_cp = None
    if req.include_stockfish and HAVE_ENGINE:
        sf = _load_stockfish()
        if sf:
            try:
                info = sf.analyse(gs.board, chess.engine.Limit(time=gs.engine_tc_ms / 1000.0))
                score = info.get("score")
                if score is not None:
                    # Normalize to side-to-move perspective
                    sf_cp = score.white().score(mate_score=100000) if gs.board.turn == chess.WHITE else score.black().score(mate_score=100000)
                    sf_cp = int(sf_cp)
            except Exception:
                sf_cp = None

    _jsonl_write(WEBUI_LOG, {"ts": _now_ts(), "type": "eval", "game_id": gs.game_id, "fen": gs.board.fen(), "matrix0_v": mat_v, "stockfish_cp": sf_cp})
    return {"fen": gs.board.fen(), "matrix0_value": mat_v, "stockfish_cp": sf_cp}


def _save_pgn(gs: GameState) -> None:
    try:
        game = chess.pgn.Game()
        game.headers["Event"] = "Matrix0 WebUI"
        game.headers["White"] = gs.white
        game.headers["Black"] = gs.black
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        node = game
        board = chess.Board()
        for u in gs.moves:
            mv = chess.Move.from_uci(u)
            node = node.add_variation(mv)
            board.push(mv)
        game.headers["Result"] = board.result(claim_draw=True)
        PGN_DIR.mkdir(parents=True, exist_ok=True)
        path = PGN_DIR / f"{gs.game_id}.pgn"
        with path.open("w", encoding="utf-8") as f:
            print(game, file=f)
    except Exception:
        pass


def _append_match_csv(row: Dict[str, object]) -> None:
    # Append to logs/webui_matches.csv with header if missing
    path = LOGS_DIR / "webui_matches.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "game_id", "tc_ms", "white", "black", "result", "moves",
        "matrix0_ms_avg", "stockfish_ms_avg"
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        vals = [
            str(row.get("game_id", "")),
            str(row.get("tc_ms", "")),
            str(row.get("white", "")),
            str(row.get("black", "")),
            str(row.get("result", "")),
            str(row.get("moves", "")),
            f"{row.get('matrix0_ms_avg', '')}",
            f"{row.get('stockfish_ms_avg', '')}",
        ]
        f.write(",".join(vals) + "\n")


@app.post("/batch-match")
def batch_match(req: BatchMatchRequest):
    # Run Matrix0 vs Stockfish for N games, alternating colors
    _load_matrix0()
    sf = _load_stockfish()
    if not sf:
        raise HTTPException(status_code=400, detail="stockfish unavailable")
    games = max(1, int(req.games))
    tc_ms = max(10, int(req.engine_tc_ms))
    limit = chess.engine.Limit(time=tc_ms / 1000.0)
    results = {"wins": 0, "losses": 0, "draws": 0}
    summaries = []

    for i in range(games):
        board = chess.Board()
        m_ms_sum = 0.0
        m_ms_moves = 0
        s_ms_sum = 0.0
        s_ms_moves = 0
        matrix0_white = (req.start_white == "matrix0") if (i % 2 == 0) else (req.start_white != "matrix0")
        white_name = "matrix0" if matrix0_white else "stockfish"
        black_name = "stockfish" if matrix0_white else "matrix0"
        move_ucis: List[str] = []

        while not board.is_game_over(claim_draw=True):
            stm_is_white = (board.turn == chess.WHITE)
            side = "matrix0" if (stm_is_white == matrix0_white) else "stockfish"
            if side == "matrix0":
                t0 = time.perf_counter()
                visits, pi, v = _matrix0_mcts.run(board)
                mv = max(visits.items(), key=lambda kv: kv[1])[0]
                m_ms_sum += (time.perf_counter() - t0) * 1000.0
                m_ms_moves += 1
            else:
                t0 = time.perf_counter()
                res = sf.play(board, limit)
                if not res.move:
                    raise HTTPException(status_code=500, detail="stockfish returned no move")
                mv = res.move
                s_ms_sum += (time.perf_counter() - t0) * 1000.0
                s_ms_moves += 1
            move_ucis.append(mv.uci())
            board.push(mv)

        result = board.result(claim_draw=True)
        if matrix0_white:
            # Matrix0 as white
            if result == "1-0":
                results["wins"] += 1
            elif result == "0-1":
                results["losses"] += 1
            else:
                results["draws"] += 1
        else:
            # Matrix0 as black
            if result == "0-1":
                results["wins"] += 1
            elif result == "1-0":
                results["losses"] += 1
            else:
                results["draws"] += 1

        game_id = f"bm_{int(_now_ts()*1000)}_{i:04d}"
        # Save PGN
        try:
            gs = GameState(game_id, _now_ts(), chess.Board(), white_name, black_name, move_ucis, tc_ms)
            _save_pgn(gs)
        except Exception:
            pass

        m_avg = (m_ms_sum / max(1, m_ms_moves)) if m_ms_moves else None
        s_avg = (s_ms_sum / max(1, s_ms_moves)) if s_ms_moves else None
        row = {
            "game_id": game_id,
            "tc_ms": tc_ms,
            "white": white_name,
            "black": black_name,
            "result": result,
            "moves": len(move_ucis),
            "matrix0_ms_avg": f"{m_avg:.1f}" if m_avg is not None else "",
            "stockfish_ms_avg": f"{s_avg:.1f}" if s_avg is not None else "",
        }
        _append_match_csv(row)
        _jsonl_write(WEBUI_LOG, {"ts": _now_ts(), "type": "batch_game_done", **row})
        summaries.append(row)

    total = sum(results.values())
    wr = results["wins"] / max(1, total)
    return {"games": total, "wins": results["wins"], "losses": results["losses"], "draws": results["draws"], "win_rate": wr, "summary": summaries}


def _read_match_csv(limit: Optional[int] = None):
    path = LOGS_DIR / "webui_matches.csv"
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            rows.append(dict(zip(header, parts)))
    if limit:
        rows = rows[-limit:]
    return rows


@app.get("/analytics/summary")
def analytics_summary():
    rows = _read_match_csv(limit=200)
    wins = losses = draws = 0
    for r in rows:
        res = r.get("result", "")
        if res == "1-0":
            # white won
            wins += 1 if r.get("white") == "matrix0" else 0
            losses += 1 if r.get("white") == "stockfish" else 0
        elif res == "0-1":
            wins += 1 if r.get("black") == "matrix0" else 0
            losses += 1 if r.get("black") == "stockfish" else 0
        elif res == "1/2-1/2":
            draws += 1
    total = wins + losses + draws
    wr = wins / max(1, total)
    return {"total": total, "wins": wins, "losses": losses, "draws": draws, "win_rate": wr}


@app.get("/pgn/list")
def list_pgn():
    PGN_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p.name for p in PGN_DIR.glob("*.pgn")])
    return {"files": files}


@app.get("/pgn/{name}")
def get_pgn(name: str):
    """Serve a PGN file by name from the webui PGN directory."""
    try:
        # Security: prevent path traversal
        name = os.path.basename(name)
        path = PGN_DIR / name
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="PGN not found")
        return FileResponse(str(path), media_type="application/x-chess-pgn", filename=name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve PGN: {e}")


@app.get("/health")
def health():
    # lightweight model info
    try:
        cfg = Config.load("config.yaml")
        model = PolicyValueNet.from_config(cfg.model())
        params = model.count_parameters()
    except Exception:
        params = None
    sf_available = _load_stockfish() is not None
    return {"stockfish": sf_available, "model_params": params, "device": _device}


@app.get("/system/metrics")
def system_metrics():
    """Return lightweight metrics for the dashboard."""

    try:
        memory_info = monitor.get_memory_usage("auto")
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("Failed to gather memory usage: %s", exc)
        memory_info = {"available": False}

    metrics: Dict[str, Any] = {
        "timestamp": _now_ts(),
        "active_games": len(GAMES),
        "memory": memory_info,
    }

    try:
        training = training_status()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("Training status lookup failed: %s", exc)
        training = {"error": str(exc), "is_training": False}

    if isinstance(training, dict):
        metrics["training"] = {
            "is_training": bool(training.get("is_training", False)),
            "progress": training.get("progress"),
        }
        if training.get("message"):
            metrics["training"]["message"] = training["message"]
        if training.get("error"):
            metrics["training"]["error"] = training["error"]
    else:
        metrics["training"] = {"is_training": False}

    try:
        ssl_info = ssl_status()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("SSL status lookup failed: %s", exc)
        ssl_info = {"error": str(exc), "enabled": False, "tasks": []}

    ssl_metrics = {"enabled": False, "task_count": 0}
    if isinstance(ssl_info, dict):
        ssl_metrics["enabled"] = bool(ssl_info.get("enabled", False))
        tasks = ssl_info.get("tasks") or []
        if isinstance(tasks, list):
            ssl_metrics["task_count"] = len(tasks)
        else:
            try:
                ssl_metrics["task_count"] = int(tasks)
            except Exception:
                ssl_metrics["task_count"] = 0
        if ssl_info.get("error"):
            ssl_metrics["error"] = ssl_info["error"]
    metrics["ssl"] = ssl_metrics

    try:
        tournaments_info = list_tournaments()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("Tournament summary lookup failed: %s", exc)
        tournaments_info = {"error": str(exc), "active_tournaments": [], "completed_tournaments": []}

    tournaments = {"active": 0, "completed": 0}
    if isinstance(tournaments_info, dict):
        active = tournaments_info.get("total_active")
        if active is None and isinstance(tournaments_info.get("active_tournaments"), list):
            active = len(tournaments_info["active_tournaments"])
        completed = tournaments_info.get("total_completed")
        if completed is None and isinstance(tournaments_info.get("completed_tournaments"), list):
            completed = len(tournaments_info["completed_tournaments"])
        tournaments["active"] = int(active or 0)
        tournaments["completed"] = int(completed or 0)
        if tournaments_info.get("error"):
            tournaments["error"] = tournaments_info["error"]
    metrics["tournaments"] = tournaments

    return metrics


@app.get("/ssl/status")
def ssl_status():
    """Get SSL configuration and current status."""
    try:
        cfg = Config.load("config.yaml")
        model_config = cfg.model()

        ssl_info = {
            "enabled": model_config.get("self_supervised", False),
            "tasks": model_config.get("ssl_tasks", []),
            "curriculum": model_config.get("ssl_curriculum", True),
            "ssl_head_count": len(model_config.get("ssl_tasks", [])),
            "config": {
                "ssl_weight": cfg.training().get("ssl_weight", 0.04),
                "ssl_warmup_steps": cfg.training().get("ssl_warmup_steps", 1500),
                "ssl_target_weight": cfg.training().get("ssl_target_weight", 1.0),
            }
        }

        # Try to get model parameter counts for SSL heads
        if _matrix0_model is not None and hasattr(_matrix0_model, 'ssl_heads'):
            ssl_head_params = {}
            ssl_head_analysis = {}

            for task_name, head in _matrix0_model.ssl_heads.items():
                params = sum(p.numel() for p in head.parameters())
                ssl_head_params[task_name] = params

                # Analyze head structure
                ssl_head_analysis[task_name] = {
                    "parameters": params,
                    "structure": str(head).split('\n')[0],  # First line of head description
                    "trainable": sum(p.numel() for p in head.parameters() if p.requires_grad)
                }

            ssl_info["head_parameters"] = ssl_head_params
            ssl_info["total_ssl_params"] = sum(ssl_head_params.values())
            ssl_info["head_analysis"] = ssl_head_analysis

            # Add SSL task weights from config
            ssl_weights = {}
            training_config = cfg.training()
            for task in ssl_info["tasks"]:
                weight_key = f"ssl_{task}_weight"
                ssl_weights[task] = training_config.get(weight_key, 1.0)

            ssl_info["task_weights"] = ssl_weights

        return ssl_info
    except Exception as e:
        return {"error": str(e), "enabled": False}


@app.get("/ssl/performance")
def ssl_performance():
    """Get SSL performance metrics and training progress."""
    try:
        # Read recent training logs for SSL performance
        ssl_metrics = {
            "ssl_loss_history": [],
            "task_contributions": {},
            "learning_progress": {}
        }

        train_log = BASE_DIR / "logs" / "matrix0.log"
        if train_log.exists():
            with open(train_log, 'r') as f:
                lines = f.readlines()[-50:]  # Last 50 lines for SSL analysis

            for line in lines:
                if "TRAINING_HB" in line:
                    import re
                    # Support both legacy "SSL:" and new "SSL(EMA):"
                    match = re.search(r'Step (\d+)/(\d+).* (?:SSL\(EMA\)|SSL):\s*([0-9.]+)', line)
                    if match:
                        step, total_steps, ssl_loss = match.groups()
                        ssl_metrics["ssl_loss_history"].append({
                            "step": int(step),
                            "ssl_loss": float(ssl_loss)
                        })

        # Calculate SSL performance statistics
        if ssl_metrics["ssl_loss_history"]:
            losses = [m["ssl_loss"] for m in ssl_metrics["ssl_loss_history"]]
            ssl_metrics["statistics"] = {
                "current_ssl_loss": losses[-1],
                "avg_ssl_loss": sum(losses) / len(losses),
                "ssl_loss_trend": "decreasing" if len(losses) > 1 and losses[-1] < losses[0] else "stable",
                "min_ssl_loss": min(losses),
                "max_ssl_loss": max(losses)
            }

        return ssl_metrics
    except Exception as e:
        return {"error": str(e)}


@app.get("/training/status")
def training_status():
    """Get current training status and metrics."""
    try:
        # Read recent training logs
        training_data = []

        # Check if training log exists
        train_log = BASE_DIR / "logs" / "matrix0.log"
        if train_log.exists():
            with open(train_log, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines for better history

            for line in lines:
                if "TRAINING_HB" in line:
                    # Parse training heartbeat lines (support legacy and new EMA format)
                    import re
                    # New format example:
                    # TRAINING_HB: Step S/T (updates=U) | BatchLoss: x.xxx | EMA: x.xxx | Policy(EMA): x.xxx | Value(EMA): x.xxx | SSL(EMA): x.xxx | LR: x.xxxxxx
                    m_new = re.search(
                        r'Step\s+(\d+)/(\d+).*?EMA:\s*([0-9.]+).*?Policy\(EMA\):\s*([0-9.]+).*?Value\(EMA\):\s*([0-9.]+).*?(?:SSL\(EMA\)|SSL):\s*([0-9.]+).*?LR:\s*([0-9.]+)',
                        line
                    )
                    m_old = re.search(
                        r'Step\s+(\d+)/(\d+).*?Loss:\s*([0-9.]+).*?Policy:\s*([0-9.]+).*?Value:\s*([0-9.]+).*?(?:SSL\(EMA\)|SSL):\s*([0-9.]+).*?LR:\s*([0-9.]+)',
                        line
                    )
                    match = m_new or m_old
                    if match:
                        step, total_steps, loss, policy, value, ssl, lr = match.groups()
                        training_data.append({
                            "step": int(step),
                            "total_steps": int(total_steps),
                            "loss": float(loss),
                            "policy_loss": float(policy),
                            "value_loss": float(value),
                            "ssl_loss": float(ssl),
                            "learning_rate": float(lr)
                        })

        if training_data:
            latest = training_data[-1]

            # Calculate training statistics
            steps = [d["step"] for d in training_data]
            losses = [d["loss"] for d in training_data]
            ssl_losses = [d["ssl_loss"] for d in training_data]
            learning_rates = [d["learning_rate"] for d in training_data]

            return {
                "is_training": True,
                "current_step": latest["step"],
                "total_steps": latest["total_steps"],
                "progress": (latest["step"] / latest["total_steps"]) * 100,
                "latest_metrics": {
                    "loss": latest["loss"],
                    "policy_loss": latest["policy_loss"],
                    "value_loss": latest["value_loss"],
                    "ssl_loss": latest["ssl_loss"],
                    "learning_rate": latest["learning_rate"]
                },
                "recent_history": training_data[-10:],  # Last 10 entries for better visualization
                "statistics": {
                    "avg_loss": sum(losses) / len(losses),
                    "avg_ssl_loss": sum(ssl_losses) / len(ssl_losses),
                    "loss_trend": "decreasing" if len(losses) > 1 and losses[-1] < losses[0] else "stable",
                    "ssl_trend": "decreasing" if len(ssl_losses) > 1 and ssl_losses[-1] < ssl_losses[0] else "stable",
                    "lr_range": [min(learning_rates), max(learning_rates)]
                }
            }
        else:
            return {
                "is_training": False,
                "message": "No recent training data found"
            }

    except Exception as e:
        return {"error": str(e), "is_training": False}


@app.get("/model/analysis")
def model_analysis():
    """Get detailed model analysis."""
    try:
        cfg = Config.load("config.yaml")
        model = PolicyValueNet.from_config(cfg.model())

        total_params = sum(p.numel() for p in model.parameters())

        # Parameter count by layer type
        param_breakdown = {}
        for name, module in model.named_modules():
            if hasattr(module, 'parameters') and list(module.parameters()):
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    layer_type = type(module).__name__
                    param_breakdown[layer_type] = param_breakdown.get(layer_type, 0) + param_count

        # SSL head analysis
        ssl_heads = {}
        if hasattr(model, 'ssl_heads'):
            for task_name, head in model.ssl_heads.items():
                ssl_heads[task_name] = {
                    "parameters": sum(p.numel() for p in head.parameters()),
                    "structure": str(head)
                }

        return {
            "total_parameters": total_params,
            "parameter_breakdown": param_breakdown,
            "ssl_heads": ssl_heads,
            "architecture": {
                "channels": cfg.model().get("channels", 320),
                "blocks": cfg.model().get("blocks", 24),
                "attention_heads": cfg.model().get("attention_heads", 20),
                "policy_size": cfg.model().get("policy_size", 4672),
                "ssl_enabled": cfg.model().get("self_supervised", False),
                "ssl_tasks": cfg.model().get("ssl_tasks", [])
            }
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/config/view")
def config_view():
    """Get current configuration."""
    try:
        cfg = Config.load("config.yaml")
        return {
            "model": cfg.model().__dict__ if hasattr(cfg.model(), '__dict__') else cfg.model(),
            "training": cfg.training().__dict__ if hasattr(cfg.training(), '__dict__') else cfg.training(),
            "selfplay": cfg.selfplay().__dict__ if hasattr(cfg.selfplay(), '__dict__') else cfg.selfplay(),
            "eval": cfg.eval().__dict__ if hasattr(cfg.eval(), '__dict__') else cfg.eval(),
            "mcts": cfg.mcts().__dict__ if hasattr(cfg.mcts(), '__dict__') else cfg.mcts()
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/admin/purge")
def admin_purge():
    """Endpoint to purge finished or stale games."""
    removed = _cleanup_games()
    return {"removed": removed, "active": len(GAMES)}


# Tournament Management Endpoints

@app.post("/tournament/create")
async def create_tournament(req: TournamentCreateRequest):
    """Create and start a new tournament."""
    try:
        # Create tournament configuration
        config = TournamentConfig(
            name=req.name,
            format=TournamentFormat(req.format),
            engines=req.engines,
            num_games_per_pairing=req.num_games_per_pairing,
            time_control=f"{req.time_control_ms}ms",
            concurrency=req.max_concurrency,
            output_dir="webui_tournaments",
            save_pgns=True,
            calculate_ratings=True
        )

        # Create and start tournament
        tournament = Tournament(config)
        tournament_id = f"tournament_{int(_now_ts() * 1000)}"

        # Store active tournament
        ACTIVE_TOURNAMENTS[tournament_id] = tournament

        # Run tournament asynchronously
        import asyncio
        asyncio.create_task(run_tournament_async(tournament_id, tournament))

        return {
            "tournament_id": tournament_id,
            "status": "started",
            "config": {
                "name": req.name,
                "format": req.format,
                "engines": req.engines,
                "games_per_pairing": req.num_games_per_pairing,
                "time_control_ms": req.time_control_ms,
                "concurrency": req.max_concurrency
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create tournament: {str(e)}")


async def run_tournament_async(tournament_id: str, tournament: Tournament):
    """Run tournament asynchronously and store results."""
    try:
        results = await tournament.run_tournament()

        # Move to completed tournaments
        COMPLETED_TOURNAMENTS[tournament_id] = results
        ACTIVE_TOURNAMENTS.pop(tournament_id, None)

        # Update ratings based on tournament results
        await update_ratings_from_tournament(results)

    except Exception as e:
        logger.error(f"Tournament {tournament_id} failed: {e}")
        ACTIVE_TOURNAMENTS.pop(tournament_id, None)


@app.get("/tournament/list")
def list_tournaments():
    """List all active and completed tournaments."""
    active = []
    for tid, tournament in ACTIVE_TOURNAMENTS.items():
        active.append({
            "id": tid,
            "name": tournament.config.name,
            "status": "running",
            "engines": tournament.config.engines,
            "format": tournament.config.format.value,
            "start_time": tournament.start_time,
            "duration": tournament.duration
        })

    completed = []
    for tid, results in COMPLETED_TOURNAMENTS.items():
        completed.append({
            "id": tid,
            "name": results.get("tournament_name", "Unknown"),
            "status": "completed",
            "engines": results.get("engines", []),
            "format": results.get("format", "unknown"),
            "start_time": results.get("start_time"),
            "end_time": results.get("end_time"),
            "duration": results.get("duration", 0),
            "total_games": results.get("total_games", 0)
        })

    return {
        "active_tournaments": active,
        "completed_tournaments": completed,
        "total_active": len(active),
        "total_completed": len(completed)
    }


@app.get("/tournament/{tournament_id}/status")
def get_tournament_status(tournament_id: str):
    """Get status of a specific tournament."""
    # Check active tournaments
    if tournament_id in ACTIVE_TOURNAMENTS:
        tournament = ACTIVE_TOURNAMENTS[tournament_id]
        return {
            "id": tournament_id,
            "status": "running",
            "name": tournament.config.name,
            "progress": len(tournament.completed_games) / max(1, len(tournament.completed_games) + len(tournament.rounds) * 10),  # Estimate
            "engines": tournament.config.engines,
            "format": tournament.config.format.value,
            "games_completed": len(tournament.completed_games),
            "standings": [
                {
                    "engine": standing.engine_name,
                    "points": standing.points,
                    "games_played": standing.games_played,
                    "wins": standing.wins,
                    "losses": standing.losses,
                    "draws": standing.draws
                }
                for standing in tournament.standings.values()
            ]
        }

    # Check completed tournaments
    if tournament_id in COMPLETED_TOURNAMENTS:
        results = COMPLETED_TOURNAMENTS[tournament_id]
        return {
            "id": tournament_id,
            "status": "completed",
            "name": results.get("tournament_name"),
            "engines": results.get("engines", []),
            "format": results.get("format"),
            "standings": results.get("standings", []),
            "game_results": results.get("game_results", []),
            "statistics": results.get("statistics", {}),
            "duration": results.get("duration", 0)
        }

    raise HTTPException(status_code=404, detail="Tournament not found")


@app.get("/ratings")
def get_ratings():
    """Get current ELO and Glicko-2 ratings for all engines."""
    ratings = {}
    for engine, rating_data in ENGINE_RATINGS.items():
        ratings[engine] = {
            "elo": rating_data["elo"],
            "glicko_rating": rating_data["glicko"].rating,
            "glicko_rd": rating_data["glicko"].rd,
            "glicko_sigma": rating_data["glicko"].sigma
        }

    return {"ratings": ratings}


@app.post("/ratings/update")
def update_ratings(req: EloCalculationRequest):
    """Update ratings based on game results."""
    try:
        updated_ratings = {}

        for game in req.game_results:
            white = game["white"]
            black = game["black"]
            result_str = game["result"]

            # Convert result string to score (1.0 for white win, 0.5 for draw, 0.0 for black win)
            if result_str == "1-0":
                white_score = 1.0
                black_score = 0.0
            elif result_str == "0-1":
                white_score = 0.0
                black_score = 1.0
            else:  # Draw
                white_score = 0.5
                black_score = 0.5

            # Update ELO ratings
            if white in req.engine_ratings and black in req.engine_ratings:
                white_elo, black_elo = update_elo(
                    req.engine_ratings[white],
                    req.engine_ratings[black],
                    white_score
                )

                # Update global ratings
                ENGINE_RATINGS[white]["elo"] = white_elo
                ENGINE_RATINGS[black]["elo"] = black_elo

                # Update Glicko-2 ratings
                white_glicko = update_glicko2_player(
                    ENGINE_RATINGS[white]["glicko"],
                    [(req.engine_ratings[black], 350.0)],  # Assume opponent RD
                    [white_score]
                )
                ENGINE_RATINGS[white]["glicko"] = white_glicko

                updated_ratings[white] = {"elo": white_elo, "glicko": white_glicko.rating}
                updated_ratings[black] = {"elo": black_elo, "glicko": ENGINE_RATINGS[black]["glicko"].rating}

        return {"updated_ratings": updated_ratings, "message": f"Updated ratings for {len(updated_ratings)} engines"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update ratings: {str(e)}")


async def update_ratings_from_tournament(results: Dict[str, Any]):
    """Update ratings based on tournament results."""
    try:
        game_results = []
        current_ratings = {engine: ENGINE_RATINGS.get(engine, {"elo": 1500.0})["elo"]
                          for engine in results.get("engines", [])}

        for game in results.get("game_results", []):
            game_results.append({
                "white": game["white"],
                "black": game["black"],
                "result": game["result"]
            })

        if game_results:
            await update_ratings(EloCalculationRequest(
                engine_ratings=current_ratings,
                game_results=game_results
            ))

    except Exception as e:
        logger.error(f"Failed to update ratings from tournament: {e}")

if __name__ == "__main__":
    # Launch with: python webui/server.py  (uses uvicorn if available)
    try:
        import uvicorn
        uvicorn.run("webui.server:app", host="127.0.0.1", port=8000, reload=False)
    except Exception:
        print("Install uvicorn to run the WebUI: pip install uvicorn fastapi")
