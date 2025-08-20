from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List
import time

import chess
import chess.pgn
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from azchess.config import Config, select_device
from azchess.model import PolicyValueNet
from azchess.mcts import MCTS, MCTSConfig

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


app = FastAPI(title="Matrix0 WebUI", version="1.0")

# Serve static frontend
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# Lazy-loaded engines
_matrix0_model = None
_matrix0_mcts = None
_device = None
_stockfish = None
_stockfish_path = (BASE_DIR / "engines" / "bin" / "stockfish")


def _load_matrix0(cfg_path: str = "config.yaml", ckpt: Optional[str] = None, device_pref: str = "cpu"):
    global _matrix0_model, _matrix0_mcts, _device
    if _matrix0_model is not None and _matrix0_mcts is not None:
        return
    cfg = Config.load(cfg_path)
    # Default to CPU to avoid interfering with training; allow override via env MATRIX0_WEBUI_DEVICE
    dev_env = os.environ.get("MATRIX0_WEBUI_DEVICE", device_pref)
    _device = select_device(dev_env) if dev_env != "cpu" else "cpu"
    model = PolicyValueNet.from_config(cfg.model())
    model.eval()
    # Load best checkpoint by default
    ckpt_path = ckpt or cfg.engines().get("matrix0", {}).get("checkpoint", str(CHECKPOINTS_DIR / "best.pt"))
    if ckpt_path and Path(ckpt_path).exists():
        import torch
        state = torch.load(ckpt_path, map_location=_device)
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
    if not _stockfish_path.exists():
        return None
    try:
        _stockfish = chess.engine.SimpleEngine.popen_uci(str(_stockfish_path))
        return _stockfish
    except Exception:
        return None


# In-memory games (local only)
GAMES: Dict[str, GameState] = {}
# Active websocket connections per game_id
WS_CONNECTIONS: Dict[str, List[WebSocket]] = {}


async def _broadcast(game_id: str, payload: Dict[str, object]) -> None:
    """Send a JSON payload to all websockets for a given game."""
    if game_id not in WS_CONNECTIONS:
        return
    data = json.dumps(payload)
    dead: List[WebSocket] = []
    for ws in WS_CONNECTIONS.get(game_id, []):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    if dead:
        for ws in dead:
            try:
                WS_CONNECTIONS[game_id].remove(ws)
            except ValueError:
                pass
        if not WS_CONNECTIONS[game_id]:
            WS_CONNECTIONS.pop(game_id, None)


async def _send_state(game_id: str) -> None:
    """Broadcast current board state for a game."""
    gs = GAMES.get(game_id)
    if not gs:
        return
    finished = gs.board.is_game_over(claim_draw=True)
    payload = {
        "type": "state",
        "fen": gs.board.fen(),
        "turn": "w" if gs.board.turn == chess.WHITE else "b",
        "game_over": finished,
        "result": gs.board.result(claim_draw=True) if finished else None,
    }
    await _broadcast(game_id, payload)


async def _send_eval(game_id: str, mat_v: Optional[float], sf_cp: Optional[int]) -> None:
    """Broadcast evaluation information for a game."""
    await _broadcast(
        game_id,
        {
            "type": "eval",
            "fen": GAMES[game_id].board.fen() if game_id in GAMES else None,
            "matrix0_value": mat_v,
            "stockfish_cp": sf_cp,
        },
    )


@app.websocket("/ws/{game_id}")
async def ws_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    WS_CONNECTIONS.setdefault(game_id, []).append(websocket)
    # Send current state if game exists
    if game_id in GAMES:
        await _send_state(game_id)
    try:
        while True:
            # Keep the connection open; incoming messages are ignored
            await websocket.receive_text()
    except WebSocketDisconnect:
        try:
            WS_CONNECTIONS[game_id].remove(websocket)
            if not WS_CONNECTIONS[game_id]:
                WS_CONNECTIONS.pop(game_id, None)
        except Exception:
            pass


@app.on_event("shutdown")
def _cleanup() -> None:
    # Close stockfish if running
    global _stockfish
    try:
        if _stockfish:
            _stockfish.quit()
    except Exception:
        pass


@app.post("/new")
def new_game(req: NewGameRequest):
    _load_matrix0()
    game_id = f"g_{int(_now_ts()*1000)}"
    board = chess.Board(req.fen) if req.fen else chess.Board()
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
    _jsonl_write(WEBUI_LOG, {"ts": _now_ts(), "type": "new_game", "game_id": game_id, "white": req.white, "black": req.black, "fen": board.fen(), "tc_ms": gs.engine_tc_ms})
    return {"game_id": game_id, "fen": board.fen(), "turn": "w" if board.turn == chess.WHITE else "b"}


@app.post("/move")
async def play_move(req: MoveRequest):
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
    await _send_state(gs.game_id)
    return {
        "fen": gs.board.fen(),
        "turn": "w" if gs.board.turn == chess.WHITE else "b",
        "game_over": gs.board.is_game_over(claim_draw=True),
        "result": gs.board.result(claim_draw=True) if gs.board.is_game_over(claim_draw=True) else None,
    }


@app.post("/engine-move")
async def engine_move(req: EngineMoveRequest):
    if req.game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    gs = GAMES[req.game_id]
    start = time.perf_counter()
    mv = None
    engine = req.engine
    if engine == "matrix0":
        _load_matrix0()
        visits, pi, v = _matrix0_mcts.run(gs.board)
        mv = max(visits.items(), key=lambda kv: kv[1])[0]
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

    finished = gs.board.is_game_over(claim_draw=True)
    if finished:
        _save_pgn(gs)
    await _send_state(gs.game_id)
    return {
        "uci": mv.uci(),
        "fen": gs.board.fen(),
        "turn": "w" if gs.board.turn == chess.WHITE else "b",
        "game_over": finished,
        "result": gs.board.result(claim_draw=True) if finished else None,
    }


@app.post("/eval")
async def eval_position(req: EvalRequest):
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
    await _send_eval(gs.game_id, mat_v, sf_cp)
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

if __name__ == "__main__":
    # Launch with: python webui/server.py  (uses uvicorn if available)
    try:
        import uvicorn
        uvicorn.run("webui.server:app", host="127.0.0.1", port=8000, reload=False)
    except Exception:
        print("Install uvicorn to run the WebUI: pip install uvicorn fastapi")
