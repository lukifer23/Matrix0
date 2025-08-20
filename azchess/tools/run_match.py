from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import chess
import chess.pgn

from azchess.config import Config, select_device
from azchess.model import PolicyValueNet
from azchess.mcts import MCTS, MCTSConfig


BASE_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = BASE_DIR / "logs"
PGN_DIR = LOGS_DIR / "webui_pgn"
CSV_PATH = LOGS_DIR / "webui_matches.csv"


def jsonl_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def append_csv_row(row: dict) -> None:
    header = [
        "game_id", "tc_ms", "white", "black", "result", "moves",
        "matrix0_ms_avg", "stockfish_ms_avg"
    ]
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", encoding="utf-8") as f:
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


def save_pgn(game_id: str, moves: List[str], white: str, black: str) -> None:
    game = chess.pgn.Game()
    game.headers["Event"] = "Matrix0 Batch Eval"
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    node = game
    board = chess.Board()
    for u in moves:
        mv = chess.Move.from_uci(u)
        node = node.add_variation(mv)
        board.push(mv)
    game.headers["Result"] = board.result(claim_draw=True)
    PGN_DIR.mkdir(parents=True, exist_ok=True)
    with (PGN_DIR / f"{game_id}.pgn").open("w", encoding="utf-8") as f:
        print(game, file=f)


def main():
    ap = argparse.ArgumentParser(description="Run Matrix0 vs Stockfish batch matches (eval-only)")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--tc-ms", type=int, default=100)
    ap.add_argument("--stockfish", type=str, default=str(BASE_DIR / "engines" / "bin" / "stockfish"))
    ap.add_argument("--device", type=str, default=os.environ.get("MATRIX0_WEBUI_DEVICE", "cpu"))
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = select_device(args.device) if args.device != "cpu" else "cpu"
    # Load model
    model = PolicyValueNet.from_config(cfg.model())
    model.eval()
    ckpt_path = cfg.engines().get("matrix0", {}).get("checkpoint", str(BASE_DIR / "checkpoints" / "best.pt"))
    if ckpt_path and Path(ckpt_path).exists():
        import torch
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_ema", state.get("model", {})))
    model = model.to(device)
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
    mcts = MCTS(model, MCTSConfig.from_dict(mcfg_dict), device)

    # Stockfish
    import chess.engine
    sf_path = Path(args.stockfish)
    if not sf_path.exists():
        raise SystemExit(f"Stockfish not found at {sf_path}")
    sf = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    limit = chess.engine.Limit(time=max(10, int(args.tc_ms)) / 1000.0)

    # Run games
    wins = losses = draws = 0
    for i in range(max(1, int(args.games))):
        board = chess.Board()
        matrix0_white = (i % 2 == 0)
        white_name = "matrix0" if matrix0_white else "stockfish"
        black_name = "stockfish" if matrix0_white else "matrix0"
        m_ms_sum = s_ms_sum = 0.0
        m_ms_moves = s_ms_moves = 0
        move_ucis: List[str] = []

        while not board.is_game_over(claim_draw=True):
            is_white = (board.turn == chess.WHITE)
            side = "matrix0" if (is_white == matrix0_white) else "stockfish"
            if side == "matrix0":
                t0 = time.perf_counter()
                visits, pi, v = mcts.run(board)
                mv = max(visits.items(), key=lambda kv: kv[1])[0]
                m_ms_sum += (time.perf_counter() - t0) * 1000.0
                m_ms_moves += 1
            else:
                t0 = time.perf_counter()
                res = sf.play(board, limit)
                if not res.move:
                    sf.quit()
                    raise SystemExit("Stockfish returned no move")
                mv = res.move
                s_ms_sum += (time.perf_counter() - t0) * 1000.0
                s_ms_moves += 1
            move_ucis.append(mv.uci())
            board.push(mv)

        result = board.result(claim_draw=True)
        if matrix0_white:
            wins += 1 if result == "1-0" else 0
            losses += 1 if result == "0-1" else 0
        else:
            wins += 1 if result == "0-1" else 0
            losses += 1 if result == "1-0" else 0
        draws += 1 if result == "1/2-1/2" else 0

        game_id = f"cli_{int(time.time()*1000)}_{i:04d}"
        save_pgn(game_id, move_ucis, white_name, black_name)
        m_avg = (m_ms_sum / max(1, m_ms_moves)) if m_ms_moves else None
        s_avg = (s_ms_sum / max(1, s_ms_moves)) if s_ms_moves else None
        row = {
            "game_id": game_id,
            "tc_ms": int(args.tc_ms),
            "white": white_name,
            "black": black_name,
            "result": result,
            "moves": len(move_ucis),
            "matrix0_ms_avg": f"{m_avg:.1f}" if m_avg is not None else "",
            "stockfish_ms_avg": f"{s_avg:.1f}" if s_avg is not None else "",
        }
        append_csv_row(row)
        jsonl_write(LOGS_DIR / "webui.jsonl", {"ts": time.time(), "type": "cli_batch_game_done", **row})

    sf.quit()
    total = wins + losses + draws
    wr = wins / max(1, total)
    print(f"Games: {total} | W/L/D: {wins}/{losses}/{draws} | WR={wr:.3f}")


if __name__ == "__main__":
    main()

