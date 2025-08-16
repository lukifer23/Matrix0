from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Tuple
from time import perf_counter

import numpy as np
import chess

import torch

from .config import Config, select_device
from .model import PolicyValueNet
from .mcts import MCTS, MCTSConfig
from .data_manager import DataManager
from .encoding import encode_board, move_to_index


def selfplay_worker(proc_id: int, cfg_dict: dict, ckpt_path: str | None, games: int, q: Queue | None = None):
    random.seed(1234 + proc_id)
    np.random.seed(1234 + proc_id)

    device = select_device(cfg_dict.get("device", "auto"))

    model = PolicyValueNet.from_config(cfg_dict["model"]).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])

    sp_mcts_cfg = cfg_dict["selfplay"].copy()
    sp_mcts_cfg["selection_jitter"] = float(sp_mcts_cfg.get("selection_jitter", 0.0))
    mcts = MCTS(model, MCTSConfig(**sp_mcts_cfg), device=device)
    data_manager = DataManager(base_dir=cfg_dict.get("data_dir", "data"))

    for g in range(games):
        board = chess.Board()
        states: List[np.ndarray] = []
        pis: List[np.ndarray] = []
        turns: List[int] = []
        t0 = perf_counter()
        # Opening diversity: random uniform opening plies
        opening_plies = int(cfg_dict["selfplay"].get("opening_random_plies", 0))
        if opening_plies > 0:
            for _ in range(opening_plies):
                if board.is_game_over():
                    break
                legal = list(board.legal_moves)
                if not legal:
                    break
                # Record state and uniform policy over legals
                states.append(encode_board(board))
                pi_open = np.zeros(4672, dtype=np.float32)
                for lm in legal:
                    pi_open[move_to_index(board, lm)] = 1.0 / len(legal)
                pis.append(pi_open)
                turns.append(1 if board.turn == chess.WHITE else -1)
                mv = legal[int(np.random.randint(0, len(legal)))]
                board.push(mv)

        # Resign parameters
        resign_thr = float(cfg_dict["selfplay"].get("resign_threshold", -1.0))
        resign_min_moves = int(cfg_dict["selfplay"].get("resign_min_moves", 60))
        resign_consec = int(cfg_dict["selfplay"].get("resign_consecutive", 3))
        consec_bad = 0
        resigned = False
        resign_winner = 0.0

        while not board.is_game_over() and len(states) < cfg_dict["selfplay"].get("max_game_len", 512):
            mv_no = len(states)
            temperature, sims = schedule_params(cfg_dict["selfplay"], mv_no)
            visit_counts, pi, v = mcts.run(board, num_simulations=sims)
            move = sample_move_from_counts(board, visit_counts, temperature)
            states.append(encode_board(board))
            pis.append(pi)
            turns.append(1 if board.turn == chess.WHITE else -1)
            # Resign check (from current player's perspective)
            if mv_no >= resign_min_moves and resign_thr > -1.0:
                if v <= resign_thr:
                    consec_bad += 1
                else:
                    consec_bad = 0
                if consec_bad >= resign_consec:
                    resigned = True
                    resign_winner = -1.0 if board.turn == chess.WHITE else 1.0
                    break
            board.push(move)

        z = resign_winner if resigned else game_result(board)
        # Save samples (s, pi, z)
        game_data = {
            "s": np.array(states, dtype=np.float32),
            "pi": np.array(pis, dtype=np.float32),
            "z": np.array([z * t for t in turns], dtype=np.float32),
        }
        filepath = data_manager.add_selfplay_data(game_data, worker_id=proc_id, game_id=g)
        
        if q is not None:
            q.put({
                "type": "game",
                "proc": proc_id,
                "file": filepath,
                "moves": len(states),
                "result": float(z),
                "secs": perf_counter() - t0,
                "resigned": resigned,
            })


def sample_move_from_counts(board: chess.Board, counts: Dict[chess.Move, int], temperature: float) -> chess.Move:
    moves = list(counts.keys())
    visits = np.array(list(counts.values()), dtype=np.float32)
    if temperature <= 1e-3:
        return moves[int(np.argmax(visits))]
    logits = np.log(visits + 1e-8) / max(temperature, 1e-3)
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    idx = np.random.choice(len(moves), p=probs)
    return moves[idx]


def game_result(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return -1.0
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/selfplay")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--external-engines", action="store_true", help="Use external engines for self-play")
    args = parser.parse_args()

    cfg_obj = Config.load(args.config)
    # Strict encoding enforcement via env
    try:
        import os as _os
        if bool(cfg_obj.get("strict_encoding", False)):
            _os.environ["MATRIX0_STRICT_ENCODING"] = "1"
    except Exception:
        pass
    cfg = cfg_obj.to_dict()
    workers = args.workers or cfg["selfplay"].get("num_workers", 2)
    games_per_worker = math_div_ceil(args.games, workers)

    if args.external_engines:
        # Use external engine self-play
        try:
            from .selfplay.external_engine_worker import external_engine_worker
            import asyncio
            
            print("[SelfPlay] Using external engines for self-play")
            
            # Run external engine self-play
            async def run_external_selfplay():
                # For now, run in single process to avoid async/multiprocessing complexity
                games = await external_engine_worker(0, cfg_obj, args.out, args.games)
                return games
            
            games = asyncio.run(run_external_selfplay())
            print(f"[SelfPlay] Completed {len(games)} external engine games")
            
        except ImportError as e:
            print(f"[SelfPlay] External engine support not available: {e}")
            print("[SelfPlay] Falling back to internal self-play")
            args.external_engines = False
        
        if not args.external_engines:
            # Fall back to internal self-play
            pass
    
    if not args.external_engines:
        # Original internal self-play logic
        procs: List[Process] = []
        q: Queue = Queue()
        for i in range(workers):
            p = Process(target=selfplay_worker, args=(i, cfg, args.ckpt, games_per_worker, q))
            p.start()
            procs.append(p)
        done = 0
        total = workers * games_per_worker
        try:
            while done < total:
                msg = q.get()
                if isinstance(msg, dict) and msg.get("type") == "game":
                    done += 1
                    print(f"[SelfPlay] {done}/{total} gms | p{msg['proc']} moves={msg['moves']} res={msg['result']} time={msg['secs']:.1f}s")
        finally:
            for p in procs:
                p.join()


def math_div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def schedule_params(sp_cfg: dict, move_no: int) -> Tuple[float, int]:
    temp = sp_cfg.get("temperature", 1.0)
    sims = sp_cfg.get("num_simulations", 200)
    schedule = sp_cfg.get("schedule", [])
    for entry in schedule:
        if move_no < int(entry.get("until_move", 0)):
            temp = float(entry.get("temperature", temp))
            sims = int(entry.get("num_simulations", sims))
            break
    return temp, sims


if __name__ == "__main__":
    main()
