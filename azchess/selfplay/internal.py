from __future__ import annotations

from multiprocessing import Queue
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import chess
import torch
import os
import random

from ..config import select_device
from ..model import PolicyValueNet
from ..mcts import MCTS, MCTSConfig
from ..data_manager import DataManager
from ..encoding import encode_board, move_to_index


def selfplay_worker(proc_id: int, cfg_dict: dict, ckpt_path: str | None, games: int, q: Queue | None = None):
    random.seed(1234 + proc_id)
    np.random.seed(1234 + proc_id)

    device = select_device(cfg_dict.get("device", "auto"))

    model = PolicyValueNet.from_config(cfg_dict["model"]).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_ema", state.get("model", {})))

    sp_cfg = cfg_dict["selfplay"]
    mcts = MCTS(
        model,
        MCTSConfig(
            num_simulations=int(sp_cfg.get("num_simulations", 200)),
            cpuct=float(sp_cfg.get("cpuct", 1.5)),
            dirichlet_alpha=float(sp_cfg.get("dirichlet_alpha", 0.3)),
            dirichlet_frac=float(sp_cfg.get("dirichlet_frac", 0.25)),
            tt_capacity=int(sp_cfg.get("tt_capacity", 200000)),
            selection_jitter=float(sp_cfg.get("selection_jitter", 0.0)),
        ),
        device=device,
    )
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
