
from __future__ import annotations

import random
from time import perf_counter
from typing import Dict, List, Tuple

import chess
import numpy as np
import torch
from multiprocessing import Queue

from ..config import select_device
from ..data_manager import DataManager
from ..encoding import encode_board, move_to_index
from ..mcts import MCTS, MCTSConfig
from ..model import PolicyValueNet


def selfplay_worker(proc_id: int, cfg_dict: dict, ckpt_path: str | None, games: int, q: Queue | None = None):
    random.seed(1234 + proc_id)
    np.random.seed(1234 + proc_id)

    device = select_device(cfg_dict.get("device", "auto"))

    model = PolicyValueNet.from_config(cfg_dict["model"]).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_ema", state["model"]))

    mcts = MCTS(model, MCTSConfig(**cfg_dict["selfplay"]), device=device)
    data_manager = DataManager(base_dir=cfg_dict.get("data_dir", "data"))

    for g in range(games):
        board = chess.Board()
        states: List[np.ndarray] = []
        pis: List[np.ndarray] = []
        turns: List[int] = []
        t0 = perf_counter()
        while not board.is_game_over() and len(states) < cfg_dict["selfplay"].get("max_game_len", 512):
            mv_no = len(states)
            temperature, sims = schedule_params(cfg_dict["selfplay"], mv_no)
            visit_counts, pi, v = mcts.run(board, num_simulations=sims)
            move = sample_move_from_counts(visit_counts, temperature)
            states.append(encode_board(board))
            pis.append(pi)
            turns.append(1 if board.turn == chess.WHITE else -1)
            board.push(move)

        z = game_result(board)
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
            })


def sample_move_from_counts(counts: Dict[chess.Move, int], temperature: float) -> chess.Move:
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
