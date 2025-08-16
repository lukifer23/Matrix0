from __future__ import annotations

import argparse
import random
from pathlib import Path
import numpy as np
import time
import os
import chess
import chess.pgn
import torch

from .config import Config, select_device
from .model import PolicyValueNet
from .mcts import MCTS, MCTSConfig


def _wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z*z/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z * ((p*(1-p)/n) + (z*z)/(4*n*n))**0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _save_pgn(moves: list[chess.Move], result: str, headers: dict, out_dir: str, idx: int) -> None:
    game = chess.pgn.Game()
    node = game
    for k, v in headers.items():
        game.headers[k] = str(v)
    board = chess.Board()
    for mv in moves:
        node = node.add_variation(mv)
        board.push(mv)
    game.headers["Result"] = result
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"game_{idx:04d}.pgn")
    with open(path, "w") as f:
        print(game, file=f)


def play_match(ckpt_a: str, ckpt_b: str, games: int, cfg: Config, seed: int | None = None, pgn_out: str | None = None, pgn_sample: int = 0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
    device = select_device(cfg.get("device", "auto"))
    e = cfg.eval()
    mcfg = MCTSConfig(
        num_simulations=int(e.get("num_simulations", 200)),
        cpuct=float(e.get("cpuct", 1.5)),
        dirichlet_alpha=float(e.get("dirichlet_alpha", 0.3)),
        dirichlet_frac=0.0,
        tt_capacity=int(e.get("tt_capacity", 200000)),
        selection_jitter=0.0,
    )
    model_a = PolicyValueNet.from_config(cfg.model()).to(device)
    model_b = PolicyValueNet.from_config(cfg.model()).to(device)
    state_a = torch.load(ckpt_a, map_location=device)
    state_b = torch.load(ckpt_b, map_location=device)
    if "model_ema" in state_a:
        model_a.load_state_dict(state_a["model_ema"])
    else:
        model_a.load_state_dict(state_a["model"])
    if "model_ema" in state_b:
        model_b.load_state_dict(state_b["model_ema"])
    else:
        model_b.load_state_dict(state_b["model"])
    mcts_a = MCTS(model_a, mcfg, device)
    mcts_b = MCTS(model_b, mcfg, device)

    score = 0.0
    pgn_kept = 0
    for g in range(games):
        board = chess.Board()
        if g % 2 == 1:
            engines = [mcts_b, mcts_a]
        else:
            engines = [mcts_a, mcts_b]
        trace: list[chess.Move] = []
        while not board.is_game_over():
            mcts = engines[0] if board.turn == chess.WHITE else engines[1]
            visits, pi, _ = mcts.run(board)
            move = max(visits.items(), key=lambda kv: kv[1])[0]
            board.push(move)
            trace.append(move)
        if board.result() == "1-0":
            score += 1.0
        elif board.result() == "1/2-1/2":
            score += 0.5
        if pgn_out and pgn_kept < max(0, pgn_sample):
            headers = {
                "Event": "Matrix0 Eval",
                "Site": "Local",
                "Date": time.strftime("%Y.%m.%d"),
                "Round": g + 1,
                "White": "A" if g % 2 == 0 else "B",
                "Black": "B" if g % 2 == 0 else "A",
            }
            _save_pgn(trace, board.result(), headers, pgn_out, pgn_kept)
            pgn_kept += 1
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--ckpt_b", type=str, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pgn-out", type=str, default=None)
    parser.add_argument("--pgn-sample", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    score = play_match(args.ckpt_a, args.ckpt_b, args.games, cfg, seed=args.seed, pgn_out=args.pgn_out, pgn_sample=args.pgn_sample)
    wr = score / float(args.games)
    lo, hi = _wilson_interval(wr, args.games)
    print(f"Score (A as White first): {score} / {args.games} (win_rate={wr:.3f}, CI95=[{lo:.3f},{hi:.3f}])")


if __name__ == "__main__":
    main()

