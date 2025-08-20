from __future__ import annotations

import argparse
import sys

import chess
import torch

from .config import Config, select_device
from .model import PolicyValueNet
from .mcts import MCTS, MCTSConfig


def print_board(board: chess.Board):
    print(board)
    print("Fen:", board.fen())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    # Strict encoding enforcement via env
    try:
        import os as _os
        if bool(cfg.get("strict_encoding", False)):
            _os.environ["MATRIX0_STRICT_ENCODING"] = "1"
    except Exception:
        pass
    device = select_device(cfg.get("device", "auto"))
    model = PolicyValueNet.from_config(cfg.model()).to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        if "model_ema" in state:
            model.load_state_dict(state["model_ema"])
        else:
            model.load_state_dict(state["model"])
    mcfg_dict = dict(cfg.mcts())
    mcfg_dict.update(
        {
            "num_simulations": args.sims,
            "cpuct": cfg.selfplay().get("cpuct", mcfg_dict.get("cpuct", 1.5)),
            "dirichlet_frac": 0.0,
            "tt_cleanup_frequency": int(cfg.mcts().get("tt_cleanup_frequency", 500)),
        }
    )
    mcts = MCTS(model, MCTSConfig.from_dict(mcfg_dict), device=device)

    board = chess.Board()
    print("Play vs engine. Enter moves in UCI (e.g., e2e4, e7e8q). Type 'quit' to exit.")
    print_board(board)
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            mv = input("Your move (UCI): ").strip()
            if mv.lower() in {"q", "quit", "exit"}:
                sys.exit(0)
            try:
                move = chess.Move.from_uci(mv)
            except Exception:
                print("Invalid UCI. Try again.")
                continue
            if move not in board.legal_moves:
                print("Illegal move. Try again.")
                continue
            board.push(move)
        else:
            visits, pi, _ = mcts.run(board)
            move = max(visits.items(), key=lambda kv: kv[1])[0]
            print(f"Engine plays: {move.uci()}")
            board.push(move)
        print_board(board)
    print("Game over:", board.result())


if __name__ == "__main__":
    main()
