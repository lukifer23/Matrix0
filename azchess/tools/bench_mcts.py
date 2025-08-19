from __future__ import annotations

import argparse
import time
import random
from multiprocessing import Process, Event, Queue

import numpy as np
import chess
import torch

from ..config import Config, select_device
from ..model import PolicyValueNet
from ..mcts import MCTS, MCTSConfig
from ..encoding import encode_board
from ..selfplay.inference import run_inference_server, InferenceClient


def random_board(max_plies: int = 40, seed: int | None = None) -> chess.Board:
    if seed is not None:
        random.seed(seed)
    b = chess.Board()
    for _ in range(random.randint(1, max_plies)):
        if b.is_game_over():
            break
        mv = random.choice(list(b.legal_moves))
        b.push(mv)
    return b


def bench_mcts(cfg_path: str, sims: int, boards: int, shared: bool) -> None:
    cfg = Config.load(cfg_path)
    device = select_device(cfg.get("device", "auto"))

    # Optional shared inference server
    req_q = None
    server_proc = None
    stop_event = None
    if shared and device != 'cpu':
        req_q = Queue()
        stop_event = Event()
        ckpt = cfg.training().get("checkpoint_dir", "checkpoints") + "/best.pt"
        resp_qs = [Queue()]
        server_proc = Process(target=run_inference_server, args=(device, cfg.model(), ckpt, req_q, resp_qs, stop_event))
        server_proc.start()
        backend = InferenceClient(req_q, resp_qs[0], slot=0)
    else:
        backend = None

    model = None if backend is not None else PolicyValueNet.from_config(cfg.model()).to(device)
    if model is not None:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    mcts = MCTS(model, MCTSConfig(num_simulations=sims, batch_size=32, cpuct=float(cfg.selfplay().get('cpuct', 2.5))), device, inference_backend=backend)

    # Generate boards
    bs = [random_board() for _ in range(boards)]
    t0 = time.time()
    total_sims = 0
    for b in bs:
        _vc, _pi, _v = mcts.run(b)
        total_sims += getattr(mcts, '_last_sims_run', sims)
    dt = time.time() - t0
    print(f"device={device} shared={shared} sims={sims} boards={boards} | {boards/dt:.2f} boards/s | {total_sims/dt:.1f} sims/s | time={dt:.3f}s")

    if server_proc is not None:
        stop_event.set()
        server_proc.join(timeout=5)


def main():
    ap = argparse.ArgumentParser(description="MCTS throughput benchmark")
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--sims', type=int, default=128)
    ap.add_argument('--boards', type=int, default=64)
    ap.add_argument('--shared', action='store_true', help='Use shared inference')
    args = ap.parse_args()
    bench_mcts(args.config, args.sims, args.boards, args.shared)


if __name__ == '__main__':
    main()
