from __future__ import annotations

import argparse
import time
from multiprocessing import Event, Process, Queue

import chess
import torch

from ..config import Config, select_device
from ..mcts import MCTS, MCTSConfig
from ..model import PolicyValueNet
from ..selfplay.inference import InferenceClient, run_inference_server
from ..utils.board import random_board


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

    mcfg_dict = dict(cfg.mcts())
    mcfg_dict.update(
        {
            "num_simulations": sims,
            "batch_size": 32,
            "cpuct": float(cfg.selfplay().get('cpuct', mcfg_dict.get('cpuct', 2.5))),
            "tt_cleanup_frequency": int(cfg.mcts().get("tt_cleanup_frequency", 500)),
        }
    )
    mcts = MCTS(MCTSConfig.from_dict(mcfg_dict), model, device, inference_backend=backend)

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
