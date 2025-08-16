from __future__ import annotations

import argparse
from multiprocessing import Process, Queue
from typing import List

from ..config import Config
from .worker import selfplay_worker

def math_div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b

def main():
    parser = argparse.ArgumentParser(description="Run Matrix0 self-play generation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--games", type=int, default=16, help="Total number of games to generate.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg_dict = cfg.to_dict()
    
    workers = args.workers or cfg.selfplay().get("num_workers", 2)
    games_per_worker = math_div_ceil(args.games, workers)

    print(f"Starting self-play with {workers} workers, generating {games_per_worker} games each.")
    
    procs: List[Process] = []
    q: Queue = Queue()
    for i in range(workers):
        p = Process(target=selfplay_worker, args=(i, cfg_dict, args.ckpt, games_per_worker, q))
        p.start()
        procs.append(p)
        
    done = 0
    total = workers * games_per_worker
    try:
        while done < total:
            msg = q.get()
            if isinstance(msg, dict) and msg.get("type") == "game":
                done += 1
                print(f"[SelfPlay] {done}/{total} games | "
                      f"proc {msg['proc']} | "
                      f"moves={msg['moves']} | "
                      f"res={msg['result']} | "
                      f"time={msg['secs']:.1f}s")
    finally:
        for p in procs:
            p.join()

if __name__ == "__main__":
    main()