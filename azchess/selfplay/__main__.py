from __future__ import annotations

import argparse
from multiprocessing import Process, Queue
from typing import List

from ..config import Config
from .internal import selfplay_worker, math_div_ceil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
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

    if args.external_engines:
        try:
            # Defer import to avoid hard dependency if not used
            from .external_engine_worker import external_engine_worker
            import asyncio

            async def run_external_selfplay():
                games = await external_engine_worker(0, cfg_obj, cfg["selfplay"].get("buffer_dir", "data/selfplay"), args.games)
                return games

            games = asyncio.run(run_external_selfplay())
            print(f"[SelfPlay] Completed {len(games)} external engine games")
            return
        except Exception as e:
            print(f"[SelfPlay] External engine support failed: {e}")
            print("[SelfPlay] Falling back to internal self-play")

    workers = args.workers or cfg["selfplay"].get("num_workers", 2)
    games_per_worker = math_div_ceil(args.games, workers)
    procs: List[Process] = []
    q: Queue = Queue()
    for i in range(workers):
        p = Process(target=selfplay_worker, args=(i, cfg, args.ckpt, games_per_worker, q))
        p.start()
        procs.append(p)
    import time, queue as pyqueue
    done = 0
    total = workers * games_per_worker
    last_msg_time = time.time()
    try:
        while done < total:
            try:
                msg = q.get(timeout=2.0)
            except pyqueue.Empty:
                # If a worker died early, decrement target to avoid stalling forever
                alive = sum(1 for p in procs if p.is_alive())
                if alive == 0:
                    print("[SelfPlay] All workers exited; ending early.")
                    break
                if time.time() - last_msg_time > 300:
                    raise RuntimeError("[SelfPlay] Stalled: no progress for 300s")
                continue
            if isinstance(msg, dict) and msg.get("type") == "game":
                done += 1
                last_msg_time = time.time()
                print(f"[SelfPlay] {done}/{total} gms | p{msg['proc']} moves={msg['moves']} res={msg['result']} time={msg['secs']:.1f}s")
    finally:
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
