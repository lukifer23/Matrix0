from __future__ import annotations

import argparse
from multiprocessing import Process, Queue
from typing import List

from ..config import Config
from .internal import math_div_ceil, selfplay_worker


def _join_or_terminate_workers(
    procs: List[Process],
    done: int,
    total: int,
    join_timeout: float = 30.0,
    terminate_timeout: float = 10.0,
) -> None:
    """Join workers, but do not let completed self-play hang forever on shutdown."""
    for p in procs:
        p.join(timeout=join_timeout)

    hung = [p for p in procs if p.is_alive()]
    if not hung:
        return

    if done >= total:
        print(
            f"[SelfPlay] Completed {done}/{total} games, but {len(hung)} worker(s) did not exit; terminating shutdown hang"
        )
    else:
        print(f"[SelfPlay] Terminating {len(hung)} stalled worker(s): done={done}/{total}")

    for p in hung:
        p.terminate()
    for p in hung:
        p.join(timeout=terminate_timeout)

    still_alive = [p for p in hung if p.is_alive()]
    for p in still_alive:
        if hasattr(p, "kill"):
            p.kill()
    for p in still_alive:
        p.join(timeout=terminate_timeout)


def _handle_worker_message(msg, done: int, total: int) -> tuple[int, bool]:
    """Handle one worker message and report whether it counts as progress."""
    if not isinstance(msg, dict):
        return done, False

    if msg.get("type") == "game":
        done += 1
        source = msg.get("result_source", "unknown")
        capped = " capped" if msg.get("capped") else ""
        print(
            f"[SelfPlay] {done}/{total} gms | p{msg['proc']} moves={msg['moves']} "
            f"res={msg['result']} src={source}{capped} time={msg['secs']:.1f}s"
        )
        return done, True

    if msg.get("type") == "heartbeat":
        moves = msg.get("moves", 0)
        avg_sims = float(msg.get("avg_sims", 0.0) or 0.0)
        entropy = float(msg.get("avg_policy_entropy", 0.0) or 0.0)
        print(
            f"[SelfPlay] heartbeat p{msg.get('proc', '?')} game={msg.get('game', '?')} "
            f"moves={moves} avg_sims={avg_sims:.1f} entropy={entropy:.3f}"
        )
        return done, True

    return done, False


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
            import asyncio

            from .external_engine_worker import external_engine_worker

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
        start = i * games_per_worker
        end = min(args.games, start + games_per_worker)
        assigned_games = max(0, end - start)
        if assigned_games == 0:
            continue
        p = Process(target=selfplay_worker, args=(i, cfg, args.ckpt, assigned_games, q))
        p.start()
        procs.append(p)
    import queue as pyqueue
    import time
    done = 0
    total = args.games
    last_msg_time = time.time()
    try:
        while done < total:
            try:
                msg = q.get(timeout=2.0)
            except pyqueue.Empty:
                failed = [p for p in procs if p.exitcode not in (None, 0)]
                if failed:
                    codes = [p.exitcode for p in failed]
                    raise RuntimeError(f"[SelfPlay] Worker failed before completing requested games: exitcodes={codes}, done={done}/{total}")
                if all(not p.is_alive() for p in procs):
                    raise RuntimeError(f"[SelfPlay] All workers exited before completing requested games: done={done}/{total}")
                if time.time() - last_msg_time > 300:
                    raise RuntimeError("[SelfPlay] Stalled: no progress for 300s")
                continue
            done, made_progress = _handle_worker_message(msg, done, total)
            if made_progress:
                last_msg_time = time.time()
    finally:
        _join_or_terminate_workers(procs, done, total)
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass
    failed = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    if failed and done != total:
        raise RuntimeError(f"[SelfPlay] Worker failure after join: exitcodes={failed}")
    if failed:
        print(f"[SelfPlay] Completed requested games despite worker shutdown exitcodes={failed}")
    if done != total:
        raise RuntimeError(f"[SelfPlay] Completed {done}/{total} requested games")


if __name__ == "__main__":
    main()
