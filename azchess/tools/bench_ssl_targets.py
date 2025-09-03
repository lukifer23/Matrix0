#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List

import chess
import torch

from azchess.encoding import encode_board
from azchess.ssl_algorithms import ChessSSLAlgorithms, get_ssl_algorithms


def random_board(max_plies: int = 40) -> chess.Board:
    b = chess.Board()
    plies = random.randint(0, max_plies)
    for _ in range(plies):
        if b.is_game_over(claim_draw=True):
            break
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(random.choice(moves))
    return b


def build_batch_states(n: int) -> torch.Tensor:
    arrs = []
    for _ in range(n):
        brd = random_board()
        arrs.append(encode_board(brd))
    import numpy as np
    arr = np.stack(arrs, axis=0)
    x = torch.from_numpy(arr)
    return x


def main():
    ap = argparse.ArgumentParser(description="Benchmark SSL target generation (piece, threat, pin, fork, control)")
    ap.add_argument("--batches", type=int, default=10, help="Number of batches")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"], help="Device for tensor ops")
    ap.add_argument("--tasks", type=str, default="all", help="Comma-separated subset of tasks or 'all'")
    args = ap.parse_args()

    device = args.device
    ssl: ChessSSLAlgorithms = get_ssl_algorithms()
    tasks: List[str] = ["piece", "threat", "pin", "fork", "control"] if args.tasks == "all" else [t.strip() for t in args.tasks.split(",") if t.strip()]

    totals: Dict[str, float] = {t: 0.0 for t in tasks}
    totals["all"] = 0.0
    total_positions = 0

    for _ in range(args.batches):
        x = build_batch_states(args.batch_size).to(device)
        total_positions += x.size(0)

        # End-to-end (all)
        t0 = time.perf_counter()
        _ = ssl.create_enhanced_ssl_targets(x)
        totals["all"] += (time.perf_counter() - t0)

        # Per-task
        if "piece" in tasks:
            t0 = time.perf_counter()
            _ = ssl._create_piece_targets(x)
            totals["piece"] += (time.perf_counter() - t0)
        if "threat" in tasks:
            t0 = time.perf_counter()
            _ = ssl.detect_threats_batch(x)
            totals["threat"] += (time.perf_counter() - t0)
        if "pin" in tasks:
            t0 = time.perf_counter()
            _ = ssl.detect_pins_batch(x)
            totals["pin"] += (time.perf_counter() - t0)
        if "fork" in tasks:
            t0 = time.perf_counter()
            _ = ssl.detect_forks_batch(x)
            totals["fork"] += (time.perf_counter() - t0)
        if "control" in tasks:
            t0 = time.perf_counter()
            _ = ssl.calculate_square_control_batch(x)
            totals["control"] += (time.perf_counter() - t0)

    # Report
    print("SSL Target Generation Benchmark")
    print(f"Batches: {args.batches}  Batch size: {args.batch_size}  Device: {device}")
    print(f"Total positions: {total_positions}")
    if totals["all"] > 0:
        print(f"All tasks: {totals['all']:.3f}s  ({total_positions / totals['all']:.1f} pos/s)")
    for t in tasks:
        dur = totals.get(t, 0.0)
        if dur > 0:
            print(f"  {t:8s}: {dur:.3f}s  ({total_positions / dur:.1f} pos/s)")


if __name__ == "__main__":
    main()
