from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import chess

from ..encoding import encode_board, move_to_index


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return e / s


def build_policy_from_multipv(board: chess.Board, multipv: List[dict], temp_cp: float, epsilon: float) -> np.ndarray:
    pi = np.zeros(4672, dtype=np.float32)
    if multipv:
        moves = []
        scores = []
        for item in multipv:
            try:
                mv = chess.Move.from_uci(item["move"])
                if mv not in board.legal_moves:
                    continue
                moves.append(mv)
                scores.append(float(item.get("score_cp", 0.0)))
            except Exception:
                continue
        if moves:
            logits = np.array(scores, dtype=np.float32) / max(1e-6, float(temp_cp))
            probs = softmax(logits)
            for mv, p in zip(moves, probs):
                pi[move_to_index(board, mv)] += float(p)
            # Optional smoothing to discourage zero mass
            if epsilon > 0:
                pi *= (1.0 - epsilon)
                # distribute epsilon uniformly over assigned indices
                nz = (pi > 0).sum()
                if nz > 0:
                    pi[pi > 0] += (epsilon / nz)
            return pi
    # Fallback: uniform over legals
    legal = list(board.legal_moves)
    if not legal:
        return pi
    p = 1.0 / len(legal)
    for mv in legal:
        pi[move_to_index(board, mv)] = p
    return pi


def convert_external_dir(selfplay_dir: str, replay_dir: str, shard_size: int = 16384, temp_cp: float = 200.0, epsilon: float = 0.0, backup_dir: str | None = None) -> Tuple[int, int]:
    sp = Path(selfplay_dir)
    rp = Path(replay_dir)
    rp.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(sp / "external_game_*.json")))
    if not files:
        return 0, 0
    s_buf: List[np.ndarray] = []
    pi_buf: List[np.ndarray] = []
    z_buf: List[np.ndarray] = []
    shards_written = 0
    samples_written = 0

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        meta = data.get("metadata", {})
        g = data.get("game_data", {})
        fens = g.get("fens", [])
        multipv_list = g.get("multipv", [])
        result = meta.get("result", 0.0)
        # Iterate positions
        for i, fen in enumerate(fens):
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            s = encode_board(board)
            multipv = multipv_list[i] if i < len(multipv_list) else []
            pi = build_policy_from_multipv(board, multipv, temp_cp=temp_cp, epsilon=epsilon)
            # z from perspective of side-to-move at this position
            z = float(result)
            if board.turn == chess.BLACK:
                z = -z
            s_buf.append(s.astype(np.float32))
            pi_buf.append(pi.astype(np.float32))
            z_buf.append(np.array(z, dtype=np.float32))
            if len(s_buf) >= shard_size:
                out = rp / f"ext_{shards_written:06d}.npz"
                np.savez_compressed(out, s=np.stack(s_buf), pi=np.stack(pi_buf), z=np.stack(z_buf))
                s_buf.clear(); pi_buf.clear(); z_buf.clear()
                shards_written += 1
        # Move or delete processed file
        try:
            if backup_dir:
                Path(backup_dir).mkdir(parents=True, exist_ok=True)
                Path(fpath).rename(Path(backup_dir) / Path(fpath).name)
            else:
                os.remove(fpath)
        except Exception:
            pass

    if s_buf:
        out = rp / f"ext_{shards_written:06d}.npz"
        np.savez_compressed(out, s=np.stack(s_buf), pi=np.stack(pi_buf), z=np.stack(z_buf))
        shards_written += 1
    samples_written = shards_written * shard_size if not s_buf else (shards_written - 1) * shard_size + len(s_buf)
    return shards_written, samples_written


def main():
    ap = argparse.ArgumentParser(description="Convert external engine JSON games to NPZ shards")
    ap.add_argument("--selfplay", type=str, default="data/selfplay")
    ap.add_argument("--replay", type=str, default="data/replays")
    ap.add_argument("--shard-size", type=int, default=16384)
    ap.add_argument("--temp-cp", type=float, default=200.0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--backup", type=str, default="data/backups/external")
    args = ap.parse_args()
    shards, samples = convert_external_dir(args.selfplay, args.replay, args.shard_size, args.temp_cp, args.epsilon, args.backup)
    print(f"Converted: shards={shards} samples≈{samples}")


if __name__ == "__main__":
    main()

