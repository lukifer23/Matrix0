from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import chess

from ..encoding import encode_board, move_to_index


def _one_hot_policy(board: chess.Board, move_uci: str) -> Optional[np.ndarray]:
    try:
        mv = chess.Move.from_uci(move_uci)
    except Exception:
        return None
    if mv not in board.legal_moves:
        # Try SAN if given accidentally
        try:
            mv = board.parse_san(move_uci)
        except Exception:
            return None
        if mv not in board.legal_moves:
            return None
    pi = np.zeros(4672, dtype=np.float32)
    try:
        pi[move_to_index(board, mv)] = 1.0
    except Exception:
        return None
    return pi


def _uniform_policy(board: chess.Board) -> np.ndarray:
    legal = list(board.legal_moves)
    pi = np.zeros(4672, dtype=np.float32)
    if not legal:
        return pi
    p = 1.0 / float(len(legal))
    for mv in legal:
        try:
            pi[move_to_index(board, mv)] = p
        except Exception:
            continue
    return pi


def _cp_to_value(cp: float) -> float:
    # Smooth mapping from centipawns to [-1,1]
    # Using tanh(cp / 300) is common; clamp for safety
    return float(np.tanh(cp / 300.0))


def convert_fen_bestmove(csv_path: str, shard_size: int) -> Tuple[int, int, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    s_buf: List[np.ndarray] = []
    pi_buf: List[np.ndarray] = []
    z_buf: List[np.ndarray] = []
    shards_written = 0
    samples_written = 0

    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fen = row.get("fen") or row.get("FEN")
            best = row.get("best_move") or row.get("best") or row.get("Move") or row.get("move")
            winp = row.get("winning_percentage") or row.get("win_pct") or row.get("win%")
            if not fen or not best:
                continue
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            pi = _one_hot_policy(board, best)
            if pi is None:
                # Skip illegal mappings
                continue
            # Value from winning percentage if present; interpret as side-to-move success likelihood
            if winp is not None and winp != "":
                try:
                    p = float(str(winp).replace('%','')) / 100.0
                    p = max(0.0, min(1.0, p))
                    z = 2.0 * p - 1.0
                except Exception:
                    z = 0.0
            else:
                z = 0.0
            s_buf.append(encode_board(board))
            pi_buf.append(pi)
            z_buf.append(np.array(z, dtype=np.float32))
    return shards_written, samples_written, s_buf, pi_buf, z_buf


def convert_fen_eval(csv_path: str, shard_size: int) -> Tuple[int, int, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    s_buf: List[np.ndarray] = []
    pi_buf: List[np.ndarray] = []
    z_buf: List[np.ndarray] = []
    shards_written = 0
    samples_written = 0
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fen = row.get("FEN") or row.get("fen")
            ev = row.get("Evaluation") or row.get("eval") or row.get("cp")
            if not fen or ev is None:
                continue
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            # Parse evaluations like "+56" or "-10" into centipawns
            try:
                evs = str(ev).strip()
                # Mate values like #5 are rare; approximate as large cp
                if evs.startswith('#'):
                    m = float(evs[1:]) if len(evs) > 1 else 0.0
                    cp = 10000.0 * (1.0 if m >= 0 else -1.0)
                else:
                    cp = float(evs)
            except Exception:
                continue
            # Map to value from side-to-move perspective (assuming cp from White’s perspective)
            if board.turn == chess.WHITE:
                z = _cp_to_value(cp)
            else:
                z = _cp_to_value(-cp)
            s_buf.append(encode_board(board))
            pi_buf.append(_uniform_policy(board))
            z_buf.append(np.array(z, dtype=np.float32))
    return shards_written, samples_written, s_buf, pi_buf, z_buf


def convert_puzzles(csv_path: str, shard_size: int) -> Tuple[int, int, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    s_buf: List[np.ndarray] = []
    pi_buf: List[np.ndarray] = []
    z_buf: List[np.ndarray] = []
    shards_written = 0
    samples_written = 0
    with open(csv_path, newline="") as f:
        # Try DictReader first; fall back to raw reader if no header
        peek = f.readline()
        f.seek(0)
        has_header = any(k in peek for k in ("FEN", "fen", "Moves", "moves"))
        if has_header:
            rdr = csv.DictReader(f)
            fen_col = 'FEN' if rdr.fieldnames and 'FEN' in rdr.fieldnames else ('fen' if rdr.fieldnames and 'fen' in rdr.fieldnames else None)
            moves_col = None
            for cand in ('Moves', 'moves', 'solution', 'Solution'):
                if rdr.fieldnames and cand in rdr.fieldnames:
                    moves_col = cand
                    break
            for row in rdr:
                fen = row.get(fen_col) if fen_col else row.get('FEN') or row.get('fen')
                mvseq = row.get(moves_col) if moves_col else row.get('Moves') or row.get('moves')
                if not fen or not mvseq:
                    continue
                try:
                    board = chess.Board(fen)
                except Exception:
                    continue
                parts = str(mvseq).strip().split()
                for u in parts:
                    pi = _one_hot_policy(board, u)
                    if pi is None:
                        break
                    s_buf.append(encode_board(board))
                    pi_buf.append(pi)
                    z_buf.append(np.array(0.8, dtype=np.float32))
                    board.push(chess.Move.from_uci(u))
        else:
            rdr = csv.reader(f)
            for row in rdr:
                if len(row) < 3:
                    continue
                fen = row[1]
                mvseq = row[2]
                try:
                    board = chess.Board(fen)
                except Exception:
                    continue
                parts = str(mvseq).strip().split()
                for u in parts:
                    pi = _one_hot_policy(board, u)
                    if pi is None:
                        break
                    s_buf.append(encode_board(board))
                    pi_buf.append(pi)
                    z_buf.append(np.array(0.8, dtype=np.float32))
                    board.push(chess.Move.from_uci(u))
    return shards_written, samples_written, s_buf, pi_buf, z_buf


def write_shards(out_dir: str, prefix: str, shard_size: int, s: List[np.ndarray], pi: List[np.ndarray], z: List[np.ndarray]) -> Tuple[int, int]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    shards = 0
    written = 0
    i = 0
    while i < len(s):
        j = min(i + shard_size, len(s))
        s_arr = np.stack(s[i:j])
        pi_arr = np.stack(pi[i:j])
        z_arr = np.stack(z[i:j])
        path = outp / f"{prefix}_{shards:06d}.npz"
        np.savez_compressed(path, s=s_arr, pi=pi_arr, z=z_arr)
        written += (j - i)
        shards += 1
        i = j
    return shards, written


def _parse_moves_list(cell: str) -> List[str]:
    """Parse a moves_list cell like "['1.e4', 'Nf6', '2.e5', 'Nd5']" into SAN tokens.
    Removes move numbers and keeps SAN moves only.
    """
    raw = cell.strip()
    # Remove brackets and quotes
    for ch in ['[', ']', '"', "'"]:
        raw = raw.replace(ch, '')
    tokens = [t.strip() for t in raw.split(',') if t.strip()]
    out: List[str] = []
    for tok in tokens:
        # Strip move numbers like "1.e4" → "e4"
        if '.' in tok:
            try:
                _, san = tok.split('.', 1)
                tok = san.strip()
            except ValueError:
                pass
        if tok:
            out.append(tok)
    return out


def convert_opening_san(csv_path: str, shard_size: int) -> Tuple[int, int, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    s_buf: List[np.ndarray] = []
    pi_buf: List[np.ndarray] = []
    z_buf: List[np.ndarray] = []
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            moves_list = row.get('moves_list')
            # Fallback: assemble from move1w, move1b, ... if present
            if not moves_list:
                parts: List[str] = []
                for i in range(1, 20):
                    for side in ('w', 'b'):
                        key = f'move{i}{side}'
                        val = row.get(key)
                        if val:
                            parts.append(val)
                if parts:
                    moves = parts
                else:
                    continue
            else:
                moves = _parse_moves_list(moves_list)
            if not moves:
                continue
            # Compute opening-wide expected value from win rates if available
            wwp = row.get('White_win%') or row.get('White_win_pct')
            bwp = row.get('Black_win%') or row.get('Black_win_pct')
            opening_z: Optional[float] = None
            try:
                if wwp is not None and bwp is not None:
                    pw = float(str(wwp).replace('%','')) / 100.0
                    pb = float(str(bwp).replace('%','')) / 100.0
                    opening_z = max(-1.0, min(1.0, (pw - pb)))
            except Exception:
                opening_z = None

            board = chess.Board()
            for san in moves:
                try:
                    mv = board.parse_san(san)
                except Exception:
                    break
                # record sample for current position
                s_buf.append(encode_board(board))
                pi = np.zeros(4672, dtype=np.float32)
                try:
                    pi[move_to_index(board, mv)] = 1.0
                except Exception:
                    # skip this position
                    s_buf.pop()
                    break
                pi_buf.append(pi)
                if opening_z is None:
                    z = 0.0
                else:
                    # from side-to-move perspective
                    z = opening_z if board.turn == chess.WHITE else -opening_z
                z_buf.append(np.array(z, dtype=np.float32))
                board.push(mv)
    return 0, 0, s_buf, pi_buf, z_buf


def main():
    ap = argparse.ArgumentParser(description="Convert CSV datasets (openings, evals, puzzles) into NPZ shards compatible with training")
    ap.add_argument('--csv', nargs='+', required=True, help='List of CSV files to convert')
    ap.add_argument('--format', choices=['fen_bestmove', 'fen_eval', 'puzzles', 'opening_san', 'auto'], default='auto', help='CSV format')
    ap.add_argument('--out', type=str, default='data/replays', help='Output replay directory')
    ap.add_argument('--prefix', type=str, default='extcsv', help='Output shard prefix')
    ap.add_argument('--shard-size', type=int, default=16384)
    args = ap.parse_args()

    s_all: List[np.ndarray] = []
    pi_all: List[np.ndarray] = []
    z_all: List[np.ndarray] = []

    for path in args.csv:
        # Skip missing files without error to support fluid datasets
        p = Path(path)
        if not p.exists():
            print(f'SKIP: {path} does not exist')
            continue
        fmt = args.format
        try:
            if fmt == 'auto':
                # naive sniffing by header
                with open(path, newline="") as f:
                    rdr = csv.reader(f)
                    header = next(rdr, [])
                hdr = [h.strip() for h in header]
                low = set(map(str.lower, hdr))
                if {'fen', 'best_move'} <= low:
                    fmt = 'fen_bestmove'
                elif ('evaluation' in low) and ('fen' in low):
                    fmt = 'fen_eval'
                elif ('moves' in low) and ('fen' in low):
                    fmt = 'puzzles'
                elif ('moves_list' in low) or any(h.startswith('move') for h in hdr):
                    fmt = 'opening_san'
                else:
                    # fallback: try fen_bestmove first, then fen_eval
                    fmt = 'fen_bestmove'

            if fmt == 'fen_bestmove':
                _, _, s, p, z = convert_fen_bestmove(path, args.shard_size)
            elif fmt == 'fen_eval':
                _, _, s, p, z = convert_fen_eval(path, args.shard_size)
            elif fmt == 'puzzles':
                _, _, s, p, z = convert_puzzles(path, args.shard_size)
            elif fmt == 'opening_san':
                _, _, s, p, z = convert_opening_san(path, args.shard_size)
            else:
                continue
            s_all.extend(s); pi_all.extend(p); z_all.extend(z)
        except Exception as e:
            print(f'SKIP: failed to convert {path}: {e}')
            continue

    if not s_all:
        print('No samples converted from specified CSV files')
        return

    shards, samples = write_shards(args.out, args.prefix, args.shard_size, s_all, pi_all, z_all)
    print(f'Wrote {shards} shards with {samples} samples to {args.out}')


if __name__ == '__main__':
    main()
