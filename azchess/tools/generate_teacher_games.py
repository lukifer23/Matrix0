#!/usr/bin/env python3
from __future__ import annotations

"""
Teacher-guided game/position generator for Matrix0.

Generates supervised training data by comparing the model's choice to a
Stockfish teacher and collecting positions where the model deviates or
causes a significant evaluation swing.

Saved NPZ shard keys:
  - s: (N, planes, 8, 8) float32 board encodings
  - pi: (N, policy_size) float32 teacher policy distribution (soft labels)
  - z: (N,) float32 scalar value in [-1,1] (tanh(cp/600) from side-to-move)
  - legal_mask: (N, policy_size) uint8 legal move mask
  - meta_cp_before, meta_cp_best, meta_cp_after, meta_cp_swing: (N,) float32
  - meta_topk_hit: (N,) uint8 (1 if model move in teacher top-K)

DB source tag: "teacher:<scenario>"
Output layout: data/teacher_games/<scenario>/teacher_<scenario>_*.npz
"""

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess
import chess.engine
import torch

from azchess.config import Config, select_device
from azchess.encoding import encode_board, move_to_index
from azchess.model import PolicyValueNet
from azchess.data_manager import DataManager

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    scenario: str
    games: int
    model_path: str
    sims: int
    stockfish_path: str
    threads: int
    hash_mb: int
    movetime_ms: Optional[int]
    depth: Optional[int]
    multipv: int
    topk: int
    cp_swing_mid: int
    cp_swing_end: int
    teacher_adv_thresh: int
    max_moves: int
    hb_every: int
    shard_size: int
    output_dir: str


def _score_to_cp(score: chess.engine.PovScore, turn_white: bool) -> float:
    try:
        s = score.white() if turn_white else score.black()
        if s.is_mate():
            # Map mate distances to large finite cp
            sign = 1.0 if s.mate() and s.mate() > 0 else -1.0
            return 10000.0 * sign
        return float(s.cp)
    except Exception:
        return 0.0


def _cp_to_value(cp: float) -> float:
    # Side-to-move value; squash with tanh
    return float(math.tanh(cp / 600.0))


def _softmax_cp(weights: List[Tuple[str, float]], temperature_cp: float) -> Dict[str, float]:
    if not weights:
        return {}
    T = max(1e-6, float(temperature_cp))
    xs = np.array([w for _, w in weights], dtype=np.float32) / T
    # Normalize by subtracting max for stability
    xs = xs - xs.max()
    exps = np.exp(xs)
    exps /= exps.sum() if exps.sum() > 0 else 1.0
    return {uci: float(p) for (uci, _), p in zip(weights, exps.tolist())}


def _collect_position(board: chess.Board,
                      teacher: chess.engine.SimpleEngine,
                      model: PolicyValueNet,
                      policy_size: int,
                      topk: int,
                      multipv: int,
                      movetime_ms: Optional[int],
                      depth: Optional[int],
                      cp_swing_mid: int,
                      cp_swing_end: int,
                      teacher_adv_thresh: int) -> Optional[Dict[str, np.ndarray]]:
    # Encode board
    enc = encode_board(board)
    if enc is None:
        return None
    s = torch.from_numpy(enc[None, ...]).to(model.device)
    with torch.no_grad():
        out = model(s, return_ssl=False)
        # Handle both (p,v) and (p,v,ssl_out)
        if isinstance(out, (list, tuple)):
            if len(out) >= 2:
                p_logits, v = out[0], out[1]
            else:
                raise RuntimeError("Model forward returned unexpected shape")
        else:
            raise RuntimeError("Model forward returned non-tuple result")
        p = p_logits[0].float().cpu().numpy()
        value = float(v[0].item())

    # Legal mask and model policy over legal moves
    legal_indices: List[int] = []
    legal_mask = np.zeros((policy_size,), dtype=np.uint8)
    legal_moves: List[chess.Move] = []
    for mv in board.legal_moves:
        try:
            idx = move_to_index(board, mv)
            if 0 <= idx < policy_size:
                legal_indices.append(idx)
                legal_mask[idx] = 1
                legal_moves.append(mv)
        except Exception:
            continue
    if not legal_indices:
        return None
    legal_probs = p[legal_indices]
    legal_probs = np.exp(legal_probs - legal_probs.max())
    legal_probs = legal_probs / (legal_probs.sum() if legal_probs.sum() > 0 else 1.0)
    model_entropy = -float(np.sum(legal_probs * np.log(np.clip(legal_probs, 1e-9, 1.0))))

    # Model move: greedy on legal_probs
    choice_idx = int(np.argmax(legal_probs))
    model_move = legal_moves[choice_idx]

    # Teacher analysis before move
    limit = chess.engine.Limit(time=movetime_ms / 1000.0) if movetime_ms else chess.engine.Limit(depth=int(depth or 10))
    info = teacher.analyse(board, limit=limit, multipv=int(max(multipv, topk)))

    teacher_lines: List[Tuple[str, float]] = []
    teacher_cp_best = 0.0
    for entry in info:
        pv = entry.get('pv')
        if not pv:
            continue
        move_uci = pv[0].uci()
        cp = _score_to_cp(entry.get('score'), board.turn)
        teacher_lines.append((move_uci, cp))
        if len(teacher_lines) == 1:
            teacher_cp_best = cp

    if not teacher_lines:
        return None
    topk_ucis = {uci for uci, _ in teacher_lines[:topk]}
    model_move_uci = model_move.uci()
    topk_hit = 1 if model_move_uci in topk_ucis else 0

    # CP before move
    cp_before = teacher_cp_best

    # Apply model move and evaluate after
    board.push(model_move)
    try:
        info_after = teacher.analyse(board, limit=limit, multipv=1)
        cp_after = _score_to_cp(info_after[0].get('score'), board.turn ^ True)
    finally:
        board.pop()

    cp_swing = float(cp_before - cp_after)

    # Decide if this position is a training candidate
    phase_endgame = (len(board.move_stack) >= 60)
    swing_thresh = float(cp_swing_end if phase_endgame else cp_swing_mid)
    keep = (topk_hit == 0) or (abs(cp_swing) >= swing_thresh) or (abs(cp_before) >= float(teacher_adv_thresh) and model_entropy >= 1.25)
    if not keep:
        return None

    # Teacher policy distribution over top moves (soft labels)
    teacher_dist = _softmax_cp(teacher_lines[:max(topk, multipv)], temperature_cp=180.0)
    pi = np.zeros((policy_size,), dtype=np.float32)
    for uci, prob in teacher_dist.items():
        try:
            mv = chess.Move.from_uci(uci)
            idx = move_to_index(board, mv)
            if 0 <= idx < policy_size:
                pi[idx] = float(prob)
        except Exception:
            continue
    # Ensure non-empty distribution
    if pi.sum() <= 0:
        # Fallback: one-hot teacher best
        try:
            mv = chess.Move.from_uci(teacher_lines[0][0])
            idx = move_to_index(board, mv)
            if 0 <= idx < policy_size:
                pi[idx] = 1.0
        except Exception:
            return None

    # Value target from teacher cp_before
    z = np.array([_cp_to_value(cp_before)], dtype=np.float32)

    return {
        's': enc.astype(np.float32, copy=False),
        'pi': pi,
        'z': z,
        'legal_mask': legal_mask,
        'meta': np.array([model_entropy, cp_before, teacher_cp_best, cp_after, cp_swing, float(topk_hit)], dtype=np.float32)
    }


def run(cfg: TeacherConfig):
    # Minimal logging setup
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-7s %(message)s')

    # Load config to get planes/policy size
    conf = Config.load("config.yaml")
    model_cfg = conf.model()
    policy_size = int(model_cfg.get('policy_size', 4672))
    device = select_device(conf)

    # Load model
    model = PolicyValueNet.from_config(model_cfg)
    chk = torch.load(cfg.model_path, map_location='cpu', weights_only=False)
    # Prefer EMA if present for smoother targets
    sd = chk.get('model_ema') or chk.get('model_state_dict') or chk.get('model')
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.device = device  # convenience

    # Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(cfg.stockfish_path)
    try:
        engine.configure({'Threads': str(cfg.threads), 'Hash': str(cfg.hash_mb), 'MultiPV': str(cfg.multipv)})
    except Exception:
        pass

    # Output dir
    out_root = Path(cfg.output_dir) / cfg.scenario
    out_root.mkdir(parents=True, exist_ok=True)

    collected: List[Dict[str, np.ndarray]] = []
    sample_meta: List[np.ndarray] = []
    accepted = 0
    last_hb = 0

    try:
        for g in range(1, cfg.games + 1):
            board = chess.Board()

            for ply in range(cfg.max_moves):
                if board.is_game_over():
                    break
                rec = _collect_position(
                    board, engine, model, policy_size,
                    cfg.topk, cfg.multipv, cfg.movetime_ms, cfg.depth,
                    cfg.cp_swing_mid, cfg.cp_swing_end, cfg.teacher_adv_thresh
                )
                if rec is not None:
                    collected.append({'s': rec['s'], 'pi': rec['pi'], 'z': rec['z'], 'legal_mask': rec['legal_mask']})
                    sample_meta.append(rec['meta'])
                    accepted += 1

                # Make the actual move: follow model's greedy policy (consistent with selection above)
                # We recompute here to avoid side-effects from previous rec call.
                enc = encode_board(board)
                if enc is None:
                    break
                with torch.no_grad():
                    out2 = model(torch.from_numpy(enc[None, ...]).to(model.device), return_ssl=False)
                    if isinstance(out2, (list, tuple)):
                        p_logits2 = out2[0]
                    else:
                        raise RuntimeError("Model forward returned non-tuple result")
                p = p_logits2[0].float().cpu().numpy()
                legals: List[Tuple[chess.Move, int, float]] = []
                for mv in board.legal_moves:
                    try:
                        idx = move_to_index(board, mv)
                        legals.append((mv, idx, p[idx]))
                    except Exception:
                        continue
                if not legals:
                    break
                # Greedy move
                legals.sort(key=lambda x: x[2], reverse=True)
                board.push(legals[0][0])

            # Heartbeat
            if cfg.hb_every and accepted - last_hb >= cfg.hb_every:
                logger.info(f"HB: accepted={accepted} games_done={g}/{cfg.games}")
                last_hb = accepted

            # Flush shard when big enough or last game
            if accepted >= cfg.shard_size or (g == cfg.games and accepted > 0):
                s = np.stack([x['s'] for x in collected], axis=0)
                pi = np.stack([x['pi'] for x in collected], axis=0)
                z = np.concatenate([x['z'] for x in collected], axis=0)
                lm = np.stack([x['legal_mask'] for x in collected], axis=0)
                meta = np.stack(sample_meta, axis=0)
                ts = time.strftime('%Y%m%d_%H%M%S')
                out_path = out_root / f"teacher_{cfg.scenario}_{ts}_{accepted}.npz"
                np.savez_compressed(out_path, s=s, pi=pi, z=z, legal_mask=lm,
                                    meta=meta,
                                    meta_keys=np.array(['entropy','cp_before','cp_best','cp_after','cp_swing','topk_hit']))
                logger.info(f"Saved teacher shard: {out_path.name} (N={accepted})")
                # Register in DB
                try:
                    dm = DataManager(base_dir='data')
                    dm.import_replay_dir(str(out_root), source=f"teacher:{cfg.scenario}", move_files=False)
                except Exception as e:
                    logger.warning(f"DB import failed: {e}")
                # Reset buffers
                collected.clear()
                sample_meta.clear()
                accepted = 0
                last_hb = 0

    finally:
        try:
            engine.quit()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description='Generate teacher-guided datasets (Matrix0 vs Stockfish)')
    ap.add_argument('--scenario', required=True, help='Scenario tag, e.g., teacher_multipv, teacher_cp_swing')
    ap.add_argument('--games', type=int, default=20, help='Number of short games to play')
    ap.add_argument('--model', required=True, help='Path to Matrix0 checkpoint (.pt)')
    ap.add_argument('--sims', type=int, default=0, help='Reserved for MCTS sims (not used in this version)')
    ap.add_argument('--stockfish-path', default='stockfish', help='Path to Stockfish executable')
    ap.add_argument('--threads', type=int, default=2, help='UCI Threads')
    ap.add_argument('--hash-mb', type=int, default=256, help='UCI Hash in MB')
    ap.add_argument('--movetime-ms', type=int, default=25, help='Per-position movetime in ms')
    ap.add_argument('--depth', type=int, default=None, help='Analysis depth (overrides movetime if set)')
    ap.add_argument('--multipv', type=int, default=4, help='Stockfish MultiPV lines')
    ap.add_argument('--topk', type=int, default=3, help='Top-K teacher moves to consider a match')
    ap.add_argument('--cp-swing-mid', type=int, default=120, help='CP swing threshold (midgame)')
    ap.add_argument('--cp-swing-end', type=int, default=80, help='CP swing threshold (endgame)')
    ap.add_argument('--teacher-adv-thresh', type=int, default=120, help='Teacher advantage to keep high-entropy positions')
    ap.add_argument('--max-moves', type=int, default=120, help='Max plies per game')
    ap.add_argument('--hb-every', type=int, default=250, help='Heartbeat after N accepted positions')
    ap.add_argument('--shard-size', type=int, default=2048, help='Positions per shard before saving')
    ap.add_argument('--output-dir', default='data/teacher_games', help='Root output directory')

    args = ap.parse_args()
    cfg = TeacherConfig(
        scenario=args.scenario,
        games=int(args.games),
        model_path=args.model,
        sims=int(args.sims),
        stockfish_path=args.stockfish_path,
        threads=int(args.threads),
        hash_mb=int(args.hash_mb),
        movetime_ms=args.movetime_ms,
        depth=args.depth,
        multipv=int(args.multipv),
        topk=int(args.topk),
        cp_swing_mid=int(args.cp_swing_mid),
        cp_swing_end=int(args.cp_swing_end),
        teacher_adv_thresh=int(args.teacher_adv_thresh),
        max_moves=int(args.max_moves),
        hb_every=int(args.hb_every),
        shard_size=int(args.shard_size),
        output_dir=args.output_dir,
    )
    run(cfg)


if __name__ == '__main__':
    main()

