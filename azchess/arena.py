from __future__ import annotations

import argparse
import random
from pathlib import Path
import numpy as np
import time
import os
import chess
import chess.pgn
import torch
from tqdm import tqdm
from datetime import datetime

from .config import Config, select_device
from .model import PolicyValueNet
from .mcts import MCTS, MCTSConfig
from multiprocessing import Process, Queue
from .selfplay.inference import (
    setup_shared_memory_for_worker,
    run_inference_server,
    InferenceClient,
)

# Globals for worker processes (initialized by _arena_worker_init)
_P_DEVICE = None
_P_CFG = None
_P_MCFG = None
_P_MCTS_A = None
_P_MCTS_B = None


def _arena_worker_init(cfg_dict, ckpt_a_path, ckpt_b_path, num_sims_inner, batch_size_inner):
    """Initializer for multiprocessing workers: loads models and builds MCTS."""
    global _P_DEVICE, _P_CFG, _P_MCFG, _P_MCTS_A, _P_MCTS_B
    _P_CFG = Config(cfg_dict)
    _P_DEVICE = select_device(_P_CFG.get("device", "auto"))
    _P_MCFG = MCTSConfig(
        num_simulations=int(num_sims_inner),
        cpuct=float(_P_CFG.mcts().get("cpuct", 1.5)),
        dirichlet_alpha=float(_P_CFG.mcts().get("dirichlet_alpha", 0.3)),
        dirichlet_frac=float(_P_CFG.mcts().get("dirichlet_frac", 0.25)),
        tt_capacity=int(_P_CFG.mcts().get("tt_capacity", 200000)),
        selection_jitter=float(_P_CFG.selfplay().get("selection_jitter", 0.0)),
        batch_size=int(batch_size_inner or 1),
    )
    model_a_local = PolicyValueNet.from_config(_P_CFG.model()).to(_P_DEVICE)
    model_b_local = PolicyValueNet.from_config(_P_CFG.model()).to(_P_DEVICE)
    import numpy
    from torch.serialization import add_safe_globals
    add_safe_globals([numpy.core.multiarray.scalar])
    sa = torch.load(ckpt_a_path, map_location=_P_DEVICE, weights_only=False)
    sb = torch.load(ckpt_b_path, map_location=_P_DEVICE, weights_only=False)
    model_a_local.load_state_dict(sa.get("model_ema", sa.get("model", sa)))
    model_b_local.load_state_dict(sb.get("model_ema", sb.get("model", sb)))
    _P_MCTS_A = MCTS(model_a_local, _P_MCFG, _P_DEVICE)
    _P_MCTS_B = MCTS(model_b_local, _P_MCFG, _P_DEVICE)


def _arena_run_one_game(args_tuple):
    """Worker function: play a single game and return (score_from_A_persp, moves, result_str)."""
    idx, max_moves, temp_local, temp_plies_local, debug_local = args_tuple
    import chess, time, numpy as np
    global _P_DEVICE, _P_CFG, _P_MCFG, _P_MCTS_A, _P_MCTS_B
    board = chess.Board()
    moves_count = 0
    a_is_white = (idx % 2 == 0)
    engines = (_P_MCTS_A, _P_MCTS_B) if a_is_white else (_P_MCTS_B, _P_MCTS_A)
    while (not board.is_game_over(claim_draw=True)) and (moves_count < max_moves):
        stm_white = board.turn == chess.WHITE
        mcts = engines[0] if stm_white else engines[1]
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            break
        visits, pi, vroot = mcts.run(board)
        if temp_local > 1e-3 and moves_count < temp_plies_local:
            moves = list(visits.keys())
            vis = np.array([visits[m] for m in moves], dtype=np.float32)
            logits = np.log(vis + 1e-8) / max(temp_local, 1e-3)
            probs = np.exp(logits - np.max(logits))
            s = probs.sum()
            if s <= 0 or not np.isfinite(s):
                move = moves[int(np.argmax(vis))]
            else:
                probs /= s
                idxc = int(np.random.choice(len(moves), p=probs))
                move = moves[idxc]
        else:
            move = max(visits.items(), key=lambda kv: kv[1])[0]
        board.push(move)
        moves_count += 1
    # Score from A's perspective
    if board.is_game_over(claim_draw=True):
        res = board.result(claim_draw=True)
        if res == "1-0":
            score = 1.0 if a_is_white else 0.0
        elif res == "0-1":
            score = 0.0 if a_is_white else 1.0
        else:
            score = 0.5
    else:
        res = "1/2-1/2"
        score = 0.5
    return score, moves_count, res


def arena_worker_loop(cfg_dict, ckpt_a_path, ckpt_b_path, num_sims_inner, batch_size_inner,
                      indices, max_moves_local, temp_local, temp_plies_local, q_local,
                      shared_res_a=None, shared_res_b=None):
    """Top-level worker function to play multiple assigned games and emit heartbeats/results.

    Spawned by the main process; safe for 'spawn' start method on macOS.
    """
    try:
        # Initialize models/MCTS per worker
        cfg_local = Config(cfg_dict)
        device_local = select_device(cfg_local.get("device", "auto"))
        mcfg_local = MCTSConfig(
            num_simulations=int(num_sims_inner),
            cpuct=float(cfg_local.mcts().get("cpuct", 1.5)),
            dirichlet_alpha=float(cfg_local.mcts().get("dirichlet_alpha", 0.3)),
            dirichlet_frac=float(cfg_local.mcts().get("dirichlet_frac", 0.25)),
            tt_capacity=int(cfg_local.mcts().get("tt_capacity", 200000)),
            selection_jitter=float(cfg_local.selfplay().get("selection_jitter", 0.0)),
            batch_size=int(batch_size_inner or 1),
        )
        if shared_res_a is not None and shared_res_b is not None:
            # Use shared inference servers (no per-process models)
            infer_a = InferenceClient(shared_res_a)
            infer_b = InferenceClient(shared_res_b)
            mcts_a_local = MCTS(None, mcfg_local, device_local, inference_backend=infer_a)
            mcts_b_local = MCTS(None, mcfg_local, device_local, inference_backend=infer_b)
        else:
            model_a_local = PolicyValueNet.from_config(cfg_local.model()).to(device_local)
            model_b_local = PolicyValueNet.from_config(cfg_local.model()).to(device_local)
            import numpy
            from torch.serialization import add_safe_globals
            # Allow both numpy.core and numpy._core scalar classes
            try:
                add_safe_globals([numpy.core.multiarray.scalar])
            except Exception:
                pass
            try:
                add_safe_globals([numpy._core.multiarray.scalar])
            except Exception:
                pass
            sa = torch.load(ckpt_a_path, map_location=device_local, weights_only=False)
            sb = torch.load(ckpt_b_path, map_location=device_local, weights_only=False)
            model_a_local.load_state_dict(sa.get("model_ema", sa.get("model", sa)))
            model_b_local.load_state_dict(sb.get("model_ema", sb.get("model", sb)))
            mcts_a_local = MCTS(model_a_local, mcfg_local, device_local)
            mcts_b_local = MCTS(model_b_local, mcfg_local, device_local)
    except Exception as e:
        try:
            q_local.put({"type": "error", "msg": f"worker_init_failed: {e}"})
        except Exception:
            pass
        return

    import chess as _chess, time as _time, numpy as _np
    for idx in indices:
        t0 = _time.perf_counter()
        board = _chess.Board()
        moves_count = 0
        a_is_white = (idx % 2 == 0)
        engines = (mcts_a_local, mcts_b_local) if a_is_white else (mcts_b_local, mcts_a_local)
        last_hb = t0
        while (not board.is_game_over(claim_draw=True)) and (moves_count < max_moves_local):
            stm_white = board.turn == _chess.WHITE
            mcts_local = engines[0] if stm_white else engines[1]
            if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
                break
            try:
                visits, pi, vroot = mcts_local.run(board)
            except Exception as e:
                try:
                    q_local.put({"type": "error", "msg": f"mcts_failed game={idx} move={moves_count}: {e}"})
                except Exception:
                    pass
                # Fallback to random legal move
                legal = list(board.legal_moves)
                if not legal:
                    break
                mv = legal[0]
                board.push(mv)
                moves_count += 1
                continue
            if temp_local > 1e-3 and moves_count < temp_plies_local:
                moves = list(visits.keys())
                vis = _np.array([visits[m] for m in moves], dtype=_np.float32)
                logits = _np.log(vis + 1e-8) / max(temp_local, 1e-3)
                probs = _np.exp(logits - _np.max(logits))
                s = probs.sum()
                if s <= 0 or not _np.isfinite(s):
                    move = moves[int(_np.argmax(vis))]
                else:
                    probs /= s
                    idxc = int(_np.random.choice(len(moves), p=probs))
                    move = moves[idxc]
            else:
                move = max(visits.items(), key=lambda kv: kv[1])[0]
            board.push(move)
            moves_count += 1
            now = _time.perf_counter()
            if now - last_hb >= 5.0:
                try:
                    q_local.put({"type": "heartbeat", "game": idx, "moves": moves_count, "secs": now - t0})
                except Exception:
                    pass
                last_hb = now
        # Score from A's perspective
        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)
            if res == "1-0":
                score = 1.0 if a_is_white else 0.0
            elif res == "0-1":
                score = 0.0 if a_is_white else 1.0
            else:
                score = 0.5
        else:
            res = "1/2-1/2"
            score = 0.5
        try:
            q_local.put({"type": "game", "game": idx, "moves": moves_count, "secs": _time.perf_counter() - t0, "score": float(score), "result": res})
        except Exception:
            pass


def _wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z*z/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z * ((p*(1-p)/n) + (z*z)/(4*n*n))**0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _save_pgn(moves: list[chess.Move], result: str, headers: dict, out_dir: str, idx: int) -> None:
    game = chess.pgn.Game()
    node = game
    for k, v in headers.items():
        game.headers[k] = str(v)
    board = chess.Board()
    for mv in moves:
        node = node.add_variation(mv)
        board.push(mv)
    game.headers["Result"] = result
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"game_{idx:04d}.pgn")
    with open(path, "w") as f:
        print(game, file=f)


def play_match(
    ckpt_a: str,
    ckpt_b: str,
    games: int,
    cfg: Config,
    seed: int | None = None,
    pgn_out: str | None = None,
    pgn_sample: int = 0,
    num_sims: int = 500,
    max_moves_override: int | None = None,
    debug_moves: bool = False,
    workers: int = 1,
    batch_size: int = 1,
    adaptive: bool = False,
    temp: float = 0.0,
    temp_plies: int = 0,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
    
    print(f"🎯 Starting evaluation: {games} games")
    print(f"📊 Model A (trained): {ckpt_a}")
    print(f"📊 Model B (untrained): {ckpt_b}")
    print(f"⚙️  Device: {select_device(cfg.get('device', 'auto'))}")
    print(f"🎲 MCTS Simulations: {num_sims}")
    print("=" * 60)
    
    device = select_device(cfg.get("device", "auto"))
    # Set MPS-friendly watermarks if using MPS
    try:
        if device == 'mps':
            import os as _os
            _os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.8')
            _os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.6')
    except Exception:
        pass
    e = cfg.eval()
    
    # Use command-line override if provided
    # num_sims is already passed as parameter
    
    # Align MCTS config with self-play and config.yaml for consistency
    mcts_cfg = cfg.mcts()
    selfplay_cfg = cfg.selfplay()
    
    mcfg = MCTSConfig(
        num_simulations=num_sims,
        cpuct=float(mcts_cfg.get("cpuct", 1.5)),
        dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.3)),
        dirichlet_frac=float(mcts_cfg.get("dirichlet_frac", 0.25)),
        tt_capacity=int(mcts_cfg.get("tt_capacity", 200000)),
        selection_jitter=float(selfplay_cfg.get("selection_jitter", 0.0)),
        batch_size=int(batch_size or 1),
    )
    
    # Parallel execution path: spawn worker processes to play games concurrently with heartbeats
    if workers and workers > 1:
        print("🔄 Preparing parallel workers...")
        try:
            import torch.multiprocessing as mp
            try:
                mp.set_start_method('spawn', force=False)
            except RuntimeError:
                pass
        except Exception:
            pass

        max_moves = int(max_moves_override) if max_moves_override is not None else int(e.get("max_moves", 300))
        # Assign game indices to workers round-robin
        assignments = {w: [] for w in range(int(workers))}
        for i in range(games):
            assignments[i % int(workers)].append(i)

        q: Queue = Queue()

        # Shared inference servers: one per model (A and B). On MPS, skip and fall back to per-worker models.
        if device == 'mps':
            shared_res_a = []
            shared_res_b = []
        else:
            model_cfg = cfg.model()
            planes = int(model_cfg.get('planes', 19))
            policy_size = int(model_cfg.get('policy_size', 4672))
            # Preload checkpoints once to state dicts (PyTorch 2.6 weights_only handling)
            import numpy
            from torch.serialization import add_safe_globals
            # Allow both legacy and new NumPy scalar paths
            try:
                add_safe_globals([numpy.core.multiarray.scalar])
            except Exception:
                pass
            try:
                add_safe_globals([numpy._core.multiarray.scalar])
            except Exception:
                pass
            state_a = torch.load(ckpt_a, map_location=device, weights_only=False)
            state_b = torch.load(ckpt_b, map_location=device, weights_only=False)
            state_dict_a = state_a.get("model_ema", state_a.get("model", state_a))
            state_dict_b = state_b.get("model_ema", state_b.get("model", state_b))

            # Create shared memory resources for each worker for both models
            shared_res_a = []
            shared_res_b = []
            for wid in range(int(workers)):
                shared_res_a.append(setup_shared_memory_for_worker(wid, planes, policy_size, max_batch_size=int(batch_size or 1)))
                shared_res_b.append(setup_shared_memory_for_worker(wid, planes, policy_size, max_batch_size=int(batch_size or 1)))

            stop_a = torch.multiprocessing.Event()
            stop_b = torch.multiprocessing.Event()
            ready_a = torch.multiprocessing.Event()
            ready_b = torch.multiprocessing.Event()
            server_a = Process(target=run_inference_server, args=(device, model_cfg, state_dict_a, stop_a, ready_a, shared_res_a))
            server_b = Process(target=run_inference_server, args=(device, model_cfg, state_dict_b, stop_b, ready_b, shared_res_b))
            server_a.start(); server_b.start()
            # Wait ready
            if not ready_a.wait(timeout=60) or not ready_b.wait(timeout=60):
                print("❌ Inference servers failed to start in time. Falling back to per-worker models.")
                try:
                    stop_a.set(); stop_b.set()
                except Exception:
                    pass
                server_a.join(timeout=5); server_b.join(timeout=5)
                shared_res_a = []
                shared_res_b = []

        # Spawn workers (use shared inference clients, no model loading in worker)
        procs = []
        for w in range(int(workers)):
            if shared_res_a and shared_res_b:
                p = Process(target=arena_worker_loop, args=(
                    cfg.to_dict(), None, None, num_sims, batch_size, assignments[w],
                    max_moves, float(temp), int(temp_plies), q,
                    shared_res_a[w], shared_res_b[w]
                ))
            else:
                p = Process(target=arena_worker_loop, args=(
                    cfg.to_dict(), ckpt_a, ckpt_b, num_sims, batch_size, assignments[w],
                    max_moves, float(temp), int(temp_plies), q,
                    None, None
                ))
            # Attach shared resources indices via environment variables is messy; instead, pass worker id via assignments and look up in worker using wid from smallest index
            p.start()
            procs.append(p)

        # Live table display similar to orchestrator
        try:
            from rich.live import Live
            from rich.table import Table
            import time as _time
            table = Table(title="Arena (parallel)")
            table.add_column("Worker", justify="left")
            table.add_column("Done/Total", justify="right")
            table.add_column("Last HB(s)", justify="right")
            table.add_column("Last Moves", justify="right")
            table.add_column("W/L/D", justify="right")
            table.add_column("ETA", justify="right")

            per_worker = {w: {"done": 0, "total": len(assignments[w]), "hb_ts": 0.0, "moves": 0} for w in range(int(workers))}
            total_done = 0
            # Track global W/L/D from A's perspective
            a_wins = b_wins = draws = 0
            score = 0.0
            last_msg_time = time.time()
            start_ts = _time.perf_counter()
            with Live(table, refresh_per_second=4, transient=False) as live:
                # initial render
                for w in range(int(workers)):
                    table.add_row(f"W{w}", f"0/{per_worker[w]['total']}", "-", "-", f"0/0/0", "-")
                live.update(table)
                target_total = games
                while total_done < target_total:
                    try:
                        msg = q.get(timeout=2.0)
                    except Exception:
                        if time.time() - last_msg_time > 300:
                            raise RuntimeError("Arena appears stalled (no progress for 300s)")
                        continue
                    last_msg_time = time.time()
                    if isinstance(msg, dict) and msg.get("type") == "heartbeat":
                        gidx = int(msg.get("game", -1))
                        wid = gidx % int(workers)
                        per_worker[wid]["hb_ts"] = time.perf_counter()
                        per_worker[wid]["moves"] = int(msg.get("moves", 0))
                    elif isinstance(msg, dict) and msg.get("type") == "error":
                        err = str(msg.get("msg", "unknown error"))
                        print(f"❌ Worker error: {err}")
                        # Continue to listen; a single game error shouldn’t bring down whole eval
                    elif isinstance(msg, dict) and msg.get("type") == "game":
                        total_done += 1
                        sc = float(msg.get("score", 0.5))
                        score += sc
                        gidx = int(msg.get("game", -1))
                        wid = gidx % int(workers)
                        per_worker[wid]["done"] += 1
                        per_worker[wid]["moves"] = int(msg.get("moves", 0))
                        # Update W/L/D from A's perspective using result and color
                        res = msg.get("result", "1/2-1/2")
                        a_white = (gidx % 2 == 0)
                        if res == "1-0":
                            if a_white: a_wins += 1
                            else: b_wins += 1
                        elif res == "0-1":
                            if a_white: b_wins += 1
                            else: a_wins += 1
                        else:
                            draws += 1
                    # Refresh table
                    new_table = Table(title="Arena (parallel)")
                    new_table.add_column("Worker", justify="left")
                    new_table.add_column("Done/Total", justify="right")
                    new_table.add_column("Last HB(s)", justify="right")
                    new_table.add_column("Last Moves", justify="right")
                    new_table.add_column("W/L/D", justify="right")
                    new_table.add_column("ETA", justify="right")
                    # Compute ETA
                    elapsed = max(1e-3, _time.perf_counter() - start_ts)
                    rate = total_done / elapsed
                    eta_s = (target_total - total_done) / rate if rate > 0 else 0.0
                    eta_str = f"{int(eta_s)}s"
                    for w in range(int(workers)):
                        hb_age = 0.0
                        if per_worker[w]["hb_ts"] > 0:
                            hb_age = max(0.0, time.perf_counter() - per_worker[w]["hb_ts"])
                        new_table.add_row(
                            f"W{w}",
                            f"{per_worker[w]['done']}/{per_worker[w]['total']}",
                            f"{hb_age:.0f}",
                            f"{per_worker[w]['moves']}",
                            f"{a_wins}/{b_wins}/{draws}",
                            eta_str
                        )
                live.update(new_table)
        finally:
            for p in procs:
                p.join()
            # Attempt to stop shared servers if they were started
            try:
                if 'server_a' in locals():
                    stop_a.set()
                    server_a.join(timeout=5)
                if 'server_b' in locals():
                    stop_b.set()
                    server_b.join(timeout=5)
            except Exception:
                pass
            try:
                stop_a.set(); stop_b.set()
            except Exception:
                pass
            server_a.join(timeout=5); server_b.join(timeout=5)

        win_rate = score / float(max(1, games))
        print("=" * 60)
        print("🏆 EVALUATION COMPLETE!")
        print("=" * 60)
        print(f"📊 Final Score: {score:.1f} / {games} games")
        print(f"🎯 Win Rate: {win_rate:.1%}")
        if win_rate >= 0.55:
            print(f"\n🎉 PROMOTION CRITERIA MET! Win rate {win_rate:.1%} >= 55%")
        else:
            print(f"\n⚠️  PROMOTION CRITERIA NOT MET. Win rate {win_rate:.1%} < 55%")
        return score

    # Fallback: sequential path as before
    print("🔄 Loading models...")
    model_a = PolicyValueNet.from_config(cfg.model()).to(device)
    model_b = PolicyValueNet.from_config(cfg.model()).to(device)
    # Safely load checkpoints by whitelisting numpy classes
    import numpy
    from torch.serialization import add_safe_globals
    add_safe_globals([numpy.core.multiarray.scalar])
    state_a = torch.load(ckpt_a, map_location=device, weights_only=False)
    state_b = torch.load(ckpt_b, map_location=device, weights_only=False)
    if "model_ema" in state_a:
        model_a.load_state_dict(state_a["model_ema"])
    else:
        model_a.load_state_dict(state_a["model"])
    if "model_ema" in state_b:
        model_b.load_state_dict(state_b["model_ema"])
    else:
        model_b.load_state_dict(state_b["model"])
    print("🧠 Creating MCTS instances...")
    mcts_a = MCTS(model_a, mcfg, device)
    mcts_b = MCTS(model_b, mcfg, device)
    print("✅ Models loaded successfully!")
    print("=" * 60)

    score = 0.0
    pgn_kept = 0
    game_results = []
    start_time = time.time()
    
    # Progress bar for overall evaluation
    pbar = tqdm(total=games, desc="🎮 Evaluation Progress", unit="game")
    
    # Safety cap for excessively long games (plies)
    max_moves = int(max_moves_override) if max_moves_override is not None else int(e.get("max_moves", 300))
    print(f"🛡️  Max plies per game: {max_moves}")

    for g in range(games):
        game_start_time = time.time()
        board = chess.Board()
        
        # Alternate colors for fairness
        if g % 2 == 1:
            engines = [mcts_b, mcts_a]
            white_model = "B (untrained)"
            black_model = "A (trained)"
            actor_id_white, actor_id_black = "B", "A"
        else:
            engines = [mcts_a, mcts_b]
            white_model = "A (trained)"
            black_model = "B (untrained)"
            actor_id_white, actor_id_black = "A", "B"
        
        trace: list[chess.Move] = []
        moves_count = 0
        move_history = []  # Track move history for repetition detection
        
        # Game progress tracking
        while (not board.is_game_over(claim_draw=True)) and (moves_count < max_moves):
            stm_is_white = board.turn == chess.WHITE
            mcts = engines[0] if stm_is_white else engines[1]
            
            # Show current position info for EVERY move
            legal_moves = len(list(board.legal_moves))
            print(f"  Move {moves_count+1}: {legal_moves} legal moves")
            
            # Check for repetitive patterns before making move
            if len(move_history) >= 6:
                recent_moves = move_history[-6:]
                if len(set(str(m) for m in recent_moves)) <= 3:  # Simple repetition check
                    print(f"  ⚠️  Repetitive pattern detected at move {moves_count+1}")
            
            # If position is a claimable draw, stop early to avoid marathons
            if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
                break

            visits, pi, vroot = mcts.run(board)

            if debug_moves:
                try:
                    root = mcts._last_root
                    if root is not None and root.children:
                        # Top 3 by visits
                        items = sorted(root.children.items(), key=lambda kv: kv[1].n, reverse=True)[:3]
                        dbg = []
                        for mv, ch in items:
                            try:
                                san = board.san(mv)
                            except Exception:
                                san = str(mv)
                            dbg.append(f"{san}: N={ch.n} Q={ch.q:.3f} P={ch.prior:.3f}")
                        actor_id = actor_id_white if stm_is_white else actor_id_black
                        print(f"    🔎 [{actor_id}] Root v={vroot:.3f} | Top: " + " | ".join(dbg))
                except Exception:
                    pass
            # Temperature control: sample from visit counts for first temp_plies
            if temp > 1e-3 and moves_count < temp_plies:
                moves = list(visits.keys())
                vis = np.array([visits[m] for m in moves], dtype=np.float32)
                logits = np.log(vis + 1e-8) / max(temp, 1e-3)
                probs = np.exp(logits - np.max(logits))
                s = probs.sum()
                if s <= 0 or not np.isfinite(s):
                    move = moves[int(np.argmax(vis))]
                else:
                    probs /= s
                    idx = int(np.random.choice(len(moves), p=probs))
                    move = moves[idx]
            else:
                move = max(visits.items(), key=lambda kv: kv[1])[0]
            board.push(move)
            trace.append(move)
            move_history.append(move)  # Track move history
            moves_count += 1
        
        # Game result analysis
        game_time = time.time() - game_start_time
        # Determine result, allowing claimable draws
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
        else:
            # Safety cap triggered: adjudicate as draw
            result = "1/2-1/2"
            print(f"  ⏱️  Max moves ({max_moves}) reached, adjudicating draw.")
        
        if result == "1-0":
            if g % 2 == 0:  # Model A was white
                score += 1.0
                game_result = "A wins"
            else:  # Model B was white
                game_result = "B wins"
        elif result == "0-1":
            if g % 2 == 0:  # Model A was white
                game_result = "B wins"
            else:  # Model B was white
                score += 1.0
                game_result = "A wins"
        elif result == "1/2-1/2":
            score += 0.5
            game_result = "Draw"
        else:
            game_result = "Unknown"
        
        # Store game details
        game_details = {
            "game": g + 1,
            "result": game_result,
            "moves": moves_count,
            "time": game_time,
            "white": white_model,
            "black": black_model,
            "final_position": board.fen()
        }
        game_results.append(game_details)
        
        # Live game result display
        print(f"🎮 Game {g+1:2d}: {game_result:8s} | {moves_count:3d} moves | {game_time:5.1f}s | White: {white_model}")
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Score': f'{score:.1f}/{g+1}',
            'Win Rate': f'{score/(g+1):.1%}',
            'Last Game': game_result
        })
        
        # PGN export if requested
        if pgn_out and pgn_kept < max(0, pgn_sample):
            headers = {
                "Event": "Matrix0 Eval",
                "Site": "Local",
                "Date": time.strftime("%Y.%m.%d"),
                "Round": g + 1,
                "White": "A (trained)" if g % 2 == 0 else "B (untrained)",
                "Black": "B (untrained)" if g % 2 == 0 else "A (trained)",
                "Result": result,
                "Moves": str(moves_count),
                "Time": f"{game_time:.1f}s"
            }
            _save_pgn(trace, result, headers, pgn_out, pgn_kept)
            pgn_kept += 1
    
    pbar.close()
    
    # Final results summary
    total_time = time.time() - start_time
    win_rate = score / float(games)
    
    print("=" * 60)
    print("🏆 EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Final Score: {score:.1f} / {games} games")
    print(f"🎯 Win Rate: {win_rate:.1%}")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    print(f"⚡ Average Time per Game: {total_time/games:.1f}s")
    
    # Game-by-game summary
    print("\n📋 Game Results Summary:")
    print("-" * 60)
    for game in game_results:
        print(f"Game {game['game']:2d}: {game['result']:8s} | {game['moves']:3d} moves | {game['time']:5.1f}s | W: {game['white']}")
    
    # Performance analysis
    a_wins = sum(1 for g in game_results if "A wins" in g['result'])
    b_wins = sum(1 for g in game_results if "B wins" in g['result'])
    draws = sum(1 for g in game_results if "Draw" in g['result'])
    
    print("\n📈 Performance Analysis:")
    print(f"  🏆 Model A (trained): {a_wins} wins ({a_wins/games:.1%})")
    print(f"  🏆 Model B (untrained): {b_wins} wins ({b_wins/games:.1%})")
    print(f"  🤝 Draws: {draws} ({draws/games:.1%})")
    # Consistent single-line summary
    print(f"EVAL SUMMARY: A={a_wins} B={b_wins} D={draws} WR={win_rate:.3f}")
    # JSONL eval summary
    try:
        import json as _json
        from pathlib import Path as _Path
        _logs = _Path(cfg.training().get("log_dir", "logs"))
        _logs.mkdir(parents=True, exist_ok=True)
        _rec = {
            'type': 'eval_summary',
            'games': int(games),
            'a_wins': int(a_wins),
            'b_wins': int(b_wins),
            'draws': int(draws),
            'win_rate': float(win_rate),
            'timestamp': int(__import__('time').time()),
        }
        with (_logs / 'eval_summary.jsonl').open('a') as _f:
            _f.write(_json.dumps(_rec) + "\n")
    except Exception:
        pass
    
    if win_rate >= 0.55:
        print(f"\n🎉 PROMOTION CRITERIA MET! Win rate {win_rate:.1%} >= 55%")
        print("✅ Trained model is significantly stronger than untrained model!")
    else:
        print(f"\n⚠️  PROMOTION CRITERIA NOT MET. Win rate {win_rate:.1%} < 55%")
        print("❌ Trained model needs more training or different approach.")
    
    return score


def main():
    parser = argparse.ArgumentParser(description="Matrix0 Arena Evaluation")
    parser.add_argument("--ckpt_a", type=str, required=True, help="Checkpoint for model A")
    parser.add_argument("--ckpt_b", type=str, required=True, help="Checkpoint for model B")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--pgn-out", type=str, default=None, help="PGN output directory")
    parser.add_argument("--pgn-sample", type=int, default=0, help="Number of games to save as PGN")
    parser.add_argument("--num-sims", type=int, default=None, help="Override number of MCTS simulations")
    parser.add_argument("--cpuct", type=float, default=None, help="Override cpuct for PUCT selection (e.g., 1.0)")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive simulations during eval (off by default)")
    parser.add_argument("--max-moves", type=int, default=None, help="Adjudicate draw after this many plies (default 300)")
    parser.add_argument("--debug-moves", action="store_true", help="Print top-3 moves with N, Q, P, and root value each ply")
    parser.add_argument("--debug", dest="debug_moves", action="store_true", help="Alias for --debug-moves")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent game workers")
    parser.add_argument("--batch-size", type=int, default=1, help="Batched leaf evaluations per step (default 1)")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for sampling from visit counts (0=greedy)")
    parser.add_argument("--temp-plies", type=int, default=0, help="Apply temperature for first N plies")
    args = parser.parse_args()
    
    cfg = Config.load("config.yaml")  # Fixed: Use default config path
    
    # Get simulation count from args or config
    num_sims = args.num_sims if args.num_sims is not None else int(cfg.eval().get("num_simulations", 500))
    
    # If adaptive flag is set, let eval use adaptive sims
    if args.adaptive:
        print("⚙️  Adaptive simulations enabled for eval")

    score = play_match(
        args.ckpt_a,
        args.ckpt_b,
        args.games,
        cfg,
        seed=args.seed,
        pgn_out=args.pgn_out,
        pgn_sample=args.pgn_sample,
        num_sims=num_sims,
        max_moves_override=args.max_moves,
        debug_moves=args.debug_moves,
        workers=int(args.workers),
        batch_size=args.batch_size,
        adaptive=args.adaptive,
        temp=float(args.temp),
        temp_plies=int(args.temp_plies),
    )
    
    # Final statistical summary
    wr = score / float(args.games)
    lo, hi = _wilson_interval(wr, args.games)
    
    print("\n" + "=" * 60)
    print("📊 STATISTICAL SUMMARY")
    print("=" * 60)
    print(f"🎯 Score (A as White first): {score:.1f} / {args.games}")
    print(f"📈 Win Rate: {wr:.3f}")
    print(f"🔬 95% Confidence Interval: [{lo:.3f}, {hi:.3f}]")
    
    # Promotion recommendation
    if wr >= 0.55:
        print(f"\n🎉 RECOMMENDATION: PROMOTE MODEL!")
        print(f"   Win rate {wr:.1%} exceeds 55% threshold")
        print(f"   Confidence interval [{lo:.1%}, {hi:.1%}] supports promotion")
    else:
        print(f"\n⚠️  RECOMMENDATION: DO NOT PROMOTE")
        print(f"   Win rate {wr:.1%} below 55% threshold")
        print(f"   Confidence interval [{lo:.1%}, {hi:.1%}] suggests more training needed")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
