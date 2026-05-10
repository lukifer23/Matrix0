from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import chess
import chess.pgn
import chess.polyglot
import chess.syzygy
import numpy as np
import torch
from torch.multiprocessing import Queue

from azchess.logging_utils import setup_logging

from ..config import select_device
from ..data_manager import DataManager
from ..draw import should_adjudicate_draw
from ..encoding import encode_board, move_to_index, move_encoder
from ..mcts import MCTS, MCTSConfig
from ..model import PolicyValueNet
from ..ssl_algorithms import ChessSSLAlgorithms
from .inference import InferenceClient


def math_div_ceil(a: int, b: int) -> int:
    """Integer division with ceiling."""
    return (a + b - 1) // b
logger = setup_logging(level=logging.INFO)
OPENING_BOOK: List[chess.Board] = []


def build_value_targets(
    result_source: str,
    z: float,
    turns: List[int],
    search_values: List[float],
    capped_value_weight: float,
) -> tuple[List[float], float, bool]:
    """Build side-to-move value targets and sample weight for a completed shard."""
    if result_source in ("capped", "unfinished") and capped_value_weight > 0.0 and len(search_values) == len(turns):
        targets = [float(np.clip(v, -1.0, 1.0)) for v in search_values]
        return targets, float(capped_value_weight), True

    targets = [float(z * turn) for turn in turns]
    value_weight = 0.0 if result_source in ("capped", "unfinished") else 1.0
    return targets, float(value_weight), False

def load_opening_book(pgn_path: str):
    """Loads positions from a PGN file into the global opening book."""
    global OPENING_BOOK
    if OPENING_BOOK: # Already loaded
        return

    pgn_file = Path(pgn_path)
    if not pgn_file.exists():
        logger.debug(f"Opening book not found at {pgn_path}, starting from scratch.")
        return
    
    max_positions = 50000 # Limit memory usage
    with open(pgn_file) as f:
        while len(OPENING_BOOK) < max_positions:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                # Add positions from the first 20 moves (10 full moves)
                for i, move in enumerate(game.mainline_moves()):
                    if i >= 20:
                        break
                    board.push(move)
                    OPENING_BOOK.append(board.copy())
            except Exception:
                continue
    logger.debug(f"Loaded {len(OPENING_BOOK)} positions from opening book: {pgn_path}")

def get_opening_position() -> chess.Board:
    """Gets a random opening position, or the starting position if book is empty."""
    if not OPENING_BOOK:
        return chess.Board()
    return random.choice(OPENING_BOOK)

def apply_polyglot_opening(board: chess.Board, polyglot_path: str, max_plies: int = 8) -> None:
    """Advance the board using a polyglot opening book up to max_plies, sampling randomly.

    If no entries are found for a position, stops early. This mutates the board in-place.
    """
    p = Path(polyglot_path)
    if not p.exists() or max_plies <= 0:
        return
    try:
        with chess.polyglot.open_reader(str(p)) as reader:
            plies = 0
            while plies < max_plies and not board.is_game_over():
                entries = list(reader.find_all(board))
                if not entries:
                    break
                move = random.choice(entries).move
                board.push(move)
                plies += 1
    except Exception:
        # Fail silently; fall back to random opening plies if configured
        return


def selfplay_worker(proc_id: int, cfg_dict: dict, ckpt_path: str | None, games: int, q: Queue | None = None,
                    shared_memory_resource: Dict[str, Any] | None = None):
    # Set up logging for this worker
    lvl_name = os.environ.get('MATRIX0_WORKER_LOG_LEVEL', 'INFO').upper()  # Changed to INFO for debugging
    level = getattr(logging, lvl_name, logging.INFO)
    logger = setup_logging(level=level, name=f"selfplay_worker_{proc_id}")

    logger.info(f"Worker {proc_id} starting with PID {os.getpid()}")
    logger.info(f"Worker {proc_id} config: games={games}, device={select_device(cfg_dict.get('device', 'auto'))}")
    logger.info(f"Worker {proc_id} checkpoint: {ckpt_path}")

    # Log system info
    import psutil
    process = psutil.Process()
    logger.info(f"Worker {proc_id} memory usage: {process.memory_info().rss / 1024 / 1024:.1f}MB")
    logger.info(f"Worker {proc_id} CPU usage: {process.cpu_percent()}%")
    
    base_seed = int(cfg_dict.get("seed", 1234))
    random.seed(base_seed + proc_id)
    np.random.seed(base_seed + proc_id)

    # Select device dynamically (cuda > mps > cpu)
    device = select_device(cfg_dict.get("device", "auto"))

    # Avoid CPU over-subscription when using GPU/MPS
    try:
        if device != "cpu":
            # Prefer explicit selfplay.worker_threads, then mps preset override, else fallback
            threads_cfg = None
            try:
                threads_cfg = int(cfg_dict.get('selfplay', {}).get('worker_threads', None) or 0)
            except Exception:
                threads_cfg = None
            if not threads_cfg:
                try:
                    dev_sel = select_device(cfg_dict.get('device', 'auto'))
                    threads_cfg = int(cfg_dict.get('presets', {}).get(dev_sel, {}).get('worker_threads', 0) or 0)
                except Exception:
                    threads_cfg = None
            if not threads_cfg:
                try:
                    threads_cfg = int(cfg_dict.get('num_threads', 0) or 0)
                except Exception:
                    threads_cfg = None
            torch.set_num_threads(int(threads_cfg) if threads_cfg and threads_cfg > 0 else 1)
    except Exception:
        pass

    # Compact logging for worker process: silence console, log to file
    import os as _os
    if _os.environ.get("MATRIX0_COMPACT_LOG") == "1":
        try:
            import logging as _logging
            root = _logging.getLogger()
            root.setLevel(_logging.WARNING)
            wl = _logging.getLogger(__name__)
            wl.setLevel(_logging.WARNING)
            wl.propagate = False
            import logging.handlers as _lh
            _os.makedirs("logs", exist_ok=True)
            fh = _lh.RotatingFileHandler(f"logs/worker_{proc_id}.log", maxBytes=3_000_000, backupCount=2)
            fh.setLevel(_logging.INFO)
            fh.setFormatter(_logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
            if not any(isinstance(h, _lh.RotatingFileHandler) for h in wl.handlers):
                wl.addHandler(fh)
        except Exception:
            pass

    use_shared_infer = (shared_memory_resource is not None) and (device != "cpu")
    if use_shared_infer:
        model = None  # model hosted in server
        infer_backend = InferenceClient(shared_memory_resource)
        logger.debug("Using shared inference backend")
    else:
        infer_backend = None
        model = PolicyValueNet.from_config(cfg_dict["model"]).to(device)
        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                sd = state.get("model_ema", state.get("model", state))
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing:
                    total_expected = len(sd) + len(missing)
                    logger.warning(
                        f"Worker {proc_id}: Missing keys during load (initialized): {len(missing)}/{total_expected} keys "
                        f"({len(sd)} loaded successfully)"
                    )
                    logger.warning(f"Worker {proc_id}: Missing keys: {sorted(list(missing))}")
                if unexpected:
                    logger.warning(f"Worker {proc_id}: Unexpected keys during load (ignored): {len(unexpected)} keys")
                    logger.warning(f"Worker {proc_id}: Unexpected keys: {sorted(list(unexpected))}")
                allow_partial = bool(cfg_dict.get("selfplay", {}).get("allow_partial_checkpoint_load", False))
                if (missing or unexpected) and not allow_partial:
                    raise RuntimeError(
                        "Checkpoint did not match model exactly. "
                        "Set selfplay.allow_partial_checkpoint_load=true only for an intentional migration."
                    )
                logger.info(f"Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}") from e
        else:
            logger.info("No checkpoint provided, using untrained model")

    sp_cfg = cfg_dict["selfplay"]
    # Use unified draw configuration merging top-level and selfplay.draw
    try:
        from ..config import Config as _Cfg
        _cfg_obj = _Cfg(cfg_dict)
        draw_cfg = _cfg_obj.draw()
    except Exception:
        draw_cfg = cfg_dict.get("draw", {})

    # Detect value orientation (side-to-move vs absolute White) and pass to MCTS
    # This is critical for correct value target generation in self-play
    def _detect_value_from_white() -> bool:
        try:
            import copy as _copy
            b1 = chess.Board()  # standard start, White to move
            b2 = _copy.deepcopy(b1)
            b2.turn = chess.BLACK  # same position but Black to move
            arr = np.stack([encode_board(b1), encode_board(b2)], axis=0)
            if use_shared_infer:
                p_np, v_np = infer_backend.infer_np(arr)
                v1, v2 = float(v_np[0]), float(v_np[1])
            else:
                with torch.no_grad():
                    x = torch.from_numpy(arr).to(device)
                    model.eval()
                    device_type = device.split(':')[0]
                    use_amp = device_type in ("cuda", "mps")
                    with torch.autocast(device_type=device_type, enabled=use_amp):
                        _, v_t = model(x)
                    v_np = v_t.detach().cpu().to(torch.float32).numpy().flatten()
                    v1, v2 = float(v_np[0]), float(v_np[1])
            # If model is side-to-move oriented, values should flip sign when turn flips
            # Compare |v2 + v1| vs |v2 - v1|
            # Side-to-move: v2 ≈ -v1 (opposite signs) → |v2 + v1| < |v2 - v1|
            # Absolute white: v2 ≈ v1 (same sign) → |v2 - v1| < |v2 + v1|
            flip_diff = abs(v2 + v1)
            same_diff = abs(v2 - v1)
            if flip_diff < same_diff:
                detected = False  # side-to-move
                logger.info(f"Worker {proc_id}: Detected side-to-move value orientation (v1={v1:.3f}, v2={v2:.3f}, flip_diff={flip_diff:.3f} < same_diff={same_diff:.3f})")
            else:
                detected = True   # absolute white perspective
                logger.info(f"Worker {proc_id}: Detected absolute white value orientation (v1={v1:.3f}, v2={v2:.3f}, same_diff={same_diff:.3f} < flip_diff={flip_diff:.3f})")
            return detected
        except Exception as e:
            logger.warning(f"Worker {proc_id}: Value orientation detection failed: {e}, defaulting to side-to-move")
            return False
    
    # Respect an explicit config override. Auto-detection is only a compatibility
    # path for older configs that do not declare value orientation.
    mcts_orientation_cfg = cfg_dict.get("mcts", {}) or {}
    if "value_from_white" in mcts_orientation_cfg:
        value_from_white = bool(mcts_orientation_cfg["value_from_white"])
        value_from_white_source = "config"
    else:
        value_from_white = _detect_value_from_white()
        value_from_white_source = "auto"
    logger.info(f"Worker {proc_id}: Using value_from_white={value_from_white} (source={value_from_white_source})")
    
    # Load opening book
    book_path = sp_cfg.get("book_path")
    if book_path:
        load_opening_book(book_path)
    
    # Initialize tablebases if enabled
    tb_cfg = cfg_dict.get("tablebases", {})
    tablebase_reader = None
    if tb_cfg.get("enabled", False):
        tb_path = Path(tb_cfg.get("path", ""))
        if tb_path.exists() and tb_path.is_dir():
            try:
                tablebase_reader = chess.syzygy.open_tablebase(tb_path)
                logger.info(f"Syzygy tablebase initialized at path: {tb_path}")
            except Exception as e:
                logger.error(f"Failed to initialize tablebase at {tb_path}: {e}")
        else:
            logger.warning(f"Tablebase path not found or not a directory: {tb_path}")

    # Set simulation count from config (no more hardcoded limits)
    sims = sp_cfg.get("num_simulations", 800)  # Use config default, not hardcoded 50
    logger.info(f"Worker {proc_id}: sims={sims} games={games}")

    # Build MCTS config with robust fallbacks
    base_mcts = dict(cfg_dict.get("mcts", {}))
    if not base_mcts:
        # Fallback to defaults section if provided
        base_mcts = dict(cfg_dict.get("mcts_defaults", {}))
    # Final safety: ensure minimal keys exist
    base_mcts.setdefault("num_simulations", int(sp_cfg.get("num_simulations", 800)))
    base_mcts.setdefault("cpuct", float(sp_cfg.get("cpuct", 2.5)))
    base_mcts.setdefault("dirichlet_alpha", float(sp_cfg.get("dirichlet_alpha", 0.3)))
    base_mcts.setdefault("dirichlet_frac", float(sp_cfg.get("dirichlet_frac", 0.25)))
    # Get unified inference batch size from config helper
    try:
        from ..config import Config as _Cfg
        _cfg_obj = _Cfg(cfg_dict)
        unified_batch_size = _cfg_obj.inference_batch_size()
    except Exception:
        unified_batch_size = 96
    base_mcts.setdefault("inference_batch_size", unified_batch_size)
    base_mcts.setdefault("selection_jitter", float(sp_cfg.get("selection_jitter", 0.01)))

    mcfg_dict = dict(base_mcts)
    mcfg_dict.update(
        {
            "num_simulations": int(sp_cfg.get("num_simulations", mcfg_dict.get("num_simulations", 800))),
            "cpuct": float(sp_cfg.get("cpuct", mcfg_dict.get("cpuct", 2.5))),
            "dirichlet_alpha": float(sp_cfg.get("dirichlet_alpha", mcfg_dict.get("dirichlet_alpha", 0.3))),
            "dirichlet_frac": float(sp_cfg.get("dirichlet_frac", mcfg_dict.get("dirichlet_frac", 0.25))),
            "tt_capacity": int(mcfg_dict.get("tt_capacity", 2000000)),
            "selection_jitter": float(sp_cfg.get("selection_jitter", mcfg_dict.get("selection_jitter", 0.01))),
            "inference_batch_size": int(mcfg_dict.get("inference_batch_size", unified_batch_size)),
            "fpu": float(sp_cfg.get("fpu", mcfg_dict.get("fpu", 0.5))),
            "parent_q_init": bool(sp_cfg.get("parent_q_init", mcfg_dict.get("parent_q_init", True))),
            "tt_cleanup_frequency": int(mcfg_dict.get("tt_cleanup_frequency", 500)),
            "draw_penalty": float(mcfg_dict.get("draw_penalty", -0.1)),
            "value_from_white": bool(value_from_white),
        }
    )
    mcts = MCTS(
        MCTSConfig.from_dict(mcfg_dict),
        model,
        device=device,
        inference_backend=infer_backend,
    )

    # Always generate SSL targets if model has SSL enabled (precomputed for training efficiency)
    sp_flags = cfg_dict.get("selfplay", {})
    model_cfg = cfg_dict.get("model", {})
    model_has_ssl = bool(model_cfg.get("self_supervised", False))
    ssl_tasks = model_cfg.get("ssl_tasks", []) if model_has_ssl else []
    ssl_enabled = model_has_ssl and len(ssl_tasks) > 0
    ssl_algorithms = ChessSSLAlgorithms() if ssl_enabled else None

    data_manager = DataManager(
        base_dir=cfg_dict.get("data_dir", "data"),
        expected_planes=cfg_dict.get("model", {}).get("planes", 19),
    )

    try:
        for g in range(games):
            board = get_opening_position()
            # Apply polyglot opening if configured (adds variety before random plies)
            openings_cfg = cfg_dict.get("openings", {})
            polyglot_path = openings_cfg.get("polyglot", "")
            polyglot_plies = int(openings_cfg.get("max_plies", 0))
            if polyglot_path and polyglot_plies > 0:
                apply_polyglot_opening(board, polyglot_path, max_plies=polyglot_plies)
            states: List[np.ndarray] = []
            pis: List[np.ndarray] = []
            turns: List[int] = []
            search_values: List[float] = []
            sims_used: List[int] = []
            entropy_sum: float = 0.0
            entropy_count: int = 0
            legal_masks: List[np.ndarray] = []
            ssl_targets: Dict[str, List[np.ndarray]] = {}
            
            t0 = perf_counter()
            last_hb = t0
            
            # Temperature parameters for exploration
            temp_start = float(sp_cfg.get("temperature_start", 1.0))
            temp_end = float(sp_cfg.get("temperature_end", 0.1))
            temp_moves = int(sp_cfg.get("temperature_moves", 20))
    
            # Resign parameters (multi-factor)
            resign_thr = float(sp_cfg.get("resign_threshold", -0.98))
            consec_bad = 0
            # Track a small window of recent values and entropies for stability-based resignation
            recent_values: List[float] = []
            recent_entropies: List[float] = []
            window_k = int(sp_cfg.get("resign_window", 4))
            min_entropy = float(sp_cfg.get("resign_min_entropy", 0.3))
            value_margin = float(sp_cfg.get("resign_value_margin", 0.05))
            resigner_color: str | None = None
            move_history = []  # Track moves for pattern detection
            resigned = False
            z: float | None = None
            result_source = "unknown"
            capped = False
            adjudicated_draw = False
            tablebase_hit = False
    
            # Opening diversity: optional random opening plies (quick uniform legal moves)
            try:
                # Prefer explicit selfplay key; fallback to openings.random_plies
                rnd_plies = int(sp_cfg.get("opening_random_plies", cfg_dict.get("openings", {}).get("random_plies", 0)))
            except Exception:
                rnd_plies = 0
            for _ in range(max(0, rnd_plies)):
                if board.is_game_over():
                    break
                legal = list(board.legal_moves)
                if not legal:
                    break
                mv = random.choice(legal)
                board.push(mv)
                move_history.append(mv)
    
            min_resign_plies = int(sp_cfg.get("min_resign_plies", 24))
            while not board.is_game_over() and len(states) < sp_cfg.get("max_game_len", 200):
                if should_adjudicate_draw(board, move_history, draw_cfg):
                    z = 0.0
                    result_source = "draw_adjudication"
                    adjudicated_draw = True
                    break
    
                move_no = board.fullmove_number
                
                # Determine temperature for the current move
                # Smooth linear schedule by move number
                if temp_moves <= 0:
                    temperature = temp_end
                else:
                    t = min(max(move_no, 0), temp_moves) / float(max(1, temp_moves))
                    temperature = temp_start + (temp_end - temp_start) * t
    
                # Throttle progress logs: switch to DEBUG to reduce output noise
                move_count = len(states) + 1
                if move_count % 50 == 1:
                    logger.debug(f"Game {g + 1}: Move {move_count}")
    
                # Capture turn perspective for this state BEFORE making a move
                # This is used to convert game result to side-to-move value targets
                turn_sign = 1 if board.turn == chess.WHITE else -1
    
                try:
                    t_move0 = perf_counter()
                    # Pass ply index to MCTS for ply-gated Dirichlet in early game
                    visit_counts, pi, v = mcts.run(board, ply=len(states))
                    t_move1 = perf_counter()
                    
                    # Validate visit counts - fail fast if MCTS search produced no valid results
                    if not visit_counts or all(count == 0 for count in visit_counts.values()):
                        error_msg = f"MCTS returned invalid visit counts: {visit_counts}. This indicates a search failure."
                        logger.error(error_msg)
                        # Fail fast: raise exception instead of falling back to random moves
                        raise RuntimeError(error_msg)
                    else:
                        # Low-visit fallback: increase temperature when search is shallow
                        try:
                            low_visit_thr = int(sp_cfg.get("low_visit_threshold", 0))
                        except Exception:
                            low_visit_thr = 0
                        max_visits = max(visit_counts.values()) if visit_counts else 0
                        temp_eff = temperature
                        if low_visit_thr > 0 and max_visits < low_visit_thr:
                            temp_eff = max(temperature, 0.8)
                        move = sample_move_from_counts(board, visit_counts, temp_eff)
                        # Track policy entropy from MCTS policy vector
                        try:
                            _pi = np.clip(pi.astype(np.float64, copy=False), 1e-12, 1.0)
                            ent = float(-np.sum(_pi * np.log(_pi)))
                            entropy_sum += ent
                            entropy_count += 1
                            recent_entropies.append(ent)
                            if len(recent_entropies) > window_k:
                                recent_entropies.pop(0)
                        except Exception:
                            pass
                
                except Exception as e:
                    error_msg = f"MCTS failed during move selection: {e}"
                    logger.error(error_msg, exc_info=True)
                    # Fail fast: raise exception instead of falling back to random moves
                    # This ensures we catch and fix MCTS issues rather than silently using bad moves
                    raise RuntimeError(error_msg) from e
                
                states.append(encode_board(board))
                pis.append(pi)
                search_values.append(v)
                # Record perspective of this position (side to move at this state)
                turns.append(turn_sign)
                # Always compute and save legal mask (precomputed for training efficiency)
                try:
                    lm = move_encoder.get_legal_actions(board).astype(np.uint8, copy=False)
                except Exception as e:
                    raise RuntimeError(f"Failed to encode legal mask for board {board.fen()}: {e}") from e
                legal_masks.append(lm)

                # Always generate SSL targets if model has SSL enabled (precomputed for training efficiency)
                if ssl_enabled and ssl_algorithms:
                    board_state = encode_board(board)
                    ssl_batch = torch.from_numpy(np.expand_dims(board_state, axis=0)).float()

                    try:
                        # Generate all SSL targets at once using enhanced method
                        all_targets = ssl_algorithms.create_enhanced_ssl_targets(ssl_batch)
                        
                        # Filter to only requested tasks and convert to numpy
                        for task in ssl_tasks:
                            if task not in ssl_targets:
                                ssl_targets[task] = []
                            
                            if task in all_targets:
                                target = all_targets[task]
                                # Handle different shapes: piece is (1, 13, 8, 8), others are (1, 8, 8)
                                if target.dim() == 4:
                                    # Multi-channel target (e.g., piece)
                                    target_np = target.squeeze(0).detach().cpu().numpy()
                                else:
                                    # Single-channel target
                                    target_np = target.squeeze(0).detach().cpu().numpy()
                                ssl_targets[task].append(target_np)
                            else:
                                raise RuntimeError(f"SSL task '{task}' not found in generated targets")
                    except Exception as e:
                        raise RuntimeError(f"Failed to generate SSL targets for board {board.fen()}: {e}") from e
                # Track sims used
                try:
                    sims_used.append(int(getattr(mcts, '_last_sims_run', 0)))
                except Exception:
                    pass
    
                # Resign check (multi-factor): consecutive low values, low entropy, stable bad trend
                if resign_thr > -1.0 and len(states) >= min_resign_plies:
                    recent_values.append(float(v))
                    if len(recent_values) > window_k:
                        recent_values.pop(0)
                    low_value = v < resign_thr
                    if low_value:
                        consec_bad += 1
                        logger.info(f"Game {g + 1}: Bad value {v:.3f} < {resign_thr:.3f} (consec: {consec_bad})")
                    else:
                        consec_bad = 0
                    resign_seq_bad = int(sp_cfg.get("resign_consecutive_bad", 5))
                    # Stability: average of recent values below (resign_thr + margin)
                    stable_bad = False
                    if len(recent_values) >= max(2, window_k // 2):
                        avg_recent = float(sum(recent_values) / len(recent_values))
                        stable_bad = avg_recent < (resign_thr + value_margin)
                    # Uncertainty: low average entropy
                    low_uncertainty = False
                    if len(recent_entropies) >= max(2, window_k // 2):
                        avg_ent = float(sum(recent_entropies) / len(recent_entropies))
                        low_uncertainty = avg_ent < min_entropy
                    if consec_bad >= resign_seq_bad and (stable_bad or low_uncertainty):
                        resigned = True
                        z = -1.0 if board.turn == chess.WHITE else 1.0
                        result_source = "resignation"
                        resigner_color = 'W' if board.turn == chess.WHITE else 'B'
                        logger.info(
                            f"Game {g + 1}: Resigned after {consec_bad} low values (stable={stable_bad} lowH={low_uncertainty}). "
                            f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}, Result: {z}"
                        )
                        break
                
                move_history.append(move)
                board.push(move)
    
                # Heartbeat to orchestrator every ~2 seconds (more responsive TUI)
                if q is not None:
                    now = perf_counter()
                    if now - last_hb >= 2.0:
                        try:
                            q.put({
                                "type": "heartbeat",
                                "proc": proc_id,
                                "game": g,
                                "moves": len(states),
                                "avg_sims": (float(sum(sims_used)) / max(1, len(sims_used)) if sims_used else 0.0),
                                "resigned": resigned,
                                "avg_policy_entropy": (entropy_sum / max(1, entropy_count)),
                            })
                        except Exception:
                            pass
                        last_hb = now
    
                # Check tablebase after the move
                if tablebase_reader:
                    piece_count = len(board.piece_map())
                    if piece_count <= tb_cfg.get("max_pieces", 7):
                        try:
                            wdl = tablebase_reader.probe_wdl(board)
                            if wdl is not None:
                                # Tablebase hit, game is over. WDL is from the perspective of the side to move.
                                # +2, +1 = win; 0 = draw; -1, -2 = loss.
                                if wdl == 0:
                                    z = 0.0
                                else:
                                    # Convert WDL to a game result (-1.0, 0.0, 1.0) from white's perspective.
                                    side_to_move = board.turn
                                    if wdl > 0: # Win for the side to move
                                        z = 1.0 if side_to_move == chess.WHITE else -1.0
                                    else: # Loss for the side to move
                                        z = -1.0 if side_to_move == chess.WHITE else 1.0
                                
                                logger.info(f"Tablebase hit ({piece_count} pieces). WDL: {wdl}. Final result: {z}")
                                result_source = "tablebase"
                                tablebase_hit = True
                                break # End the game
                        except Exception as e:
                            logger.warning(f"Tablebase probe failed: {e}")
    
            # If no real result/adjudication fired, assign explicit neutral
            # targets for capped games. Do not let an untrained search value
            # label unfinished games as decisive outcomes.
            if z is None:
                terminal = board.is_game_over(claim_draw=True)
                if terminal:
                    z = game_result(board)
                    result_source = "terminal"
                else:
                    capped = len(states) >= int(sp_cfg.get("max_game_len", 200))
                    z = 0.0
                    result_source = "capped" if capped else "unfinished"
            
            # Debug: log the final game result
            if resigned:
                logger.info(f"Game {g + 1}: Final result from resignation: {z}")
            elif capped:
                logger.info(f"Game {g + 1}: Final result from capped game: {z}")
            else:
                logger.info(f"Game {g + 1}: Final result from {result_source}: {z}")
            
            capped_value_weight = float(sp_cfg.get("capped_value_weight", 0.25))
            value_target, value_weight, value_bootstrap = build_value_targets(
                result_source=result_source,
                z=float(z),
                turns=turns,
                search_values=search_values,
                capped_value_weight=capped_value_weight,
            )
            
            # Validation: log sample values to verify orientation
            if len(value_target) > 0 and (g == 0 or len(value_target) % 50 == 0):
                sample_indices = [0, min(10, len(value_target) - 1), len(value_target) - 1]
                sample_values = [(i, value_target[i], turns[i], "white" if turns[i] > 0 else "black") 
                                for i in sample_indices if i < len(value_target)]
                logger.debug(f"Game {g + 1}: Value target samples (idx, value, turn_sign, side): {sample_values}")
                logger.debug(f"Game {g + 1}: Final result z={z:.3f} (white perspective), first value_target={value_target[0]:.3f} (side-to-move)")
    
            filepath = None
            if len(states) > 0:
                game_data = {
                    "s": np.array(states, dtype=np.float32),
                    "pi": np.array(pis, dtype=np.float32),
                    "z": np.array(value_target, dtype=np.float32),
                    "value_weight": np.full((len(states),), value_weight, dtype=np.float32),
                    # Per-game metadata arrays
                    "meta_moves": np.array([len(states)], dtype=np.int32),
                    "meta_result": np.array([z], dtype=np.float32),
                    "meta_resigned": np.array([1 if resigned else 0], dtype=np.int8),
                    "meta_capped": np.array([1 if capped else 0], dtype=np.int8),
                    "meta_terminal": np.array([1 if result_source == "terminal" else 0], dtype=np.int8),
                    "meta_tablebase": np.array([1 if tablebase_hit else 0], dtype=np.int8),
                    "meta_adjudicated_draw": np.array([1 if adjudicated_draw else 0], dtype=np.int8),
                    "meta_result_source": np.array([result_source]),
                    "meta_value_bootstrap": np.array([1 if value_bootstrap else 0], dtype=np.int8),
                    "meta_value_weight": np.array([value_weight], dtype=np.float32),
                    "meta_draw": np.array([1 if z == 0.0 else 0], dtype=np.int8),
                    "meta_avg_policy_entropy": np.array([entropy_sum / max(1, entropy_count)], dtype=np.float32),
                    "meta_avg_sims": np.array([float(sum(sims_used)) / max(1, len(sims_used)) if sims_used else 0.0], dtype=np.float32),
                    "meta_final_fen": np.array([board.fen()]),
                    "meta_final_piece_count": np.array([len(board.piece_map())], dtype=np.int16),
                    "meta_final_halfmove_clock": np.array([board.halfmove_clock], dtype=np.int16),
                    "meta_final_legal_count": np.array([board.legal_moves.count()], dtype=np.int16),
                    "meta_final_can_claim_draw": np.array(
                        [1 if board.can_claim_draw() else 0],
                        dtype=np.int8,
                    ),
                }
                # Always include legal masks (precomputed for training efficiency)
                if legal_masks:
                    try:
                        game_data["legal_mask"] = np.stack(legal_masks, axis=0).astype(np.uint8, copy=False)
                    except Exception:
                        pass

                # Add SSL targets if they were generated
                if ssl_targets:
                    for task, targets in ssl_targets.items():
                        if targets:
                            game_data[f"ssl_{task}"] = np.stack(targets, axis=0).astype(np.float32)
                try:
                    filepath = data_manager.add_selfplay_data(game_data, worker_id=proc_id, game_id=g)
                except Exception as e:
                    logger.error(f"Failed to save self-play game w{proc_id} g{g}: {e}", exc_info=True)
                    filepath = None
            else:
                logger.warning(f"Game {g + 1}: No states collected; skipping save to avoid empty shard")
    
            game_time = perf_counter() - t0
            avg_ms_per_move = (game_time * 1000.0 / max(1, len(states)))
            avg_sims = (float(sum(sims_used)) / max(1, len(sims_used))) if sims_used else 0.0
            logger.info(f"Game {g + 1} completed: {len(states)} moves, result: {z:.3f}, time: {game_time:.1f}s, avg_entropy: {entropy_sum / max(1, entropy_count):.3f}")
            
            if q is not None:
                q.put({
                    "type": "game",
                    "proc": proc_id,
                    "file": filepath,
                    "moves": len(states),
                    "result": float(z),
                    "secs": game_time,
                    "resigned": resigned,
                    "resigner": resigner_color,
                    "draw": bool(z == 0.0),
                    "capped": bool(capped),
                    "result_source": result_source,
                    "avg_policy_entropy": (entropy_sum / max(1, entropy_count)),
                    "avg_ms_per_move": avg_ms_per_move,
                    "avg_sims": avg_sims,
                })
    
    
    finally:
        if tablebase_reader is not None:
            try:
                tablebase_reader.close()
            except Exception as e:
                logger.warning(f"Failed to close tablebase: {e}")


def sample_move_from_counts(board: chess.Board, counts: Dict[chess.Move, int], temperature: float) -> chess.Move:
    if not counts:
        raise RuntimeError("Cannot sample move from empty MCTS visit counts")

    moves = list(counts.keys())
    visits = np.array(list(counts.values()), dtype=np.float32)
    
    if np.any(np.isnan(visits)) or np.all(visits == 0):
        raise RuntimeError(f"Invalid MCTS visit counts: {counts}")
    
    if temperature < 1e-3:
        # Deterministic: pick the best move
        return moves[np.argmax(visits)]
    
    # Probabilistic: sample from distribution
    visit_dist = visits**(1.0 / temperature)
    
    # Ensure we don't have division by zero or NaN
    visit_sum = visit_dist.sum()
    if visit_sum <= 0 or np.isnan(visit_sum):
        raise RuntimeError(f"Invalid temperature-adjusted MCTS visit distribution: counts={counts}, temperature={temperature}")
    
    visit_dist /= visit_sum
    if np.any(np.isnan(visit_dist)):
        raise RuntimeError(f"MCTS visit distribution contains NaN: counts={counts}, temperature={temperature}")
    
    idx = np.random.choice(len(moves), p=visit_dist)
    return moves[idx]


def game_result(board: chess.Board) -> float:
    if board.is_checkmate():
        # board.turn is the side to move (and thus the side being checkmated)
        # If White is to move and checkmated, White loses (-1.0)
        # If Black is to move and checkmated, Black loses (1.0 for White)
        return -1.0 if board.turn == chess.WHITE else 1.0
    
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return -1.0
    return 0.0
