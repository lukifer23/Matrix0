from __future__ import annotations

from torch.multiprocessing import Queue
from time import perf_counter
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging

import numpy as np
import chess
import torch
import os
import random
import chess.pgn
import chess.syzygy

from ..config import select_device
from ..model import PolicyValueNet
from ..mcts import MCTS, MCTSConfig
from ..data_manager import DataManager
from .inference import InferenceClient
from ..encoding import encode_board, move_to_index
import chess.polyglot


def math_div_ceil(a: int, b: int) -> int:
    """Integer division with ceiling."""
    return (a + b - 1) // b

logger = logging.getLogger(__name__)
OPENING_BOOK: List[chess.Board] = []

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
    lvl_name = os.environ.get('MATRIX0_WORKER_LOG_LEVEL', 'WARNING').upper()
    level = getattr(logging, lvl_name, logging.WARNING)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(f"selfplay_worker_{proc_id}")
    
    base_seed = int(cfg_dict.get("seed", 1234))
    random.seed(base_seed + proc_id)
    np.random.seed(base_seed + proc_id)

    # Select device dynamically (cuda > mps > cpu)
    device = select_device(cfg_dict.get("device", "auto"))

    # Avoid CPU over-subscription when using GPU/MPS
    try:
        if device != "cpu":
            torch.set_num_threads(max(1, os.cpu_count() // 2))
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
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device)
                if "model_ema" in state:
                    model.load_state_dict(state["model_ema"])
                elif "model" in state:
                    model.load_state_dict(state["model"])
                else:
                    # Handle case where checkpoint is just the model state dict
                    model.load_state_dict(state)
                logger.info(f"Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
                logger.info("Using untrained model")
        else:
            logger.info("No checkpoint provided, using untrained model")

    sp_cfg = cfg_dict["selfplay"]
    
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

    # Set reasonable simulation count for testing
    sims = sp_cfg.get("num_simulations", 50)  # Default to 50 instead of 800
    logger.info(f"Worker {proc_id}: sims={sims} games={games}")

    mcts_cfg = cfg_dict["mcts"]
    # Allow selfplay overrides for common MCTS params
    mcts = MCTS(
        model,
        MCTSConfig(
            num_simulations=int(sp_cfg.get("num_simulations", mcts_cfg.get("num_simulations", 800))),
            cpuct=float(sp_cfg.get("cpuct", mcts_cfg.get("cpuct", 2.5))),
            dirichlet_alpha=float(sp_cfg.get("dirichlet_alpha", mcts_cfg.get("dirichlet_alpha", 0.3))),
            dirichlet_frac=float(sp_cfg.get("dirichlet_frac", mcts_cfg.get("dirichlet_frac", 0.25))),
            tt_capacity=int(mcts_cfg.get("tt_capacity", 2000000)),
            selection_jitter=float(sp_cfg.get("selection_jitter", mcts_cfg.get("selection_jitter", 0.01))),
            batch_size=int(sp_cfg.get("batch_size", mcts_cfg.get("batch_size", 32))),
            fpu=float(sp_cfg.get("fpu", mcts_cfg.get("fpu", 0.5))),
            parent_q_init=bool(sp_cfg.get("parent_q_init", mcts_cfg.get("parent_q_init", True))),
            tt_cleanup_frequency=int(mcts_cfg.get("tt_cleanup_frequency", 500)),
            draw_penalty=float(mcts_cfg.get("draw_penalty", -0.1)),
        ),
        device=device,
        inference_backend=infer_backend,
    )
    data_manager = DataManager(base_dir=cfg_dict.get("data_dir", "data"))

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
        
        t0 = perf_counter()
        last_hb = t0
        
        # Temperature parameters for exploration
        temp_start = float(sp_cfg.get("temperature_start", 1.0))
        temp_end = float(sp_cfg.get("temperature_end", 0.1))
        temp_moves = int(sp_cfg.get("temperature_moves", 20))

        # Resign parameters
        resign_thr = float(sp_cfg.get("resign_threshold", -0.98))
        consec_bad = 0
        move_history = []  # Track moves for pattern detection
        resigned = False

        # Opening diversity: optional random opening plies (quick uniform legal moves)
        try:
            rnd_plies = int(sp_cfg.get("opening_random_plies", 0))
        except Exception:
            rnd_plies = 0
        for _ in range(max(0, rnd_plies)):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(random.choice(legal))

        while not board.is_game_over() and len(states) < sp_cfg.get("max_game_len", 200):
            # Check for early termination to prevent draws
            # Optional early draw adjudication (configurable)
            if bool(sp_cfg.get("early_draw_enabled", False)):
                if _should_terminate_early(
                    board,
                    move_history,
                    min_plies_before_check=int(sp_cfg.get("early_draw_min_plies", 60)),
                    recent_window=int(sp_cfg.get("early_draw_window", 24)),
                    min_unique_in_window=int(sp_cfg.get("early_draw_min_unique", 8)),
                ):
                    break

            move_no = board.fullmove_number
            # Early draw adjudication is handled in _should_terminate_early; still allow claimable draws
            if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
                break
            
            # Determine temperature for the current move
            if move_no < temp_moves:
                temperature = temp_start
            else:
                temperature = temp_end

            # Throttle progress logs: switch to DEBUG to reduce output noise
            move_count = len(states) + 1
            if move_count % 50 == 1:
                logger.debug(f"Game {g + 1}: Move {move_count}")

            try:
                t_move0 = perf_counter()
                visit_counts, pi, v = mcts.run(board)
                t_move1 = perf_counter()
                
                # Validate visit counts
                if not visit_counts or all(count == 0 for count in visit_counts.values()):
                    logger.warning(f"Invalid visit counts: {visit_counts}")
                    # Fallback to random legal move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                    else:
                        logger.error("No legal moves available")
                        break
                else:
                    move = sample_move_from_counts(board, visit_counts, temperature)
                    
            except Exception as e:
                logger.error(f"MCTS failed: {e}")
                # Fallback to random legal move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                else:
                    logger.error("No legal moves available")
                    break
                # Provide safe defaults for policy and value on failure
                # Uniform over legal moves
                legal = list(board.legal_moves)
                pi = np.zeros(4672, dtype=np.float32)
                if len(legal) > 0:
                    p = 1.0 / float(len(legal))
                    for mv in legal:
                        try:
                            pi[move_to_index(board, mv)] = p
                        except Exception:
                            pass
                v = 0.0
            
            states.append(encode_board(board))
            pis.append(pi)
            turns.append(1 if board.turn == chess.WHITE else -1)
            search_values.append(v)
            # Track sims used
            try:
                sims_used.append(int(getattr(mcts, '_last_sims_run', 0)))
            except Exception:
                pass

            # Resign check
            if resign_thr > -1.0:
                if v < resign_thr:
                    consec_bad += 1
                else:
                    consec_bad = 0
                if consec_bad >= 3:
                    resigned = True
                    # Assign a proper game result on resignation (from white's perspective).
                    # The side to move is resigning and thus loses.
                    z = -1.0 if board.turn == chess.WHITE else 1.0
                    break
            
            move_history.append(move)
            board.push(move)

            # Heartbeat to orchestrator every 5 seconds
            if q is not None:
                now = perf_counter()
                if now - last_hb >= 5.0:
                    try:
                        q.put({
                            "type": "heartbeat",
                            "proc": proc_id,
                            "game": g,
                            "moves": len(states)
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
                            break # End the game
                    except Exception as e:
                        logger.warning(f"Tablebase probe failed: {e}")

        # If the game didn't end by tablebase, get the result normally
        if 'z' not in locals():
            z = game_result(board)
        
        # Blend final game result with search values for a more stable target
        value_target = []
        for i in range(len(states)):
            final_z = z * turns[i]
            search_z = search_values[i]
            blended_z = 0.7 * final_z + 0.3 * search_z
            value_target.append(blended_z)

        game_data = {
            "s": np.array(states, dtype=np.float32),
            "pi": np.array(pis, dtype=np.float32),
            "z": np.array(value_target, dtype=np.float32),
        }
        try:
            filepath = data_manager.add_selfplay_data(game_data, worker_id=proc_id, game_id=g)
        except Exception as e:
            logger.error(f"Failed to save self-play game w{proc_id} g{g}: {e}")
            filepath = None

        game_time = perf_counter() - t0
        avg_ms_per_move = (game_time * 1000.0 / max(1, len(states)))
        avg_sims = (float(sum(sims_used)) / max(1, len(sims_used))) if sims_used else 0.0
        logger.info(f"Game {g + 1} completed: {len(states)} moves, result: {z:.3f}, time: {game_time:.1f}s")
        
        if q is not None:
            q.put({
                "type": "game",
                "proc": proc_id,
                "file": filepath,
                "moves": len(states),
                "result": float(z),
                "secs": game_time,
                "resigned": resigned,
                "avg_ms_per_move": avg_ms_per_move,
                "avg_sims": avg_sims,
            })


def sample_move_from_counts(board: chess.Board, counts: Dict[chess.Move, int], temperature: float) -> chess.Move:
    if not counts:
        # Should not happen in a normal game, but as a fallback
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        return legal_moves[0]

    moves = list(counts.keys())
    visits = np.array(list(counts.values()), dtype=np.float32)
    
    # Handle edge cases: all zero visits or NaN values
    if np.any(np.isnan(visits)) or np.all(visits == 0):
        # Fallback to uniform distribution over legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        return np.random.choice(legal_moves)
    
    if temperature < 1e-3:
        # Deterministic: pick the best move
        return moves[np.argmax(visits)]
    
    # Probabilistic: sample from distribution
    visit_dist = visits**(1.0 / temperature)
    
    # Ensure we don't have division by zero or NaN
    visit_sum = visit_dist.sum()
    if visit_sum <= 0 or np.isnan(visit_sum):
        # Fallback to uniform distribution
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        return np.random.choice(legal_moves)
    
    visit_dist /= visit_sum
    
    # Final safety check for NaN
    if np.any(np.isnan(visit_dist)):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        return np.random.choice(legal_moves)
    
    idx = np.random.choice(len(moves), p=visit_dist)
    return moves[idx]


def game_result(board: chess.Board) -> float:
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return -1.0
    return 0.0


def _should_terminate_early(
    board: chess.Board,
    move_history: List[chess.Move],
    min_plies_before_check: int = 40,
    recent_window: int = 16,
    min_unique_in_window: int = 5,
) -> bool:
    """Heuristic early-draw adjudication to prevent marathons.

    - Only begins checking after `min_plies_before_check` plies
    - Looks at the last `recent_window` moves; if fewer than `min_unique_in_window`
      unique moves appear, the game is probably shuffling → adjudicate draw.
    - Still respects claimable draws and insufficient material.
    """
    plies = len(move_history)
    if plies < min_plies_before_check:
        return False

    # Check for repetitive patterns (potential draws)
    recent = move_history[-recent_window:]
    if len(set(recent)) < min_unique_in_window:
        return True

    # Threefold repetition or fifty-move rule
    if board.is_repetition(3) or board.can_claim_fifty_moves():
        return True

    # Clear insufficient material
    if board.is_insufficient_material():
        return True

    # Hard cap
    if plies > 200:
        return True

    return False
