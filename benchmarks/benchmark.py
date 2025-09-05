#!/usr/bin/env python3
# Matrix0 Benchmark System
"""
Main benchmark script for evaluating Matrix0 models against UCI-compliant chess engines.

Usage:
    python benchmarks/benchmark.py --config benchmarks/configs/default.yaml
    python benchmarks/benchmark.py --model checkpoints/v2_base.pt --engine stockfish --games 50
"""

import argparse
import json
import logging
import sys
import time
from multiprocessing import Event as MPEvent, Process
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn

import psutil
import yaml

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from azchess.logging_utils import setup_logging

logger = setup_logging(level=logging.INFO)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - Matrix0 model loading disabled")

from benchmarks.config import (BenchmarkConfig, ConfigManager, TestScenario,
                               UCIEngineConfig)
from benchmarks.metrics import (BenchmarkMetrics, GameMetrics, MetricsAnalyzer,
                                MetricsCollector)
from benchmarks.uci_bridge import EngineManager
from azchess.encoding import move_to_index
from rich.table import Table
from rich.live import Live
from rich import box
from azchess.selfplay.inference import (
    InferenceClient,
    run_inference_server,
    setup_shared_memory_for_worker,
)

if TORCH_AVAILABLE:
    from azchess.config import Config, select_device as _select_device
    from azchess.model.resnet import PolicyValueNet
    from azchess.mcts import MCTS, MCTSConfig



class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, config: BenchmarkConfig, live_enabled: bool = False, pgn_dir: Optional[str] = None, hb_interval: float = 5.0, mcts_sims: Optional[int] = None, shared_infer: bool = False, quiet: bool = False, seed: Optional[int] = None, device: Optional[str] = None):
        self.config = config
        self.engine_manager = EngineManager()
        self.metrics_collector = MetricsCollector(
            sample_interval=config.performance_config.sample_interval
        )
        self.live_enabled = bool(live_enabled)
        self.pgn_dir: Optional[Path] = Path(pgn_dir) if pgn_dir else None
        self.hb_interval: float = float(hb_interval)
        self.live: Optional[Live] = None
        # Optional CLI override for Matrix0 MCTS simulations
        self.cli_sims: Optional[int] = int(mcts_sims) if mcts_sims is not None else None
        # Optional shared inference backend to isolate MPS work in a separate process (stability on macOS)
        self.shared_infer_enabled: bool = bool(shared_infer)
        self._infer_proc: Optional[Process] = None
        self._infer_stop = None  # type: ignore[assignment]
        self._infer_ready = None  # type: ignore[assignment]
        self._infer_client: Optional[InferenceClient] = None
        self._shared_res = None
        self.quiet: bool = bool(quiet)
        # Seed if provided
        if seed is not None:
            try:
                import random as _random
                _random.seed(int(seed))
                import numpy as _np
                _np.random.seed(int(seed))
                if TORCH_AVAILABLE:
                    torch.manual_seed(int(seed))
                logger.info(f"Seeding RNGs with {seed}")
            except Exception:
                pass

        # Load Matrix0 model
        self.model = None
        self.mcts = None
        if TORCH_AVAILABLE:
            # Prefer unified device selection for consistency with the rest of the project
            forced_device = (device or os.environ.get("MATRIX0_DEVICE", "")).lower().strip()
            dev_req = forced_device if forced_device in ("cpu", "mps", "cuda") else "auto"
            try:
                dev_str = _select_device(dev_req)
            except Exception:
                dev_str = "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else "cpu"
            self.device = torch.device(dev_str)
            logger.info(f"Using device: {self.device}")
        else:
            self.device = torch.device("cpu")
            logger.warning("PyTorch not available, using CPU")

        logger.info(f"Initialized benchmark runner with {len(config.scenarios)} scenarios")
        logger.info(f"System resources - CPU cores: {len(psutil.cpu_percent(percpu=True))} | Memory: {psutil.virtual_memory().total // (1024**3)}GB | Device: {self.device}")

    def _build_live_table(self, scenario: TestScenario, current_game: int, total_games: int, games: List[GameMetrics], game_status: str = "") -> Table:
        # Aggregate Matrix0-centric results from games list
        m_wins = 0
        m_losses = 0
        draws = 0
        total_moves = 0
        all_move_times: List[float] = []
        for g in games:
            total_moves += int(g.total_moves)
            for mv in g.move_times:
                t = mv.get('time', 0.0)
                if isinstance(t, (int, float)):
                    all_move_times.append(float(t))
            if g.result == "1/2-1/2":
                draws += 1
            elif g.winner == "white" and g.white_engine == "Matrix0":
                m_wins += 1
            elif g.winner == "black" and g.black_engine == "Matrix0":
                m_wins += 1
            else:
                # loss for Matrix0 when other side wins
                if g.result in ("1-0", "0-1"):
                    m_losses += 1

        played = len(games)
        win_rate = (m_wins + 0.5 * draws) / played if played > 0 else 0.0
        # Low-N Elo estimate with 95% CI
        elo_str = "-"
        if played > 0:
            import math
            s = min(max(win_rate, 1e-6), 1.0 - 1e-6)
            elo = 400.0 * math.log10(s / (1.0 - s))
            se = math.sqrt(max(s * (1.0 - s) / float(played), 1e-12))
            lo_s = min(max(s - 1.96 * se, 1e-6), 1.0 - 1e-6)
            hi_s = min(max(s + 1.96 * se, 1e-6), 1.0 - 1e-6)
            elo_lo = 400.0 * math.log10(lo_s / (1.0 - lo_s))
            elo_hi = 400.0 * math.log10(hi_s / (1.0 - hi_s))
            elo_str = f"{elo:.0f} [{elo_lo:.0f},{elo_hi:.0f}]"

        avg_moves = (total_moves / played) if played > 0 else 0.0
        avg_t = (sum(all_move_times) / len(all_move_times)) if all_move_times else 0.0
        var_t = MetricsAnalyzer._calculate_variance(all_move_times) if all_move_times else 0.0

        table = Table(title=f"Matrix0 vs {scenario.engine_config.name} — {scenario.time_control}", box=box.SIMPLE_HEAVY)
        table.add_column("Progress", justify="left")
        table.add_column("W", justify="right")
        table.add_column("D", justify="right")
        table.add_column("L", justify="right")
        table.add_column("Win%", justify="right")
        table.add_column("Elo±", justify="right")
        table.add_column("AvgMoves", justify="right")
        table.add_column("AvgT/move", justify="right")
        table.add_column("VarT", justify="right")
        table.add_row(
            f"{min(current_game, total_games)}/{total_games}",
            str(m_wins),
            str(draws),
            str(m_losses),
            f"{(100.0*win_rate):.1f}",
            elo_str,
            f"{avg_moves:.1f}",
            f"{avg_t:.3f}s",
            f"{var_t:.4f}"
        )

        if game_status:
            table.add_row(f"{game_status}", "", "", "", "", "", "", "", "")
        return table

    def _refresh_live(self, scenario: TestScenario, current_game: int, total_games: int, games: List[GameMetrics], game_status: str = "") -> None:
        if not self.live_enabled or self.live is None:
            return
        table = self._build_live_table(scenario, current_game, total_games, games, game_status)
        self.live.update(table)

    def load_model(self, checkpoint_path: str) -> bool:
        """Load Matrix0 model from checkpoint."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available - cannot load Matrix0 model")
            return False

        try:
            logger.info(f"Loading Matrix0 model from {checkpoint_path}")

            # Load model config
            model_config_obj = Config.load('config.yaml').model()

            # Create model
            self.model = PolicyValueNet.from_config(model_config_obj)
            self.model.to(self.device)
            self.model.eval()

            # Load checkpoint (prefer EMA, then model, then model_state_dict)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state = checkpoint.get('model_ema') or checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint
            self.model.load_state_dict(state, strict=False)

            logger.info(f"Successfully loaded Matrix0 model ({sum(p.numel() for p in self.model.parameters()):,} parameters)")

            # Initialize MCTS with benchmark-appropriate settings
            # Derive MCTS defaults from config.yaml, allow CLI override of sims, and disable noise for eval
            mcfg = Config.load('config.yaml').mcts() if TORCH_AVAILABLE else {}
            mcts_kwargs = dict(
                num_simulations=(self.cli_sims if self.cli_sims is not None else int(mcfg.get('num_simulations', 800))),
                cpuct=float(mcfg.get('cpuct', 2.2)),
                dirichlet_alpha=float(mcfg.get('dirichlet_alpha', 0.3)),
                dirichlet_frac=0.0,  # disabled for evaluation
                tt_capacity=int(mcfg.get('tt_capacity', 100000)),
                selection_jitter=0.0,
                enable_entropy_noise=False,
            )

            # Safety: avoid multi-threaded MPS in-process inference; will override if shared inference is enabled
            if str(self.device) == "mps":
                mcts_kwargs.update(num_threads=1, parallel_simulations=False)
                logger.info("MPS detected — forcing single-threaded MCTS for stability (will enable batching if shared inference)")

            # Optional shared inference server (mirrors orchestrator stability path)
            inference_backend = None
            if self.shared_infer_enabled and str(self.device) != "cpu":
                try:
                    import os as _os
                    _os.environ.setdefault("MATRIX0_COMPACT_LOG", "1")
                    # Derive planes/policy_size robustly
                    planes = 19
                    policy_size = 4672
                    try:
                        planes = int(getattr(getattr(self.model, 'cfg', None), 'planes', planes))
                        policy_size = int(getattr(getattr(self.model, 'cfg', None), 'policy_size', policy_size))
                    except Exception:
                        pass
                    if isinstance(model_config_obj, dict):
                        planes = int(model_config_obj.get('planes', planes))
                        policy_size = int(model_config_obj.get('policy_size', policy_size))

                    self._shared_res = setup_shared_memory_for_worker(worker_id=0, planes=planes, policy_size=policy_size, max_batch_size=32)
                    self._infer_stop = MPEvent()
                    self._infer_ready = MPEvent()

                    # Capture the exact state dict (prefer EMA) we loaded and ensure CPU tensors for interprocess transfer
                    state_for_server = checkpoint.get('model_ema') or checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint
                    try:
                        if isinstance(state_for_server, dict):
                            state_for_server_cpu = {}
                            for k, v in state_for_server.items():
                                try:
                                    import torch as _torch
                                    if _torch.is_tensor(v):
                                        state_for_server_cpu[k] = v.detach().to('cpu')
                                    else:
                                        state_for_server_cpu[k] = v
                                except Exception:
                                    state_for_server_cpu[k] = v
                        else:
                            state_for_server_cpu = state_for_server
                    except Exception:
                        state_for_server_cpu = state_for_server

                    self._infer_proc = Process(target=run_inference_server, args=(str(self.device), model_config_obj if isinstance(model_config_obj, dict) else model_config_obj, state_for_server_cpu, self._infer_stop, self._infer_ready, [self._shared_res]))
                    self._infer_proc.start()
                    if not self._infer_ready.wait(timeout=60):
                        if not self.quiet:
                            logger.error("Inference server failed to start in time; continuing without shared backend")
                    else:
                        self._infer_client = InferenceClient(self._shared_res)
                        inference_backend = self._infer_client
                        logger.info("Shared inference server initialized for benchmark")
                        # With shared inference, enable batched parallel simulations to reduce per-infer overhead
                        try:
                            mcts_kwargs.update(num_threads=4, parallel_simulations=True, simulation_batch_size=16)
                            logger.info("Enabled parallel MCTS simulations with batching (threads=4, batch=16)")
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"Failed to initialize shared inference backend: {e}")
                    inference_backend = None

            # Finalize MCTS config (after possible shared-infer overrides)
            mcts_config = MCTSConfig(**mcts_kwargs)
            self.mcts = MCTS(mcts_config, self.model, self.device, inference_backend=inference_backend)
            logger.info(f"MCTS initialized with {mcts_config.num_simulations} simulations per move on device: {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def run_benchmark(self) -> BenchmarkMetrics:
        """Run the complete benchmark suite."""
        logger.info(f"Starting benchmark: {self.config.name}")
        logger.info(f"Description: {self.config.description}")

        benchmark_metrics = BenchmarkMetrics(
            benchmark_name=self.config.name,
            start_time=time.time(),
            end_time=0.0  # Will be updated when benchmark completes
        )

        # Optional live view context per scenario
        # Run each scenario
        for scenario in self.config.scenarios:
            logger.info(f"Running scenario: {scenario.name}")

            live_ctx = Live(self._build_live_table(scenario, 0, scenario.num_games, benchmark_metrics.games, "Ready"), refresh_per_second=4) if self.live_enabled else None
            if live_ctx:
                self.live = live_ctx
                self.live.start()
            try:
                # Load model if needed
                if self.model is None:
                    if not self.load_model(scenario.model_checkpoint):
                        logger.error(f"Skipping scenario {scenario.name} - model load failed")
                        continue

                # Add and start engine
                logger.info(f"Adding engine {scenario.engine_config.name}...")
                if not self.engine_manager.add_engine(scenario.engine_config):
                    logger.error(f"Skipping scenario {scenario.name} - engine start failed")
                    continue
                logger.info(f"Engine {scenario.engine_config.name} started successfully")

                # Run scenario
                scenario_metrics = self._run_scenario(scenario)
                benchmark_metrics.games.extend(scenario_metrics)
            finally:
                # Clean up engine and live view
                self.engine_manager.remove_engine(scenario.engine_config.name)
                if live_ctx:
                    live_ctx.stop()
                    self.live = None

        # Finalize benchmark
        benchmark_metrics.end_time = time.time()

        # Calculate aggregate statistics
        self._calculate_aggregate_stats(benchmark_metrics)

        # Export results
        self._export_results(benchmark_metrics)

        logger.info("✅ Benchmark completed!")
        logger.info(f"⏱️  Total duration: {benchmark_metrics.duration:.1f} seconds")
        logger.info(f"📊 Total games played: {benchmark_metrics.total_games}")
        # Shutdown shared inference server if running
        try:
            if self._infer_proc is not None:
                if self._infer_stop is not None:
                    self._infer_stop.set()
                self._infer_proc.join(timeout=5)
                self._infer_proc = None
                self._infer_client = None
                self._shared_res = None
        except Exception:
            pass
        return benchmark_metrics

    def _run_scenario(self, scenario: TestScenario) -> List[GameMetrics]:
        """Run a single test scenario."""
        logger.info(f"Running {scenario.num_games} games against {scenario.engine_config.name}")

        games = []
        engine = self.engine_manager.get_engine(scenario.engine_config.name)

        if not engine:
            logger.error(f"Engine {scenario.engine_config.name} not available")
            return games

        # Start metrics collection
        if self.config.performance_config.track_cpu or self.config.performance_config.track_memory:
            self.metrics_collector.start_collection()

        try:
            for game_id in range(scenario.num_games):
                logger.info(f"🎯 Game {game_id + 1}/{scenario.num_games} - Starting...")
                start_game_time = time.time()

                # Play game
                game_metrics = self._play_game(scenario, game_id)
                if game_metrics:
                    games.append(game_metrics)
                    game_duration = time.time() - start_game_time
                    logger.info(f"   📈 Score: {game_metrics.result} | Moves: {game_metrics.total_moves}")

                # Progress logging
                if (game_id + 1) % 1 == 0:  # Show progress after every game for debugging
                    completed = len([g for g in games if g.total_moves > 0])
                    logger.info(f"📊 Progress: {completed}/{game_id + 1} games completed")

        finally:
            # Stop metrics collection
            if self.config.performance_config.track_cpu or self.config.performance_config.track_memory:
                collected_metrics = self.metrics_collector.stop_collection()

                # Distribute system metrics across games
                if collected_metrics and games:
                    metrics_per_game = len(collected_metrics) // len(games)
                    for i, game in enumerate(games):
                        start_idx = i * metrics_per_game
                        end_idx = (i + 1) * metrics_per_game if i < len(games) - 1 else len(collected_metrics)
                        game.system_metrics = collected_metrics[start_idx:end_idx]

        return games

    def _play_game(self, scenario: TestScenario, game_id: int) -> Optional[GameMetrics]:
        """Play a single game between Matrix0 and UCI engine."""
        try:
            logger.info(f"🎮 Initializing game {game_id + 1}")
            engine = self.engine_manager.get_engine(scenario.engine_config.name)
            if not engine:
                logger.error(f"Engine {scenario.engine_config.name} not available")
                return None

            # Alternate sides: even-indexed games -> Matrix0 plays White, odd -> Matrix0 plays Black
            matrix0_white = (game_id % 2 == 0)

            game_metrics = GameMetrics(
                game_id=f"{scenario.name}_game_{game_id}",
                start_time=time.time(),
                end_time=0.0,  # Will be updated when game ends
                total_moves=0,  # Will be updated as game progresses
                result="*",  # Will be updated when game ends
                winner="unknown",  # Will be updated when game ends
                white_engine=("Matrix0" if matrix0_white else scenario.engine_config.name),
                black_engine=(scenario.engine_config.name if matrix0_white else "Matrix0")
            )

            # Initialize engines
            logger.info("Setting up engines...")
            self.model.eval()  # Matrix0 doesn't need new_game
            engine.new_game()
            logger.info("Engines initialized successfully")

            # Set up initial position
            logger.info("Setting up starting position...")
            current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            moves = []
            game_metrics.move_times = []
            logger.info("Game setup complete - starting moves")

            # Play game
            termination_reason = ""
            last_hb = time.time()
            for move_num in range(scenario.max_moves):
                is_white_turn = (move_num % 2 == 0)
                is_matrix0_turn = (is_white_turn and matrix0_white) or ((not is_white_turn) and (not matrix0_white))
                side_label = ("White" if is_white_turn else "Black")
                to_move_label = f"{side_label} ({'Matrix0' if is_matrix0_turn else scenario.engine_config.name})"
                logger.info(f"Move {move_num + 1}: {to_move_label} to move")
                start_time = time.time()

                try:
                    # Live heartbeat update
                    if self.live_enabled and self.live is not None and (time.time() - last_hb) >= self.hb_interval:
                        status = f"Game {game_id + 1}/{scenario.num_games} • Move {move_num + 1} • {to_move_label}"
                        # Update outer progress table using all completed games so far
                        self._refresh_live(scenario, game_id + 1, scenario.num_games, [*[]], status)
                        last_hb = time.time()
                    # Validate board and detect terminal positions before making a move
                    board = chess.Board(current_fen)
                    for move_uci in moves:
                        try:
                            board.push_uci(move_uci)
                        except Exception:
                            termination_reason = "invalid_move_in_history"
                            break
                    if termination_reason:
                        break
                    if board.is_game_over():
                        termination_reason = "terminal_position"
                if not self.quiet:
                    logger.info("Position is terminal before move; ending game.")
                        break
                    if is_matrix0_turn:
                        if not self.quiet:
                            logger.info("🤖 Matrix0 thinking...")
                        move, move_time, diag = self._get_matrix0_move(current_fen, moves)
                        if not self.quiet:
                            logger.info(f"🤖 Matrix0 played: {move}")
                    else:
                        if not self.quiet:
                            logger.info("♟️  Stockfish thinking...")
                        engine.set_position(current_fen, moves)
                        move, move_time = engine.go(scenario.time_control)
                        diag = None
                        if not self.quiet:
                            logger.info(f"♟️  Stockfish played: {move}")

                    end_time = time.time()
                    actual_time = end_time - start_time

                    # Record move time + diagnostics
                    mt_rec = {
                        "move_num": move_num + 1,
                        "player": "white" if move_num % 2 == 0 else "black",
                        "move": move,
                        "time": actual_time,
                        "timestamp": end_time
                    }
                    # Attach diagnostics on Matrix0 moves
                    if is_matrix0_turn and isinstance(diag, dict):
                        if diag.get('empty_visits'):
                            game_metrics.mcts_empty_visits += 1
                        ent = diag.get('root_policy_entropy')
                        if isinstance(ent, (int, float)):
                            game_metrics.root_policy_entropy_sum += float(ent)
                            game_metrics.root_policy_entropy_samples += 1
                        mt_rec['root_entropy'] = ent
                        mt_rec['mcts_empty'] = bool(diag.get('empty_visits', False))
                    game_metrics.move_times.append(mt_rec)

                    # Update game state
                    moves.append(move)
                    # Note: FEN updating would require chess library integration
                    # For now, we'll keep the initial FEN and rely on move list

                    # Update timing totals
                    if is_white_turn:
                        game_metrics.white_total_time += actual_time
                    else:
                        game_metrics.black_total_time += actual_time

                    # Check for game end or invalid move
                    if not move or move == "0000":
                        legal = list(board.legal_moves)
                        if not legal:
                            termination_reason = "terminal_position"
                            logger.info("Game ended - no legal moves (terminal).")
                        else:
                            termination_reason = "invalid_move"
                            logger.info("Game ended - invalid move received")
                        break

                except Exception as e:
                    logger.error(f"Error during move {move_num + 1}: {e}")
                    logger.info("Ending game due to error")
                    break

            # Calculate final statistics
            game_metrics.end_time = time.time()
            game_metrics.total_moves = len(moves)

            # Determine result based on termination_reason
            if len(moves) >= scenario.max_moves:
                game_metrics.result = "1/2-1/2"
                game_metrics.winner = "draw"
                logger.info(f"🏁 Game ended by move limit ({scenario.max_moves} moves)")
            elif termination_reason.startswith("invalid"):
                # Determine who made the invalid move
                if len(moves) % 2 == 1:  # White made the last move (even indices)
                    game_metrics.result = "0-1"
                    game_metrics.winner = "black"
                    logger.info(f"🏁 Game ended - White (Matrix0) made invalid move!")
                else:
                    game_metrics.result = "1-0"
                    game_metrics.winner = "white"
                    logger.info(f"🏁 Game ended - Black ({scenario.engine_config.name}) made invalid move!")
            elif termination_reason == "terminal_position":
                # Use chess rules to decide result
                current_board = chess.Board()
                for mv in moves:
                    try:
                        current_board.push_uci(mv)
                    except Exception:
                        break
                if current_board.is_checkmate():
                    # Side to move is checkmated in python-chess
                    if current_board.turn:
                        game_metrics.result = "0-1"
                        game_metrics.winner = "black"
                    else:
                        game_metrics.result = "1-0"
                        game_metrics.winner = "white"
                elif current_board.is_stalemate() or current_board.is_insufficient_material() or current_board.can_claim_threefold_repetition() or current_board.can_claim_fifty_moves():
                    game_metrics.result = "1/2-1/2"
                    game_metrics.winner = "draw"
            else:
                # Check for checkmate or stalemate
                current_board = chess.Board()
                for move in moves:
                    current_board.push_uci(move)

                if current_board.is_checkmate():
                    # Side to move is checkmated
                    if current_board.turn:
                        game_metrics.result = "0-1"
                        game_metrics.winner = "black"
                        logger.info(f"🏁 Checkmate! Black ({game_metrics.black_engine}) wins!")
                    else:
                        game_metrics.result = "1-0"
                        game_metrics.winner = "white"
                        logger.info(f"🏁 Checkmate! White ({game_metrics.white_engine}) wins!")
                elif current_board.is_stalemate():
                    game_metrics.result = "1/2-1/2"
                    game_metrics.winner = "draw"
                    logger.info(f"🏁 Stalemate! Game is a draw")
                else:
                    game_metrics.result = "1/2-1/2"  # Default to draw
                    game_metrics.winner = "draw"
                    logger.info(f"🏁 Game ended - result unclear, recorded as draw")

            # Calculate averages
            if game_metrics.total_moves > 0:
                game_metrics.white_avg_time_per_move = game_metrics.white_total_time / ((game_metrics.total_moves + 1) // 2)
                game_metrics.black_avg_time_per_move = game_metrics.black_total_time / (game_metrics.total_moves // 2)

            # Clear winner announcement
            if game_metrics.winner == "white":
                winner_name = game_metrics.white_engine
                loser_name = game_metrics.black_engine
                logger.info(f"🎉 WINNER: {winner_name} (White) defeated {loser_name}!")
            elif game_metrics.winner == "black":
                winner_name = game_metrics.black_engine
                loser_name = game_metrics.white_engine
                logger.info(f"🎉 WINNER: {winner_name} (Black) defeated {loser_name}!")
            else:
                winner_name = "Draw"
                logger.info(f"🤝 DRAW: {game_metrics.white_engine} vs {game_metrics.black_engine}")

            logger.info(f"Game completed: {game_metrics.total_moves} moves, result: {game_metrics.result} ({winner_name})")

            # Optional PGN export
            try:
                if self.pgn_dir:
                    self.pgn_dir.mkdir(parents=True, exist_ok=True)
                    game = chess.pgn.Game()
                    game.headers["Event"] = scenario.name
                    game.headers["White"] = game_metrics.white_engine
                    game.headers["Black"] = game_metrics.black_engine
                    game.headers["Result"] = game_metrics.result
                    board = chess.Board()
                    node = game
                    for mv in moves:
                        try:
                            move_obj = chess.Move.from_uci(mv)
                            node = node.add_main_variation(move_obj)
                            board.push(move_obj)
                        except Exception:
                            break
                    # Validate/Correct PGN result header vs reconstructed board
                    if board.is_game_over(claim_draw=True):
                        true_res = board.result(claim_draw=True)
                        if str(true_res) != str(game.headers.get("Result", "")):
                            logger.warning(f"PGN result mismatch: header={game.headers.get('Result')} actual={true_res}; correcting header")
                            game.headers["Result"] = true_res
                    out_path = self.pgn_dir / f"{scenario.name}_game_{game_id}.pgn"
                    with open(out_path, 'w') as f:
                        print(game, file=f)
            except Exception as e:
                logger.warning(f"PGN export failed for game {game_id}: {e}")

            return game_metrics

        except Exception as e:
            logger.error(f"Error playing game {game_id}: {e}")
            return None

    def _get_matrix0_move(self, fen: str, moves: List[str]) -> Tuple[str, float, Dict[str, Any]]:
        """Get move from Matrix0 model using MCTS with configured simulations.

        Returns: (move_uci, move_time_seconds, diagnostics)
        diagnostics: { 'empty_visits': bool, 'root_policy_entropy': float }
        """
        try:
            import chess
            from azchess.encoding import encode_board

            start_time = time.time()
            logger.info(f"🔍 Creating board position from {len(moves)} moves...")

            # Create chess board from FEN and moves
            board = chess.Board(fen)
            for move_uci in moves:
                board.push_uci(move_uci)

            mcts_start = time.time()

            try:
                # Run MCTS with configured simulations per move for Matrix0
                num_simulations = int(self.cli_sims) if self.cli_sims is not None else int(getattr(self.mcts.cfg, 'num_simulations', 200))
                logger.info(f"🧠 Running MCTS with {num_simulations} simulations...")
                # Ensure MCTS uses the same device as the model
                visits, policy, value = self.mcts.run(board, num_simulations=num_simulations)

                mcts_time = time.time() - mcts_start
                logger.info(f"🧠 MCTS completed in {mcts_time:.1f}s ({num_simulations} simulations)")
            except Exception as e:
                logger.error(f"MCTS failed after {time.time() - mcts_start:.1f}s: {e}")
                raise

            # Diagnostics: compute legal-only entropy from returned policy
            root_entropy = None
            try:
                legal = list(board.legal_moves)
                if legal and policy is not None and len(policy) >= 4672:
                    import math
                    import numpy as np
                    mass = []
                    for mv in legal:
                        try:
                            idx = move_to_index(board, mv)
                            mass.append(float(policy[idx]) if 0 <= idx < len(policy) else 0.0)
                        except Exception:
                            mass.append(0.0)
                    s = sum(mass)
                    if s > 0:
                        p = [x / s for x in mass]
                        root_entropy = -sum((x * math.log(max(x, 1e-12)) for x in p))
            except Exception:
                root_entropy = None

            # Get best move from visit counts
            if visits:
                best_move = max(visits.items(), key=lambda x: x[1])[0]
                move_uci = best_move.uci()
                move_time = time.time() - start_time

                logger.debug(f"Matrix0 move: {move_uci} in {move_time:.3f}s")
                return move_uci, move_time, {"empty_visits": False, "root_policy_entropy": root_entropy}
            else:
                logger.warning("No moves found from MCTS, using policy-based fallback")
                # Fallback: choose legal move with highest policy probability
                legal_moves = list(board.legal_moves)
                if legal_moves and policy is not None and len(policy) >= 4672:
                    best = None
                    best_score = -1.0
                    for mv in legal_moves:
                        try:
                            idx = move_to_index(board, mv)
                            score = float(policy[idx]) if 0 <= idx < len(policy) else 0.0
                        except Exception:
                            score = 0.0
                        if score > best_score:
                            best_score = score
                            best = mv
                    if best is not None:
                        move_uci = best.uci()
                        move_time = time.time() - start_time
                        return move_uci, move_time, {"empty_visits": True, "root_policy_entropy": root_entropy}
                # Final fallback
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    best = legal_moves[0]
                    return best.uci(), time.time() - start_time, {"empty_visits": True, "root_policy_entropy": root_entropy}
                logger.error("No legal moves available (terminal position)")
                return "", 0.0, {"empty_visits": True, "root_policy_entropy": root_entropy}

        except Exception as e:
            logger.error(f"Error getting Matrix0 move: {e}")
            return "", 0.0, {"empty_visits": True, "root_policy_entropy": None}

    def _calculate_aggregate_stats(self, benchmark_metrics: BenchmarkMetrics):
        """Calculate aggregate statistics for the benchmark."""
        games = benchmark_metrics.games

        if not games:
            return

        benchmark_metrics.total_games = len(games)
        benchmark_metrics.white_wins = sum(1 for game in games if game.result == "1-0")
        benchmark_metrics.black_wins = sum(1 for game in games if game.result == "0-1")
        benchmark_metrics.draws = sum(1 for game in games if game.result == "1/2-1/2")

        # Engine-centric aggregation (Matrix0 vs Opponent)
        engine_names = set()
        for g in games:
            engine_names.add(g.white_engine)
            engine_names.add(g.black_engine)

        engine_stats: Dict[str, Dict[str, Any]] = {}
        for name in engine_names:
            engine_stats[name] = {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
            }

        for g in games:
            if g.winner == "white":
                winner = g.white_engine
                loser = g.black_engine
                engine_stats[winner]["wins"] += 1
                engine_stats[loser]["losses"] += 1
            elif g.winner == "black":
                winner = g.black_engine
                loser = g.white_engine
                engine_stats[winner]["wins"] += 1
                engine_stats[loser]["losses"] += 1
            else:
                # Draw counts for both engines
                engine_stats[g.white_engine]["draws"] += 1
                engine_stats[g.black_engine]["draws"] += 1

        # Finalize win rates per engine
        for name, s in engine_stats.items():
            total_played = s["wins"] + s["losses"] + s["draws"]
            if total_played > 0:
                s["win_rate"] = (s["wins"] + 0.5 * s["draws"]) / float(total_played)

        benchmark_metrics.engine_stats = engine_stats

        # Calculate averages
        total_duration = sum(game.duration for game in games)
        total_moves = sum(game.total_moves for game in games)

        benchmark_metrics.avg_game_duration = total_duration / len(games)
        benchmark_metrics.avg_moves_per_game = total_moves / len(games)

        # Time per move
        all_move_times = []
        for game in games:
            all_move_times.extend([move.get('time', 0) for move in game.move_times])

        if all_move_times:
            benchmark_metrics.avg_time_per_move = sum(all_move_times) / len(all_move_times)

        # System resources (if available)
        all_cpu = []
        all_memory = []
        peak_memory = 0

        for game in games:
            for metrics in game.system_metrics:
                all_cpu.append(metrics.cpu_percent)
                all_memory.append(metrics.memory_percent)
                peak_memory = max(peak_memory, metrics.memory_used_gb)

        if all_cpu:
            benchmark_metrics.avg_cpu_usage = sum(all_cpu) / len(all_cpu)
        if all_memory:
            benchmark_metrics.avg_memory_usage = sum(all_memory) / len(all_memory)
        benchmark_metrics.peak_memory_usage = peak_memory

    def _export_results(self, benchmark_metrics: BenchmarkMetrics):
        """Export benchmark results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export main results
        main_results_path = output_dir / f"{self.config.name.replace(' ', '_').lower()}_results.json"
        MetricsAnalyzer.export_metrics(benchmark_metrics, str(main_results_path))

        # Export summary report
        summary_path = output_dir / f"{self.config.name.replace(' ', '_').lower()}_summary.json"
        summary = MetricsAnalyzer.analyze_game_metrics(benchmark_metrics.games)
        # Attach engine-centric aggregation to summary for clarity
        summary["by_engine"] = benchmark_metrics.engine_stats

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Export configuration
        config_path = output_dir / f"{self.config.name.replace(' ', '_').lower()}_config.yaml"
        ConfigManager.save_config(self.config, str(config_path))

        logger.info(f"Results exported to {output_dir}")
        logger.info(f"Main results: {main_results_path}")
        logger.info(f"Summary: {summary_path}")
        logger.info(f"Config: {config_path}")


def create_default_config():
    """Create a default benchmark configuration."""
    from benchmarks.config import PREDEFINED_ENGINES

    # Use predefined engines if available
    engines = {}
    for name, config in PREDEFINED_ENGINES.items():
        engines[name] = {
            "command": config.command,
            "working_dir": config.working_dir,
            "options": config.options
        }

    return {
        "name": "Matrix0 Benchmark",
        "description": "Standard benchmark against Stockfish",
        "output_dir": "benchmarks/results",
        "engines": engines,
        "scenarios": [
            {
                "name": "Stockfish_Benchmark",
                "engine": "stockfish",
                "model_checkpoint": "checkpoints/v2_base.pt",
                "num_games": 10,
                "time_control": "30+0.3",
                "concurrency": 1
            }
        ],
        "performance": {
            "track_cpu": True,
            "track_memory": True,
            "track_gpu": True,
            "sample_interval": 0.1
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Matrix0 Benchmark System")
    parser.add_argument("--config", type=str, help="Benchmark configuration file")
    parser.add_argument("--model", type=str, help="Matrix0 model checkpoint")
    parser.add_argument("--engine", type=str, help="UCI engine name")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    parser.add_argument("--time-control", type=str, default="30+0.3", help="Time control")
    parser.add_argument("--mcts-sims", type=int, default=None, help="Override Matrix0 MCTS simulations per move (recommended 800–1600)")
    parser.add_argument("--engine-option", action='append', default=None, help="UCI engine option override, e.g. --engine-option 'Skill Level=8' --engine-option 'UCI_Elo=1500'")
    parser.add_argument("--output", type=str, default="benchmarks/results", help="Output directory")
    parser.add_argument("--live", action="store_true", help="Enable live TUI updates per game")
    parser.add_argument("--export-pgns", type=str, default=None, help="Directory to export PGN files per game")
    parser.add_argument("--hb-interval", type=float, default=5.0, help="Heartbeat interval in seconds for live TUI")
    parser.add_argument("--shared-infer", action="store_true", help="Run Matrix0 inference in a separate process (MPS stability)")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-move logging (compact output)")
    parser.add_argument("--seed", type=int, default=None, help="Seed RNGs for reproducible matches")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu|mps|cuda")

    args = parser.parse_args()

    try:
        if args.config:
            # Load from config file
            config = ConfigManager.load_config(args.config)
        elif args.model and args.engine:
            # Create config from command line args
            from benchmarks.config import TestScenario, UCIEngineConfig

            # Build engine options from CLI overrides
            options = {}
            if args.engine_option:
                for kv in args.engine_option:
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        options[k.strip()] = v.strip()
            # Resolve engine binary path from env or global config if available
            eng = args.engine.lower()
            cmd = eng
            try:
                from azchess.config import Config as _GCfg
                gcfg = _GCfg.load('config.yaml')
                geng = gcfg.engines().get(eng, {})
                path_cfg = geng.get('path') or geng.get('command')
                if path_cfg:
                    cmd = str(path_cfg)
            except Exception:
                pass
            # Env overrides
            env_map = {'stockfish': 'STOCKFISH_PATH', 'lc0': 'LC0_PATH', 'komodo': 'KOMODO_PATH'}
            env_var = env_map.get(eng)
            if env_var and os.environ.get(env_var):
                cmd = os.environ[env_var]

            engine_config = UCIEngineConfig(
                name=args.engine,
                command=cmd,
                options=options
            )

            scenario = TestScenario(
                name=f"{args.engine}_benchmark",
                engine_config=engine_config,
                model_checkpoint=args.model,
                num_games=args.games,
                time_control=args.time_control
            )

            config = BenchmarkConfig(
                name="Command Line Benchmark",
                description=f"Benchmark against {args.engine}",
                output_dir=args.output,
                scenarios=[scenario]
            )
        else:
            # Create default config
            default_config_data = create_default_config()
            config_path = "benchmarks/configs/default.yaml"
            Path("benchmarks/configs").mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                yaml.dump(default_config_data, f, default_flow_style=False)

            config = ConfigManager.load_config(config_path)

        # Run benchmark
        runner = BenchmarkRunner(config, live_enabled=bool(args.live), pgn_dir=args.export_pgns, hb_interval=float(args.hb_interval), mcts_sims=args.mcts_sims, shared_infer=bool(args.shared_infer), quiet=bool(args.quiet), seed=args.seed, device=args.device)
        results = runner.run_benchmark()

        # Print summary (engine-centric)
        print("\n📊 Benchmark Summary (Matrix0-centric):")
        print(f"Games played: {results.total_games}")
        # Engine-centric lines
        m = results.engine_stats.get("Matrix0", {"wins": 0, "losses": 0, "draws": 0, "win_rate": 0.0})
        # Find opponent name(s) excluding Matrix0
        opponents = [name for name in results.engine_stats.keys() if name != "Matrix0"]
        opp_display = opponents[0] if opponents else "Opponent"
        print(f"Matrix0 vs {opp_display}: W {m['wins']} / D {m['draws']} / L {m['losses']} | Win% {(100.0*m['win_rate']):.1f}")
        # Optional color-based for debugging
        print(f"(Color split) White wins: {results.white_wins} | Black wins: {results.black_wins} | Draws: {results.draws}")
        # Optional: print averages if available
        if results.total_games > 0:
            print(f"Avg duration: {results.avg_game_duration:.1f}s")
            print(f"Avg moves: {results.avg_moves_per_game:.1f}")
            if results.avg_time_per_move:
                print(f"Avg time/move: {results.avg_time_per_move:.3f}s")
            # Elo estimate (Matrix0-centric, low-N)
            try:
                import math
                s = min(max(m['win_rate'], 1e-6), 1.0 - 1e-6)
                elo = 400.0 * math.log10(s / (1.0 - s))
                se = math.sqrt(max(s * (1.0 - s) / float(results.total_games), 1e-12))
                lo_s = min(max(s - 1.96 * se, 1e-6), 1.0 - 1e-6)
                hi_s = min(max(s + 1.96 * se, 1e-6), 1.0 - 1e-6)
                elo_lo = 400.0 * math.log10(lo_s / (1.0 - lo_s))
                elo_hi = 400.0 * math.log10(hi_s / (1.0 - hi_s))
                print(f"Elo estimate (low-N): {elo:.0f} [{elo_lo:.0f}, {elo_hi:.0f}]")
            except Exception:
                pass

        # Clean up
        runner.engine_manager.stop_all()

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
