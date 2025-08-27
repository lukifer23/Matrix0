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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess

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

if TORCH_AVAILABLE:
    from azchess.config import Config
    from azchess.model.resnet import PolicyValueNet
    from azchess.mcts import MCTS, MCTSConfig



class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine_manager = EngineManager()
        self.metrics_collector = MetricsCollector(
            sample_interval=config.performance_config.sample_interval
        )

        # Load Matrix0 model
        self.model = None
        self.mcts = None
        if TORCH_AVAILABLE:
            if torch.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon) device")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU device (MPS not available)")
        else:
            self.device = torch.device("cpu")
            logger.warning("PyTorch not available, using CPU")

        logger.info(f"Initialized benchmark runner with {len(config.scenarios)} scenarios")
        logger.info(f"System resources - CPU cores: {len(psutil.cpu_percent(percpu=True))} | Memory: {psutil.virtual_memory().total // (1024**3)}GB | Device: {self.device}")

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

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            logger.info(f"Successfully loaded Matrix0 model ({sum(p.numel() for p in self.model.parameters()):,} parameters)")

            # Initialize MCTS with benchmark-appropriate settings
            mcts_config = MCTSConfig(
                num_simulations=200,  # Match config.yaml setting
                cpuct=2.2,
                dirichlet_alpha=0.3,
                dirichlet_frac=0.0,
                tt_capacity=100000,
                selection_jitter=0.0
            )

            self.mcts = MCTS(mcts_config, self.model, self.device)
            logger.info(f"MCTS initialized with 200 simulations per move on device: {self.device}")
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

        # Run each scenario
        for scenario in self.config.scenarios:
            logger.info(f"Running scenario: {scenario.name}")

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

            # Clean up engine
            self.engine_manager.remove_engine(scenario.engine_config.name)

        # Finalize benchmark
        benchmark_metrics.end_time = time.time()

        # Calculate aggregate statistics
        self._calculate_aggregate_stats(benchmark_metrics)

        # Export results
        self._export_results(benchmark_metrics)

        logger.info("✅ Benchmark completed!")
        logger.info(f"⏱️  Total duration: {benchmark_metrics.duration:.1f} seconds")
        logger.info(f"📊 Total games played: {benchmark_metrics.total_games}")
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

            game_metrics = GameMetrics(
                game_id=f"{scenario.name}_game_{game_id}",
                start_time=time.time(),
                end_time=0.0,  # Will be updated when game ends
                total_moves=0,  # Will be updated as game progresses
                result="*",  # Will be updated when game ends
                winner="unknown",  # Will be updated when game ends
                white_engine="Matrix0",
                black_engine=scenario.engine_config.name
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
            for move_num in range(scenario.max_moves):
                logger.info(f"Move {move_num + 1}: {'White (Matrix0)' if move_num % 2 == 0 else 'Black (Stockfish)'} to move")
                start_time = time.time()

                try:
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
                        logger.info("Position is terminal before move; ending game.")
                        break
                    if move_num % 2 == 0:  # White to move (Matrix0)
                        logger.info("🤖 Matrix0 thinking...")
                        move, move_time = self._get_matrix0_move(current_fen, moves)
                        logger.info(f"🤖 Matrix0 played: {move}")
                    else:  # Black to move (UCI engine)
                        logger.info("♟️  Stockfish thinking...")
                        engine.set_position(current_fen, moves)
                        move, move_time = engine.go(scenario.time_control)
                        logger.info(f"♟️  Stockfish played: {move}")

                    end_time = time.time()
                    actual_time = end_time - start_time

                    # Record move time
                    game_metrics.move_times.append({
                        "move_num": move_num + 1,
                        "player": "white" if move_num % 2 == 0 else "black",
                        "move": move,
                        "time": actual_time,
                        "timestamp": end_time
                    })

                    # Update game state
                    moves.append(move)
                    # Note: FEN updating would require chess library integration
                    # For now, we'll keep the initial FEN and rely on move list

                    # Update timing totals
                    if move_num % 2 == 0:
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
                    if current_board.turn:  # Black to move -> white delivered mate
                        game_metrics.result = "1-0"
                        game_metrics.winner = "white"
                    else:
                        game_metrics.result = "0-1"
                        game_metrics.winner = "black"
                elif current_board.is_stalemate() or current_board.is_insufficient_material() or current_board.can_claim_threefold_repetition() or current_board.can_claim_fifty_moves():
                    game_metrics.result = "1/2-1/2"
                    game_metrics.winner = "draw"
            else:
                # Check for checkmate or stalemate
                current_board = chess.Board()
                for move in moves:
                    current_board.push_uci(move)

                if current_board.is_checkmate():
                    if current_board.turn:  # Black to move, so White won
                        game_metrics.result = "1-0"
                        game_metrics.winner = "white"
                        logger.info(f"🏁 Checkmate! White (Matrix0) wins!")
                    else:
                        game_metrics.result = "0-1"
                        game_metrics.winner = "black"
                        logger.info(f"🏁 Checkmate! Black ({scenario.engine_config.name}) wins!")
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
                winner_name = "Matrix0"
                logger.info(f"🎉 WINNER: Matrix0 (White) defeated {scenario.engine_config.name}!")
            elif game_metrics.winner == "black":
                winner_name = scenario.engine_config.name
                logger.info(f"🎉 WINNER: {scenario.engine_config.name} (Black) defeated Matrix0!")
            else:
                winner_name = "Draw"
                logger.info(f"🤝 DRAW: Matrix0 vs {scenario.engine_config.name}")

            logger.info(f"Game completed: {game_metrics.total_moves} moves, result: {game_metrics.result} ({winner_name})")
            return game_metrics

        except Exception as e:
            logger.error(f"Error playing game {game_id}: {e}")
            return None

    def _get_matrix0_move(self, fen: str, moves: List[str]) -> Tuple[str, float]:
        """Get move from Matrix0 model using MCTS with 200 simulations."""
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
                # Run MCTS with 200 simulations per move for Matrix0
                num_simulations = 200  # Target: 200 sims per move
                logger.info(f"🧠 Running MCTS with {num_simulations} simulations...")
                # Ensure MCTS uses the same device as the model
                visits, policy, value = self.mcts.run(board, num_simulations=num_simulations)

                mcts_time = time.time() - mcts_start
                logger.info(f"🧠 MCTS completed in {mcts_time:.1f}s ({num_simulations} simulations)")
            except Exception as e:
                logger.error(f"MCTS failed after {time.time() - mcts_start:.1f}s: {e}")
                raise

            # Get best move from visit counts
            if visits:
                best_move = max(visits.items(), key=lambda x: x[1])[0]
                move_uci = best_move.uci()
                move_time = time.time() - start_time

                logger.debug(f"Matrix0 move: {move_uci} in {move_time:.3f}s")
                return move_uci, move_time
            else:
                logger.warning("No moves found from MCTS, using fallback")
                # Fallback: random legal move to avoid deterministic blunders
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    import random
                    fallback_move = random.choice(legal_moves)
                    move_uci = fallback_move.uci()
                    move_time = time.time() - start_time
                    return move_uci, move_time
                else:
                    logger.error("No legal moves available (terminal position)")
                    return "", 0.0

        except Exception as e:
            logger.error(f"Error getting Matrix0 move: {e}")
            return "", 0.0

    def _calculate_aggregate_stats(self, benchmark_metrics: BenchmarkMetrics):
        """Calculate aggregate statistics for the benchmark."""
        games = benchmark_metrics.games

        if not games:
            return

        benchmark_metrics.total_games = len(games)
        benchmark_metrics.white_wins = sum(1 for game in games if game.result == "1-0")
        benchmark_metrics.black_wins = sum(1 for game in games if game.result == "0-1")
        benchmark_metrics.draws = sum(1 for game in games if game.result == "1/2-1/2")

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
    parser.add_argument("--output", type=str, default="benchmarks/results", help="Output directory")

    args = parser.parse_args()

    try:
        if args.config:
            # Load from config file
            config = ConfigManager.load_config(args.config)
        elif args.model and args.engine:
            # Create config from command line args
            from benchmarks.config import TestScenario, UCIEngineConfig

            engine_config = UCIEngineConfig(
                name=args.engine,
                command=args.engine.lower(),
                options={}
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
        runner = BenchmarkRunner(config)
        results = runner.run_benchmark()

        # Print summary
        print("\n📊 Benchmark Summary:")
        print(f"Games played: {results.total_games}")
        print(f"White wins: {results.white_wins}")
        print(f"Black wins: {results.black_wins}")
        print(f"Draws: {results.draws}")
        # Optional: print averages if available
        if results.total_games > 0:
            print(f"Avg duration: {results.avg_game_duration:.1f}s")
            print(f"Avg moves: {results.avg_moves_per_game:.1f}")
            if results.avg_time_per_move:
                print(f"Avg time/move: {results.avg_time_per_move:.3f}s")
            # Elo estimate (low-N): convert score to Elo diff and print a 95% CI (normal approx)
            try:
                import math
                score = (results.white_wins + 0.5 * results.draws) / float(results.total_games)
                s = min(max(score, 1e-6), 1.0 - 1e-6)
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
