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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from azchess.logging_utils import setup_logging

logger = setup_logging(level=logging.INFO)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - Matrix0 model loading disabled")

from benchmarks.config import (
    BenchmarkConfig,
    ConfigManager,
    TestScenario,
    UCIEngineConfig,
)
from benchmarks.metrics import (
    BenchmarkMetrics,
    GameMetrics,
    MetricsAnalyzer,
    MetricsCollector,
)
from benchmarks.uci_bridge import EngineManager

if TORCH_AVAILABLE:
    from azchess.config import Config
    from azchess.model.resnet import PolicyValueNet



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
        if TORCH_AVAILABLE:
            self.device = "mps" if torch.mps.is_available() else "cpu"
        else:
            self.device = "cpu"

        logger.info(f"Initialized benchmark runner with {len(config.scenarios)} scenarios")

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
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            logger.info(f"Successfully loaded Matrix0 model ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
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
            start_time=time.time()
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
            if not self.engine_manager.add_engine(scenario.engine_config):
                logger.error(f"Skipping scenario {scenario.name} - engine start failed")
                continue

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

        logger.info(f"Benchmark completed in {benchmark_metrics.duration:.1f} seconds")
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
                logger.info(f"Game {game_id + 1}/{scenario.num_games}")

                # Play game
                game_metrics = self._play_game(scenario, game_id)
                if game_metrics:
                    games.append(game_metrics)

                # Progress logging
                if (game_id + 1) % 10 == 0:
                    completed = len([g for g in games if g.total_moves > 0])
                    logger.info(f"Progress: {completed}/{game_id + 1} games completed")

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
            engine = self.engine_manager.get_engine(scenario.engine_config.name)
            if not engine:
                return None

            game_metrics = GameMetrics(
                game_id=f"{scenario.name}_game_{game_id}",
                start_time=time.time(),
                white_engine="Matrix0",
                black_engine=scenario.engine_config.name
            )

            # Initialize engines
            self.model.eval()  # Matrix0 doesn't need new_game
            engine.new_game()

            # Set up initial position
            current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            moves = []

            game_metrics.move_times = []

            # Play game
            for move_num in range(scenario.max_moves):
                start_time = time.time()

                if move_num % 2 == 0:  # White to move (Matrix0)
                    move, move_time = self._get_matrix0_move(current_fen, moves)
                else:  # Black to move (UCI engine)
                    engine.set_position(current_fen, moves)
                    move, move_time = engine.go(scenario.time_control)

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

                # Update timing totals
                if move_num % 2 == 0:
                    game_metrics.white_total_time += actual_time
                else:
                    game_metrics.black_total_time += actual_time

                # Check for game end (simplified - just check if we got a valid move)
                if not move or move == "0000":
                    break

            # Calculate final statistics
            game_metrics.end_time = time.time()
            game_metrics.total_moves = len(moves)

            # Determine result (simplified - assume draw if max moves reached)
            if len(moves) >= scenario.max_moves:
                game_metrics.result = "1/2-1/2"
                game_metrics.winner = "draw"
            else:
                # This would need proper game result detection
                game_metrics.result = "1/2-1/2"  # Default to draw
                game_metrics.winner = "draw"

            # Calculate averages
            if game_metrics.total_moves > 0:
                game_metrics.white_avg_time_per_move = game_metrics.white_total_time / ((game_metrics.total_moves + 1) // 2)
                game_metrics.black_avg_time_per_move = game_metrics.black_total_time / (game_metrics.total_moves // 2)

            logger.info(f"Game completed: {game_metrics.total_moves} moves, result: {game_metrics.result}")
            return game_metrics

        except Exception as e:
            logger.error(f"Error playing game {game_id}: {e}")
            return None

    def _get_matrix0_move(self, fen: str, moves: List[str]) -> Tuple[str, float]:
        """Get move from Matrix0 model."""
        try:
            start_time = time.time()

            # This is a simplified implementation
            # In a real implementation, you'd need to:
            # 1. Convert FEN to model input format
            # 2. Run model inference
            # 3. Convert model output to UCI move format
            # 4. Apply time controls

            # For now, return a dummy move
            move = "e2e4"  # Example move
            move_time = time.time() - start_time

            return move, move_time

        except Exception as e:
            logger.error(f"Error getting Matrix0 move: {e}")
            return "0000", 0.0

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
        print(".3f")
        print(".2f")
        print(".3f")
        print(".1f")

        # Clean up
        runner.engine_manager.stop_all()

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
