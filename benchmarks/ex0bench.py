#!/usr/bin/env python3
# EX0Bench - External Engine Benchmarking for Matrix0
"""
EX0Bench: Streamlined external engine benchmarking for Matrix0.

Focuses on Stockfish vs LC0 head-to-head comparisons with easy UCI engine plugin support.
Provides quick, clean interface while leveraging the full Matrix0 benchmarking infrastructure.

Usage:
    python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 50
    python benchmarks/ex0bench.py --engine1 stockfish --engine2 matrix0 --games 25
    python benchmarks/ex0bench.py --uci-engine /path/to/engine --games 20
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Import torch for device handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from azchess.logging_utils import setup_logging
from benchmarks.config import BenchmarkConfig, UCIEngineConfig, TestScenario
from benchmarks.benchmark import BenchmarkRunner
from benchmarks.results import ResultsAnalyzer

logger = setup_logging(level=logging.INFO)


class EX0Bench:
    """EX0Bench - External Engine Benchmarking System"""

    def __init__(self):
        self.engines = self._get_default_engines()
        self.results_dir = Path("benchmarks/results")

    def _get_default_engines(self) -> Dict[str, Dict[str, Any]]:
        """Get default engine configurations for common engines."""
        return {
            "stockfish": {
                "command": "stockfish",
                "options": {
                    "Threads": "4",
                    "Hash": "512",
                    "Skill Level": "20"  # Full strength
                },
                "time_control": "30+0.3"
            },
            "stockfish_club": {
                "command": "stockfish",
                "options": {
                    "Threads": "4",
                    "Hash": "512",
                    "Skill Level": "4"  # Club level
                },
                "time_control": "30+0.3"
            },
            "stockfish_weak": {
                "command": "stockfish",
                "options": {
                    "Threads": "2",
                    "Hash": "256",
                    "Skill Level": "0"  # Weak level
                },
                "time_control": "10+0.1"
            },
            "lc0": {
                "command": "/opt/homebrew/bin/lc0",  # macOS default
                "options": {
                    "Threads": "4",
                    "NNCacheSize": "2000000",
                    "Backend": "metal",  # Apple Silicon
                    "CPuct": "1.745000"
                },
                "time_control": "30+0.3"
            },
            "matrix0": {
                "type": "internal",
                "checkpoint": "checkpoints/best.pt"
            }
        }

    def create_external_engine_config(self,
                                      engine1: str,
                                      engine2: str,
                                      games: int = 50,
                                      time_control: str = "30+0.3",
                                      output_name: Optional[str] = None) -> Dict[str, Any]:
        """Create configuration for pure external engine battles (no Matrix0)."""

        # Generate output name if not provided
        if not output_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_name = f"ex0bench_external_{engine1}_vs_{engine2}_{timestamp}"

        config = {
            "name": f"EX0Bench External: {engine1} vs {engine2}",
            "description": f"Pure external engine battle between {engine1} and {engine2}",
            "output_dir": str(self.results_dir),

            "engines": {},
            "scenarios": []
        }

        # Add engine configurations
        for engine_name in [engine1, engine2]:
            if engine_name in self.engines:
                engine_config = self.engines[engine_name].copy()
                config["engines"][engine_name] = engine_config
            else:
                logger.warning(f"Engine '{engine_name}' not found in defaults. "
                             f"Available: {list(self.engines.keys())}")
                return None

        # Create test scenarios (both colors for each engine)
        for i, (eng1, eng2) in enumerate([(engine1, engine2), (engine2, engine1)]):
            scenario = {
                "name": f"{eng1}_vs_{eng2}_external_game_{i+1}",
                "engine": eng1,
                "opponent": eng2,
                "num_games": games // 2,  # Split games between colors
                "time_control": time_control,
                "concurrency": 2,
                "max_moves": 200,
                "random_openings": True,
                "opening_plies": 8,
                "external_only": True  # Flag for external engine battle
            }
            config["scenarios"].append(scenario)

        # Add performance monitoring
        config["performance"] = {
            "track_cpu": True,
            "track_memory": True,
            "track_gpu": False,  # No GPU needed for external engines
            "sample_interval": 0.1
        }

        return config

    def create_head_to_head_config(self,
                                   engine1: str,
                                   engine2: str,
                                   games: int = 50,
                                   time_control: str = "30+0.3",
                                   output_name: Optional[str] = None) -> Dict[str, Any]:
        """Create configuration for head-to-head engine comparison."""

        # Generate output name if not provided
        if not output_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_name = f"ex0bench_{engine1}_vs_{engine2}_{timestamp}"

        config = {
            "name": f"EX0Bench: {engine1} vs {engine2}",
            "description": f"Head-to-head comparison between {engine1} and {engine2}",
            "output_dir": str(self.results_dir),

            "engines": {},
            "scenarios": []
        }

        # Add engine configurations
        for engine_name in [engine1, engine2]:
            if engine_name in self.engines:
                engine_config = self.engines[engine_name].copy()
                config["engines"][engine_name] = engine_config
            else:
                logger.warning(f"Engine '{engine_name}' not found in defaults. "
                             f"Available: {list(self.engines.keys())}")
                return None

        # Create test scenarios (both colors for each engine)
        for i, (eng1, eng2) in enumerate([(engine1, engine2), (engine2, engine1)]):
            scenario = {
                "name": f"{eng1}_vs_{eng2}_game_{i+1}",
                "engine": eng1,
                "opponent": eng2,
                "num_games": games // 2,  # Split games between colors
                "time_control": time_control,
                "concurrency": 2,
                "max_moves": 200,
                "random_openings": True,
                "opening_plies": 8
            }
            config["scenarios"].append(scenario)

        # Add performance monitoring
        config["performance"] = {
            "track_cpu": True,
            "track_memory": True,
            "track_gpu": True,
            "sample_interval": 0.1
        }

        return config

    def add_custom_uci_engine(self, name: str, path: str, options: Optional[Dict[str, Any]] = None) -> None:
        """Add a custom UCI engine to the engine registry."""
        if options is None:
            options = {"Threads": "4", "Hash": "512"}

        self.engines[name] = {
            "command": path,
            "options": options,
            "time_control": "30+0.3"
        }
        logger.info(f"Added custom UCI engine '{name}' at path: {path}")

    def run_external_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run pure external engine benchmark without Matrix0."""
        logger.info("Starting pure external engine benchmark (no Matrix0 required)")

        # Simple external engine battle implementation
        from benchmarks.uci_bridge import UCIEngine
        import chess
        import chess.pgn
        import json
        from pathlib import Path
        from types import SimpleNamespace

        # Helper function to create config object from dict
        def create_engine_config(name: str, command: str, options: Dict[str, Any] = None):
            return SimpleNamespace(
                name=name,
                command=command,
                working_dir=None,
                options=options or {}
            )

        results = {
            "benchmark_name": config["name"],
            "description": config["description"],
            "total_games": 0,
            "games": [],
            "summary": {
                "white_wins": 0,
                "black_wins": 0,
                "draws": 0
            }
        }

        # Create output directory
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        pgn_dir = output_dir / "pgns_external"
        pgn_dir.mkdir(exist_ok=True)

        # Run scenarios
        for scenario in config.get("scenarios", []):
            engine1_name = scenario["engine"]
            engine2_name = scenario["opponent"]
            num_games = scenario["num_games"]

            logger.info(f"Running scenario: {scenario['name']}")
            logger.info(f"{engine1_name} vs {engine2_name} - {num_games} games")

            # Get engine configs and create proper config objects
            engine1_dict = config["engines"][engine1_name]
            engine2_dict = config["engines"][engine2_name]

            engine1_config = create_engine_config(
                engine1_name,
                engine1_dict["command"],
                engine1_dict.get("options", {})
            )
            engine2_config = create_engine_config(
                engine2_name,
                engine2_dict["command"],
                engine2_dict.get("options", {})
            )

            # Run games
            for game_idx in range(num_games):
                logger.info(f"Game {game_idx + 1}/{num_games}: {engine1_name} (White) vs {engine2_name} (Black)")

                # Initialize engines
                engine1 = UCIEngine(engine1_config)
                engine2 = UCIEngine(engine2_config)

                try:
                    engine1.start()
                    engine2.start()

                    # Start new game
                    engine1.new_game()
                    engine2.new_game()

                    # Play game
                    board = chess.Board()
                    moves = []
                    game_result = None

                    while not board.is_game_over() and len(moves) < scenario["max_moves"]:
                        # Determine whose turn it is
                        if board.turn:  # White to move
                            current_engine = engine1
                            current_name = engine1_name
                        else:  # Black to move
                            current_engine = engine2
                            current_name = engine2_name

                        # Set position and get best move
                        try:
                            # Set current position
                            current_engine.set_position(board.fen(), [])

                            # Get best move
                            best_move_str, _ = current_engine.go(scenario["time_control"])

                            if best_move_str and best_move_str != "(none)":
                                move = chess.Move.from_uci(best_move_str)
                                if move in board.legal_moves:
                                    board.push(move)
                                    moves.append(move)
                                    logger.debug(f"{current_name}: {move}")
                                else:
                                    logger.warning(f"{current_name} returned illegal move: {best_move_str}")
                                    break
                            else:
                                logger.warning(f"{current_name} returned no move")
                                break
                        except Exception as e:
                            logger.error(f"Error getting move from {current_name}: {e}")
                            break

                    # Determine result
                    if board.is_checkmate():
                        if board.turn:  # Black just moved and checkmated white
                            game_result = "0-1"  # Black wins
                            results["summary"]["black_wins"] += 1
                        else:  # White just moved and checkmated black
                            game_result = "1-0"  # White wins
                            results["summary"]["white_wins"] += 1
                    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
                        game_result = "1/2-1/2"  # Draw
                        results["summary"]["draws"] += 1
                    else:
                        game_result = "*"  # Game interrupted

                    # Record game
                    game_data = {
                        "game_id": len(results["games"]) + 1,
                        "white_engine": engine1_name,
                        "black_engine": engine2_name,
                        "result": game_result,
                        "moves": len(moves),
                        "pgn": str(chess.pgn.Game.from_board(board))
                    }
                    results["games"].append(game_data)
                    results["total_games"] += 1

                    # Save PGN
                    pgn_game = chess.pgn.Game()
                    pgn_game.headers["White"] = engine1_name
                    pgn_game.headers["Black"] = engine2_name
                    pgn_game.headers["Result"] = game_result
                    pgn_game.headers["Date"] = time.strftime("%Y.%m.%d")
                    pgn_game.headers["Time"] = time.strftime("%H:%M:%S")

                    # Add moves to PGN
                    node = pgn_game
                    for move in moves:
                        node = node.add_variation(move)

                    pgn_file = pgn_dir / f"{engine1_name}_vs_{engine2_name}_game_{game_idx + 1}.pgn"
                    with open(pgn_file, 'w') as f:
                        f.write(str(pgn_game))

                    logger.info(f"Game {game_idx + 1} completed: {game_result} ({len(moves)} moves)")

                finally:
                    # Clean up engines
                    try:
                        engine1.stop()
                        engine2.stop()
                    except:
                        pass

        # Save results
        results_file = output_dir / f"ex0bench_external_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"External benchmark completed! Results saved to {results_file}")
        return results

    def run_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the benchmark using the existing Matrix0 benchmarking infrastructure."""

        # Check if this is a pure external engine battle
        if config.get("scenarios", [{}])[0].get("external_only", False):
            return self.run_external_benchmark(config)

        # Convert dictionary config to BenchmarkConfig
        from benchmarks.config import UCIEngineConfig, TestScenario, PerformanceConfig

        # Create engine configurations
        engines = {}
        for engine_name, engine_data in config.get('engines', {}).items():
            engines[engine_name] = UCIEngineConfig(
                name=engine_name,
                command=engine_data['command'],
                options=engine_data.get('options', {})
            )

        # Create test scenarios
        scenarios = []
        for scenario_data in config.get('scenarios', []):
            # Find the engine config for this scenario
            engine_name = scenario_data['engine']
            if engine_name in engines:
                scenario = TestScenario(
                    name=scenario_data['name'],
                    engine_config=engines[engine_name],
                    model_checkpoint=scenario_data.get('model_checkpoint', 'checkpoints/best.pt'),
                    num_games=scenario_data['num_games'],
                    time_control=scenario_data['time_control'],
                    max_moves=scenario_data.get('max_moves', 200),
                    concurrency=scenario_data.get('concurrency', 2),
                    random_openings=scenario_data.get('random_openings', True),
                    opening_plies=scenario_data.get('opening_plies', 8)
                )
                scenarios.append(scenario)

        # Create performance config
        perf_config = PerformanceConfig(
            track_cpu=config.get('performance', {}).get('track_cpu', True),
            track_memory=config.get('performance', {}).get('track_memory', True),
            track_gpu=config.get('performance', {}).get('track_gpu', True),
            sample_interval=config.get('performance', {}).get('sample_interval', 0.1)
        )

        # Create BenchmarkConfig
        benchmark_config = BenchmarkConfig(
            name=config['name'],
            description=config['description'],
            output_dir=config['output_dir'],
            scenarios=scenarios,
            performance_config=perf_config
        )

        # Create and run benchmark
        runner = BenchmarkRunner(benchmark_config)
        results = runner.run_benchmark()

        return results

    def generate_quick_report(self, results) -> str:
        """Generate a quick summary report of the benchmark results."""

        # Handle BenchmarkMetrics object
        if hasattr(results, 'games') and hasattr(results, 'total_games'):
            # This is a BenchmarkMetrics object
            games = results.games or []
            total_games = results.total_games
            white_wins = getattr(results, 'white_wins', 0)
            black_wins = getattr(results, 'black_wins', 0)
            draws = getattr(results, 'draws', 0)
        elif isinstance(results, dict) and "scenarios" in results:
            # This is a legacy dictionary format
            games = []
            total_games = 0
            white_wins = 0
            black_wins = 0
            draws = 0
            # Fallback to old format handling
        else:
            return "No results available or unsupported format"

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("EX0BENCH QUICK REPORT")
        report_lines.append("=" * 60)

        # Report individual games if available
        if games:
            report_lines.append(f"\n📊 Individual Games ({len(games)} total):")
            for i, game in enumerate(games[:10]):  # Show first 10 games
                if hasattr(game, 'result'):
                    result = game.result
                    moves = getattr(game, 'moves', 0)
                    duration = getattr(game, 'duration', 0)
                    report_lines.append(f"  Game {i+1}: {result} in {moves} moves ({duration:.1f}s)")

            if len(games) > 10:
                report_lines.append(f"  ... and {len(games) - 10} more games")

        # Overall statistics
        report_lines.append(f"\n{'='*30} OVERALL STATISTICS {'='*30}")
        report_lines.append(f"Total Games: {total_games}")
        report_lines.append(f"White Wins: {white_wins}")
        report_lines.append(f"Black Wins: {black_wins}")
        report_lines.append(f"Draws: {draws}")

        if total_games > 0:
            white_win_rate = white_wins / total_games * 100
            black_win_rate = black_wins / total_games * 100
            draw_rate = draws / total_games * 100

            report_lines.append(f"White Win Rate: {white_win_rate:.1f}%")
            report_lines.append(f"Black Win Rate: {black_win_rate:.1f}%")
            report_lines.append(f"Draw Rate: {draw_rate:.1f}%")

        # Performance stats if available
        if hasattr(results, 'avg_game_duration') and results.avg_game_duration > 0:
            report_lines.append(f"Average Game Duration: {results.avg_game_duration:.1f}s")
        if hasattr(results, 'avg_moves_per_game') and results.avg_moves_per_game > 0:
            report_lines.append(f"Average Moves per Game: {results.avg_moves_per_game:.1f}")
        if hasattr(results, 'avg_time_per_move') and results.avg_time_per_move > 0:
            report_lines.append(f"Average Time per Move: {results.avg_time_per_move:.3f}s")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="EX0Bench - External Engine Benchmarking for Matrix0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stockfish vs LC0 (default engines)
  python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 50

  # Matrix0 vs Stockfish Club level
  python benchmarks/ex0bench.py --engine1 matrix0 --engine2 stockfish_club --games 25

  # Custom UCI engine vs Stockfish
  python benchmarks/ex0bench.py --uci-engine /path/to/myengine --engine2 stockfish --games 20

  # Quick test with weak engines
  python benchmarks/ex0bench.py --engine1 stockfish_weak --engine2 matrix0 --games 10 --time 10+0.1
        """
    )

    # Engine selection
    parser.add_argument("--engine1", type=str,
                       choices=["stockfish", "stockfish_club", "stockfish_weak", "lc0", "matrix0"],
                       help="First engine (from defaults)")
    parser.add_argument("--engine2", type=str,
                       choices=["stockfish", "stockfish_club", "stockfish_weak", "lc0", "matrix0"],
                       help="Second engine (from defaults)")

    # Custom UCI engine
    parser.add_argument("--uci-engine", type=str,
                       help="Path to custom UCI engine (alternative to --engine1)")
    parser.add_argument("--uci-name", type=str, default="custom_engine",
                       help="Name for custom UCI engine")

    # Benchmark settings
    parser.add_argument("--games", type=int, default=50,
                       help="Total number of games (default: 50)")
    parser.add_argument("--time", type=str, default="30+0.3",
                       help="Time control (default: 30+0.3)")
    parser.add_argument("--concurrency", type=int, default=2,
                       help="Games to run in parallel (default: 2)")

    # Output options
    parser.add_argument("--output", type=str,
                       help="Output directory for results (default: benchmarks/results)")
    parser.add_argument("--name", type=str,
                       help="Custom name for this benchmark run")
    parser.add_argument("--external-only", action="store_true",
                       help="Run pure external engine battles (no Matrix0 model required)")

    args = parser.parse_args()

    # Initialize EX0Bench
    ex0bench = EX0Bench()

    # Handle custom UCI engine
    if args.uci_engine:
        ex0bench.add_custom_uci_engine(args.uci_name, args.uci_engine)
        engine1 = args.uci_name
        if not args.engine2:
            logger.error("Must specify --engine2 when using --uci-engine")
            sys.exit(1)
        engine2 = args.engine2
    else:
        if not args.engine1 or not args.engine2:
            logger.error("Must specify both --engine1 and --engine2, or use --uci-engine")
            sys.exit(1)
        engine1 = args.engine1
        engine2 = args.engine2

    # Create benchmark configuration
    logger.info(f"Creating EX0Bench configuration: {engine1} vs {engine2}")

    # Check if this is a pure external engine battle (no Matrix0 involved)
    is_external_only = args.external_only or \
                      (engine1 in ['stockfish', 'stockfish_club', 'stockfish_weak', 'lc0'] and \
                       engine2 in ['stockfish', 'stockfish_club', 'stockfish_weak', 'lc0'] and \
                       'matrix0' not in [engine1, engine2])

    if is_external_only:
        logger.info("Pure external engine battle detected - no Matrix0 model required!")
        config = ex0bench.create_external_engine_config(
            engine1=engine1,
            engine2=engine2,
            games=args.games,
            time_control=args.time,
            output_name=args.name
        )
    else:
        # Use traditional head-to-head with Matrix0
        config = ex0bench.create_head_to_head_config(
            engine1=engine1,
            engine2=engine2,
            games=args.games,
            time_control=args.time,
            output_name=args.name
        )

    if not config:
        logger.error("Failed to create benchmark configuration")
        sys.exit(1)

    # Update concurrency if specified
    for scenario in config["scenarios"]:
        scenario["concurrency"] = args.concurrency

    # Update output directory if specified
    if args.output:
        config["output_dir"] = args.output
        ex0bench.results_dir = Path(args.output)

    # Display configuration
    print("\n" + "="*60)
    print("EX0BENCH CONFIGURATION")
    print("="*60)
    print(f"Engine 1: {engine1}")
    print(f"Engine 2: {engine2}")
    print(f"Total Games: {args.games}")
    print(f"Time Control: {args.time}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {config['output_dir']}")
    print("="*60)

    # Run benchmark
    logger.info("Starting EX0Bench benchmark...")
    try:
        results = ex0bench.run_benchmark(config)

        # Generate and display quick report
        report = ex0bench.generate_quick_report(results)
        print("\n" + report)

        logger.info("EX0Bench completed successfully!")

        # Print summary for quick reference
        print(f"\n🎯 Benchmark Summary:")
        print(f"   📊 Results saved to: {config['output_dir']}")
        print(f"   📈 Full results: {config['output_dir']}/ex0bench_*_results.json")
        print(f"   🎮 PGN games: {config['output_dir']}/pgns_*/*.pgn")

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
