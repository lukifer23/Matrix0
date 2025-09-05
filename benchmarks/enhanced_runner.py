#!/usr/bin/env python3
# Enhanced Matrix0 Benchmark Runner
"""
Advanced benchmark runner with Apple Silicon optimizations, SSL tracking,
tournament support, and comprehensive performance monitoring.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from benchmarks.config import BenchmarkConfig, ConfigManager
from benchmarks.benchmark import BenchmarkRunner
from benchmarks.engine_manager import EnhancedEngineManager
from benchmarks.ssl_tracker import SSLTracker
from benchmarks.tournament import Tournament, TournamentConfig, TournamentFormat, run_tournament

if TORCH_AVAILABLE:
    from azchess.config import Config as Matrix0Config, select_device as _select_device


class EnhancedBenchmarkRunner:
    """Enhanced benchmark runner with all advanced features."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.engine_manager = EnhancedEngineManager()
        self.ssl_tracker = SSLTracker()
        self.matrix0_model = None
        self.mcts_config = None

        # Load configuration
        self._load_config()

        logger.info("Enhanced Benchmark Runner initialized")

    def _load_config(self):
        """Load and validate configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            self.config = ConfigManager.load_config(self.config_path)
            logger.info(f"Loaded configuration: {self.config.name}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    async def discover_and_validate_engines(self) -> Dict[str, Any]:
        """Discover and validate available engines."""
        logger.info("Discovering and validating engines...")

        # Discover engines
        discovered = await self.engine_manager.discover_engines()

        # Validate engines
        validation_results = await self.engine_manager.validate_all_engines()

        # Generate configurations
        engine_configs = await self.engine_manager.create_engine_configs()

        logger.info(f"Engine discovery complete: {len(discovered)} engines found")

        return {
            "discovered_engines": discovered,
            "validation_results": validation_results,
            "engine_configs": engine_configs
        }

    def load_matrix0_model(self, checkpoint_path: str) -> bool:
        """Load Matrix0 model with SSL support."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available - cannot load Matrix0 model")
            return False

        try:
            logger.info(f"Loading Matrix0 model from {checkpoint_path}")

            # Load model config
            model_config_obj = Matrix0Config.load('config.yaml').model()

            # Create model
            from azchess.model.resnet import PolicyValueNet
            self.matrix0_model = PolicyValueNet.from_config(model_config_obj)

            # Determine device via unified selector
            try:
                dev_str = _select_device('auto')
            except Exception:
                dev_str = 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
            device = torch.device(dev_str)
            logger.info(f"Using device for Matrix0: {device}")

            self.matrix0_model.to(device)
            self.matrix0_model.eval()

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = checkpoint.get('model_ema') or checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint
            self.matrix0_model.load_state_dict(state, strict=False)

            logger.info(f"Successfully loaded Matrix0 model with SSL support")
            return True

        except Exception as e:
            logger.error(f"Failed to load Matrix0 model: {e}")
            return False

    async def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Run a specific benchmark scenario."""
        logger.info(f"Running scenario: {scenario_name}")

        # Find the scenario
        scenario = None
        for s in self.config.scenarios:
            if s.name == scenario_name:
                scenario = s
                break

        if not scenario:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        # Check if this is a tournament scenario
        if hasattr(scenario, 'tournament_format'):
            return await self._run_tournament_scenario(scenario)
        else:
            return await self._run_standard_scenario(scenario)

    async def _run_standard_scenario(self, scenario) -> Dict[str, Any]:
        """Run a standard benchmark scenario."""
        logger.info(f"Running standard scenario: {scenario.name}")

        # Create benchmark runner
        runner = BenchmarkRunner(
            self.config,
            live_enabled=False,  # Disable live display for enhanced runner
            pgn_dir=f"{self.config.output_dir}/pgns_{scenario.name.lower()}"
        )

        # Load model if needed
        if self.matrix0_model is None:
            if not self.load_matrix0_model(scenario.model_checkpoint):
                raise RuntimeError("Failed to load Matrix0 model")

        # Run the benchmark
        results = runner.run_benchmark()

        # Enhanced analysis
        analysis = self._analyze_results(results, scenario)

        return {
            "scenario": scenario.name,
            "results": results,
            "analysis": analysis,
            "ssl_performance": self.ssl_tracker.get_ssl_performance_summary() if hasattr(scenario, 'ssl_tracking') and scenario.ssl_tracking else None
        }

    async def _run_tournament_scenario(self, scenario) -> Dict[str, Any]:
        """Run a tournament-style benchmark scenario."""
        logger.info(f"Running tournament scenario: {scenario.name}")

        # Create tournament configuration
        tournament_config = TournamentConfig(
            name=f"{self.config.name} - {scenario.name}",
            format=getattr(TournamentFormat, scenario.tournament_format.upper()),
            engines=scenario.engines,
            num_games_per_pairing=getattr(scenario, 'num_games_per_pairing', 1),
            time_control=getattr(scenario, 'time_control', '30+0.3'),
            max_moves=getattr(scenario, 'max_moves', 200),
            concurrency=getattr(scenario, 'concurrency', 2),
            output_dir=self.config.output_dir
        )

        # Run tournament
        tournament_results = await run_tournament(tournament_config)

        # Enhanced tournament analysis
        analysis = self._analyze_tournament_results(tournament_results)

        return {
            "scenario": scenario.name,
            "tournament_results": tournament_results,
            "analysis": analysis
        }

    def _analyze_results(self, results, scenario) -> Dict[str, Any]:
        """Perform enhanced analysis of benchmark results."""
        analysis = {
            "scenario_name": scenario.name,
            "total_games": results.total_games,
            "win_rate": results.white_wins / results.total_games if results.total_games > 0 else 0,
            "avg_game_time": results.avg_game_duration,
            "avg_moves": results.avg_moves_per_game,
            "resource_usage": {
                "avg_cpu": results.avg_cpu_usage,
                "avg_memory": results.avg_memory_usage,
                "peak_memory": results.peak_memory_usage
            }
        }

        # Engine performance analysis
        if hasattr(scenario, 'engines') and scenario.engines:
            analysis["engine_performance"] = {}
            for engine in scenario.engines:
                if engine in results.engine_stats:
                    stats = results.engine_stats[engine]
                    analysis["engine_performance"][engine] = {
                        "win_rate": stats.get("win_rate", 0),
                        "games_played": stats.get("wins", 0) + stats.get("losses", 0) + stats.get("draws", 0)
                    }

        # SSL analysis if enabled
        if hasattr(scenario, 'ssl_tracking') and scenario.ssl_tracking:
            analysis["ssl_analysis"] = self.ssl_tracker.get_ssl_performance_summary()

        # Apple Silicon performance if enabled
        if hasattr(scenario, 'apple_silicon_optimized') and scenario.apple_silicon_optimized:
            analysis["apple_silicon_metrics"] = self._analyze_apple_silicon_performance(results)

        return analysis

    def _analyze_tournament_results(self, tournament_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tournament results."""
        analysis = {
            "tournament_name": tournament_results.get("tournament_name", ""),
            "format": tournament_results.get("format", ""),
            "total_games": tournament_results.get("total_games", 0),
            "duration": tournament_results.get("duration", 0),
            "rankings": tournament_results.get("final_rankings", []),
            "standings": tournament_results.get("standings", [])
        }

        # Calculate tournament statistics
        if "statistics" in tournament_results:
            stats = tournament_results["statistics"]
            analysis["tournament_stats"] = {
                "avg_game_time": stats.get("avg_game_time", 0),
                "avg_moves_per_game": stats.get("avg_moves_per_game", 0),
                "first_move_advantage": stats.get("first_move_advantage", 0)
            }

        return analysis

    def _analyze_apple_silicon_performance(self, results) -> Dict[str, Any]:
        """Analyze Apple Silicon performance metrics."""
        apple_metrics = {
            "mps_available": torch.backends.mps.is_available() if TORCH_AVAILABLE else False,
            "mps_memory_usage": None,
            "mps_utilization": None
        }

        if TORCH_AVAILABLE and torch.backends.mps.is_available():
            try:
                # Analyze MPS metrics from system metrics
                mps_memory_usage = []
                mps_utilization = []

                for game in results.games:
                    for sys_metric in game.system_metrics:
                        if hasattr(sys_metric, 'mps_memory_allocated_gb') and sys_metric.mps_memory_allocated_gb is not None:
                            mps_memory_usage.append(sys_metric.mps_memory_allocated_gb)
                        if hasattr(sys_metric, 'mps_utilization') and sys_metric.mps_utilization is not None:
                            mps_utilization.append(sys_metric.mps_utilization)

                if mps_memory_usage:
                    apple_metrics["mps_memory_usage"] = {
                        "avg": sum(mps_memory_usage) / len(mps_memory_usage),
                        "max": max(mps_memory_usage),
                        "min": min(mps_memory_usage)
                    }

                if mps_utilization:
                    apple_metrics["mps_utilization"] = {
                        "avg": sum(mps_utilization) / len(mps_utilization),
                        "max": max(mps_utilization),
                        "min": min(mps_utilization)
                    }

            except Exception as e:
                logger.warning(f"Failed to analyze Apple Silicon metrics: {e}")

        return apple_metrics

    async def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all scenarios in the configuration."""
        logger.info("Running all scenarios...")

        all_results = {}

        for scenario in self.config.scenarios:
            try:
                logger.info(f"Starting scenario: {scenario.name}")
                result = await self.run_scenario(scenario.name)
                all_results[scenario.name] = result

            except Exception as e:
                logger.error(f"Failed to run scenario {scenario.name}: {e}")
                all_results[scenario.name] = {"error": str(e)}

        # Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report(all_results)

        return {
            "individual_results": all_results,
            "comprehensive_report": comprehensive_report
        }

    def _generate_comprehensive_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive report across all scenarios."""
        report = {
            "benchmark_suite": self.config.name,
            "total_scenarios": len(all_results),
            "completed_scenarios": sum(1 for r in all_results.values() if "error" not in r),
            "failed_scenarios": sum(1 for r in all_results.values() if "error" in r),
            "scenarios": {}
        }

        # Aggregate metrics across scenarios
        total_games = 0
        total_time = 0
        engine_performance = {}

        for scenario_name, result in all_results.items():
            if "error" in result:
                report["scenarios"][scenario_name] = {"status": "failed", "error": result["error"]}
                continue

            scenario_data = result.get("analysis", {})

            # Aggregate totals
            if "total_games" in scenario_data:
                total_games += scenario_data["total_games"]
            if "avg_game_time" in scenario_data:
                # Estimate total time (approximate)
                games = scenario_data.get("total_games", 0)
                avg_time = scenario_data.get("avg_game_time", 0)
                total_time += games * avg_time

            # Collect engine performance
            if "engine_performance" in scenario_data:
                for engine, perf in scenario_data["engine_performance"].items():
                    if engine not in engine_performance:
                        engine_performance[engine] = []
                    engine_performance[engine].append(perf)

            report["scenarios"][scenario_name] = {
                "status": "completed",
                "games": scenario_data.get("total_games", 0),
                "win_rate": scenario_data.get("win_rate", 0),
                "avg_game_time": scenario_data.get("avg_game_time", 0)
            }

        # Calculate aggregate engine performance
        report["aggregate_performance"] = {}
        for engine, performances in engine_performance.items():
            if performances:
                avg_win_rate = sum(p.get("win_rate", 0) for p in performances) / len(performances)
                total_games = sum(p.get("games_played", 0) for p in performances)
                report["aggregate_performance"][engine] = {
                    "avg_win_rate": avg_win_rate,
                    "total_games": total_games
                }

        report["totals"] = {
            "games": total_games,
            "estimated_time": total_time
        }

        return report


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Matrix0 Benchmark Runner")
    parser.add_argument("--config", required=True, help="Benchmark configuration file")
    parser.add_argument("--scenario", help="Run specific scenario")
    parser.add_argument("--discover-engines", action="store_true", help="Discover and validate engines")
    parser.add_argument("--output-dir", help="Output directory for results")

    args = parser.parse_args()

    try:
        runner = EnhancedBenchmarkRunner(args.config)

        if args.discover_engines:
            print("🔍 Discovering engines...")
            engine_info = await runner.discover_and_validate_engines()

            print(f"📊 Discovered {len(engine_info['discovered_engines'])} engines:")
            for name, info in engine_info['discovered_engines'].items():
                status = "✅" if engine_info['validation_results'].get(name, False) else "❌"
                print(f"  {status} {name}: {info.version} ({info.estimated_elo} ELO)")

        elif args.scenario:
            print(f"🎯 Running scenario: {args.scenario}")
            result = await runner.run_scenario(args.scenario)
            print(f"✅ Scenario completed: {len(result.get('results', {}).get('games', []))} games")

        else:
            print("🚀 Running complete benchmark suite...")
            results = await runner.run_all_scenarios()
            print(f"✅ Benchmark suite completed: {results['comprehensive_report']['completed_scenarios']} scenarios")

        print("🎉 Enhanced benchmark runner completed successfully!")

    except Exception as e:
        logger.error(f"Benchmark runner failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
