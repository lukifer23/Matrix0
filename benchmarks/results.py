# Benchmark Results Analysis and Reporting
"""
Analyze benchmark results and generate comprehensive reports.
Create visualizations and statistical analysis of performance data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from benchmarks.metrics import BenchmarkMetrics, GameMetrics

logger = logging.getLogger(__name__)

# Optional imports for visualizations
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, interactive plots disabled")


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""
    benchmark_name: str
    summary_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]
    engine_comparison: Dict[str, Any]
    game_analysis: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "summary_stats": self.summary_stats,
            "performance_stats": self.performance_stats,
            "engine_comparison": self.engine_comparison,
            "game_analysis": self.game_analysis,
            "recommendations": self.recommendations
        }


class ResultsAnalyzer:
    """Analyze benchmark results and generate reports."""

    @staticmethod
    def analyze_results(results_file: str) -> BenchmarkReport:
        """Analyze benchmark results from JSON file."""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Results file not found: {results_file}")
            return BenchmarkReport(
                benchmark_name="Error",
                summary_stats={"error": "Results file not found"},
                performance_stats={},
                engine_comparison={},
                game_analysis={},
                recommendations=["Check results file path"]
            )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in results file: {e}")
            return BenchmarkReport(
                benchmark_name="Error",
                summary_stats={"error": "Invalid JSON format"},
                performance_stats={},
                engine_comparison={},
                game_analysis={},
                recommendations=["Check results file format"]
            )

        # Create BenchmarkMetrics from data
        try:
            metrics = BenchmarkMetrics(**data)
        except Exception as e:
            logger.error(f"Failed to parse benchmark data: {e}")
            return BenchmarkReport(
                benchmark_name="Error",
                summary_stats={"error": "Failed to parse benchmark data"},
                performance_stats={},
                engine_comparison={},
                game_analysis={},
                recommendations=["Check benchmark data format"]
            )

        # Generate analysis
        analyzer = ResultsAnalyzer()

        summary_stats = analyzer._calculate_summary_stats(metrics)
        performance_stats = analyzer._calculate_performance_stats(metrics)
        engine_comparison = analyzer._compare_engines(metrics)
        game_analysis = analyzer._analyze_games(metrics)
        recommendations = analyzer._generate_recommendations(metrics)

        return BenchmarkReport(
            benchmark_name=metrics.benchmark_name,
            summary_stats=summary_stats,
            performance_stats=performance_stats,
            engine_comparison=engine_comparison,
            game_analysis=game_analysis,
            recommendations=recommendations
        )

    def _calculate_summary_stats(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Calculate summary statistics."""
        games = metrics.games

        if not games:
            return {"error": "No games found"}

        total_games = len(games)
        total_moves = sum(game.total_moves for game in games)
        total_duration = sum(game.duration for game in games)

        # Results breakdown
        white_wins = sum(1 for game in games if game.result == "1-0")
        black_wins = sum(1 for game in games if game.result == "0-1")
        draws = sum(1 for game in games if game.result == "1/2-1/2")

        return {
            "total_games": total_games,
            "total_moves": total_moves,
            "total_duration_hours": total_duration / 3600,
            "results": {
                "white_wins": white_wins,
                "black_wins": black_wins,
                "draws": draws,
                "win_rate": white_wins / total_games if total_games > 0 else 0,
                "draw_rate": draws / total_games if total_games > 0 else 0
            },
            "averages": {
                "game_duration": total_duration / total_games if total_games > 0 else 0,
                "moves_per_game": total_moves / total_games if total_games > 0 else 0
            }
        }

    def _calculate_performance_stats(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Calculate performance statistics."""
        games = metrics.games

        if not games:
            return {"error": "No games found"}

        # Move timing analysis
        all_move_times = []
        for game in games:
            all_move_times.extend([move.get('time', 0) for move in game.move_times])

        if not all_move_times:
            return {"error": "No move timing data"}

        # System resource analysis
        all_cpu = []
        all_memory = []
        peak_memory = 0

        for game in games:
            for sys_metrics in game.system_metrics:
                all_cpu.append(sys_metrics.get('cpu_percent', 0))
                all_memory.append(sys_metrics.get('memory_percent', 0))
                peak_memory = max(peak_memory, sys_metrics.get('memory_used_gb', 0))

        return {
            "timing": {
                "avg_time_per_move": np.mean(all_move_times),
                "min_time_per_move": np.min(all_move_times),
                "max_time_per_move": np.max(all_move_times),
                "std_time_per_move": np.std(all_move_times),
                "median_time_per_move": np.median(all_move_times)
            },
            "resources": {
                "avg_cpu_usage": np.mean(all_cpu) if all_cpu else 0,
                "avg_memory_usage": np.mean(all_memory) if all_memory else 0,
                "peak_memory_gb": peak_memory,
                "std_cpu_usage": np.std(all_cpu) if all_cpu else 0,
                "std_memory_usage": np.std(all_memory) if all_memory else 0
            }
        }

    def _compare_engines(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Compare performance across different engines."""
        games = metrics.games

        if not games:
            return {"error": "No games found"}

        # Group games by opponent engine
        engine_results = {}
        for game in games:
            opponent = game.black_engine
            if opponent not in engine_results:
                engine_results[opponent] = {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "total_moves": 0,
                    "avg_game_time": 0
                }

            engine_results[opponent]["games"] += 1
            engine_results[opponent]["total_moves"] += game.total_moves
            engine_results[opponent]["avg_game_time"] += game.duration

            if game.result == "1-0":
                engine_results[opponent]["wins"] += 1
            elif game.result == "0-1":
                engine_results[opponent]["losses"] += 1
            else:
                engine_results[opponent]["draws"] += 1

        # Calculate averages and rates
        for engine, stats in engine_results.items():
            games_count = stats["games"]
            if games_count > 0:
                stats["avg_moves"] = stats["total_moves"] / games_count
                stats["avg_game_time"] = stats["avg_game_time"] / games_count
                stats["win_rate"] = stats["wins"] / games_count
                stats["draw_rate"] = stats["draws"] / games_count
                stats["loss_rate"] = stats["losses"] / games_count

        return engine_results

    def _analyze_games(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Analyze individual game characteristics."""
        games = metrics.games

        if not games:
            return {"error": "No games found"}

        # Game length analysis
        game_lengths = [game.total_moves for game in games]
        game_durations = [game.duration for game in games]

        # Move time patterns
        move_time_patterns = {}
        for game in games:
            for move in game.move_times:
                move_num = move.get('move_num', 0)
                time_taken = move.get('time', 0)

                if move_num not in move_time_patterns:
                    move_time_patterns[move_num] = []
                move_time_patterns[move_num].append(time_taken)

        # Calculate move time statistics by move number
        move_time_stats = {}
        for move_num, times in move_time_patterns.items():
            if times:
                move_time_stats[move_num] = {
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times),
                    "count": len(times)
                }

        return {
            "game_length_stats": {
                "avg_length": np.mean(game_lengths),
                "min_length": np.min(game_lengths),
                "max_length": np.max(game_lengths),
                "std_length": np.std(game_lengths)
            },
            "game_duration_stats": {
                "avg_duration": np.mean(game_durations),
                "min_duration": np.min(game_durations),
                "max_duration": np.max(game_durations),
                "std_duration": np.std(game_durations)
            },
            "move_time_by_position": move_time_stats
        }

    def _generate_recommendations(self, metrics: BenchmarkMetrics) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []

        if not metrics.games:
            return ["No games completed - check benchmark configuration"]

        # Win rate analysis
        win_rate = metrics.win_rate
        if win_rate > 0.6:
            recommendations.append("Strong performance - consider increasing difficulty")
        elif win_rate < 0.3:
            recommendations.append("Consider model improvements or reduced difficulty")
        else:
            recommendations.append("Balanced performance - good baseline achieved")

        # Timing analysis
        avg_time_per_move = metrics.avg_time_per_move
        if avg_time_per_move > 2.0:
            recommendations.append(".2f")
        elif avg_time_per_move < 0.1:
            recommendations.append(".2f")

        # Resource analysis
        if metrics.avg_cpu_usage > 80:
            recommendations.append(".1f")
        if metrics.avg_memory_usage > 85:
            recommendations.append(".1f")

        # Game quality analysis
        avg_moves = metrics.avg_moves_per_game
        if avg_moves < 20:
            recommendations.append("Games ending too quickly - check engine configurations")
        elif avg_moves > 100:
            recommendations.append("Very long games - consider adjusting time controls")

        return recommendations if recommendations else ["Performance within normal ranges"]


class ReportGenerator:
    """Generate visual reports from benchmark data."""

    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, results_file: str) -> str:
        """Generate a comprehensive report from benchmark results."""
        report = ResultsAnalyzer.analyze_results(results_file)

        # Generate text report
        text_report = self._generate_text_report(report)
        report_path = self.output_dir / f"{report.benchmark_name.replace(' ', '_').lower()}_report.txt"

        with open(report_path, 'w') as f:
            f.write(text_report)

        # Generate JSON report
        json_path = self.output_dir / f"{report.benchmark_name.replace(' ', '_').lower()}_report.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        # Generate visualizations if available
        if PLOTLY_AVAILABLE:
            self._generate_visualizations(report)

        logger.info(f"Generated report: {report_path}")
        return str(report_path)

    def _generate_text_report(self, report: BenchmarkReport) -> str:
        """Generate a text-based report."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"BENCHMARK REPORT: {report.benchmark_name}")
        lines.append("=" * 80)
        lines.append("")

        # Summary statistics
        lines.append("📊 SUMMARY STATISTICS")
        lines.append("-" * 40)
        summary = report.summary_stats
        if "error" not in summary:
            lines.append(f"Total Games: {summary['total_games']}")
            lines.append(f"Total Duration: {summary['total_duration_hours']:.2f} hours")
            results = summary['results']
            lines.append(f"White Wins: {results['white_wins']} ({results['win_rate']:.1%})")
            lines.append(f"Black Wins: {results['black_wins']}")
            lines.append(f"Draws: {results['draws']} ({results['draw_rate']:.1%})")
            averages = summary['averages']
            lines.append(f"Average Game Duration: {averages['game_duration']:.1f} seconds")
            lines.append(f"Average Moves per Game: {averages['moves_per_game']:.1f}")
        lines.append("")

        # Performance statistics
        lines.append("⚡ PERFORMANCE STATISTICS")
        lines.append("-" * 40)
        perf = report.performance_stats
        if "error" not in perf:
            timing = perf['timing']
            lines.append(f"Average Time per Move: {timing['avg_time_per_move']:.3f} seconds")
            lines.append(f"Move Time Range: {timing['min_time_per_move']:.3f} - {timing['max_time_per_move']:.3f} seconds")
            lines.append(f"Move Time Std Dev: {timing['std_time_per_move']:.3f} seconds")

            resources = perf['resources']
            lines.append(f"Average CPU Usage: {resources['avg_cpu_usage']:.1f}%")
            lines.append(f"Average Memory Usage: {resources['avg_memory_usage']:.1f}%")
            lines.append(f"Peak Memory Usage: {resources['peak_memory_gb']:.2f} GB")
        lines.append("")

        # Engine comparison
        lines.append("🤖 ENGINE COMPARISON")
        lines.append("-" * 40)
        for engine, stats in report.engine_comparison.items():
            lines.append(f"Engine: {engine}")
            lines.append(f"  Games: {stats['games']}")
            lines.append(".1%")
            lines.append(".1%")
            lines.append(".1f")
            lines.append(".1f")
            lines.append("")
        lines.append("")

        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        return "\n".join(lines)

    def _generate_visualizations(self, report: BenchmarkReport):
        """Generate visual plots and charts."""
        if not PLOTLY_AVAILABLE:
            return

        # Create results distribution plot
        summary = report.summary_stats
        if "results" in summary:
            results = summary["results"]

            fig = go.Figure(data=[
                go.Bar(name='White Wins', x=['Results'], y=[results['white_wins']], marker_color='lightblue'),
                go.Bar(name='Black Wins', x=['Results'], y=[results['black_wins']], marker_color='lightcoral'),
                go.Bar(name='Draws', x=['Results'], y=[results['draws']], marker_color='lightgray')
            ])

            fig.update_layout(
                title="Game Results Distribution",
                barmode='stack',
                xaxis_title="Game Outcomes",
                yaxis_title="Number of Games"
            )

            plot_path = self.output_dir / f"{report.benchmark_name.replace(' ', '_').lower()}_results.html"
            fig.write_html(str(plot_path))

        # Create timing analysis plot
        perf = report.performance_stats
        if "timing" in perf:
            timing = perf["timing"]

            fig = go.Figure()
            fig.add_trace(go.Box(
                y=[timing['avg_time_per_move']],
                name="Average Move Time",
                marker_color='lightblue'
            ))

            fig.update_layout(
                title="Move Timing Analysis",
                yaxis_title="Time (seconds)",
                showlegend=False
            )

            timing_path = self.output_dir / f"{report.benchmark_name.replace(' ', '_').lower()}_timing.html"
            fig.write_html(str(timing_path))

        logger.info(f"Generated visualizations in {self.output_dir}")


def generate_comparison_report(results_files: List[str], output_dir: str = "benchmarks/results"):
    """Generate a comparison report from multiple benchmark results."""
    analyzer = ResultsAnalyzer()
    reports = []

    for results_file in results_files:
        try:
            report = analyzer.analyze_results(results_file)
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to analyze {results_file}: {e}")

    if not reports:
        logger.error("No valid reports found")
        return

    # Generate comparison summary
    comparison = {
        "benchmarks": [r.benchmark_name for r in reports],
        "summary": {}
    }

    # Compare key metrics
    for report in reports:
        summary = report.summary_stats
        if "results" in summary:
            results = summary["results"]
            comparison["summary"][report.benchmark_name] = {
                "games": summary["total_games"],
                "win_rate": results["win_rate"],
                "draw_rate": results["draw_rate"],
                "avg_game_time": summary["averages"]["game_duration"]
            }

    # Save comparison
    output_path = Path(output_dir) / "benchmark_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Generated comparison report: {output_path}")
    return str(output_path)
