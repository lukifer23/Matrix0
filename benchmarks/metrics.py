# Performance Metrics Collection
"""
Comprehensive performance monitoring for benchmark runs.
Collects CPU, memory, GPU usage, and timing metrics.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU metrics will be limited")


@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float

    # GPU metrics (if available)
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None

    # Process-specific metrics
    process_cpu_percent: Optional[float] = None
    process_memory_mb: Optional[float] = None
    process_threads: Optional[int] = None


@dataclass
class GameMetrics:
    """Metrics for a single game."""
    game_id: str
    start_time: float
    end_time: float
    total_moves: int
    result: str  # "1-0", "0-1", "1/2-1/2"
    winner: str  # "white", "black", "draw"

    # Engine-specific metrics
    white_engine: str
    black_engine: str
    white_total_time: float = 0.0
    black_total_time: float = 0.0
    white_avg_time_per_move: float = 0.0
    black_avg_time_per_move: float = 0.0

    # Move-by-move timing
    move_times: List[Dict[str, Any]] = None

    # System resource usage during game
    system_metrics: List[SystemMetrics] = None

    def __post_init__(self):
        if self.move_times is None:
            self.move_times = []
        if self.system_metrics is None:
            self.system_metrics = []

    @property
    def duration(self) -> float:
        """Total game duration in seconds."""
        return self.end_time - self.start_time

    @property
    def avg_time_per_move(self) -> float:
        """Average time per move for both players."""
        if self.total_moves == 0:
            return 0.0
        return (self.white_total_time + self.black_total_time) / self.total_moves


@dataclass
class BenchmarkMetrics:
    """Complete benchmark run metrics."""
    benchmark_name: str
    start_time: float
    end_time: float

    # Game results
    games: List[GameMetrics] = None

    # Aggregate statistics
    total_games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0

    # Performance statistics
    avg_game_duration: float = 0.0
    avg_moves_per_game: float = 0.0
    avg_time_per_move: float = 0.0

    # System resource statistics
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0

    # Engine-specific statistics
    engine_stats: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.games is None:
            self.games = []
        if self.engine_stats is None:
            self.engine_stats = {}

    @property
    def duration(self) -> float:
        """Total benchmark duration in seconds."""
        return self.end_time - self.start_time

    @property
    def win_rate(self) -> float:
        """Win rate for the primary engine (white)."""
        if self.total_games == 0:
            return 0.0
        return self.white_wins / self.total_games

    @property
    def draw_rate(self) -> float:
        """Draw rate."""
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games


class MetricsCollector:
    """Collects system and performance metrics during benchmark runs."""

    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.collecting = False
        self.metrics: List[SystemMetrics] = []
        self.collection_thread: Optional[threading.Thread] = None

    def start_collection(self):
        """Start collecting system metrics."""
        if self.collecting:
            return

        self.collecting = True
        self.metrics = []

        self.collection_thread = threading.Thread(target=self._collect_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        logger.info("Started metrics collection")

    def stop_collection(self) -> List[SystemMetrics]:
        """Stop collecting and return collected metrics."""
        if not self.collecting:
            return []

        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)

        logger.info(f"Stopped metrics collection, collected {len(self.metrics)} samples")
        return self.metrics

    def _collect_loop(self):
        """Main collection loop."""
        process = psutil.Process()

        while self.collecting:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                memory_available_gb = memory.available / (1024**3)

                # Process-specific metrics
                process_cpu_percent = process.cpu_percent(interval=None)
                process_memory_mb = process.memory_info().rss / (1024**2)
                process_threads = process.num_threads()

                # GPU metrics (if PyTorch available)
                gpu_memory_used_gb = None
                gpu_memory_total_gb = None
                gpu_utilization = None

                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.get_device_properties(0)
                        gpu_memory_total_gb = gpu_memory.total_memory / (1024**3)

                        if torch.cuda.memory_allocated(0) > 0:
                            gpu_memory_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
                            gpu_utilization = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else None
                    except Exception as e:
                        logger.debug(f"Could not collect GPU metrics: {e}")

                # Create metrics snapshot
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_gb=memory_used_gb,
                    memory_available_gb=memory_available_gb,
                    gpu_memory_used_gb=gpu_memory_used_gb,
                    gpu_memory_total_gb=gpu_memory_total_gb,
                    gpu_utilization=gpu_utilization,
                    process_cpu_percent=process_cpu_percent,
                    process_memory_mb=process_memory_mb,
                    process_threads=process_threads
                )

                self.metrics.append(metrics)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

            time.sleep(self.sample_interval)


class MetricsAnalyzer:
    """Analyzes collected benchmark metrics."""

    @staticmethod
    def analyze_game_metrics(games: List[GameMetrics]) -> Dict[str, Any]:
        """Analyze metrics from multiple games."""
        if not games:
            return {}

        total_games = len(games)
        total_moves = sum(game.total_moves for game in games)
        total_duration = sum(game.duration for game in games)

        # Game results
        white_wins = sum(1 for game in games if game.result == "1-0")
        black_wins = sum(1 for game in games if game.result == "0-1")
        draws = sum(1 for game in games if game.result == "1/2-1/2")

        # Timing analysis
        all_move_times = []
        for game in games:
            all_move_times.extend([move.get('time', 0) for move in game.move_times])

        avg_time_per_move = sum(all_move_times) / len(all_move_times) if all_move_times else 0
        min_time_per_move = min(all_move_times) if all_move_times else 0
        max_time_per_move = max(all_move_times) if all_move_times else 0

        # System resource analysis
        all_cpu_usage = []
        all_memory_usage = []
        peak_memory = 0

        for game in games:
            for metrics in game.system_metrics:
                all_cpu_usage.append(metrics.cpu_percent)
                all_memory_usage.append(metrics.memory_percent)
                peak_memory = max(peak_memory, metrics.memory_used_gb)

        avg_cpu_usage = sum(all_cpu_usage) / len(all_cpu_usage) if all_cpu_usage else 0
        avg_memory_usage = sum(all_memory_usage) / len(all_memory_usage) if all_memory_usage else 0

        return {
            "total_games": total_games,
            "total_moves": total_moves,
            "total_duration": total_duration,
            "avg_game_duration": total_duration / total_games if total_games > 0 else 0,
            "avg_moves_per_game": total_moves / total_games if total_games > 0 else 0,

            "results": {
                "white_wins": white_wins,
                "black_wins": black_wins,
                "draws": draws,
                "win_rate": white_wins / total_games if total_games > 0 else 0,
                "draw_rate": draws / total_games if total_games > 0 else 0
            },

            "timing": {
                "avg_time_per_move": avg_time_per_move,
                "min_time_per_move": min_time_per_move,
                "max_time_per_move": max_time_per_move,
                "time_variance": MetricsAnalyzer._calculate_variance(all_move_times)
            },

            "resources": {
                "avg_cpu_usage": avg_cpu_usage,
                "avg_memory_usage": avg_memory_usage,
                "peak_memory_gb": peak_memory
            }
        }

    @staticmethod
    def _calculate_variance(values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    @staticmethod
    def calculate_elo_difference(win_rate: float, num_games: int) -> Dict[str, Any]:
        """Calculate ELO difference based on win rate."""
        if num_games < 30:
            return {"error": "Insufficient games for reliable ELO calculation"}

        # Simplified ELO calculation
        # Win rate of 0.5 means equal strength
        # Each 0.01 difference in win rate ≈ 10 ELO points
        elo_difference = (win_rate - 0.5) * 1000

        # Confidence interval (simplified)
        confidence = min(1.0, num_games / 100.0)  # 100 games for full confidence

        return {
            "elo_difference": elo_difference,
            "confidence": confidence,
            "interpretation": f"{'+' if elo_difference > 0 else ''}{elo_difference:.0f} ELO points"
        }

    @staticmethod
    def export_metrics(metrics: BenchmarkMetrics, output_path: str):
        """Export metrics to JSON file."""
        data = asdict(metrics)

        # Convert datetime objects to timestamps
        for game in data['games']:
            game['start_time'] = game['start_time']
            game['end_time'] = game['end_time']
            for move in game['move_times']:
                move['timestamp'] = move.get('timestamp', 0)
            for sys_metrics in game['system_metrics']:
                sys_metrics['timestamp'] = sys_metrics['timestamp']

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported metrics to {output_path}")


# Global metrics collector instance
metrics_collector = MetricsCollector()
