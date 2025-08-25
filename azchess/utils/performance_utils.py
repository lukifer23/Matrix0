"""
Unified Performance Monitoring and Metrics Collection for Matrix0
Provides consistent performance tracking, metrics collection, and monitoring across all modules.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a single performance metric measurement."""
    name: str
    value: float
    timestamp: float
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}={self.value:.4f} ({self.category})"


@dataclass
class TimingMeasurement:
    """Represents a timing measurement with start/end times."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    category: str = "timing"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get the duration if measurement is complete."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def complete(self) -> float:
        """Complete the measurement and return duration."""
        self.end_time = time.perf_counter()
        duration = self.duration
        return duration if duration is not None else 0.0

    def __str__(self) -> str:
        duration = self.duration
        if duration is not None:
            return f"{self.name}: {duration:.4f}s"
        else:
            return f"{self.name}: in progress"


class PerformanceMonitor:
    """Unified performance monitoring system."""

    def __init__(self, max_history: int = 1000):
        self._lock = threading.RLock()
        self._active_measurements: Dict[str, TimingMeasurement] = {}
        self._metrics_history: deque = deque(maxlen=max_history)
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._alert_callbacks: List[Callable] = []
        self._thresholds: Dict[str, float] = {}

    def start_timing(self, name: str, category: str = "timing", **metadata) -> TimingMeasurement:
        """Start a timing measurement."""
        measurement = TimingMeasurement(
            name=name,
            start_time=time.perf_counter(),
            category=category,
            metadata=metadata
        )

        with self._lock:
            self._active_measurements[name] = measurement

        return measurement

    def end_timing(self, name: str) -> Optional[float]:
        """End a timing measurement and return duration."""
        with self._lock:
            if name not in self._active_measurements:
                logger.warning(f"No active timing measurement found for: {name}")
                return None

            measurement = self._active_measurements.pop(name)
            duration = measurement.complete()

            # Record as metric
            self.record_metric(name, duration, "timing", **measurement.metadata)

            return duration

    def record_metric(self, name: str, value: float, category: str = "general", **metadata):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            category=category,
            metadata=metadata
        )

        with self._lock:
            self._metrics_history.append(metric)

            # Check thresholds and trigger alerts
            if name in self._thresholds:
                threshold = self._thresholds[name]
                if category == "timing" and value > threshold:
                    self._trigger_alert(f"Timing threshold exceeded for {name}: {value:.4f}s > {threshold}s")
                elif value > threshold:
                    self._trigger_alert(f"Metric threshold exceeded for {name}: {value:.4f} > {threshold}")

    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float):
        """Set a gauge metric."""
        with self._lock:
            self._gauges[name] = value

    def set_threshold(self, metric_name: str, threshold: float):
        """Set a threshold for alerting on a metric."""
        with self._lock:
            self._thresholds[metric_name] = threshold

    def add_alert_callback(self, callback: Callable[[str], None]):
        """Add a callback for performance alerts."""
        with self._lock:
            self._alert_callbacks.append(callback)

    def _trigger_alert(self, message: str):
        """Trigger performance alerts."""
        logger.warning(f"PERFORMANCE_ALERT: {message}")
        for callback in self._alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in performance alert callback: {e}")

    def get_statistics(self, category: Optional[str] = None,
                      time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            # Filter metrics by category and time window
            metrics = list(self._metrics_history)

            if category:
                metrics = [m for m in metrics if m.category == category]

            if time_window:
                cutoff_time = time.time() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if not metrics:
                return {"error": "No metrics available"}

            # Group by name
            by_name = {}
            for metric in metrics:
                if metric.name not in by_name:
                    by_name[metric.name] = []
                by_name[metric.name].append(metric.value)

            # Calculate statistics
            stats = {
                "total_metrics": len(metrics),
                "time_range": {
                    "start": min(m.timestamp for m in metrics),
                    "end": max(m.timestamp for m in metrics)
                },
                "by_metric": {}
            }

            for name, values in by_name.items():
                stats["by_metric"][name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "latest": values[-1] if values else 0
                }

            # Add counters and gauges
            stats["counters"] = dict(self._counters)
            stats["gauges"] = dict(self._gauges)
            stats["active_measurements"] = list(self._active_measurements.keys())

            return stats

    def get_summary_report(self, time_window: float = 300) -> str:
        """Generate a human-readable performance summary."""
        stats = self.get_statistics(time_window=time_window)

        if "error" in stats:
            return "No performance data available"

        report = ".1f"
        report += f"Total metrics recorded: {stats['total_metrics']}\n"
        report += f"Active timing measurements: {len(stats['active_measurements'])}\n"

        if stats['active_measurements']:
            report += f"Active measurements: {', '.join(stats['active_measurements'])}\n"

        if stats['by_metric']:
            report += "\nMetric Summary:\n"
            for name, metric_stats in stats['by_metric'].items():
                report += f"  {name}: mean={metric_stats['mean']:.4f}, "
                report += f"count={metric_stats['count']}, "
                report += f"latest={metric_stats['latest']:.4f}\n"

        if stats['counters']:
            report += "\nCounters:\n"
            for name, value in stats['counters'].items():
                report += f"  {name}: {value}\n"

        if stats['gauges']:
            report += "\nGauges:\n"
            for name, value in stats['gauges'].items():
                report += f"  {name}: {value:.4f}\n"

        return report

    def reset(self):
        """Reset all performance data."""
        with self._lock:
            self._active_measurements.clear()
            self._metrics_history.clear()
            self._counters.clear()
            self._gauges.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Convenience functions for global access
def start_timing(name: str, category: str = "timing", **metadata) -> TimingMeasurement:
    """Start a timing measurement."""
    return performance_monitor.start_timing(name, category, **metadata)


def end_timing(name: str) -> Optional[float]:
    """End a timing measurement."""
    return performance_monitor.end_timing(name)


def record_metric(name: str, value: float, category: str = "general", **metadata):
    """Record a performance metric."""
    performance_monitor.record_metric(name, value, category, **metadata)


def increment_counter(name: str, value: int = 1):
    """Increment a counter."""
    performance_monitor.increment_counter(name, value)


def set_gauge(name: str, value: float):
    """Set a gauge value."""
    performance_monitor.set_gauge(name, value)


def set_performance_threshold(metric_name: str, threshold: float):
    """Set a performance threshold for alerting."""
    performance_monitor.set_threshold(metric_name, threshold)


def add_performance_alert_callback(callback: Callable[[str], None]):
    """Add a callback for performance alerts."""
    performance_monitor.add_alert_callback(callback)


def get_performance_stats(category: Optional[str] = None, time_window: Optional[float] = None) -> Dict[str, Any]:
    """Get performance statistics."""
    return performance_monitor.get_statistics(category, time_window)


def get_performance_report(time_window: float = 300) -> str:
    """Get a performance summary report."""
    return performance_monitor.get_summary_report(time_window)


# Context manager for timing
class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, name: str, category: str = "timing", record_on_exit: bool = True, **metadata):
        self.name = name
        self.category = category
        self.record_on_exit = record_on_exit
        self.metadata = metadata
        self.measurement = None

    def __enter__(self):
        self.measurement = start_timing(self.name, self.category, **self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.measurement:
            duration = end_timing(self.name)
            if self.record_on_exit and duration is not None:
                logger.debug(f"{self.name} completed in {duration:.4f}s")


def time_operation(name: str, category: str = "timing", **metadata):
    """Decorator for timing function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimingContext(name, category, **metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator
