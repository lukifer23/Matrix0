"""
Memory Monitoring and Alerting System for Matrix0
Provides comprehensive memory monitoring, alerting, and automatic optimization.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .memory import (clear_memory_cache, emergency_memory_cleanup,
                     get_memory_usage)

logger = logging.getLogger(__name__)


@dataclass
class MemoryAlert:
    """Represents a memory alert with details."""
    alert_type: str  # 'warning', 'critical', 'info'
    message: str
    memory_usage_gb: float
    memory_limit_gb: float
    timestamp: float
    device: str
    action_taken: Optional[str] = None


class MemoryMonitor:
    """Comprehensive memory monitoring and alerting system."""

    def __init__(self,
                 warning_threshold: float = 0.80,  # 80% memory usage
                 critical_threshold: float = 0.90,  # 90% memory usage
                 check_interval: float = 30.0,      # Check every 30 seconds
                 alert_cooldown: float = 300.0):    # 5 minutes between alerts

        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.alert_cooldown = alert_cooldown

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Alert tracking
        self.last_alert_time = 0
        self.recent_alerts = deque(maxlen=50)
        self.alert_callbacks: list[Callable[[MemoryAlert], None]] = []

        # Memory history for trend analysis
        self.memory_history = deque(maxlen=100)
        self.history_timestamps = deque(maxlen=100)

        # Device-specific limits
        self.device_limits: Dict[str, float] = {
            'cpu': 32.0,      # Assume 32GB system memory
            'mps': 24.0,      # MPS memory limit
            'cuda': 16.0,     # Default CUDA memory limit
        }

    def start_monitoring(self, device: str = 'auto') -> None:
        """Start the memory monitoring thread."""
        if self.is_monitoring:
            logger.warning("Memory monitoring is already running")
            return

        self.is_monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(device,),
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitor_thread.start()
        logger.info(f"Memory monitoring started for device: {device}")

    def stop_monitoring(self) -> None:
        """Stop the memory monitoring thread."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.stop_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        logger.info("Memory monitoring stopped")

    def _monitor_loop(self, device: str) -> None:
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                self._check_memory_and_alert(device)
                self.stop_event.wait(self.check_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                self.stop_event.wait(self.check_interval)

    def _check_memory_and_alert(self, device: str) -> None:
        """Check memory usage and generate alerts if needed."""
        try:
            memory_info = get_memory_usage(device)
            memory_usage_gb = memory_info.get('memory_gb', 0)
            memory_limit_gb = self.device_limits.get(device, 32.0)

            # Record memory usage for trend analysis
            current_time = time.time()
            self.memory_history.append(memory_usage_gb)
            self.history_timestamps.append(current_time)

            # Calculate memory ratio and human-readable usage string
            if memory_limit_gb > 0:
                memory_ratio = memory_usage_gb / memory_limit_gb
                usage_message = (
                    f"memory {memory_usage_gb:.1f}/{memory_limit_gb:.1f} GB "
                    f"({memory_ratio:.0%})"
                )
            else:
                memory_ratio = 0
                usage_message = f"memory {memory_usage_gb:.1f} GB (limit unknown)"

            # Check for alerts
            alert = None
            action_taken = None

            if memory_ratio >= self.critical_threshold:
                # Critical memory usage
                if current_time - self.last_alert_time > self.alert_cooldown:
                    alert = MemoryAlert(
                        alert_type='critical',
                        message=usage_message,
                        memory_usage_gb=memory_usage_gb,
                        memory_limit_gb=memory_limit_gb,
                        timestamp=current_time,
                        device=device
                    )

                    # Take emergency action
                    try:
                        emergency_memory_cleanup(device)
                        action_taken = "emergency_cleanup"
                        logger.critical("EMERGENCY MEMORY CLEANUP performed due to critical usage")
                    except Exception as e:
                        action_taken = f"emergency_cleanup_failed: {e}"
                        logger.error(f"Emergency cleanup failed: {e}")

            elif memory_ratio >= self.warning_threshold:
                # High memory usage warning
                if current_time - self.last_alert_time > self.alert_cooldown:
                    alert = MemoryAlert(
                        alert_type='warning',
                        message=usage_message,
                        memory_usage_gb=memory_usage_gb,
                        memory_limit_gb=memory_limit_gb,
                        timestamp=current_time,
                        device=device
                    )

                    # Take preventive action
                    try:
                        clear_memory_cache(device)
                        action_taken = "cache_cleanup"
                        logger.warning("Memory cache cleared due to high usage")
                    except Exception as e:
                        action_taken = f"cache_cleanup_failed: {e}"
                        logger.error(f"Cache cleanup failed: {e}")

            # Send alert if generated
            if alert:
                alert.action_taken = action_taken
                self._send_alert(alert)
                self.last_alert_time = current_time

        except Exception as e:
            logger.error(f"Error checking memory: {e}")

    def _send_alert(self, alert: MemoryAlert) -> None:
        """Send alert to all registered callbacks."""
        self.recent_alerts.append(alert)

        # Log the alert
        log_method = {
            'critical': logger.critical,
            'warning': logger.warning,
            'info': logger.info
        }.get(alert.alert_type, logger.info)

        log_method(f"MEMORY_ALERT: {alert.message}")

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable[[MemoryAlert], None]) -> None:
        """Add a callback function to be called when alerts are generated."""
        self.alert_callbacks.append(callback)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            'is_monitoring': self.is_monitoring,
            'recent_alerts_count': len(self.recent_alerts),
            'alert_callbacks_count': len(self.alert_callbacks),
            'memory_history_size': len(self.memory_history),
        }

        if self.memory_history:
            stats.update({
                'current_memory_gb': self.memory_history[-1] if self.memory_history else 0,
                'avg_memory_gb': sum(self.memory_history) / len(self.memory_history),
                'max_memory_gb': max(self.memory_history) if self.memory_history else 0,
                'min_memory_gb': min(self.memory_history) if self.memory_history else 0,
            })

        return stats

    def get_recent_alerts(self, limit: int = 10) -> list[MemoryAlert]:
        """Get recent memory alerts."""
        return list(self.recent_alerts)[-limit:]

    def set_device_limit(self, device: str, limit_gb: float) -> None:
        """Set memory limit for a specific device."""
        self.device_limits[device] = limit_gb
        logger.info(f"Set memory limit for {device}: {limit_gb}GB")


# Global monitor instance
memory_monitor = MemoryMonitor()


def start_memory_monitoring(device: str = 'auto', **kwargs) -> None:
    """Convenience function to start memory monitoring."""
    # Update monitor settings if provided
    for key, value in kwargs.items():
        if hasattr(memory_monitor, key):
            setattr(memory_monitor, key, value)

    memory_monitor.start_monitoring(device)


def stop_memory_monitoring() -> None:
    """Convenience function to stop memory monitoring."""
    memory_monitor.stop_monitoring()


def get_memory_stats() -> Dict[str, Any]:
    """Convenience function to get memory statistics."""
    return memory_monitor.get_memory_stats()


def add_memory_alert_callback(callback: Callable[[MemoryAlert], None]) -> None:
    """Convenience function to add alert callback."""
    memory_monitor.add_alert_callback(callback)
