"""
Unified Logging Utilities for Matrix0
Centralizes logging configuration and utilities for consistent logging across modules.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Type
from dataclasses import dataclass
from functools import wraps

# Default logging format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEBUG_FORMAT = '%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s'


@dataclass
class LogEvent:
    """Structured log event for consistent logging."""
    level: str
    message: str
    module: str
    category: str = "general"
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


class LogContext:
    """Context for structured logging."""

    def __init__(self, module: str, category: str = "general", **context_data):
        self.module = module
        self.category = category
        self.context_data = context_data.copy()
        self.logger = get_logger(module)

    def log(self, level: str, message: str, **extra_data):
        """Log a structured message."""
        data = self.context_data.copy()
        data.update(extra_data)

        event = LogEvent(
            level=level,
            message=message,
            module=self.module,
            category=self.category,
            data=data if data else None
        )

        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[{self.category}] {message}", extra={'structured_data': data})

    def debug(self, message: str, **data):
        self.log('DEBUG', message, **data)

    def info(self, message: str, **data):
        self.log('INFO', message, **data)

    def warning(self, message: str, **data):
        self.log('WARNING', message, **data)

    def error(self, message: str, **data):
        self.log('ERROR', message, **data)

    def critical(self, message: str, **data):
        self.log('CRITICAL', message, **data)


class Matrix0Logger:
    """Unified logging configuration for Matrix0."""

    def __init__(self):
        self._loggers: Dict[str, logging.Logger] = {}
        self._default_level = logging.INFO
        self._lock = threading.RLock()

    def setup_logging(self, log_dir: Optional[str] = None, level: int = logging.INFO,
                     format_string: Optional[str] = None, console: bool = True,
                     file_logging: bool = True) -> logging.Logger:
        """Setup comprehensive logging configuration."""
        self._default_level = level

        # Create root logger
        root_logger = logging.getLogger('azchess')
        root_logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set format
        if level == logging.DEBUG:
            format_str = format_string or DEBUG_FORMAT
        else:
            format_str = format_string or DEFAULT_FORMAT

        formatter = logging.Formatter(format_str)

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if file_logging and log_dir:
            log_path = Path(log_dir) / 'matrix0.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        return root_logger

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with consistent configuration."""
        with self._lock:
            if name in self._loggers:
                return self._loggers[name]

            logger = logging.getLogger(f'azchess.{name}')
            logger.setLevel(self._default_level)

            # Cache the logger
            self._loggers[name] = logger
            return logger

    def create_context(self, module: str, category: str = "general", **context_data) -> LogContext:
        """Create a structured logging context."""
        return LogContext(module, category, **context_data)


# Global logger manager instance
logger_manager = Matrix0Logger()


# Convenience functions for global access
def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO,
                 format_string: Optional[str] = None, console: bool = True,
                 file_logging: bool = True) -> logging.Logger:
    """Setup logging with unified configuration."""
    return logger_manager.setup_logging(log_dir, level, format_string, console, file_logging)


def get_logger(name: str) -> logging.Logger:
    """Get a consistently configured logger."""
    return logger_manager.get_logger(name)


def create_log_context(module: str, category: str = "general", **context_data) -> LogContext:
    """Create a structured logging context."""
    return logger_manager.create_context(module, category, **context_data)


def set_logging_level(level: int) -> None:
    """Set logging level for all loggers."""
    logger_manager._default_level = level
    root_logger = logging.getLogger('azchess')
    root_logger.setLevel(level)

    # Update all cached loggers
    for logger in logger_manager._loggers.values():
        logger.setLevel(level)


def log_system_info() -> None:
    """Log system information for debugging."""
    try:
        import platform

        logger = get_logger('system')

        logger.info("System Information:")
        logger.info(f"  Platform: {platform.platform()}")
        logger.info(f"  Python: {platform.python_version()}")

        try:
            import torch
            logger.info(f"  PyTorch: {torch.__version__}")
            logger.info(f"  CUDA available: {torch.cuda.is_available()}")
            logger.info(f"  MPS available: {torch.backends.mps.is_available()}")

            if torch.cuda.is_available():
                logger.info(f"  CUDA devices: {torch.cuda.device_count()}")

        except Exception as e:
            logger.info(f"  PyTorch info unavailable: {e}")

    except Exception as e:
        print(f"Could not log system info: {e}")


# Print statement replacement utilities
def log_print(message: str, level: str = "info", logger_name: str = "general") -> None:
    """Replace print statements with proper logging."""
    logger = get_logger(logger_name)
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)


def log_progress(current: int, total: int, message: str = "", logger_name: str = "progress") -> None:
    """Log progress with consistent formatting."""
    logger = get_logger(logger_name)
    percent = (current / total) * 100 if total > 0 else 0
    logger.info(f"Progress: {current}/{total} ({percent:.1f}%) {message}")


def log_metrics(metrics: Dict[str, Any], prefix: str = "", logger_name: str = "metrics") -> None:
    """Log metrics with consistent formatting."""
    logger = get_logger(logger_name)
    metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in metrics.items())
    logger.info(f"{prefix}{metrics_str}")


# Decorator for function logging
def log_function(level: str = "info", logger_name: Optional[str] = None):
    """Decorator to log function entry and exit."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = logger_name or func.__module__
            logger = get_logger(name)
            log_method = getattr(logger, level.lower(), logger.info)

            log_method(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                log_method(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise

        return wrapper
    return decorator


# Context manager for operation logging
class LogOperation:
    """Context manager for logging operations."""

    def __init__(self, operation: str, logger_name: str = "operations", level: str = "info"):
        self.operation = operation
        self.logger_name = logger_name
        self.level = level
        self.logger = get_logger(logger_name)
        self.log_method = getattr(self.logger, level.lower(), self.logger.info)

    def __enter__(self):
        self.log_method(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.log_method(f"Completed {self.operation}")
        else:
            self.logger.error(f"Failed {self.operation}: {exc_val}")


# Performance logging integration
def log_performance_stats(logger_name: str = "performance", time_window: float = 300) -> None:
    """Log performance statistics."""
    try:
        from .performance_utils import get_performance_report
        report = get_performance_report(time_window)
        logger = get_logger(logger_name)
        logger.info(f"Performance Report:\n{report}")
    except ImportError:
        pass  # Performance monitoring not available


def log_error_details(error: Exception, logger_name: str = "errors", include_traceback: bool = True) -> None:
    """Log detailed error information."""
    import traceback

    logger = get_logger(logger_name)
    logger.error(f"Error: {error}")
    logger.error(f"Error type: {type(error).__name__}")

    if include_traceback:
        logger.error(f"Traceback:\n{traceback.format_exc()}")


def set_logging_level(level: int) -> None:
    """Set logging level for all loggers."""
    logger_manager.set_level(level)


def log_system_info() -> None:
        """Log system information for debugging."""
        try:
            import platform
            import torch

            logger = self.get_logger('system')

            logger.info("System Information:")
            logger.info(f"  Platform: {platform.platform()}")
            logger.info(f"  Python: {platform.python_version()}")

            try:
                logger.info(f"  PyTorch: {torch.__version__}")
                logger.info(f"  CUDA available: {torch.cuda.is_available()}")
                logger.info(f"  MPS available: {torch.backends.mps.is_available()}")

                if torch.cuda.is_available():
                    logger.info(f"  CUDA devices: {torch.cuda.device_count()}")

            except Exception as e:
                logger.info(f"  PyTorch info unavailable: {e}")

        except Exception as e:
            print(f"Could not log system info: {e}")


# Global instance
logger_manager = Matrix0Logger()


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO,
                 format_string: Optional[str] = None, console: bool = True,
                 file_logging: bool = True) -> logging.Logger:
    """Setup logging with unified configuration."""
    return logger_manager.setup_logging(log_dir, level, format_string, console, file_logging)


def get_logger(name: str) -> logging.Logger:
    """Get a consistently configured logger."""
    return logger_manager.get_logger(name)


def set_logging_level(level: int) -> None:
    """Set logging level for all loggers."""
    logger_manager.set_level(level)


def log_system_info() -> None:
    """Log system information."""
    logger_manager.log_system_info()
