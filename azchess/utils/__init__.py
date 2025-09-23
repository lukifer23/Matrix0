"""Unified Utility helpers for azchess."""

# Original utilities
from .board import random_board
from .checkpoint import (CheckpointManager, get_checkpoint_info,
                         load_checkpoint, save_checkpoint, validate_checkpoint)
from .config_utils import ConfigManager  # New unified config management
from .config_utils import (ConfigPath, ConfigUtils, config_get,
                           config_get_section, config_get_typed,
                           config_manager, get_global_config,
                           get_nested_config, log_config_summary,
                           merge_configs, safe_config_get, set_global_config,
                           validate_config_requirements,
                           validate_config_section)
from .device import (DeviceManager, get_device_info, select_device,
                     setup_device, validate_device)
from .error_utils import (ConfigurationError, ErrorCategory, ErrorContext,
                          ErrorHandler, ErrorResult, ErrorSeverity,
                          InferenceError, Matrix0Error, MemoryError,
                          TensorError, TrainingError, ValidationError,
                          create_error_context, error_handler,
                          get_error_statistics, handle_matrix0_error,
                          safe_operation, with_error_handling)
from .logging_utils import (LogContext, LogOperation, Matrix0Logger,
                            create_log_context, get_logger, log_error_details,
                            log_function, log_metrics, log_performance_stats,
                            log_print, log_progress, log_system_info,
                            set_logging_level, setup_logging)
# Unified utilities
from .memory import (MemoryManager, clear_memory_cache,
                     emergency_memory_cleanup, get_memory_usage)
from .memory_monitor import (MemoryAlert, MemoryMonitor,
                             add_memory_alert_callback, get_memory_stats,
                             remove_memory_alert_callback,
                             start_memory_monitoring, stop_memory_monitoring)
from .model_loader import load_model_and_mcts
from .performance_utils import (PerformanceMetric, PerformanceMonitor,
                                TimingContext, TimingMeasurement,
                                add_performance_alert_callback, end_timing,
                                get_performance_report, get_performance_stats,
                                increment_counter, performance_monitor,
                                record_metric, set_gauge,
                                set_performance_threshold, start_timing,
                                time_operation)
from .tensor import (TensorUtils, check_tensor_health, create_tensor,
                     ensure_contiguous, ensure_contiguous_array,
                     log_tensor_stats, safe_to_device, validate_tensor_shapes)

__all__ = [
    # Original utilities
    "random_board",
    "load_model_and_mcts",

    # Memory management
    "MemoryManager",
    "clear_memory_cache",
    "get_memory_usage",
    "emergency_memory_cleanup",

    # Memory monitoring
    "MemoryMonitor",
    "MemoryAlert",
    "start_memory_monitoring",
    "stop_memory_monitoring",
    "get_memory_stats",
    "add_memory_alert_callback",
    "remove_memory_alert_callback",

    # Device management
    "DeviceManager",
    "select_device",
    "validate_device",
    "get_device_info",
    "setup_device",

    # Tensor utilities
    "TensorUtils",
    "ensure_contiguous",
    "ensure_contiguous_array",
    "validate_tensor_shapes",
    "check_tensor_health",
    "safe_to_device",
    "create_tensor",
    "log_tensor_stats",

    # Checkpoint management
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "get_checkpoint_info",
    "validate_checkpoint",

    # Configuration utilities
    "ConfigUtils",
    "ConfigManager",
    "ConfigPath",
    "safe_config_get",
    "get_nested_config",
    "validate_config_section",
    "merge_configs",
    "log_config_summary",
    # New unified config management
    "set_global_config",
    "get_global_config",
    "config_get",
    "config_get_typed",
    "config_get_section",
    "validate_config_requirements",
    "config_manager",

    # Logging utilities
    "Matrix0Logger",
    "LogContext",
    "LogOperation",
    "setup_logging",
    "get_logger",
    "create_log_context",
    "set_logging_level",
    "log_system_info",
    "log_print",
    "log_progress",
    "log_metrics",
    "log_function",
    "log_performance_stats",
    "log_error_details",

    # Error handling utilities
    "ErrorHandler",
    "ErrorContext",
    "ErrorResult",
    "ErrorSeverity",
    "ErrorCategory",
    "Matrix0Error",
    "ConfigurationError",
    "TensorError",
    "MemoryError",
    "InferenceError",
    "TrainingError",
    "ValidationError",
    "create_error_context",
    "handle_matrix0_error",
    "with_error_handling",
    "safe_operation",
    "get_error_statistics",
    "error_handler",

    # Performance utilities
    "PerformanceMonitor",
    "PerformanceMetric",
    "TimingMeasurement",
    "TimingContext",
    "start_timing",
    "end_timing",
    "record_metric",
    "increment_counter",
    "set_gauge",
    "set_performance_threshold",
    "add_performance_alert_callback",
    "get_performance_stats",
    "get_performance_report",
    "time_operation",
    "performance_monitor"
]
