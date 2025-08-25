"""Unified Utility helpers for azchess."""

# Original utilities
from .board import random_board
from .model_loader import load_model_and_mcts

# Unified utilities
from .memory import (
    MemoryManager,
    clear_memory_cache,
    get_memory_usage,
    emergency_memory_cleanup
)
from .memory_monitor import (
    MemoryMonitor,
    MemoryAlert,
    start_memory_monitoring,
    stop_memory_monitoring,
    get_memory_stats,
    add_memory_alert_callback
)
from .device import (
    DeviceManager,
    select_device,
    validate_device,
    get_device_info,
    setup_device
)
from .tensor import (
    TensorUtils,
    ensure_contiguous,
    ensure_contiguous_array,
    validate_tensor_shapes,
    check_tensor_health,
    safe_to_device,
    create_tensor,
    log_tensor_stats
)
from .checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    get_checkpoint_info,
    validate_checkpoint
)
from .config_utils import (
    ConfigUtils,
    ConfigManager,
    ConfigPath,
    safe_config_get,
    get_nested_config,
    validate_config_section,
    merge_configs,
    log_config_summary,
    # New unified config management
    set_global_config,
    get_global_config,
    config_get,
    config_get_typed,
    config_get_section,
    validate_config_requirements,
    config_manager
)
from .logging_utils import (
    Matrix0Logger,
    LogContext,
    LogOperation,
    setup_logging,
    get_logger,
    create_log_context,
    set_logging_level,
    log_system_info,
    log_print,
    log_progress,
    log_metrics,
    log_function,
    log_performance_stats,
    log_error_details
)
from .error_utils import (
    ErrorHandler,
    ErrorContext,
    ErrorResult,
    ErrorSeverity,
    ErrorCategory,
    Matrix0Error,
    ConfigurationError,
    TensorError,
    MemoryError,
    InferenceError,
    TrainingError,
    ValidationError,
    create_error_context,
    handle_matrix0_error,
    with_error_handling,
    safe_operation,
    get_error_statistics,
    error_handler
)
from .performance_utils import (
    PerformanceMonitor,
    PerformanceMetric,
    TimingMeasurement,
    TimingContext,
    start_timing,
    end_timing,
    record_metric,
    increment_counter,
    set_gauge,
    set_performance_threshold,
    add_performance_alert_callback,
    get_performance_stats,
    get_performance_report,
    time_operation,
    performance_monitor
)

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
