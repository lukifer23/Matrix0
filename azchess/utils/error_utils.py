"""
Unified Error Handling System for Matrix0
Provides consistent error handling patterns, classification, and recovery mechanisms.
"""

from __future__ import annotations

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    CONFIGURATION = "configuration"
    TENSOR_OPERATION = "tensor_operation"
    MEMORY = "memory"
    INFERENCE = "inference"
    TRAINING = "training"
    MCTS = "mcts"
    DATA_LOADING = "data_loading"
    CHECKPOINT = "checkpoint"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    module: str
    function: str
    operation: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3
    context_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorResult:
    """Result of error handling operation."""
    success: bool
    error: Optional[Exception] = None
    context: Optional[ErrorContext] = None
    recovery_action: Optional[str] = None
    retry_recommended: bool = False


class Matrix0Error(Exception):
    """Base exception class for Matrix0 errors."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context_data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context_data = context_data or {}
        self.timestamp = time.time()

    def __str__(self):
        return f"[{self.category.value}:{self.severity.value}] {super().__str__()}"


class ConfigurationError(Matrix0Error):
    """Configuration-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, **kwargs)


class TensorError(Matrix0Error):
    """Tensor operation errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.TENSOR_OPERATION, ErrorSeverity.HIGH, **kwargs)


class MemoryError(Matrix0Error):
    """Memory-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.MEMORY, ErrorSeverity.CRITICAL, **kwargs)


class InferenceError(Matrix0Error):
    """Neural network inference errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.INFERENCE, ErrorSeverity.HIGH, **kwargs)


class TrainingError(Matrix0Error):
    """Training-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.TRAINING, ErrorSeverity.MEDIUM, **kwargs)


class ValidationError(Matrix0Error):
    """Data validation errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, **kwargs)


class ErrorHandler:
    """Unified error handler with recovery mechanisms."""

    def __init__(self):
        self.error_history: List[ErrorResult] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self._setup_default_strategies()

    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        # Memory error recovery
        self.recovery_strategies[ErrorCategory.MEMORY] = [
            self._memory_cleanup_strategy,
            self._memory_emergency_strategy
        ]

        # Tensor error recovery
        self.recovery_strategies[ErrorCategory.TENSOR_OPERATION] = [
            self._tensor_validation_strategy,
            self._tensor_recovery_strategy
        ]

        # Configuration error recovery
        self.recovery_strategies[ErrorCategory.CONFIGURATION] = [
            self._config_fallback_strategy
        ]

        # Inference error recovery
        self.recovery_strategies[ErrorCategory.INFERENCE] = [
            self._inference_retry_strategy,
            self._inference_fallback_strategy
        ]

    def _memory_cleanup_strategy(self, error: Exception, context: ErrorContext) -> bool:
        """Memory cleanup recovery strategy."""
        try:
            from .memory import clear_memory_cache
            clear_memory_cache('auto')
            logger.info("Memory cleanup recovery applied")
            return True
        except Exception:
            return False

    def _memory_emergency_strategy(self, error: Exception, context: ErrorContext) -> bool:
        """Emergency memory recovery strategy."""
        try:
            from .memory import emergency_memory_cleanup
            emergency_memory_cleanup('auto')
            logger.warning("Emergency memory cleanup applied")
            return True
        except Exception:
            return False

    def _tensor_validation_strategy(self, error: Exception, context: ErrorContext) -> bool:
        """Tensor validation recovery strategy."""
        # This would implement tensor validation and correction
        logger.debug("Tensor validation recovery attempted")
        return False

    def _tensor_recovery_strategy(self, error: Exception, context: ErrorContext) -> bool:
        """Tensor recovery strategy."""
        # This would implement tensor recovery mechanisms
        logger.debug("Tensor recovery attempted")
        return False

    def _config_fallback_strategy(self, error: Exception, context: ErrorContext) -> bool:
        """Configuration fallback recovery strategy."""
        logger.debug("Configuration fallback attempted")
        return False

    def _inference_retry_strategy(self, error: Exception, context: ErrorContext) -> bool:
        """Inference retry recovery strategy."""
        if context.retry_count < context.max_retries:
            logger.debug(f"Inference retry {context.retry_count + 1}/{context.max_retries}")
            return True
        return False

    def _inference_fallback_strategy(self, error: Exception, context: ErrorContext) -> bool:
        """Inference fallback recovery strategy."""
        logger.debug("Inference fallback attempted")
        return False

    def handle_error(self, error: Exception, context: ErrorContext) -> ErrorResult:
        """Handle an error with recovery attempts."""
        result = ErrorResult(
            success=False,
            error=error,
            context=context
        )

        # Log the error
        self._log_error(error, context)

        # Attempt recovery
        if context.category in self.recovery_strategies:
            for strategy in self.recovery_strategies[context.category]:
                try:
                    if strategy(error, context):
                        result.success = True
                        result.recovery_action = strategy.__name__
                        logger.info(f"Error recovery successful: {strategy.__name__}")
                        break
                except Exception as recovery_error:
                    logger.debug(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")

        # Check if retry is recommended
        if not result.success and context.retry_count < context.max_retries:
            result.retry_recommended = True

        # Store in history
        self.error_history.append(result)

        return result

    def _log_error(self, error: Exception, context: ErrorContext) -> None:
        """Log error with appropriate severity."""
        error_msg = f"[{context.category.value}:{context.severity.value}] {context.module}.{context.function}: {error}"

        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(error_msg)
            logger.debug(f"Error context: {context.context_data}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(error_msg)
            logger.debug(f"Error context: {context.context_data}")
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(error_msg)
        else:
            logger.info(error_msg)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        stats = {
            'total_errors': len(self.error_history),
            'successful_recoveries': len([r for r in self.error_history if r.success]),
            'by_category': {},
            'by_severity': {},
            'recent_errors': []
        }

        for result in self.error_history[-10:]:  # Last 10 errors
            if result.context:
                category = result.context.category.value
                severity = result.context.severity.value

                stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
                stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

                stats['recent_errors'].append({
                    'category': category,
                    'severity': severity,
                    'success': result.success,
                    'recovery': result.recovery_action
                })

        return stats


# Global error handler instance
error_handler = ErrorHandler()


def create_error_context(module: str, function: str, operation: str,
                        category: ErrorCategory = ErrorCategory.UNKNOWN,
                        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                        max_retries: int = 3,
                        context_data: Optional[Dict[str, Any]] = None) -> ErrorContext:
    """Create an error context for error handling."""
    return ErrorContext(
        module=module,
        function=function,
        operation=operation,
        category=category,
        severity=severity,
        timestamp=time.time(),
        max_retries=max_retries,
        context_data=context_data or {}
    )


def handle_matrix0_error(error: Exception, context: ErrorContext) -> ErrorResult:
    """Handle a Matrix0 error with unified error handling."""
    return error_handler.handle_error(error, context)


def with_error_handling(category: ErrorCategory = ErrorCategory.UNKNOWN,
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       max_retries: int = 3,
                       fallback_value: Any = None):
    """Decorator for unified error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            context = create_error_context(
                module=func.__module__,
                function=func.__name__,
                operation=func.__name__,
                category=category,
                severity=severity,
                max_retries=max_retries
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                result = handle_matrix0_error(e, context)

                if result.success:
                    # Recovery was successful, but we need to retry the function
                    if result.retry_recommended and context.retry_count < max_retries:
                        context.retry_count += 1
                        logger.debug(f"Retrying {func.__name__} (attempt {context.retry_count})")
                        return wrapper(*args, **kwargs)
                    else:
                        logger.debug(f"Recovery successful for {func.__name__}")
                        return fallback_value
                else:
                    # No recovery possible, return fallback
                    logger.debug(f"No recovery possible for {func.__name__}, using fallback")
                    return fallback_value

        return wrapper
    return decorator


def safe_operation(operation_name: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  fallback_value: Any = None):
    """Context manager for safe operations."""
    return ErrorHandlingContext(operation_name, category, severity, fallback_value)


class ErrorHandlingContext:
    """Context manager for error handling."""

    def __init__(self, operation: str, category: ErrorCategory, severity: ErrorSeverity,
                 fallback_value: Any = None):
        self.operation = operation
        self.category = category
        self.severity = severity
        self.fallback_value = fallback_value
        self.context: Optional[ErrorContext] = None
        self._result = None

    def __enter__(self):
        # Create context when entering
        import inspect
        frame = inspect.currentframe()
        module = frame.f_back.f_globals.get('__name__', 'unknown')
        function = frame.f_back.f_code.co_name

        self.context = create_error_context(
            module=module,
            function=function,
            operation=self.operation,
            category=self.category,
            severity=self.severity
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.context is not None:
            self._result = handle_matrix0_error(exc_val, self.context)
            if self._result.success:
                # Recovery successful, suppress the exception
                return True
        # Return False to let the exception propagate if no recovery
        return False

    @property
    def result(self):
        """Get the result of error handling."""
        return self._result


def get_error_statistics() -> Dict[str, Any]:
    """Get error statistics from the global error handler."""
    return error_handler.get_error_statistics()
