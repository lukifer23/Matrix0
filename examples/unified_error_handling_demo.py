#!/usr/bin/env python3
"""
Unified Error Handling System Demo
Showcases the comprehensive error handling and recovery system for Matrix0.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azchess.utils import (
    ConfigurationError,
    ErrorCategory,
    ErrorSeverity,
    InferenceError,
    Matrix0Error,
    MemoryError,
    TensorError,
    TrainingError,
    ValidationError,
    create_error_context,
    error_handler,
    get_error_statistics,
    handle_matrix0_error,
    safe_operation,
    with_error_handling,
)


def demo_error_classification():
    """Demonstrate error classification and custom exceptions."""
    print("=" * 60)
    print("Error Classification Demo")
    print("=" * 60)

    print("Matrix0 Error Types:")
    print("• Matrix0Error: Base exception class")
    print("• ConfigurationError: Configuration-related errors")
    print("• TensorError: Tensor operation errors")
    print("• MemoryError: Memory-related errors")
    print("• InferenceError: Neural network inference errors")
    print("• TrainingError: Training-related errors")
    print("• ValidationError: Data validation errors")

    print("\nError Categories:")
    for category in ErrorCategory:
        print(f"• {category.name}: {category.value}")

    print("\nError Severities:")
    for severity in ErrorSeverity:
        print(f"• {severity.name}: {severity.value}")

    print("\n✅ Error classification demonstrated")


def demo_error_creation_and_handling():
    """Demonstrate error creation and handling."""
    print("\n" + "=" * 60)
    print("Error Creation and Handling Demo")
    print("=" * 60)

    # Create different types of errors
    errors = [
        ConfigurationError("Invalid configuration value", context_data={"key": "batch_size", "value": -1}),
        TensorError("Tensor shape mismatch", context_data={"expected": (32, 64), "actual": (16, 128)}),
        MemoryError("Out of memory during training", context_data={"usage_gb": 12.5, "limit_gb": 8.0}),
        InferenceError("Model inference failed", context_data={"model": "resnet", "batch_size": 64}),
        TrainingError("Training loss became NaN", context_data={"epoch": 42, "step": 1000}),
        ValidationError("Invalid board encoding", context_data={"position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"})
    ]

    for error in errors:
        print(f"\nError: {error}")
        print(f"Category: {error.category.value}")
        print(f"Severity: {error.severity.value}")
        print(f"Context: {error.context_data}")

        # Create error context
        context = create_error_context(
            module="demo",
            function="demo_error_creation_and_handling",
            operation="error_testing",
            category=error.category,
            severity=error.severity,
            context_data=error.context_data
        )

        # Handle the error
        result = handle_matrix0_error(error, context)
        print(f"Handling result: Success={result.success}, Recovery={result.recovery_action}")

    print("\n✅ Error creation and handling demonstrated")


def demo_decorator_error_handling():
    """Demonstrate decorator-based error handling."""
    print("\n" + "=" * 60)
    print("Decorator Error Handling Demo")
    print("=" * 60)

    @with_error_handling(
        category=ErrorCategory.TENSOR_OPERATION,
        severity=ErrorSeverity.HIGH,
        max_retries=2,
        fallback_value="fallback_result"
    )
    def risky_tensor_operation(should_fail=False):
        """A function that might fail with tensor operations."""
        if should_fail:
            raise TensorError("Simulated tensor operation failure")
        return "success"

    @with_error_handling(
        category=ErrorCategory.MEMORY,
        severity=ErrorSeverity.CRITICAL,
        max_retries=1,
        fallback_value=None
    )
    def risky_memory_operation(should_fail=False):
        """A function that might fail with memory issues."""
        if should_fail:
            raise MemoryError("Simulated out of memory error")
        return "memory_success"

    print("Testing successful operations:")
    result1 = risky_tensor_operation(should_fail=False)
    result2 = risky_memory_operation(should_fail=False)
    print(f"Tensor operation: {result1}")
    print(f"Memory operation: {result2}")

    print("\nTesting failed operations with recovery:")
    result3 = risky_tensor_operation(should_fail=True)
    result4 = risky_memory_operation(should_fail=True)
    print(f"Failed tensor operation result: {result3}")
    print(f"Failed memory operation result: {result4}")

    print("\n✅ Decorator error handling demonstrated")


def demo_context_manager_error_handling():
    """Demonstrate context manager error handling."""
    print("\n" + "=" * 60)
    print("Context Manager Error Handling Demo")
    print("=" * 60)

    def safe_tensor_computation(should_fail=False):
        """Safe tensor computation with error handling."""
        with safe_operation("tensor_computation", ErrorCategory.TENSOR_OPERATION,
                          ErrorSeverity.MEDIUM, fallback_value="default_tensor") as ctx:
            if should_fail:
                raise TensorError("Tensor computation failed")
            return "tensor_result"

    def safe_inference_operation(should_fail=False):
        """Safe inference operation with error handling."""
        with safe_operation("model_inference", ErrorCategory.INFERENCE,
                          ErrorSeverity.HIGH, fallback_value=None) as ctx:
            if should_fail:
                raise InferenceError("Model inference timeout")
            return "inference_result"

    print("Testing successful operations:")
    result1 = safe_tensor_computation(should_fail=False)
    result2 = safe_inference_operation(should_fail=False)
    print(f"Tensor computation: {result1}")
    print(f"Inference operation: {result2}")

    print("\nTesting failed operations:")
    result3 = safe_tensor_computation(should_fail=True)
    result4 = safe_inference_operation(should_fail=True)
    print(f"Failed tensor computation: {result3}")
    print(f"Failed inference operation: {result4}")

    print("\n✅ Context manager error handling demonstrated")


def demo_error_statistics():
    """Demonstrate error statistics and monitoring."""
    print("\n" + "=" * 60)
    print("Error Statistics Demo")
    print("=" * 60)

    # Generate some test errors
    test_errors = [
        (TensorError("Test tensor error"), ErrorCategory.TENSOR_OPERATION, ErrorSeverity.MEDIUM),
        (MemoryError("Test memory error"), ErrorCategory.MEMORY, ErrorSeverity.HIGH),
        (ConfigurationError("Test config error"), ErrorCategory.CONFIGURATION, ErrorSeverity.LOW),
        (InferenceError("Test inference error"), ErrorCategory.INFERENCE, ErrorSeverity.MEDIUM),
        (ValidationError("Test validation error"), ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
    ]

    print("Generating test errors...")
    for error, category, severity in test_errors:
        context = create_error_context(
            module="demo",
            function="demo_error_statistics",
            operation="error_testing",
            category=category,
            severity=severity
        )
        handle_matrix0_error(error, context)

    # Get error statistics
    stats = get_error_statistics()

    print(f"\nError Statistics:")
    print(f"• Total errors: {stats['total_errors']}")
    print(f"• Successful recoveries: {stats['successful_recoveries']}")
    print(f"• By category: {stats['by_category']}")
    print(f"• By severity: {stats['by_severity']}")

    print(f"\nRecent errors ({len(stats['recent_errors'])}):")
    for i, error in enumerate(stats['recent_errors'], 1):
        print(f"  {i}. {error['category']}:{error['severity']} - Success: {error['success']}")

    print("\n✅ Error statistics demonstrated")


def demo_recovery_strategies():
    """Demonstrate error recovery strategies."""
    print("\n" + "=" * 60)
    print("Error Recovery Strategies Demo")
    print("=" * 60)

    print("Built-in Recovery Strategies:")
    print("• Memory Error Recovery:")
    print("  - Memory cleanup (clear GPU caches)")
    print("  - Emergency cleanup (force garbage collection)")
    print("  - Memory pressure monitoring")
    print()
    print("• Tensor Error Recovery:")
    print("  - Tensor validation and health checking")
    print("  - Shape validation and correction")
    print("  - NaN/Inf detection and handling")
    print()
    print("• Inference Error Recovery:")
    print("  - Retry with backoff")
    print("  - Fallback to uniform policy")
    print("  - Model validation and reloading")
    print()
    print("• Configuration Error Recovery:")
    print("  - Default value fallback")
    print("  - Configuration validation")
    print("  - Type conversion and correction")

    # Test memory recovery
    print("\nTesting memory recovery strategy...")
    memory_error = MemoryError("Test memory pressure")
    context = create_error_context(
        module="demo",
        function="demo_recovery_strategies",
        operation="memory_test",
        category=ErrorCategory.MEMORY,
        severity=ErrorSeverity.CRITICAL
    )

    result = handle_matrix0_error(memory_error, context)
    print(f"Memory recovery result: Success={result.success}, Action={result.recovery_action}")

    print("\n✅ Error recovery strategies demonstrated")


def main():
    """Run all unified error handling demonstrations."""
    print("Matrix0 Unified Error Handling System Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive error handling, classification,")
    print("and recovery system for Matrix0.")
    print()

    # Run all demos
    demo_error_classification()
    demo_error_creation_and_handling()
    demo_decorator_error_handling()
    demo_context_manager_error_handling()
    demo_error_statistics()
    demo_recovery_strategies()

    print("\n" + "=" * 60)
    print("🎉 All unified error handling demonstrations completed!")
    print("=" * 60)
    print("\nThe unified error handling system provides:")
    print("• Comprehensive error classification and categorization")
    print("• Custom exception types for different error scenarios")
    print("• Automatic error recovery with multiple strategies")
    print("• Decorator and context manager error handling")
    print("• Error statistics and monitoring")
    print("• Structured error context and logging")
    print("• Thread-safe error handling operations")
    print("\nError handling patterns available:")
    print("• @with_error_handling decorator for functions")
    print("• safe_operation context manager for blocks")
    print("• Direct error handling with create_error_context")
    print("• Custom recovery strategies by error category")
    print("• Error statistics and performance monitoring")
    print("\nThe error handling system ensures:")
    print("• Consistent error processing across modules")
    print("• Automatic recovery from common failure modes")
    print("• Comprehensive error logging and diagnostics")
    print("• Production-ready reliability and stability")
    print("• Minimal performance impact on successful operations")
    print("\nMatrix0 is now equipped with enterprise-grade error handling! 🛡️")


if __name__ == "__main__":
    main()
