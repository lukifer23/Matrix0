#!/usr/bin/env python3
"""
Unified Logging System Demo
Showcases the comprehensive logging utilities and standardized patterns for Matrix0.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azchess.utils import (
    LogOperation,
    create_log_context,
    get_logger,
    log_error_details,
    log_function,
    log_metrics,
    log_print,
    log_progress,
    log_system_info,
    set_logging_level,
    setup_logging,
)


def demo_basic_logging():
    """Demonstrate basic logging patterns."""
    print("=" * 60)
    print("Basic Logging Demo")
    print("=" * 60)

    # Setup logging
    logger = setup_logging(level=20)  # INFO level

    # Get named loggers
    training_logger = get_logger("training")
    mcts_logger = get_logger("mcts")
    inference_logger = get_logger("inference")

    print("Loggers created:")
    print("• training logger")
    print("• mcts logger")
    print("• inference logger")

    # Log different levels
    training_logger.info("Training module initialized")
    mcts_logger.debug("MCTS debug information")
    inference_logger.warning("Inference warning")
    training_logger.error("Training error occurred")

    print("\n✅ Basic logging demonstrated")


def demo_structured_logging():
    """Demonstrate structured logging with context."""
    print("\n" + "=" * 60)
    print("Structured Logging Demo")
    print("=" * 60)

    # Create structured logging contexts
    training_ctx = create_log_context("training", category="model_training", epoch=42, batch_size=192)
    mcts_ctx = create_log_context("mcts", category="search", simulations=800, cpuct=2.2)

    print("Created structured logging contexts:")
    print("• Training context (epoch=42, batch_size=192)")
    print("• MCTS context (simulations=800, cpuct=2.2)")

    # Log with structured data
    training_ctx.info("Epoch completed", loss=0.234, accuracy=0.923, lr=0.001)
    mcts_ctx.info("Search completed", visits=800, value=0.156, best_move="e2e4")

    print("\n✅ Structured logging demonstrated")


def demo_operation_logging():
    """Demonstrate operation logging with context managers."""
    print("\n" + "=" * 60)
    print("Operation Logging Demo")
    print("=" * 60)

    def simulate_training_epoch():
        with LogOperation("training_epoch", level="info"):
            time.sleep(0.1)  # Simulate work
            return {"loss": 0.234, "accuracy": 0.923}

    def simulate_mcts_search():
        with LogOperation("mcts_search", level="info"):
            time.sleep(0.05)  # Simulate work
            return {"visits": 800, "value": 0.156}

    print("Running operations with automatic logging:")
    result1 = simulate_training_epoch()
    result2 = simulate_mcts_search()

    print(f"Training result: {result1}")
    print(f"MCTS result: {result2}")

    print("\n✅ Operation logging demonstrated")


def demo_print_replacements():
    """Demonstrate print statement replacements."""
    print("\n" + "=" * 60)
    print("Print Statement Replacements Demo")
    print("=" * 60)

    print("Replacing common print patterns:")

    # Progress logging
    log_progress(25, 100, "Processing data")
    log_progress(50, 100, "Halfway complete")
    log_progress(100, 100, "Processing complete")

    # Metrics logging
    metrics = {
        "loss": 0.234,
        "accuracy": 0.923,
        "learning_rate": 0.001,
        "epoch": 42
    }
    log_metrics(metrics, prefix="Training metrics: ")

    # Simple print replacements
    log_print("This replaces a print statement", level="info")
    log_print("This is a warning message", level="warning")
    log_print("This is an error message", level="error")

    print("\n✅ Print replacements demonstrated")


def demo_function_decorator():
    """Demonstrate function logging decorator."""
    print("\n" + "=" * 60)
    print("Function Logging Decorator Demo")
    print("=" * 60)

    @log_function(level="info")
    def train_step(batch_data, model_params):
        """Simulate a training step."""
        time.sleep(0.02)  # Simulate computation
        return {"loss": 0.234, "gradients": "computed"}

    @log_function(level="debug", logger_name="custom_module")
    def validate_model(validation_data):
        """Simulate model validation."""
        time.sleep(0.01)  # Simulate validation
        return {"accuracy": 0.923, "f1_score": 0.891}

    print("Running decorated functions:")
    result1 = train_step({"size": 192}, {"lr": 0.001})
    result2 = validate_model({"size": 1000})

    print(f"Train step result: {result1}")
    print(f"Validation result: {result2}")

    print("\n✅ Function decorator demonstrated")


def demo_error_logging():
    """Demonstrate enhanced error logging."""
    print("\n" + "=" * 60)
    print("Enhanced Error Logging Demo")
    print("=" * 60)

    def risky_operation():
        """Simulate an operation that might fail."""
        if True:  # Simulate error condition
            raise ValueError("Simulated validation error in tensor processing")

    print("Testing error logging:")
    try:
        risky_operation()
    except Exception as e:
        log_error_details(e, include_traceback=False)
        print("Error details logged successfully")

    print("\n✅ Enhanced error logging demonstrated")


def demo_cross_module_consistency():
    """Demonstrate consistent logging across different modules."""
    print("\n" + "=" * 60)
    print("Cross-Module Consistency Demo")
    print("=" * 60)

    print("Simulating logging from different Matrix0 modules:")

    # Training module logging
    training_ctx = create_log_context("training", category="model", model="resnet")
    training_ctx.info("Model training started", epochs=100, batch_size=192)

    # MCTS module logging
    mcts_ctx = create_log_context("mcts", category="search", algorithm="alphazero")
    mcts_ctx.info("MCTS search initialized", simulations=800, threads=4)

    # Inference module logging
    inference_ctx = create_log_context("inference", category="prediction", device="mps")
    inference_ctx.info("Inference engine ready", batch_size=32, precision="fp16")

    # Data loading module logging
    data_ctx = create_log_context("data", category="loading", dataset="lichess")
    data_ctx.info("Dataset loaded", samples=1000000, features=13)

    print("\nAll modules use consistent logging patterns:")
    print("• Standardized logger names")
    print("• Structured context data")
    print("• Consistent message formats")
    print("• Unified error handling")
    print("• Centralized configuration")

    print("\n✅ Cross-module consistency demonstrated")


def demo_performance_integration():
    """Demonstrate integration with performance monitoring."""
    print("\n" + "=" * 60)
    print("Performance Monitoring Integration Demo")
    print("=" * 60)

    print("Logging integrates with performance monitoring:")
    print("• System information logging")
    print("• Performance statistics reporting")
    print("• Error logging with performance context")
    print("• Structured logging with timing data")
    print("• Debug logging for performance bottlenecks")

    # Note: Full performance integration requires performance monitoring to be active
    print("\n✅ Performance integration framework demonstrated")


def main():
    """Run all unified logging demonstrations."""
    print("Matrix0 Unified Logging System Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive logging utilities,")
    print("structured logging patterns, and print statement replacements.")
    print()

    # Run all demos
    demo_basic_logging()
    demo_structured_logging()
    demo_operation_logging()
    demo_print_replacements()
    demo_function_decorator()
    demo_error_logging()
    demo_cross_module_consistency()
    demo_performance_integration()

    print("\n" + "=" * 60)
    print("🎉 All unified logging demonstrations completed!")
    print("=" * 60)
    print("\nThe unified logging system provides:")
    print("• Consistent logger configuration across modules")
    print("• Structured logging with context and metadata")
    print("• Operation logging with automatic timing")
    print("• Print statement replacements with proper logging")
    print("• Function decorators for automatic logging")
    print("• Enhanced error logging with traceback support")
    print("• Cross-module consistency and standardization")
    print("• Performance monitoring integration")
    print("\nLogging patterns available:")
    print("• Basic logging: get_logger(), setup_logging()")
    print("• Structured logging: create_log_context(), LogContext")
    print("• Operation logging: LogOperation context manager")
    print("• Function logging: @log_function decorator")
    print("• Print replacements: log_print(), log_progress(), log_metrics()")
    print("• Error logging: log_error_details()")
    print("• Performance logging: log_performance_stats()")
    print("\nThe logging system enables:")
    print("• Consistent log formats across all modules")
    print("• Structured data in log messages")
    print("• Better debugging and monitoring")
    print("• Performance analysis and profiling")
    print("• Production-ready logging infrastructure")
    print("\nMatrix0 now has enterprise-grade logging! 📝")


if __name__ == "__main__":
    main()
