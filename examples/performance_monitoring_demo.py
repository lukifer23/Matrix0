#!/usr/bin/env python3
"""
Unified Performance Monitoring System Demo
Showcases the comprehensive performance monitoring and metrics collection for Matrix0.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azchess.utils import (
    TimingContext,
    add_performance_alert_callback,
    end_timing,
    get_performance_report,
    get_performance_stats,
    increment_counter,
    performance_monitor,
    record_metric,
    set_gauge,
    set_performance_threshold,
    start_timing,
    time_operation,
)


def demo_basic_timing():
    """Demonstrate basic timing measurements."""
    print("=" * 60)
    print("Basic Timing Demo")
    print("=" * 60)

    # Method 1: Manual timing
    print("Method 1: Manual timing")
    start_timing("manual_operation", category="demo")

    time.sleep(0.1)  # Simulate work

    duration = end_timing("manual_operation")
    print(f"Manual timing result: {duration:.4f}s")

    # Method 2: Context manager
    print("\nMethod 2: Context manager timing")
    with TimingContext("context_operation", category="demo"):
        time.sleep(0.05)  # Simulate work

    print("Context manager timing completed automatically")

    # Method 3: Decorator
    print("\nMethod 3: Decorator timing")

    @time_operation("decorated_function", category="demo")
    def sample_function():
        time.sleep(0.02)
        return "result"

    result = sample_function()
    print(f"Decorated function result: {result}")

    print("\n✅ Basic timing demonstrated")


def demo_metrics_collection():
    """Demonstrate metrics collection and monitoring."""
    print("\n" + "=" * 60)
    print("Metrics Collection Demo")
    print("=" * 60)

    print("Recording various types of metrics...")

    # Record some metrics
    record_metric("memory_usage", 1024.5, category="system")
    record_metric("cpu_usage", 75.2, category="system")
    record_metric("model_accuracy", 0.923, category="ml")

    # Counters
    increment_counter("requests_processed", 5)
    increment_counter("errors_encountered", 1)
    increment_counter("cache_hits", 10)

    # Gauges
    set_gauge("active_connections", 25.0)
    set_gauge("queue_length", 5.0)

    print("Metrics recorded successfully")
    print("\n✅ Metrics collection demonstrated")


def demo_performance_alerts():
    """Demonstrate performance alerts and thresholds."""
    print("\n" + "=" * 60)
    print("Performance Alerts Demo")
    print("=" * 60)

    # Custom alert handler
    def performance_alert_handler(message):
        print(f"🚨 PERFORMANCE ALERT: {message}")

    # Add alert callback
    add_performance_alert_callback(performance_alert_handler)

    print("Setting performance thresholds...")

    # Set thresholds
    set_performance_threshold("slow_operation", 0.1)  # Alert if > 0.1s
    set_performance_threshold("high_memory", 2000.0)  # Alert if > 2000 MB

    print("Testing threshold alerts...")

    # This should trigger an alert
    start_timing("slow_operation", category="demo")
    time.sleep(0.15)  # Exceed threshold
    end_timing("slow_operation")

    # This should trigger an alert
    record_metric("high_memory", 2500.0, category="system")

    print("\n✅ Performance alerts demonstrated")


def demo_statistics_and_reporting():
    """Demonstrate statistics collection and reporting."""
    print("\n" + "=" * 60)
    print("Statistics and Reporting Demo")
    print("=" * 60)

    print("Generating sample performance data...")

    # Generate some sample data
    for i in range(50):
        with TimingContext(f"operation_{i}", category="sample"):
            # Simulate variable execution time
            time.sleep(random.uniform(0.001, 0.01))

        record_metric("sample_metric", random.uniform(10, 100), category="sample")
        increment_counter("sample_counter", random.randint(1, 5))

    print("Sample data generated")
    print()

    # Get statistics
    print("Getting performance statistics...")
    all_stats = get_performance_stats()
    print(f"Total metrics collected: {all_stats['total_metrics']}")

    timing_stats = get_performance_stats(category="timing")
    if 'by_metric' in timing_stats:
        for name, stats in timing_stats['by_metric'].items():
            print(f"  {name}: mean={stats['mean']:.4f}s, count={stats['count']}")

    print()

    # Get performance report
    print("Generating performance report...")
    report = get_performance_report(time_window=60)
    print("Performance Report:")
    print(report)

    print("\n✅ Statistics and reporting demonstrated")


def demo_cross_module_integration():
    """Demonstrate how performance monitoring integrates across modules."""
    print("\n" + "=" * 60)
    print("Cross-Module Integration Demo")
    print("=" * 60)

    print("Simulating performance monitoring across different Matrix0 modules:")
    print()

    # Training module simulation
    print("1. Training Module:")
    with TimingContext("epoch_training", category="training"):
        time.sleep(0.1)  # Simulate training time

    record_metric("training_loss", 0.234, category="training")
    increment_counter("training_steps", 100)
    set_gauge("learning_rate", 0.001)

    print("   ✓ Epoch training timed and metrics recorded")
    print()

    # MCTS module simulation
    print("2. MCTS Module:")
    with TimingContext("mcts_search", category="mcts"):
        time.sleep(0.05)  # Simulate MCTS time

    record_metric("mcts_simulations_per_second", 1500.0, category="mcts")
    increment_counter("mcts_nodes_expanded", 800)
    set_gauge("mcts_tree_size", 1200.0)

    print("   ✓ MCTS search timed and metrics recorded")
    print()

    # Inference module simulation
    print("3. Inference Module:")
    with TimingContext("batch_inference", category="inference"):
        time.sleep(0.02)  # Simulate inference time

    record_metric("inference_batch_size", 32.0, category="inference")
    increment_counter("inference_requests", 50)
    set_gauge("gpu_memory_usage", 2048.0)

    print("   ✓ Batch inference timed and metrics recorded")
    print()

    # Get integrated statistics
    print("4. Integrated Statistics:")
    integrated_stats = get_performance_stats()

    print(f"   Total metrics across modules: {integrated_stats['total_metrics']}")
    print(f"   Categories tracked: {list(set(m.category for m in performance_monitor._metrics_history))}")

    print("\n✅ Cross-module integration demonstrated")


def main():
    """Run all performance monitoring demonstrations."""
    print("Matrix0 Unified Performance Monitoring System Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive performance monitoring,")
    print("metrics collection, and alerting system for Matrix0.")
    print()

    # Run all demos
    demo_basic_timing()
    demo_metrics_collection()
    demo_performance_alerts()
    demo_statistics_and_reporting()
    demo_cross_module_integration()

    print("\n" + "=" * 60)
    print("🎉 All performance monitoring demonstrations completed!")
    print("=" * 60)
    print("\nThe unified performance monitoring system provides:")
    print("• Comprehensive timing measurements with automatic recording")
    print("• Flexible metrics collection (counters, gauges, histograms)")
    print("• Configurable performance thresholds and alerts")
    print("• Detailed statistics and performance reporting")
    print("• Cross-module integration and consistency")
    print("• Thread-safe operations with minimal overhead")
    print("• Historical data retention and trend analysis")
    print("\nPerformance monitoring features:")
    print("• Manual timing: start_timing() / end_timing()")
    print("• Context managers: TimingContext")
    print("• Decorators: @time_operation")
    print("• Metrics recording: record_metric(), increment_counter(), set_gauge()")
    print("• Alerting: set_performance_threshold(), alert callbacks")
    print("• Reporting: get_performance_stats(), get_performance_report()")
    print("\nThe system enables:")
    print("• Real-time performance tracking across all modules")
    print("• Automated bottleneck detection and alerting")
    print("• Performance regression monitoring")
    print("• Comprehensive debugging and optimization support")
    print("• Production-ready performance monitoring")
    print("\nMatrix0 now has enterprise-grade performance monitoring! 📊")


if __name__ == "__main__":
    main()
