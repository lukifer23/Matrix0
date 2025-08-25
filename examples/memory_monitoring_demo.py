#!/usr/bin/env python3
"""
Memory Monitoring and Alerting System Demo
Demonstrates the comprehensive memory monitoring capabilities of Matrix0.
"""

import sys
import os
import time
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azchess.utils import (
    start_memory_monitoring, stop_memory_monitoring,
    get_memory_stats, add_memory_alert_callback, MemoryAlert,
    clear_memory_cache, get_memory_usage
)


def demo_basic_monitoring():
    """Demonstrate basic memory monitoring."""
    print("=" * 60)
    print("Basic Memory Monitoring Demo")
    print("=" * 60)

    # Get initial memory stats
    memory_info = get_memory_usage('auto')
    print(f"Initial memory usage: {memory_info['memory_gb']:.2f}GB on {memory_info.get('device', 'unknown')}")

    # Clear cache
    print("Clearing memory cache...")
    clear_memory_cache('auto')

    # Get memory stats
    stats = get_memory_stats()
    print(f"Monitoring status: {'Active' if stats['is_monitoring'] else 'Inactive'}")
    print(f"Memory history size: {stats['memory_history_size']}")

    print("✅ Basic monitoring demo complete")


def demo_advanced_monitoring():
    """Demonstrate advanced monitoring with alerts."""
    print("\n" + "=" * 60)
    print("Advanced Memory Monitoring with Alerts Demo")
    print("=" * 60)

    # Custom alert handler
    def custom_alert_handler(alert: MemoryAlert):
        print(f"\n🚨 MEMORY ALERT RECEIVED:")
        print(f"   Type: {alert.alert_type.upper()}")
        print(f"   Message: {alert.message}")
        print(f"   Memory: {alert.memory_usage_gb:.2f}GB / {alert.memory_limit_gb:.2f}GB")
        print(f"   Device: {alert.device}")
        if alert.action_taken:
            print(f"   Action Taken: {alert.action_taken}")
        print()

    # Add custom alert callback
    add_memory_alert_callback(custom_alert_handler)

    # Start monitoring with aggressive thresholds for demo
    print("Starting memory monitoring with aggressive thresholds...")
    start_memory_monitoring(
        device='auto',
        warning_threshold=0.70,   # 70% warning threshold
        critical_threshold=0.85,  # 85% critical threshold
        check_interval=5.0        # Check every 5 seconds
    )

    print("Monitoring started! The system will:")
    print("• Check memory usage every 5 seconds")
    print("• Generate WARNING alerts at 70% usage")
    print("• Generate CRITICAL alerts at 85% usage")
    print("• Automatically take corrective actions")
    print("\nMonitoring for 30 seconds...")

    # Monitor for 30 seconds
    for i in range(6):
        time.sleep(5)
        stats = get_memory_stats()
        if stats['memory_history_size'] > 0:
            current_mem = stats.get('current_memory_gb', 0)
            print(f"  [{i*5:2d}s] Current memory: {current_mem:.2f}GB, History: {stats['memory_history_size']} samples")

    # Stop monitoring
    stop_memory_monitoring()
    print("Monitoring stopped.")

    print("✅ Advanced monitoring demo complete")


def demo_memory_stress_test():
    """Demonstrate memory stress testing capabilities."""
    print("\n" + "=" * 60)
    print("Memory Stress Test Demo")
    print("=" * 60)

    print("This demo shows how the monitoring system handles memory pressure.")

    # Start monitoring with very aggressive thresholds
    start_memory_monitoring(
        device='auto',
        warning_threshold=0.50,   # 50% warning threshold
        critical_threshold=0.70,  # 70% critical threshold
        check_interval=2.0        # Check every 2 seconds
    )

    def stress_memory():
        """Simulate memory usage by creating large tensors."""
        print("Simulating memory stress...")
        try:
            import torch

            # Create some large tensors to increase memory usage
            tensors = []
            for i in range(10):
                # Create a moderately large tensor
                tensor = torch.randn(1000, 1000, dtype=torch.float32)
                tensors.append(tensor)
                print(f"  Created tensor {i+1}/10")
                time.sleep(0.5)  # Small delay

            # Hold for a bit then clean up
            print("Holding memory for 5 seconds...")
            time.sleep(5)

            # Clean up
            del tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except ImportError:
            print("PyTorch not available, skipping tensor stress test")
        except Exception as e:
            print(f"Stress test error: {e}")

    # Run stress test in separate thread
    stress_thread = threading.Thread(target=stress_memory, daemon=True)
    stress_thread.start()

    # Monitor during stress test
    for i in range(10):
        time.sleep(2)
        stats = get_memory_stats()
        if stats['memory_history_size'] > 0:
            current_mem = stats.get('current_memory_gb', 0)
            print(f"  [{i*2:2d}s] Memory: {current_mem:.2f}GB")

    # Stop monitoring
    stop_memory_monitoring()
    print("Stress test complete.")

    print("✅ Memory stress test demo complete")


def main():
    """Run all memory monitoring demonstrations."""
    print("Matrix0 Memory Monitoring and Alerting System Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive memory monitoring capabilities.")
    print()

    # Run all demos
    demo_basic_monitoring()
    demo_advanced_monitoring()
    demo_memory_stress_test()

    print("\n" + "=" * 60)
    print("🎉 All memory monitoring demonstrations completed!")
    print("=" * 60)
    print("\nThe memory monitoring system provides:")
    print("• Real-time memory usage tracking")
    print("• Configurable warning and critical thresholds")
    print("• Automatic corrective actions (cache clearing, emergency cleanup)")
    print("• Custom alert callbacks for application-specific handling")
    print("• Memory usage history and trend analysis")
    print("• Device-specific memory limits and monitoring")
    print("• Background monitoring with minimal performance impact")
    print("\nMemory monitoring helps prevent:")
    print("• Out-of-memory crashes during training")
    print("• Performance degradation due to memory pressure")
    print("• Silent memory leaks and accumulation")
    print("• GPU memory fragmentation issues")
    print("\nUse memory monitoring in production to ensure stable, efficient training!")


if __name__ == "__main__":
    main()
