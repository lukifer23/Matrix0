#!/usr/bin/env python3
"""
Quick test to verify value targets are working correctly in teacher data.
"""

import numpy as np
import os
from pathlib import Path

def test_value_targets():
    """Test that value targets in teacher shards are properly distributed."""

    # Check the most recent teacher shard
    teacher_dir = Path('data/teacher_games/ssl_enhanced_teacher_batch_4')
    if not teacher_dir.exists():
        print("❌ No teacher data directory found")
        return

    # Find the most recent shard
    shards = list(teacher_dir.glob('*.npz'))
    if not shards:
        print("❌ No teacher shards found")
        return

    latest_shard = max(shards, key=lambda x: x.stat().st_mtime)
    print(f"📊 Testing value targets in: {latest_shard.name}")

    # Load and analyze
    data = np.load(latest_shard)
    z_values = data['z'].flatten()

    print(f"Total positions: {len(z_values)}")

    # Analyze distribution
    win_count = np.sum(z_values == 1.0)
    loss_count = np.sum(z_values == -1.0)
    draw_count = np.sum(z_values == 0.0)

    print("
🎯 Value Target Distribution:"    print(f"  Wins (z=1.0): {win_count:,} ({win_count/len(z_values)*100:.1f}%)")
    print(f"  Losses (z=-1.0): {loss_count:,} ({loss_count/len(z_values)*100:.1f}%)")
    print(f"  Draws (z=0.0): {draw_count:,} ({draw_count/len(z_values)*100:.1f}%)")

    # Check if distribution looks reasonable
    total_games = len(z_values)
    if win_count > 0 and loss_count > 0:
        print("
✅ SUCCESS: Value targets are properly distributed!"        print("   Both wins and losses are present - game outcomes are being tracked correctly.")
    else:
        print("
❌ ISSUE: Value targets appear incorrect"        print("   Missing wins or losses - may need to regenerate data.")

    # Show sample values
    print("\n📋 Sample z values:")
    print(f"   First 10: {z_values[:10]}")
    print(f"   Last 10: {z_values[-10:]}")
    print(f"   Unique values: {np.unique(z_values)}")
if __name__ == "__main__":
    test_value_targets()
