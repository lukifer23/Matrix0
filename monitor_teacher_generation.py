#!/usr/bin/env python3
"""
Monitor teacher data generation progress and notify when complete.
"""

import time
import sqlite3
from pathlib import Path

def check_teacher_generation_progress():
    """Check the current progress of teacher data generation."""
    db_path = "data/data_metadata.db"
    if not Path(db_path).exists():
        print("❌ Database not found")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get current teacher data samples
    cursor.execute('SELECT SUM(sample_count) FROM shards WHERE source = "teacher:enhanced_best_teacher_50k" AND corrupted = FALSE')
    result = cursor.fetchone()
    current_samples = result[0] if result[0] else 0
    
    target_samples = 50000
    progress_percent = (current_samples / target_samples) * 100
    remaining_samples = target_samples - current_samples
    
    print(f"📊 Teacher Data Generation Progress:")
    print(f"   Current: {current_samples:,} / {target_samples:,} samples")
    print(f"   Progress: {progress_percent:.1f}%")
    print(f"   Remaining: {remaining_samples:,} samples")
    
    # Estimate completion time
    if current_samples > 0:
        # Rough estimate: each file has ~1000 samples, takes ~2.5 minutes
        estimated_remaining_files = remaining_samples // 1000
        estimated_time_minutes = estimated_remaining_files * 2.5
        print(f"   Estimated time to completion: {estimated_time_minutes:.0f} minutes ({estimated_time_minutes/60:.1f} hours)")
    
    conn.close()
    
    return current_samples >= target_samples

def main():
    """Monitor teacher data generation until complete."""
    print("🔍 Monitoring teacher data generation...")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            is_complete = check_teacher_generation_progress()
            
            if is_complete:
                print("\n🎉 Teacher data generation is COMPLETE!")
                print("✅ Ready to start training with 2,787 steps (3 epochs)")
                break
            
            print(f"⏳ Waiting... (checked at {time.strftime('%H:%M:%S')})")
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

if __name__ == "__main__":
    main()
