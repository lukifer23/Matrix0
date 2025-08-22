import numpy as np
import os

print("Training Data Analysis")
print("=" * 50)

# Check openings data
print("\nOpenings data:")
try:
    d = np.load('/Users/admin/Downloads/VSCode/Matrix0/data/training/openings_training_data.npz')
    print(f"Keys: {list(d.keys())}")
    for k in d.keys():
        print(f"{k}: shape {d[k].shape}, dtype {d[k].dtype}")
    d.close()
except Exception as e:
    print(f"Error loading openings data: {e}")

# Check tactical data
print("\nTactical data:")
try:
    d = np.load('/Users/admin/Downloads/VSCode/Matrix0/data/training/tactical_training_data.npz')
    print(f"Keys: {list(d.keys())}")
    for k in d.keys():
        print(f"{k}: shape {d[k].shape}, dtype {d[k].dtype}")
    d.close()
except Exception as e:
    print(f"Error loading tactical data: {e}")

print("\nData validation complete!")
