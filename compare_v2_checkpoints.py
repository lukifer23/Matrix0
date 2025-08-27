#!/usr/bin/env python3
"""
Compare v2_base.pt and v2_fresh_clean.pt to verify they're functionally identical.
"""

import torch
import os

def compare_checkpoints(path1: str, path2: str):
    """Compare two checkpoint files."""
    print("🔍 Comparing V2 Checkpoints")
    print("=" * 50)

    # Load both checkpoints
    print(f"Loading {path1}...")
    ckpt1 = torch.load(path1, map_location='cpu', weights_only=False)

    print(f"Loading {path2}...")
    ckpt2 = torch.load(path2, map_location='cpu', weights_only=False)

    # Compare top-level keys
    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())

    print(f"\n📋 Top-level keys comparison:")
    print(f"  {path1}: {sorted(keys1)}")
    print(f"  {path2}: {sorted(keys2)}")
    print(f"  Keys match: {keys1 == keys2}")

    # Compare model state dict
    if 'model_state_dict' in ckpt1 and 'model_state_dict' in ckpt2:
        model1 = ckpt1['model_state_dict']
        model2 = ckpt2['model_state_dict']

        keys_m1 = set(model1.keys())
        keys_m2 = set(model2.keys())

        print(f"\n🏗️  Model parameters comparison:")
        print(f"  {path1}: {len(keys_m1)} parameters")
        print(f"  {path2}: {len(keys_m2)} parameters")
        print(f"  Parameter keys match: {keys_m1 == keys_m2}")

        # Check for SSL heads
        ssl_keys1 = [k for k in keys_m1 if 'ssl' in k.lower()]
        ssl_keys2 = [k for k in keys_m2 if 'ssl' in k.lower()]

        print(f"\n🔬 SSL parameters:")
        print(f"  {path1}: {len(ssl_keys1)} SSL parameters - {ssl_keys1}")
        print(f"  {path2}: {len(ssl_keys2)} SSL parameters - {ssl_keys2}")

        # Compare metadata
        if 'version' in ckpt1 and 'version' in ckpt2:
            print(f"\n📝 Version info:")
            print(f"  {path1}: {ckpt1.get('version', 'N/A')}")
            print(f"  {path2}: {ckpt2.get('version', 'N/A')}")

        if 'ssl_info' in ckpt1 and 'ssl_info' in ckpt2:
            print(f"\n🔧 SSL Configuration:")
            print(f"  {path1}: {ckpt1['ssl_info']}")
            print(f"  {path2}: {ckpt2['ssl_info']}")

    print(f"\n✅ Files are functionally identical: {ckpt1 == ckpt2}")

    return ckpt1 == ckpt2

if __name__ == "__main__":
    path1 = "/Users/admin/Downloads/VSCode/Matrix0/checkpoints/v2_base.pt"
    path2 = "/Users/admin/Downloads/VSCode/Matrix0/checkpoints/v2_fresh_clean.pt"

    if os.path.exists(path1) and os.path.exists(path2):
        compare_checkpoints(path1, path2)
    else:
        print("❌ One or both checkpoint files not found")
