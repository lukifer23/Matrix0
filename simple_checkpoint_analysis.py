#!/usr/bin/env python3
"""
Simple Checkpoint Analysis Tool for Matrix0
Basic analysis without torch/matplotlib dependencies.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_checkpoint_basic(path: str) -> dict:
    """Load checkpoint using pickle (basic analysis)."""
    try:
        logger.info(f"Loading checkpoint: {path}")
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None

def analyze_checkpoint_basic(checkpoint: dict) -> dict:
    """Basic analysis of checkpoint structure."""
    analysis = {
        'keys': list(checkpoint.keys()),
        'structure': {},
        'sizes': {},
        'types': {}
    }

    for key, value in checkpoint.items():
        analysis['types'][key] = str(type(value))

        if hasattr(value, '__len__') and not isinstance(value, str):
            try:
                analysis['sizes'][key] = len(value)
            except:
                analysis['sizes'][key] = 'unknown'

        # Special handling for nested structures
        if isinstance(value, dict):
            analysis['structure'][key] = {
                'keys': list(value.keys()),
                'num_keys': len(value),
                'nested_types': {k: str(type(v)) for k, v in value.items()}
            }
            # Count parameters if it's a model state dict
            if key in ['model', 'model_state_dict'] or 'state_dict' in key:
                total_params = 0
                param_shapes = {}
                for param_key, param_value in value.items():
                    if hasattr(param_value, 'shape'):
                        param_shapes[param_key] = param_value.shape
                        try:
                            total_params += int(np.prod(param_value.shape))
                        except:
                            pass
                analysis['structure'][key]['total_params'] = total_params
                analysis['structure'][key]['param_shapes'] = param_shapes
        else:
            analysis['structure'][key] = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)

    return analysis

def compare_checkpoints_basic(cp1: dict, cp2: dict, name1: str, name2: str) -> dict:
    """Compare two checkpoints at a basic level."""
    comparison = {
        'common_keys': set(cp1.keys()) & set(cp2.keys()),
        'only_in_1': set(cp1.keys()) - set(cp2.keys()),
        'only_in_2': set(cp2.keys()) - set(cp1.keys()),
        'differences': {}
    }

    logger.info(f"\n=== Comparing {name1} vs {name2} ===")
    logger.info(f"Common keys: {len(comparison['common_keys'])}")
    logger.info(f"Only in {name1}: {list(comparison['only_in_1'])}")
    logger.info(f"Only in {name2}: {list(comparison['only_in_2'])}")

    # Compare model parameters if available
    model_keys = ['model', 'model_state_dict']
    for model_key in model_keys:
        if model_key in cp1 and model_key in cp2:
            logger.info(f"\nComparing {model_key}:")
            m1, m2 = cp1[model_key], cp2[model_key]
            common_params = set(m1.keys()) & set(m2.keys())
            only_m1 = set(m1.keys()) - set(m2.keys())
            only_m2 = set(m2.keys()) - set(m2.keys())

            logger.info(f"  Common parameters: {len(common_params)}")
            logger.info(f"  Only in {name1}: {len(only_m1)}")
            logger.info(f"  Only in {name2}: {len(only_m2)}")

            # Check for shape differences
            shape_diffs = []
            for param_key in common_params:
                if hasattr(m1[param_key], 'shape') and hasattr(m2[param_key], 'shape'):
                    if m1[param_key].shape != m2[param_key].shape:
                        shape_diffs.append((param_key, m1[param_key].shape, m2[param_key].shape))

            if shape_diffs:
                logger.warning(f"  Shape mismatches: {len(shape_diffs)}")
                for param_key, shape1, shape2 in shape_diffs[:5]:
                    logger.warning(f"    {param_key}: {shape1} vs {shape2}")
            else:
                logger.info("  No shape mismatches found")

    return comparison

def check_ssl_parameters(checkpoint: dict, name: str):
    """Check for SSL-related parameters."""
    logger.info(f"\n=== SSL Analysis for {name} ===")

    ssl_related = []
    for key in checkpoint.keys():
        if 'ssl' in key.lower():
            ssl_related.append(key)

    if ssl_related:
        logger.info(f"SSL-related keys found: {ssl_related}")
        for key in ssl_related:
            if isinstance(checkpoint[key], dict):
                logger.info(f"  {key}: {len(checkpoint[key])} parameters")
                # Look for SSL task heads
                ssl_heads = [k for k in checkpoint[key].keys() if any(task in k.lower() for task in ['piece', 'threat', 'pin', 'fork', 'control', 'pawn', 'king'])]
                if ssl_heads:
                    logger.info(f"    SSL heads: {ssl_heads}")
            else:
                logger.info(f"  {key}: {type(checkpoint[key])}")
    else:
        logger.info("No SSL-related keys found")

def main():
    """Main analysis function."""
    checkpoint_paths = [
        "/Users/admin/Downloads/VSCode/Matrix0/checkpoints/model_step_4000.pt",
        "/Users/admin/Downloads/VSCode/Matrix0/checkpoints/model_step_18000.pt",
        "/Users/admin/Downloads/VSCode/Matrix0/checkpoints/best.pt",
        "/Users/admin/Downloads/VSCode/Matrix0/checkpoints/enhanced_best.pt"
    ]

    checkpoints = {}

    # Load and analyze each checkpoint
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint = load_checkpoint_basic(path)
            if checkpoint:
                name = Path(path).stem
                analysis = analyze_checkpoint_basic(checkpoint)
                checkpoints[name] = {
                    'checkpoint': checkpoint,
                    'analysis': analysis
                }

                logger.info(f"\n=== {name} Basic Analysis ===")
                logger.info(f"Top-level keys: {analysis['keys']}")
                logger.info(f"Key types: {analysis['types']}")

                # Check for model information
                model_key = None
                if 'model' in checkpoint:
                    model_key = 'model'
                elif 'model_state_dict' in checkpoint:
                    model_key = 'model_state_dict'

                if model_key and model_key in analysis['structure']:
                    model_info = analysis['structure'][model_key]
                    if isinstance(model_info, dict) and 'total_params' in model_info:
                        logger.info(f"Model parameters: {model_info['total_params']}")
                        logger.info(f"Model keys: {len(model_info.get('keys', []))}")

                # Check SSL parameters
                check_ssl_parameters(checkpoint, name)

        else:
            logger.warning(f"Checkpoint not found: {path}")

    # Compare checkpoints
    names = list(checkpoints.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            cp1 = checkpoints[name1]['checkpoint']
            cp2 = checkpoints[name2]['checkpoint']
            compare_checkpoints_basic(cp1, cp2, name1, name2)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("CHECKPOINT ANALYSIS SUMMARY")
    logger.info("="*60)

    for name, data in checkpoints.items():
        analysis = data['analysis']
        logger.info(f"\n{name}:")
        logger.info(f"  Keys: {len(analysis['keys'])}")
        logger.info(f"  Types: {list(analysis['types'].values())}")

        # Extract model info if available
        for key, info in analysis['structure'].items():
            if isinstance(info, dict) and 'total_params' in info:
                logger.info(f"  {key}: {info['total_params']} parameters")

    logger.info("\nBasic analysis complete!")
    logger.info("\nRecommendations:")
    logger.info("1. Comparing checkpoints DOES make sense - they track training progress")
    logger.info("2. Check if SSL parameters are present and consistent")
    logger.info("3. Verify checkpoint creation script handles new SSL heads")
    logger.info("4. Consider updating checkpoint metadata for better tracking")

if __name__ == "__main__":
    main()
