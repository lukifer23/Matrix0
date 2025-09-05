#!/usr/bin/env python3
"""
Checkpoint Analysis Tool for Matrix0
Analyzes model similarities, distributions, and differences between checkpoints.
"""

import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_checkpoint_safely(path: str) -> Dict[str, Any]:
    """Load checkpoint with error handling."""
    try:
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None

def analyze_checkpoint_structure(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the structure and contents of a checkpoint."""
    analysis = {
        'keys': list(checkpoint.keys()),
        'has_model': 'model' in checkpoint or 'model_state_dict' in checkpoint,
        'has_optimizer': 'optimizer' in checkpoint or 'optimizer_state_dict' in checkpoint,
        'has_scheduler': 'scheduler' in checkpoint or 'scheduler_state_dict' in checkpoint,
        'has_ema': 'ema' in checkpoint or 'model_ema' in checkpoint,
        'metadata': {}
    }

    # Extract model information
    model_key = None
    if 'model' in checkpoint:
        model_key = 'model'
    elif 'model_state_dict' in checkpoint:
        model_key = 'model_state_dict'

    if model_key:
        model_state = checkpoint[model_key]
        analysis['model_keys'] = list(model_state.keys())
        analysis['total_params'] = sum(p.numel() for p in model_state.values() if hasattr(p, 'numel'))
        analysis['param_shapes'] = {k: v.shape if hasattr(v, 'shape') else str(type(v)) for k, v in model_state.items()}

    # Extract metadata
    for key in ['epoch', 'step', 'loss', 'lr', 'timestamp']:
        if key in checkpoint:
            analysis['metadata'][key] = checkpoint[key]

    return analysis

def compare_model_parameters(model1: Dict[str, torch.Tensor], model2: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Compare two model parameter dictionaries."""
    comparison = {
        'common_keys': set(model1.keys()) & set(model2.keys()),
        'only_in_1': set(model1.keys()) - set(model2.keys()),
        'only_in_2': set(model2.keys()) - set(model1.keys()),
        'shape_mismatches': [],
        'parameter_stats': {},
        'similarities': {}
    }

    # Check for shape mismatches
    for key in comparison['common_keys']:
        if hasattr(model1[key], 'shape') and hasattr(model2[key], 'shape'):
            if model1[key].shape != model2[key].shape:
                comparison['shape_mismatches'].append((key, model1[key].shape, model2[key].shape))

    # Compare parameter distributions for common keys
    for key in comparison['common_keys']:
        if hasattr(model1[key], 'numel') and hasattr(model2[key], 'numel'):
            p1, p2 = model1[key].flatten(), model2[key].flatten()

            comparison['parameter_stats'][key] = {
                'shape': p1.shape,
                'mean_1': float(p1.float().mean()),
                'std_1': float(p1.float().std()),
                'mean_2': float(p2.float().mean()),
                'std_2': float(p2.float().std()),
                'min_1': float(p1.min()),
                'max_1': float(p1.max()),
                'min_2': float(p2.min()),
                'max_2': float(p2.max())
            }

            # Cosine similarity
            if p1.numel() == p2.numel():
                similarity = float(torch.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)))
                comparison['similarities'][key] = similarity

    return comparison

def analyze_ssl_heads(model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyze SSL-related parameters in the model."""
    ssl_info = {
        'ssl_heads': {},
        'ssl_parameters': {},
        'total_ssl_params': 0
    }

    # Look for SSL-related parameters
    ssl_keys = [k for k in model_state.keys() if 'ssl' in k.lower()]
    ssl_info['ssl_keys'] = ssl_keys

    for key in ssl_keys:
        param = model_state[key]
        ssl_info['ssl_parameters'][key] = {
            'shape': param.shape,
            'numel': param.numel(),
            'mean': float(param.mean()),
            'std': float(param.std())
        }
        ssl_info['total_ssl_params'] += param.numel()

    return ssl_info

def plot_parameter_distributions(checkpoints: Dict[str, Dict], save_path: str = None):
    """Create plots comparing parameter distributions."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Parameter Distribution Comparison')

        colors = ['blue', 'red', 'green', 'orange']

        # Plot 1: Parameter means comparison
        ax1 = axes[0, 0]
        names = list(checkpoints.keys())
        for i, (name, data) in enumerate(checkpoints.items()):
            if 'parameter_stats' in data:
                means = [stats['mean_1'] for stats in data['parameter_stats'].values()]
                ax1.scatter(range(len(means)), means, label=name, color=colors[i], alpha=0.7)
        ax1.set_title('Parameter Means by Layer')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Mean Value')
        ax1.legend()

        # Plot 2: Parameter stds comparison
        ax2 = axes[0, 1]
        for i, (name, data) in enumerate(checkpoints.items()):
            if 'parameter_stats' in data:
                stds = [stats['std_1'] for stats in data['parameter_stats'].values()]
                ax2.scatter(range(len(stds)), stds, label=name, color=colors[i], alpha=0.7)
        ax2.set_title('Parameter Standard Deviations by Layer')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Std Deviation')
        ax2.legend()

        # Plot 3: SSL parameters comparison
        ax3 = axes[1, 0]
        ssl_params = {}
        for name, data in checkpoints.items():
            if 'ssl_info' in data and data['ssl_info']['total_ssl_params'] > 0:
                ssl_params[name] = data['ssl_info']['total_ssl_params']
        if ssl_params:
            ax3.bar(range(len(ssl_params)), list(ssl_params.values()))
            ax3.set_xticks(range(len(ssl_params)))
            ax3.set_xticklabels(list(ssl_params.keys()), rotation=45)
            ax3.set_title('SSL Parameters Count')
            ax3.set_ylabel('Parameter Count')

        # Plot 4: Model similarities heatmap (if we have comparison data)
        ax4 = axes[1, 1]
        if len(checkpoints) >= 2:
            # This would be a more complex plot - for now just show basic info
            names = list(checkpoints.keys())
            similarities = []
            for i in range(len(names)):
                row = []
                for j in range(len(names)):
                    if i == j:
                        row.append(1.0)  # Self-similarity
                    elif f'{names[i]}_vs_{names[j]}' in checkpoints:
                        # We'd need to calculate this from comparison data
                        row.append(0.8)  # Placeholder
                    else:
                        row.append(0.5)  # Placeholder
                similarities.append(row)

            im = ax4.imshow(similarities, cmap='viridis')
            ax4.set_xticks(range(len(names)))
            ax4.set_yticks(range(len(names)))
            ax4.set_xticklabels(names, rotation=45)
            ax4.set_yticklabels(names)
            ax4.set_title('Model Similarity Matrix')
            plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plots saved to: {save_path}")
        plt.show()

    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Matrix0 model checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "checkpoints/model_step_4000.pt",
            "checkpoints/model_step_18000.pt",
            "checkpoints/best.pt",
            "checkpoints/enhanced_best.pt"
        ],
        help="Paths to checkpoint files to analyze"
    )
    parser.add_argument(
        "--output",
        default="checkpoint_analysis.png",
        help="Output path for analysis plots"
    )

    args = parser.parse_args()

    # Convert relative paths to absolute if needed
    checkpoint_paths = []
    for path in args.checkpoints:
        if not os.path.isabs(path):
            # Try relative to current directory first, then script directory
            if os.path.exists(path):
                checkpoint_paths.append(path)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                abs_path = os.path.join(script_dir, path)
                if os.path.exists(abs_path):
                    checkpoint_paths.append(abs_path)
                else:
                    logger.warning(f"Checkpoint not found: {path}")
        else:
            if os.path.exists(path):
                checkpoint_paths.append(path)
            else:
                logger.warning(f"Checkpoint not found: {path}")

    checkpoints = {}
    comparisons = {}

    # Load and analyze each checkpoint
    for path in checkpoint_paths:
        checkpoint = load_checkpoint_safely(path)
        if checkpoint:
            name = Path(path).stem
            analysis = analyze_checkpoint_structure(checkpoint)
            checkpoints[name] = analysis

            # Extract model state for comparison
            model_state = None
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']

            if model_state:
                checkpoints[name]['model_state'] = model_state
                ssl_info = analyze_ssl_heads(model_state)
                checkpoints[name]['ssl_info'] = ssl_info

                logger.info(f"\n=== {name} Analysis ===")
                logger.info(f"Total parameters: {analysis.get('total_params', 'N/A')}")
                logger.info(f"SSL parameters: {ssl_info['total_ssl_params']}")
                logger.info(f"SSL keys: {ssl_info['ssl_keys']}")
                if analysis.get('metadata'):
                    logger.info(f"Metadata: {analysis['metadata']}")

    # Compare checkpoints pairwise
    names = list(checkpoints.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            if 'model_state' in checkpoints[name1] and 'model_state' in checkpoints[name2]:
                comparison = compare_model_parameters(
                    checkpoints[name1]['model_state'],
                    checkpoints[name2]['model_state']
                )
                comp_name = f'{name1}_vs_{name2}'
                comparisons[comp_name] = comparison

                logger.info(f"\n=== {name1} vs {name2} Comparison ===")
                logger.info(f"Common parameters: {len(comparison['common_keys'])}")
                logger.info(f"Only in {name1}: {len(comparison['only_in_1'])}")
                logger.info(f"Only in {name2}: {len(comparison['only_in_2'])}")
                logger.info(f"Shape mismatches: {len(comparison['shape_mismatches'])}")

                if comparison['shape_mismatches']:
                    logger.warning("Shape mismatches found:")
                    for key, shape1, shape2 in comparison['shape_mismatches'][:5]:  # Show first 5
                        logger.warning(f"  {key}: {shape1} vs {shape2}")

    # Create plots
    try:
        plot_parameter_distributions(checkpoints, save_path=args.output)
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("CHECKPOINT ANALYSIS SUMMARY")
    logger.info("="*60)

    for name, data in checkpoints.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Total params: {data.get('total_params', 'N/A')}")
        logger.info(f"  SSL params: {data['ssl_info']['total_ssl_params']}")
        logger.info(f"  SSL heads: {len([k for k in data['ssl_info']['ssl_keys'] if 'ssl' in k])}")
        if data.get('metadata'):
            logger.info(f"  Metadata: {data['metadata']}")

    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    main()
