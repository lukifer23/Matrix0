#!/usr/bin/env python3
"""
SSL Data Enhancement Tool for Matrix0

This script adds SSL targets to existing NPZ data files to enable
pre-computed SSL training and reduce training-time computation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from azchess.config import Config, select_device
from azchess.logging_utils import setup_logging
from azchess.model import PolicyValueNet

logger = setup_logging(level=logging.INFO)


class SSLDataEnhancer:
    """Enhances existing data files with SSL targets."""

    def __init__(self, config_path: str = "config.yaml", device: str = "auto"):
        self.config = Config.load(config_path)
        self.device = select_device(device)

        # Load model for SSL target generation
        self.model = PolicyValueNet.from_config(self.config.model())
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"SSL Data Enhancer initialized on device: {self.device}")
        logger.info(f"SSL tasks enabled: {self.config.model().get('ssl_tasks', ['piece'])}")

    def enhance_npz_file(self, npz_path: Path, force: bool = False) -> bool:
        """Add SSL targets to an existing NPZ file.

        Args:
            npz_path: Path to the NPZ file to enhance
            force: Whether to overwrite existing SSL targets

        Returns:
            True if file was enhanced, False otherwise
        """
        if not npz_path.exists():
            logger.warning(f"File not found: {npz_path}")
            return False

        try:
            # Load existing data
            with np.load(npz_path) as data:
                # Handle different key formats
                if 's' in data:
                    states = data['s']  # Standard format
                elif 'positions' in data:
                    states = data['positions']  # Alternative format
                else:
                    logger.error(f"No board state key found in {npz_path.name}")
                    return False

                logger.info(f"Loaded {len(states)} samples from {npz_path.name}")

                # Check if SSL targets already exist
                ssl_keys = [k for k in data.keys() if k.startswith('ssl_')]
                if ssl_keys and not force:
                    logger.info(f"SSL targets already exist in {npz_path.name}: {ssl_keys}")
                    return False

                # Convert to torch tensor
                states_tensor = torch.from_numpy(states).to(self.device, dtype=torch.float32)

                # Generate SSL targets in batches to manage memory
                batch_size = 64
                ssl_targets = self.model.create_ssl_targets_batch(states_tensor, batch_size)

                # Prepare data for saving
                save_data = {k: v for k, v in data.items()}  # Copy existing data

                # Add SSL targets with 'ssl_' prefix
                for task_name, target_tensor in ssl_targets.items():
                    key = f'ssl_{task_name}'
                    # Convert back to numpy for saving
                    save_data[key] = target_tensor.detach().cpu().numpy()
                    logger.debug(f"Added SSL target '{key}' with shape {save_data[key].shape}")

                # Save enhanced file
                temp_path = npz_path.with_suffix('.npz.tmp')
                np.savez_compressed(temp_path, **save_data)

                # Atomic move
                temp_path.replace(npz_path)

                logger.info(f"Enhanced {npz_path.name} with SSL targets: {list(ssl_targets.keys())}")
                return True

        except Exception as e:
            logger.error(f"Failed to enhance {npz_path}: {e}")
            return False

    def enhance_directory(self, directory: Path, pattern: str = "*.npz",
                         force: bool = False, max_files: Optional[int] = None) -> int:
        """Enhance all NPZ files in a directory with SSL targets.

        Args:
            directory: Directory to scan for NPZ files
            pattern: File pattern to match
            force: Whether to overwrite existing SSL targets
            max_files: Maximum number of files to process (None for all)

        Returns:
            Number of files successfully enhanced
        """
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return 0

        # Find all NPZ files
        npz_files = list(directory.glob(pattern))
        if not npz_files:
            logger.warning(f"No {pattern} files found in {directory}")
            return 0

        logger.info(f"Found {len(npz_files)} {pattern} files in {directory}")

        # Limit if requested
        if max_files:
            npz_files = npz_files[:max_files]
            logger.info(f"Processing first {max_files} files")

        # Process files with progress bar
        enhanced_count = 0
        with tqdm(total=len(npz_files), desc="Enhancing files") as pbar:
            for npz_file in npz_files:
                if self.enhance_npz_file(npz_file, force):
                    enhanced_count += 1
                pbar.update(1)

        logger.info(f"Successfully enhanced {enhanced_count}/{len(npz_files)} files")
        return enhanced_count

    def verify_ssl_targets(self, npz_path: Path) -> Dict[str, str]:
        """Verify SSL targets in an enhanced NPZ file.

        Args:
            npz_path: Path to the NPZ file to verify

        Returns:
            Dictionary with verification results
        """
        results = {"file": str(npz_path.name), "status": "not_found"}

        if not npz_path.exists():
            return results

        try:
            with np.load(npz_path) as data:
                # Handle different key formats for board states
                if 's' in data:
                    states = data['s']
                elif 'positions' in data:
                    states = data['positions']
                else:
                    results["status"] = "no_board_states"
                    return results

                ssl_keys = [k for k in data.keys() if k.startswith('ssl_')]

                results["status"] = "ok"
                results["samples"] = len(states)
                results["ssl_tasks"] = ssl_keys
                results["all_keys"] = list(data.keys())

                # Verify shapes
                shape_info = {}
                for key in ssl_keys:
                    shape_info[key] = str(data[key].shape)
                results["shapes"] = shape_info

                # Basic validation
                for key in ssl_keys:
                    target_data = data[key]
                    if target_data.size == 0:
                        results["status"] = "empty_ssl_targets"
                        break

                    # Check for NaN/Inf
                    if np.any(np.isnan(target_data)) or np.any(np.isinf(target_data)):
                        results["status"] = "invalid_ssl_targets"
                        break

        except Exception as e:
            results["status"] = f"error: {str(e)}"

        return results


def main():
    parser = argparse.ArgumentParser(description="Enhance data files with SSL targets")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--pattern", type=str, default="*.npz", help="File pattern (when input is directory)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing SSL targets")
    parser.add_argument("--max-files", type=int, help="Maximum files to process")
    parser.add_argument("--verify", action="store_true", help="Verify SSL targets instead of enhancing")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")

    args = parser.parse_args()

    enhancer = SSLDataEnhancer(args.config, args.device)
    input_path = Path(args.input)

    if args.verify:
        # Verification mode
        if input_path.is_file():
            results = enhancer.verify_ssl_targets(input_path)
            print(f"Verification results for {results['file']}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        else:
            # Verify all files in directory
            npz_files = list(input_path.glob(args.pattern))
            print(f"Verifying {len(npz_files)} files...")

            all_results = []
            for npz_file in tqdm(npz_files, desc="Verifying"):
                results = enhancer.verify_ssl_targets(npz_file)
                all_results.append(results)

            # Summary
            status_counts = {}
            for result in all_results:
                status = result['status']
                status_counts[status] = status_counts.get(status, 0) + 1

            print("\nVerification Summary:")
            for status, count in status_counts.items():
                print(f"  {status}: {count} files")

    else:
        # Enhancement mode
        if input_path.is_file():
            success = enhancer.enhance_npz_file(input_path, args.force)
            if success:
                print(f"Successfully enhanced {input_path.name}")
            else:
                print(f"Failed to enhance {input_path.name}")
        else:
            enhanced_count = enhancer.enhance_directory(
                input_path, args.pattern, args.force, args.max_files
            )
            print(f"Enhanced {enhanced_count} files in {input_path}")


if __name__ == "__main__":
    main()
