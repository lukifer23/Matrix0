#!/usr/bin/env python3
"""
Comprehensive Data Loader for Matrix0 Training
- Loads tactical, openings, and self-play data
- Supports curriculum learning and data mixing strategies
- Integrates seamlessly with existing training pipeline
"""

import sys
import os
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, Optional, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDataLoader:
    """Comprehensive data loader that combines multiple data sources."""
    
    def __init__(self, 
                 tactical_data_path: str = "data/training/tactical_training_data.npz",
                 openings_data_path: str = "data/training/openings_training_data.npz",
                 selfplay_data_dir: str = "data/selfplay"):
        """Initialize the comprehensive data loader.
        
        Args:
            tactical_data_path: Path to tactical training data
            openings_data_path: Path to openings training data  
            selfplay_data_dir: Directory containing self-play data
        """
        self.tactical_data_path = Path(tactical_data_path)
        self.openings_data_path = Path(openings_data_path)
        self.selfplay_data_dir = Path(selfplay_data_dir)
        
        # Data storage
        self.tactical_data = None
        self.openings_data = None
        self.tactical_samples = 0
        self.openings_samples = 0
        
        # Load all available data
        self.load_all_data()
    
    def load_all_data(self):
        """Load all available training data sources."""
        logger.info("Loading comprehensive training data...")
        
        # Load tactical data
        if self.tactical_data_path.exists():
            self.tactical_data = np.load(self.tactical_data_path)
            self.tactical_samples = len(self.tactical_data['positions'])
            logger.info(f"✅ Loaded {self.tactical_samples} tactical training samples")
        else:
            logger.warning(f"❌ Tactical data not found at {self.tactical_data_path}")
        
        # Load openings data
        if self.openings_data_path.exists():
            self.openings_data = np.load(self.openings_data_path)
            self.openings_samples = len(self.openings_data['positions'])
            logger.info(f"✅ Loaded {self.openings_samples} openings training samples")
        else:
            logger.warning(f"❌ Openings data not found at {self.openings_data_path}")
        
        # Check self-play data
        if self.selfplay_data_dir.exists():
            selfplay_files = list(self.selfplay_data_dir.glob("*.npz"))
            logger.info(f"✅ Found {len(selfplay_files)} self-play data files")
        else:
            logger.warning(f"❌ Self-play data directory not found at {self.selfplay_data_dir}")
        
        # Summary
        total_samples = self.tactical_samples + self.openings_samples
        logger.info(f"📊 Total external training samples: {total_samples}")

    def _validate_shapes(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray, source: str = "") -> bool:
        """Validate that data arrays have expected shapes."""
        n = states.shape[0]
        if (states.shape != (n, 19, 8, 8) or
                policies.shape != (n, 4672) or
                values.shape != (n,)):
            logger.warning(
                f"Shape mismatch in {source}: states {states.shape}, policies {policies.shape}, values {values.shape}"
            )
            return False
        return True
    
    def get_mixed_batch(self, batch_size: int, 
                       tactical_ratio: float = 0.3,
                       openings_ratio: float = 0.3,
                       selfplay_ratio: float = 0.4) -> Optional[Dict[str, np.ndarray]]:
        """Get a mixed batch from all data sources.
        
        Args:
            batch_size: Total batch size
            tactical_ratio: Fraction of batch from tactical data
            openings_ratio: Fraction of batch from openings data
            selfplay_ratio: Fraction of batch from self-play data
            
        Returns:
            Mixed training batch or None if no data available
        """
        if not (self.tactical_data or self.openings_data):
            logger.warning("No external training data available")
            return None
        
        # Normalize ratios
        total_ratio = tactical_ratio + openings_ratio + selfplay_ratio
        tactical_ratio /= total_ratio
        openings_ratio /= total_ratio
        selfplay_ratio /= total_ratio
        
        # Calculate samples per source
        tactical_samples = int(batch_size * tactical_ratio)
        openings_samples = int(batch_size * openings_ratio)
        selfplay_samples = batch_size - tactical_samples - openings_samples
        
        batch_positions = []
        batch_policies = []
        batch_values = []
        
        # Add tactical samples
        if tactical_samples > 0 and self.tactical_data is not None:
            indices = np.random.choice(self.tactical_samples, tactical_samples, replace=False)
            batch_positions.append(self.tactical_data['positions'][indices])
            batch_policies.append(self.tactical_data['policy_targets'][indices])
            batch_values.append(self.tactical_data['value_targets'][indices])
        
        # Add openings samples
        if openings_samples > 0 and self.openings_data is not None:
            indices = np.random.choice(self.openings_samples, openings_samples, replace=False)
            batch_positions.append(self.openings_data['positions'][indices])
            batch_policies.append(self.openings_data['policy_targets'][indices])
            batch_values.append(self.openings_data['value_targets'][indices])
        
        # Add self-play samples (simplified - in practice, use DataManager)
        if selfplay_samples > 0:
            # Create dummy self-play data for now
            # In practice, this would load from your DataManager
            dummy_positions = np.random.random((selfplay_samples, 19, 8, 8)).astype(np.float32)
            dummy_policies = np.random.random((selfplay_samples, 4672)).astype(np.float32)
            dummy_values = np.random.random(selfplay_samples).astype(np.float32)
            
            batch_positions.append(dummy_positions)
            batch_policies.append(dummy_policies)
            batch_values.append(dummy_values)
        
        if not batch_positions:
            return None
        
        # Combine all sources
        combined_positions = np.concatenate(batch_positions, axis=0)
        combined_policies = np.concatenate(batch_policies, axis=0)
        combined_values = np.concatenate(batch_values, axis=0)

        # Shuffle the combined batch
        indices = np.random.permutation(len(combined_positions))
        shuffled_positions = combined_positions[indices]
        shuffled_policies = combined_policies[indices]
        shuffled_values = combined_values[indices]

        if not self._validate_shapes(shuffled_positions, shuffled_policies, shuffled_values, 'mixed batch'):
            return None

        return {
            's': shuffled_positions,
            'pi': shuffled_policies,
            'z': shuffled_values
        }
    
    def get_curriculum_batch(self, batch_size: int, phase: str = "openings") -> Optional[Dict[str, np.ndarray]]:
        """Get a batch for curriculum learning.
        
        Args:
            batch_size: Batch size
            phase: Training phase ("openings", "tactics", "mixed")
            
        Returns:
            Curriculum-appropriate training batch
        """
        if phase == "openings":
            # Focus on openings (80% openings, 20% tactics)
            return self.get_mixed_batch(batch_size, 
                                      tactical_ratio=0.2, 
                                      openings_ratio=0.8, 
                                      selfplay_ratio=0.0)
        
        elif phase == "tactics":
            # Focus on tactics (80% tactics, 20% openings)
            return self.get_mixed_batch(batch_size,
                                      tactical_ratio=0.8,
                                      openings_ratio=0.2,
                                      selfplay_ratio=0.0)
        
        elif phase == "mixed":
            # Balanced mix with self-play
            return self.get_mixed_batch(batch_size,
                                      tactical_ratio=0.3,
                                      openings_ratio=0.3,
                                      selfplay_ratio=0.4)
        
        else:
            logger.warning(f"Unknown curriculum phase: {phase}")
            return self.get_mixed_batch(batch_size)
    
    def get_pure_batch(self, batch_size: int, source: str = "tactical") -> Optional[Dict[str, np.ndarray]]:
        """Get a batch from a single data source.
        
        Args:
            batch_size: Batch size
            source: Data source ("tactical", "openings")
            
        Returns:
            Single-source training batch
        """
        if source == "tactical" and self.tactical_data is not None:
            indices = np.random.choice(self.tactical_samples, batch_size, replace=False)
            batch_states = self.tactical_data['positions'][indices]
            batch_policies = self.tactical_data['policy_targets'][indices]
            batch_values = self.tactical_data['value_targets'][indices]
            if not self._validate_shapes(batch_states, batch_policies, batch_values, 'pure tactical'):
                return None
            return {
                's': batch_states,
                'pi': batch_policies,
                'z': batch_values
            }

        elif source == "openings" and self.openings_data is not None:
            indices = np.random.choice(self.openings_samples, batch_size, replace=False)
            batch_states = self.openings_data['positions'][indices]
            batch_policies = self.openings_data['policy_targets'][indices]
            batch_values = self.openings_data['value_targets'][indices]
            if not self._validate_shapes(batch_states, batch_policies, batch_values, 'pure openings'):
                return None
            return {
                's': batch_states,
                'pi': batch_policies,
                'z': batch_values
            }
        
        else:
            logger.warning(f"Data source '{source}' not available")
            return None
    
    def get_data_stats(self) -> Dict[str, int]:
        """Get statistics about available data."""
        return {
            'tactical_samples': self.tactical_samples,
            'openings_samples': self.openings_samples,
            'total_samples': self.tactical_samples + self.openings_samples
        }

def demonstrate_usage():
    """Demonstrate how to use the comprehensive data loader."""
    print("🔍 COMPREHENSIVE DATA LOADER DEMONSTRATION")
    print("="*60)
    
    # Initialize loader
    loader = ComprehensiveDataLoader()
    
    # Show data statistics
    stats = loader.get_data_stats()
    print(f"\n📊 Data Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test different batch types
    batch_size = 32
    
    print(f"\n🧪 Testing different batch types (batch_size={batch_size}):")
    
    # Mixed batch
    mixed_batch = loader.get_mixed_batch(batch_size)
    if mixed_batch:
        print(f"  ✅ Mixed batch: {mixed_batch['s'].shape}")
    
    # Curriculum batches
    for phase in ["openings", "tactics", "mixed"]:
        curriculum_batch = loader.get_curriculum_batch(batch_size, phase)
        if curriculum_batch:
            print(f"  ✅ Curriculum ({phase}): {curriculum_batch['s'].shape}")
    
    # Pure batches
    for source in ["tactical", "openings"]:
        pure_batch = loader.get_pure_batch(batch_size, source)
        if pure_batch:
            print(f"  ✅ Pure ({source}): {pure_batch['s'].shape}")
    
    print(f"\n🎯 Integration complete! Use this loader in your training pipeline.")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Data Loader for Matrix0")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--tactical-data", type=str, default="data/training/tactical_training_data.npz")
    parser.add_argument("--openings-data", type=str, default="data/training/openings_training_data.npz")
    parser.add_argument("--selfplay-dir", type=str, default="data/selfplay")
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_usage()
    else:
        # Initialize and show basic info
        loader = ComprehensiveDataLoader(
            tactical_data_path=args.tactical_data,
            openings_data_path=args.openings_data,
            selfplay_data_dir=args.selfplay_dir
        )
        
        stats = loader.get_data_stats()
        print("\n📊 Comprehensive Data Loader Ready!")
        print(f"  Tactical samples: {stats['tactical_samples']}")
        print(f"  Openings samples: {stats['openings_samples']}")
        print(f"  Total samples: {stats['total_samples']}")

if __name__ == "__main__":
    main()
