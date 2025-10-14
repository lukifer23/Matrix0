from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import chess

from .utils import clear_memory_cache, get_memory_usage, safe_config_get
from .training.ssl_targets import decode_board_from_planes
from .encoding import move_encoder

logger = logging.getLogger(__name__)


@dataclass
class DataShard:
    """Represents a single data shard with metadata."""
    path: str
    size_bytes: int
    sample_count: int
    created_at: str
    checksum: str
    version: str
    source: str | None = None
    corrupted: bool = False

    def __repr__(self) -> str:
        return f"DataShard(path={self.path}, samples={self.sample_count}, corrupted={self.corrupted})"


@dataclass
class DataStats:
    """Statistics about the data pipeline."""
    total_shards: int
    total_samples: int
    total_size_gb: float
    corrupted_shards: int
    last_updated: str


class DataManager:
    """Manages the replay buffer and data pipeline for Matrix0 training."""
    
    def __init__(self, base_dir: str = "data", max_shards: int = 128, shard_size: int = 16384, expected_planes: int = 19):
        self.base_dir = Path(base_dir)
        self.max_shards = max_shards
        self.shard_size = shard_size
        self.expected_planes = expected_planes
        
        # Ensure directories exist
        self.selfplay_dir = self.base_dir / "selfplay"
        self.replays_dir = self.base_dir / "replays"
        self.validation_dir = self.base_dir / "validation"
        self.backups_dir = self.base_dir / "backups"
        
        for dir_path in [self.selfplay_dir, self.replays_dir, self.validation_dir, self.backups_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database for tracking
        self.db_path = self.base_dir / "data_metadata.db"
        self._init_database()

        # Version tracking
        self.version = "1.0.0"
        # Warn-once flags to avoid log spam each step
        self._warned_missing_tactical = False
        self._warned_missing_openings = False
        self._warned_teacher_portion_failure = False
        # Track failed SSL teacher files to avoid repeated warnings
        self._failed_ssl_teacher_files = set()

        # Field aliases allow importing legacy or curriculum NPZ bundles that
        # use descriptive keys (e.g., positions/policy_targets/value_targets).
        self._field_aliases = {
            's': ("s", "states", "positions"),
            'pi': ("pi", "policy", "policy_targets"),
            'z': ("z", "value", "values", "value_targets"),
        }
        
    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection with WAL and busy timeout enabled."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        c = conn.cursor()
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
        except sqlite3.Error:
            logger.exception("Failed to set SQLite PRAGMA options")
            raise
        return conn

    def _init_database(self):
        """Initialize SQLite database for metadata tracking."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shards (
                path TEXT PRIMARY KEY,
                size_bytes INTEGER,
                sample_count INTEGER,
                created_at TEXT,
                checksum TEXT,
                version TEXT,
                source TEXT,
                corrupted BOOLEAN DEFAULT FALSE,
                last_accessed TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_stats (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        
        conn.commit()

        # Lightweight migration: ensure expected columns exist
        try:
            cursor.execute("PRAGMA table_info(shards)")
            cols = {row[1] for row in cursor.fetchall()}  # column names
            migrations = []
            if 'source' not in cols:
                migrations.append("ALTER TABLE shards ADD COLUMN source TEXT")
            if 'corrupted' not in cols:
                migrations.append("ALTER TABLE shards ADD COLUMN corrupted BOOLEAN DEFAULT FALSE")
            if 'last_accessed' not in cols:
                migrations.append("ALTER TABLE shards ADD COLUMN last_accessed TEXT")
            if 'version' not in cols:
                migrations.append("ALTER TABLE shards ADD COLUMN version TEXT")
            for sql in migrations:
                try:
                    cursor.execute(sql)
                except sqlite3.Error:
                    logger.exception("Migration failed: %s", sql)
                    raise
            if migrations:
                conn.commit()
        except sqlite3.Error:
            logger.exception("Database schema inspection failed")
            raise

        # Verify required tables and columns exist
        required_schema = {
            'shards': {
                'path', 'size_bytes', 'sample_count', 'created_at',
                'checksum', 'version', 'source', 'corrupted', 'last_accessed'
            },
            'data_stats': {'key', 'value', 'updated_at'}
        }
        try:
            for table, required_cols in required_schema.items():
                cursor.execute(f"PRAGMA table_info({table})")
                existing = {row[1] for row in cursor.fetchall()}
                missing = required_cols - existing
                if missing:
                    logger.exception("Missing required columns %s in table %s", missing, table)
                    raise RuntimeError(f"Missing required columns {missing} in table {table}")
        except sqlite3.Error:
            logger.exception("Failed to verify database schema")
            raise
        conn.close()

    def _with_retry(self, func, *args, **kwargs):
        """Retry a DB operation if the database is locked."""
        attempts = 0
        delay = 0.05
        while True:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if 'locked' in msg or 'busy' in msg:
                    attempts += 1
                    if attempts > 10:
                        raise
                    time.sleep(delay)
                    delay = min(0.5, delay * 2)
                else:
                    raise

    def _save_npz_shard(self, data: Dict[str, np.ndarray], dir_path: Path) -> Tuple[Path, str]:
        """Save a data shard atomically and return its path and checksum."""
        dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:8]
        filename = f"{dir_path.name}_{timestamp}_{unique}.npz"
        filepath = dir_path / filename

        attempts = 0
        while True:
            try:
                with tempfile.NamedTemporaryFile(dir=str(dir_path), suffix=".npz.tmp", delete=False) as tf:
                    tmp_path = Path(tf.name)
                    np.savez_compressed(tf, **data)
                os.replace(tmp_path, filepath)
                break
            except Exception as e:
                attempts += 1
                if attempts >= 3:
                    logger.error(f"Failed to save data shard after {attempts} attempts: {e}")
                    raise
                logger.warning(f"Save attempt {attempts} failed, retrying: {e}")
                time.sleep(0.1 * attempts)
                try:
                    if 'tmp_path' in locals() and Path(tmp_path).exists():
                        Path(tmp_path).unlink()
                except Exception:
                    pass

        checksum = self._calculate_checksum(filepath)
        return filepath, checksum
    
    def add_selfplay_data(self, data: Dict[str, np.ndarray], worker_id: int, game_id: int) -> str:
        """Add self-play data to the buffer."""
        filepath, checksum = self._save_npz_shard(data, self.selfplay_dir)
        file_size = filepath.stat().st_size
        sample_count = self._infer_sample_count(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._record_shard(str(filepath), file_size, sample_count, timestamp, checksum, source="selfplay")
        logger.info(f"Added self-play data: {filepath.name} ({sample_count} samples, {file_size/1024:.1f}KB)")
        logger.debug(f"Saved by worker {worker_id} game {game_id}")
        return str(filepath)
    
    def add_training_data(self, data: Dict[str, np.ndarray], shard_id: int, source: str = "selfplay") -> str:
        """Add processed training data to replay buffer."""
        filepath, checksum = self._save_npz_shard(data, self.replays_dir)
        file_size = filepath.stat().st_size
        sample_count = self._infer_sample_count(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._record_shard(str(filepath), file_size, sample_count, timestamp, checksum, source=source)
        # Diagnostics for legal mask persistence
        if 'legal_mask' in data:
            try:
                lm = data['legal_mask']
                ratio = float(lm.sum() / max(1, lm.size))
                logger.info(f"Added training data: {filepath.name} ({sample_count} samples, {file_size/1024:.1f}KB, legal_ratio={ratio:.3f})")
            except Exception:
                logger.info(f"Added training data: {filepath.name} ({sample_count} samples, {file_size/1024:.1f}KB, legal_mask=present)")
        else:
            logger.info(f"Added training data: {filepath.name} ({sample_count} samples, {file_size/1024:.1f}KB)")
        return str(filepath)
    
    def get_training_batch(self, batch_size: int, device: str = "cpu") -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """Get training batches from replay buffer with balanced external data mixing.

        Returns tuples (s, pi, z, legal_mask) where legal_mask may be None if not present.
        """
        all_shards = self._get_all_shards()
        valid_shards = [s for s in all_shards if not s.corrupted]

        if not valid_shards:
            raise RuntimeError("No valid training data available")

        # Separate external and self-play shards for balanced sampling
        external_shards = [s for s in valid_shards if s.source and 'stockfish' in s.source]
        selfplay_shards = [s for s in valid_shards if s.source == 'selfplay' or not s.source]

        # Calculate balanced proportions (ensure external data gets fair representation)
        total_external_samples = sum(s.sample_count for s in external_shards)
        total_selfplay_samples = sum(s.sample_count for s in selfplay_shards)

        # Target 30% external data, 70% self-play for balanced learning
        target_external_ratio = 0.3

        # Adjust ratios based on available data
        if total_external_samples == 0:
            external_ratio = 0.0
            selfplay_ratio = 1.0
        elif total_selfplay_samples == 0:
            external_ratio = 1.0
            selfplay_ratio = 0.0
        else:
            external_ratio = min(target_external_ratio, total_external_samples / (total_external_samples + total_selfplay_samples))
            selfplay_ratio = 1.0 - external_ratio

        if not external_shards:
            external_batch_size = 0
            selfplay_batch_size = batch_size
        elif not selfplay_shards:
            external_batch_size = batch_size
            selfplay_batch_size = 0
        else:
            external_batch_size = int(round(batch_size * external_ratio))
            external_batch_size = min(batch_size, external_batch_size)
            selfplay_batch_size = batch_size - external_batch_size

            if external_ratio > 0 and external_batch_size == 0:
                external_batch_size = 1
                selfplay_batch_size = max(0, batch_size - external_batch_size)

            if selfplay_ratio > 0 and selfplay_batch_size == 0:
                selfplay_batch_size = 1
                external_batch_size = max(0, batch_size - selfplay_batch_size)

        external_iter = self._iter_shard_samples(external_shards) if external_shards else None
        selfplay_iter = self._iter_shard_samples(selfplay_shards) if selfplay_shards else None

        if external_iter is None and selfplay_iter is None:
            raise RuntimeError("No valid training data available")

        def _collect_samples(sample_iter: Optional[Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]], count: int):
            collected: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
            if sample_iter is None or count <= 0:
                return collected, sample_iter
            while len(collected) < count:
                try:
                    collected.append(next(sample_iter))
                except StopIteration:
                    sample_iter = None
                    break
            return collected, sample_iter

        while True:
            ext_target = external_batch_size if external_iter else 0
            sp_target = selfplay_batch_size if selfplay_iter else 0

            if ext_target + sp_target == 0:
                if external_iter:
                    ext_target = batch_size
                elif selfplay_iter:
                    sp_target = batch_size
                else:
                    return

            ext_samples, external_iter = _collect_samples(external_iter, ext_target)
            sp_samples, selfplay_iter = _collect_samples(selfplay_iter, sp_target)

            combined = ext_samples + sp_samples
            remaining = batch_size - len(combined)

            if remaining > 0 and external_iter:
                extra, external_iter = _collect_samples(external_iter, remaining)
                combined.extend(extra)
                remaining = batch_size - len(combined)

            if remaining > 0 and selfplay_iter:
                extra, selfplay_iter = _collect_samples(selfplay_iter, remaining)
                combined.extend(extra)
                remaining = batch_size - len(combined)

            if not combined:
                return

            if len(combined) < batch_size:
                return

            states = np.stack([sample[0] for sample in combined], axis=0)
            policies = np.stack([sample[1] for sample in combined], axis=0)
            values = np.array([sample[2] for sample in combined], dtype=np.float32)

            if values.ndim == 2 and values.shape[1] == 1:
                values = values.reshape(values.shape[0])

            legal_entries = [sample[3] for sample in combined]
            if all(mask is not None for mask in legal_entries):
                legal_mask = np.stack(legal_entries, axis=0)
                if legal_mask.ndim > 2:
                    legal_mask = legal_mask.reshape(legal_mask.shape[0], -1)
                if legal_mask.dtype != np.uint8:
                    legal_mask = legal_mask.astype(np.uint8, copy=False)
                if not legal_mask.flags['C_CONTIGUOUS']:
                    legal_mask = np.ascontiguousarray(legal_mask)
            else:
                legal_mask = None

            states = np.ascontiguousarray(states, dtype=np.float32)
            policies = np.ascontiguousarray(policies, dtype=np.float32)
            values = np.ascontiguousarray(values, dtype=np.float32)

            if legal_mask is not None:
                yield (states, policies, values, legal_mask)
            else:
                yield (states, policies, values)

    def _iter_shard_samples(self, shards: List[DataShard]) -> Optional[Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]]:
        if not shards:
            return None

        shard_paths = [s.path for s in shards]

        def generator() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
            local_paths = list(shard_paths)
            while True:
                np.random.shuffle(local_paths)
                made_progress = False
                for shard_path in local_paths:
                    try:
                        with np.load(shard_path, mmap_mode='r') as data:
                            states, policies, values, legal_mask_all, ssl_targets = self._extract_training_arrays(data)

                            if values.ndim == 2 and values.shape[1] == 1:
                                values = values.reshape(values.shape[0])

                            if not self._validate_shapes(states, policies, values, self.expected_planes, shard_path):
                                self._mark_shard_corrupted(shard_path)
                                continue

                            legal_mask_processed: Optional[np.ndarray] = None
                            if legal_mask_all is not None:
                                try:
                                    if legal_mask_all.ndim > 2:
                                        legal_mask_all = legal_mask_all.reshape(legal_mask_all.shape[0], -1)
                                    if legal_mask_all.dtype != np.uint8:
                                        legal_mask_all = legal_mask_all.astype(np.uint8, copy=False)
                                    legal_mask_processed = legal_mask_all
                                except Exception:
                                    legal_mask_processed = None

                            indices = np.random.permutation(len(states))
                            states = np.ascontiguousarray(states[indices])
                            policies = np.ascontiguousarray(policies[indices])
                            values = np.ascontiguousarray(values[indices])

                            if states.dtype != np.float32:
                                states = states.astype(np.float32, copy=False)
                            if policies.dtype != np.float32:
                                policies = policies.astype(np.float32, copy=False)
                            if values.dtype != np.float32:
                                values = values.astype(np.float32, copy=False)

                            legal_batches: Optional[np.ndarray] = None
                            if legal_mask_processed is not None:
                                legal_mask_processed = legal_mask_processed[indices]
                                if not legal_mask_processed.flags['C_CONTIGUOUS']:
                                    legal_mask_processed = np.ascontiguousarray(legal_mask_processed)
                                legal_batches = legal_mask_processed

                            for key in ssl_targets:
                                ssl_targets[key] = ssl_targets[key][indices]

                            for idx in range(len(states)):
                                made_progress = True
                                legal_entry: Optional[np.ndarray] = None
                                if legal_batches is not None:
                                    legal_entry = legal_batches[idx]
                                yield (states[idx], policies[idx], values[idx], legal_entry)
                    except Exception as e:
                        logger.error(f"Error loading shard {shard_path}: {e}", exc_info=True)
                        self._mark_shard_corrupted(shard_path)
                        continue

                if not made_progress:
                    return

        return generator()
    
    def get_external_training_batch(self, batch_size: int, source: str = "mixed") -> Optional[Dict[str, np.ndarray]]:
        """Get training batches from external training data sources.
        
        Args:
            batch_size: Size of training batch
            source: Data source ("tactical", "openings", "mixed")
            
        Returns:
            Training batch dict with keys 's', 'pi', 'z' or None if no data
        """
        try:
            # Prefer Stockfish-tagged shards when available
            if source == "tactical":
                return self._get_tactical_batch(batch_size)
            elif source == "openings":
                return self._get_openings_batch(batch_size)
            elif source == "mixed":
                # Try balanced Stockfish mix first, fallback to legacy mixed
                sf = self._get_stockfish_mixed_batch(batch_size)
                return sf if sf is not None else self._get_mixed_external_batch(batch_size)
            else:
                logger.warning(f"Unknown external data source: {source}")
                return None
        except Exception as e:
            logger.error(f"Error getting external training batch: {e}")
            return None
    
    def get_curriculum_batch(self, batch_size: int, phase: str = "mixed") -> Optional[Dict[str, np.ndarray]]:
        """Get curriculum-appropriate training batch.
        
        Args:
            batch_size: Size of training batch
            phase: Training phase ("openings", "tactics", "mixed")
            
        Returns:
            Training batch dict with keys 's', 'pi', 'z' or None if no data
        """
        try:
            if phase == "openings":
                # Focus on openings (80% openings, 20% tactics)
                return self._get_curriculum_openings_batch(batch_size)
            elif phase == "tactics":
                # Focus on tactics (80% tactics, 20% openings)
                return self._get_curriculum_tactics_batch(batch_size)
            elif phase == "mixed":
                # Balanced mix with self-play
                return self._get_curriculum_mixed_batch(batch_size)
            else:
                logger.warning(f"Unknown curriculum phase: {phase}")
                return self._get_mixed_external_batch(batch_size)
        except Exception as e:
            logger.error(f"Error getting curriculum batch: {e}")
            return None
    
    def _get_tactical_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get batch from tactical training data."""
        tactical_path = Path(self.base_dir) / "tactical" / "tactical_positions.npz"
        if not tactical_path.exists():
            if not self._warned_missing_tactical:
                logger.warning("Tactical training data not found")
                self._warned_missing_tactical = True
            return None
        
        try:
            with np.load(tactical_path, allow_pickle=True) as data:
                total_positions = len(data['positions'])
                if total_positions == 0:
                    logger.warning("Tactical training data is empty")
                    return None
                draw_size = min(batch_size, total_positions)
                replace = batch_size > total_positions
                target_size = batch_size if replace else draw_size
                indices = np.random.choice(total_positions, target_size, replace=replace)
                batch_states = data['positions'][indices]  # curriculum format: (N, 19, 8, 8)
                batch_policies = data['policy_targets'][indices]  # curriculum format: (N, 4672)
                batch_values = data['value_targets'][indices]  # curriculum format: (N,)
                if not self._validate_shapes(batch_states, batch_policies, batch_values, self.expected_planes, tactical_path):
                    return None
                out: Dict[str, np.ndarray] = {
                    's': batch_states,  # Map 'positions' -> 's'
                    'pi': batch_policies,  # Map 'policy_targets' -> 'pi'
                    'z': batch_values  # Map 'value_targets' -> 'z'
                }
                # Generate legal_mask for curriculum data using proper board reconstruction
                legal_masks = []
                for i in range(len(batch_states)):
                    try:
                        # Convert board encoding back to chess.Board for legal move computation
                        board_encoding = batch_states[i]  # (19, 8, 8) - channels first
                        board = decode_board_from_planes(board_encoding)

                        # Get proper legal move mask using the move encoder
                        legal_mask = move_encoder.get_legal_actions(board).astype(np.uint8)

                        legal_masks.append(legal_mask)
                    except Exception as e:
                        logger.warning(f"Failed to compute legal mask for tactical sample {i}: {e}")
                        # Fallback: mark all moves as potentially legal (will be filtered by policy masking)
                        legal_mask = np.ones(4672, dtype=np.uint8)
                        legal_masks.append(legal_mask)

                out['legal_mask'] = np.array(legal_masks)
                return out
        except Exception as e:
            logger.error(f"Error loading tactical data: {e}")
            return None
    
    def _get_openings_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get batch from openings training data."""
        openings_path = Path(self.base_dir) / "openings" / "openings_positions.npz"
        if not openings_path.exists():
            if not self._warned_missing_openings:
                logger.warning("Openings training data not found")
                self._warned_missing_openings = True
            return None
        
        try:
            with np.load(openings_path, allow_pickle=True) as data:
                total_positions = len(data['positions'])
                if total_positions == 0:
                    logger.warning("Openings training data is empty")
                    return None
                draw_size = min(batch_size, total_positions)
                replace = batch_size > total_positions
                target_size = batch_size if replace else draw_size
                indices = np.random.choice(total_positions, target_size, replace=replace)
                batch_states = data['positions'][indices]  # curriculum format: (N, 19, 8, 8)
                batch_policies = data['policy_targets'][indices]  # curriculum format: (N, 4672)
                batch_values = data['value_targets'][indices]  # curriculum format: (N,)
                if not self._validate_shapes(batch_states, batch_policies, batch_values, self.expected_planes, openings_path):
                    return None
                out: Dict[str, np.ndarray] = {
                    's': batch_states,  # Map 'positions' -> 's'
                    'pi': batch_policies,  # Map 'policy_targets' -> 'pi'
                    'z': batch_values  # Map 'value_targets' -> 'z'
                }
                # Generate legal_mask for curriculum data using proper board reconstruction
                legal_masks = []
                for i in range(len(batch_states)):
                    try:
                        # Convert board encoding back to chess.Board for legal move computation
                        board_encoding = batch_states[i]  # (19, 8, 8) - channels first
                        board = decode_board_from_planes(board_encoding)

                        # Get proper legal move mask using the move encoder
                        legal_mask = move_encoder.get_legal_actions(board).astype(np.uint8)

                        legal_masks.append(legal_mask)
                    except Exception as e:
                        logger.warning(f"Failed to compute legal mask for openings sample {i}: {e}")
                        # Fallback: mark all moves as potentially legal (will be filtered by policy masking)
                        legal_mask = np.ones(4672, dtype=np.uint8)
                        legal_masks.append(legal_mask)

                out['legal_mask'] = np.array(legal_masks)
                return out
        except Exception as e:
            logger.error(f"Error loading openings data: {e}")
            return None
    
    def _get_mixed_external_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get mixed batch from external training data sources."""
        tactical_batch = self._get_tactical_batch(batch_size // 2)
        openings_batch = self._get_openings_batch(batch_size // 2)

        if not tactical_batch and not openings_batch:
            return None

        if not tactical_batch:
            return openings_batch
        if not openings_batch:
            return tactical_batch

        # Combine batches
        combined_batch: Dict[str, np.ndarray] = {}
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = np.concatenate([
                tactical_batch[key], openings_batch[key]
            ], axis=0)

        # Handle SSL targets
        ssl_keys_tactical = [k for k in tactical_batch.keys() if k.startswith('ssl_')]
        ssl_keys_openings = [k for k in openings_batch.keys() if k.startswith('ssl_')]

        if ssl_keys_tactical and ssl_keys_openings:
            for ssl_key in ssl_keys_tactical:
                if ssl_key in openings_batch:
                    try:
                        combined_batch[ssl_key] = np.concatenate([
                            tactical_batch[ssl_key], openings_batch[ssl_key]
                        ], axis=0)
                    except Exception as e:
                        logger.warning(f"Failed to concatenate SSL target {ssl_key}: {e}")

        # Shuffle the combined batch
        indices = np.random.permutation(len(combined_batch['s']))
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = combined_batch[key][indices]

        # Include legal_mask only if present in both sources
        if ('legal_mask' in tactical_batch) and ('legal_mask' in openings_batch):
            try:
                lm_comb = np.concatenate([tactical_batch['legal_mask'], openings_batch['legal_mask']], axis=0)
                combined_batch['legal_mask'] = lm_comb[indices]
            except Exception:
                if 'legal_mask' in combined_batch:
                    del combined_batch['legal_mask']

        # Shuffle SSL targets
        for key in list(combined_batch.keys()):
            if key.startswith('ssl_'):
                combined_batch[key] = combined_batch[key][indices]

        # CRITICAL: Explicit cleanup to prevent memory accumulation
        # The original batches contain potentially 10K+ samples each
        # but we only need the final ~192 samples
        del tactical_batch, openings_batch, indices

        # Gate memory cleanup to avoid churn: only clear when near threshold or periodically
        try:
            usage = get_memory_usage('auto')
            mem_gb = float(usage.get('memory_gb', 0.0) or 0.0)
            # Default threshold ~85% of config memory limit if available, else use 12GB heuristic
            threshold_gb = float(os.environ.get('MATRIX0_MEM_THRESHOLD_GB', '12'))
            if mem_gb >= threshold_gb:
                clear_memory_cache('auto')
        except Exception:
            pass  # Ignore cleanup errors

        if not self._validate_shapes(combined_batch['s'], combined_batch['pi'], combined_batch['z'], self.expected_planes, 'mixed external'):
            return None
        return combined_batch
    
    def _get_curriculum_openings_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get curriculum batch focused on openings (80% openings, 20% tactics)."""
        openings_size = int(batch_size * 0.8)
        tactical_size = batch_size - openings_size
        
        openings_batch = self._get_openings_batch(openings_size)
        tactical_batch = self._get_tactical_batch(tactical_size) if tactical_size > 0 else None
        
        if not openings_batch:
            return tactical_batch
        
        if not tactical_batch:
            return openings_batch
        
        # Combine with openings focus
        combined_batch: Dict[str, np.ndarray] = {}
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = np.concatenate([
                openings_batch[key], tactical_batch[key]
            ], axis=0)

        # Handle SSL targets
        ssl_keys_openings = [k for k in openings_batch.keys() if k.startswith('ssl_')]
        ssl_keys_tactical = [k for k in tactical_batch.keys() if k.startswith('ssl_')]

        if ssl_keys_openings and ssl_keys_tactical:
            for ssl_key in ssl_keys_openings:
                if ssl_key in tactical_batch:
                    try:
                        combined_batch[ssl_key] = np.concatenate([
                            openings_batch[ssl_key], tactical_batch[ssl_key]
                        ], axis=0)
                    except Exception as e:
                        logger.warning(f"Failed to concatenate SSL target {ssl_key}: {e}")

        if ('legal_mask' in openings_batch) and ('legal_mask' in tactical_batch):
            try:
                lm = np.concatenate([openings_batch['legal_mask'], tactical_batch['legal_mask']], axis=0)
                combined_batch['legal_mask'] = lm
            except Exception:
                if 'legal_mask' in combined_batch:
                    del combined_batch['legal_mask']

        # Shuffle SSL targets
        indices = np.random.permutation(len(combined_batch['s']))
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = combined_batch[key][indices]
        if 'legal_mask' in combined_batch:
            combined_batch['legal_mask'] = combined_batch['legal_mask'][indices]
        for key in list(combined_batch.keys()):
            if key.startswith('ssl_'):
                combined_batch[key] = combined_batch[key][indices]
        if not self._validate_shapes(combined_batch['s'], combined_batch['pi'], combined_batch['z'], self.expected_planes, 'curriculum openings'):
            return None
        return combined_batch
    
    def _get_curriculum_tactics_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get curriculum batch focused on tactics (80% tactics, 20% openings)."""
        tactical_size = int(batch_size * 0.8)
        openings_size = batch_size - tactical_size
        
        tactical_batch = self._get_tactical_batch(tactical_size)
        openings_batch = self._get_openings_batch(openings_size) if openings_size > 0 else None
        
        if not tactical_batch:
            return openings_batch
        
        if not openings_batch:
            return tactical_batch
        
        # Combine with tactics focus
        combined_batch: Dict[str, np.ndarray] = {}
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = np.concatenate([
                tactical_batch[key], openings_batch[key]
            ], axis=0)

        # Handle SSL targets
        ssl_keys_tactical = [k for k in tactical_batch.keys() if k.startswith('ssl_')]
        ssl_keys_openings = [k for k in openings_batch.keys() if k.startswith('ssl_')]

        if ssl_keys_tactical and ssl_keys_openings:
            for ssl_key in ssl_keys_tactical:
                if ssl_key in openings_batch:
                    try:
                        combined_batch[ssl_key] = np.concatenate([
                            tactical_batch[ssl_key], openings_batch[ssl_key]
                        ], axis=0)
                    except Exception as e:
                        logger.warning(f"Failed to concatenate SSL target {ssl_key}: {e}")

        if ('legal_mask' in tactical_batch) and ('legal_mask' in openings_batch):
            try:
                lm = np.concatenate([tactical_batch['legal_mask'], openings_batch['legal_mask']], axis=0)
                combined_batch['legal_mask'] = lm
            except Exception:
                if 'legal_mask' in combined_batch:
                    del combined_batch['legal_mask']

        # Shuffle SSL targets
        indices = np.random.permutation(len(combined_batch['s']))
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = combined_batch[key][indices]
        if 'legal_mask' in combined_batch:
            combined_batch['legal_mask'] = combined_batch['legal_mask'][indices]
        for key in list(combined_batch.keys()):
            if key.startswith('ssl_'):
                combined_batch[key] = combined_batch[key][indices]
        if not self._validate_shapes(combined_batch['s'], combined_batch['pi'], combined_batch['z'], self.expected_planes, 'curriculum tactics'):
            return None
        return combined_batch
    
    def _load_ssl_teacher_data(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Load teacher data that contains SSL targets."""
        import glob

        # Find all teacher NPZ files
        teacher_pattern = os.path.join(self.base_dir, "teacher_games", "**", "*.npz")
        teacher_files = glob.glob(teacher_pattern, recursive=True)

        # Filter for files that contain SSL targets
        ssl_teacher_files = []
        for file_path in teacher_files:
            try:
                with np.load(file_path, mmap_mode='r') as data:
                    ssl_keys = [k for k in data.keys() if k.startswith('ssl_')]
                    if ssl_keys:
                        ssl_teacher_files.append((file_path, ssl_keys))
            except Exception:
                continue

        if not ssl_teacher_files:
            logger.debug("No SSL-enabled teacher files found")
            return None

        # Shuffle and select files to get the required batch size
        np.random.shuffle(ssl_teacher_files)

        collected_data = {k: [] for k in ['s', 'pi', 'z']}
        collected_ssl = {}
        collected_legal_mask = []

        samples_needed = batch_size

        for file_path, ssl_keys in ssl_teacher_files:
            if samples_needed <= 0:
                break

            try:
                with np.load(file_path, mmap_mode='r') as data:
                    available_samples = len(data['s'])
                    take_samples = min(samples_needed, available_samples)

                    # Load basic data
                    for key in ['s', 'pi', 'z']:
                        collected_data[key].append(data[key][:take_samples])

                    # Load legal mask if present
                    if 'legal_mask' in data:
                        collected_legal_mask.append(data['legal_mask'][:take_samples])

                    # Load SSL targets
                    for ssl_key in ssl_keys:
                        if ssl_key not in collected_ssl:
                            collected_ssl[ssl_key] = []

                        ssl_data = data[ssl_key][:take_samples]

                        # Convert control task from single-channel to 3-channel format
                        if ssl_key == 'ssl_control':
                            # Convert from (batch, 8, 8) with values [-1, 0, 1]
                            # to (batch, 3, 8, 8) with one-hot encoding
                            batch_size = ssl_data.shape[0]
                            control_3d = np.zeros((batch_size, 3, 8, 8), dtype=np.float32)

                            # Channel 0: black control (where data == -1)
                            control_3d[:, 0, :, :] = (ssl_data == -1).astype(np.float32)
                            # Channel 1: neutral (where data == 0)
                            control_3d[:, 1, :, :] = (ssl_data == 0).astype(np.float32)
                            # Channel 2: white control (where data == 1)
                            control_3d[:, 2, :, :] = (ssl_data == 1).astype(np.float32)

                            collected_ssl[ssl_key].append(control_3d)
                        else:
                            collected_ssl[ssl_key].append(ssl_data)

                    samples_needed -= take_samples
                    logger.debug(f"Loaded {take_samples} samples from {os.path.basename(file_path)}")

            except Exception as e:
                # Only warn once per failed file to avoid log spam
                if file_path not in self._failed_ssl_teacher_files:
                    self._failed_ssl_teacher_files.add(file_path)
                    logger.warning(f"Failed to load SSL teacher data from {os.path.basename(file_path)}: {e}")
                continue

        if not collected_data['s']:
            return None

        # Concatenate all collected data
        result = {}
        for key in ['s', 'pi', 'z']:
            result[key] = np.concatenate(collected_data[key], axis=0)

        if collected_legal_mask:
            result['legal_mask'] = np.concatenate(collected_legal_mask, axis=0)

        # Concatenate SSL targets
        for ssl_key, ssl_arrays in collected_ssl.items():
            result[ssl_key] = np.concatenate(ssl_arrays, axis=0)

        # Shuffle the final batch
        if len(result['s']) > 1:
            indices = np.random.permutation(len(result['s']))
            for key in result:
                result[key] = result[key][indices]

        # Trim to exact batch size if we got more than requested
        actual_size = len(result['s'])
        if actual_size > batch_size:
            for key in result:
                result[key] = result[key][:batch_size]
        elif actual_size < batch_size:
            logger.debug(f"Only loaded {actual_size} samples, less than requested {batch_size}")
            # Don't return partial batches that are too small
            return None

        logger.debug(f"Loaded SSL teacher batch with {len(result['s'])} samples, SSL targets: {[k for k in result.keys() if k.startswith('ssl_')]}")
        return result

    def _get_curriculum_mixed_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get curriculum batch with balanced mix and self-play + external (Stockfish + Teacher) data.

        Strategy: half batch from external sources, half from self-play.
        External half prioritizes Teacher (if present) and Stockfish.
        """
        ext_half = max(1, batch_size // 2)
        parts: List[Dict[str, np.ndarray]] = []

        # 1) Teacher portion (up to 60% of external half if available)
        teacher_take = max(0, int(round(ext_half * 0.6)))
        if teacher_take > 0:
            try:
                # Try to load teacher data with SSL targets directly
                teacher_data = self._load_ssl_teacher_data(teacher_take)
                if teacher_data:
                    parts.append(teacher_data)
                else:
                    # Fallback to original method if SSL loading fails
                    gen_t = self.get_training_batch_by_source_prefixes(teacher_take, ["teacher:"])
                    bt = next(gen_t, None)
                    if bt is not None:
                        s, pi, z, lm = bt if len(bt) == 4 else (*bt, None)
                        d: Dict[str, np.ndarray] = {"s": s, "pi": pi, "z": z}
                        if lm is not None:
                            d["legal_mask"] = lm
                        parts.append(d)
            except Exception as e:
                if not self._warned_teacher_portion_failure:
                    self._warned_teacher_portion_failure = True
                    logger.warning(f"Failed to load teacher portion: {e}")

        # 2) Stockfish portion fills the rest of external half
        remaining = ext_half - (parts[0]['s'].shape[0] if parts else 0)
        if remaining > 0:
            sf = self._get_stockfish_mixed_batch(remaining)
            if sf is not None:
                parts.append(sf)
                remaining -= sf['s'].shape[0]
        # 2b) DB-tagged 'external' shards (e.g., imported via extra_replay_dirs)
        if remaining > 0:
            try:
                gen_e = self.get_training_batch_by_source_prefixes(remaining, ["external"])
                be = next(gen_e, None)
                if be is not None:
                    s, pi, z, lm = be if len(be) == 4 else (*be, None)
                    d: Dict[str, np.ndarray] = {"s": s, "pi": pi, "z": z}
                    if lm is not None:
                        d["legal_mask"] = lm
                    parts.append(d)
                    remaining -= s.shape[0]
            except Exception:
                pass
        
        # 3) Fallback to legacy mixed external if still short
        got = sum(p['s'].shape[0] for p in parts) if parts else 0
        if got < ext_half:
            legacy = self._get_mixed_external_batch(ext_half - got)
            if legacy is not None:
                parts.append(legacy)
        
        # Merge external parts if any
        external_batch = None
        if parts:
            external_batch = {}
            for key in ['s', 'pi', 'z']:
                external_batch[key] = np.concatenate([p[key] for p in parts if key in p], axis=0)

            # Handle SSL targets in external parts
            ssl_keys_by_part = [[k for k in p.keys() if k.startswith('ssl_')] for p in parts]
            if all(ssl_keys_by_part) and len(set(tuple(k) for k in ssl_keys_by_part)) == 1:
                # All parts have the same SSL keys
                ssl_keys = ssl_keys_by_part[0]
                for ssl_key in ssl_keys:
                    try:
                        external_batch[ssl_key] = np.concatenate([p[ssl_key] for p in parts], axis=0)
                    except Exception as e:
                        logger.warning(f"Failed to concatenate external SSL target {ssl_key}: {e}")

            if all('legal_mask' in p for p in parts):
                try:
                    external_batch['legal_mask'] = np.concatenate([p['legal_mask'] for p in parts], axis=0)
                except Exception:
                    external_batch.pop('legal_mask', None)

        # Try to get self-play data for the other half
        try:
            # Get a batch from the regular replay buffer
            sp_generator = self.get_training_batch(batch_size - ext_half, "cpu")
            sp_batch = next(sp_generator)
            sp_dict = {
                's': sp_batch[0],
                'pi': sp_batch[1], 
                'z': sp_batch[2]
            }
            # Pass through optional legal_mask if present
            if isinstance(sp_batch, (list, tuple)) and len(sp_batch) >= 4 and sp_batch[3] is not None:
                sp_dict['legal_mask'] = sp_batch[3]
        except Exception:
            sp_dict = None
        
        if not external_batch and not sp_dict:
            return None
        
        if not external_batch:
            return sp_dict
        if not sp_dict:
            return external_batch
        
        # Combine external and self-play data
        combined_batch = {}

        # Concatenate basic fields
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = np.concatenate([
                external_batch[key], sp_dict[key]
            ], axis=0)

        # Handle SSL targets - check if both sources have SSL targets
        ssl_keys_external = [k for k in external_batch.keys() if k.startswith('ssl_')]
        ssl_keys_selfplay = [k for k in sp_dict.keys() if k.startswith('ssl_')]

        if ssl_keys_external and ssl_keys_selfplay:
            # Both have SSL targets - concatenate them
            for ssl_key in ssl_keys_external:
                if ssl_key in sp_dict:
                    try:
                        combined_batch[ssl_key] = np.concatenate([
                            external_batch[ssl_key], sp_dict[ssl_key]
                        ], axis=0)
                        logger.debug(f"Concatenated SSL target: {ssl_key}")
                    except Exception as e:
                        logger.warning(f"Failed to concatenate SSL target {ssl_key}: {e}")
        elif ssl_keys_external and not ssl_keys_selfplay:
            # Only external has SSL targets - keep them (selfplay samples will get SSL targets generated on-demand)
            for ssl_key in ssl_keys_external:
                combined_batch[ssl_key] = external_batch[ssl_key]
                logger.debug(f"Preserved SSL target from external data: {ssl_key}")
        elif not ssl_keys_external and ssl_keys_selfplay:
            # Only selfplay has SSL targets - this shouldn't happen but handle it
            for ssl_key in ssl_keys_selfplay:
                combined_batch[ssl_key] = sp_dict[ssl_key]
                logger.debug(f"Preserved SSL target from selfplay data: {ssl_key}")

        # Include legal_mask only if present in both sources for full coverage
        if ('legal_mask' in external_batch) and ('legal_mask' in sp_dict):
            try:
                combined_batch['legal_mask'] = np.concatenate([
                    external_batch['legal_mask'], sp_dict['legal_mask']
                ], axis=0)
            except Exception as e:
                # If concatenation fails due to shape mismatch, drop mask to remain robust
                logger.warning(f"Legal mask concatenation failed: {e}")
                if 'legal_mask' in combined_batch:
                    del combined_batch['legal_mask']

        # Ensure all arrays have the same length before shuffling
        batch_size = len(combined_batch['s'])

        # Find the minimum length across all arrays to ensure consistency
        min_length = batch_size
        for key in combined_batch.keys():
            min_length = min(min_length, len(combined_batch[key]))

        # Trim all arrays to the minimum length
        if min_length != batch_size:
            logger.warning(f"Arrays have different lengths, trimming all to {min_length}")
            for key in list(combined_batch.keys()):
                combined_batch[key] = combined_batch[key][:min_length]
            batch_size = min_length

        # Shuffle the combined batch
        if batch_size > 1:
            indices = np.random.permutation(batch_size)
            for key in list(combined_batch.keys()):
                combined_batch[key] = combined_batch[key][indices]
        if not self._validate_shapes(combined_batch['s'], combined_batch['pi'], combined_batch['z'], self.expected_planes, 'curriculum mixed'):
            return None
        return combined_batch

    def _get_stockfish_mixed_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Balanced batch sampled from Stockfish-imported shards by domain.

        Mix: 30% openings, 30% tactical, 20% king_safety/weaknesses, 20% endgames/positional.
        Falls back gracefully if some buckets are missing.
        """
        try:
            # Define prefixes per bucket
            buckets = [
                (int(max(1, round(batch_size * 0.30))), ["stockfish:openings/"]),
                (int(max(1, round(batch_size * 0.30))), ["stockfish:tactical/"]),
                (int(max(1, round(batch_size * 0.10))), ["stockfish:king_safety/"]),
                (int(max(1, round(batch_size * 0.10))), ["stockfish:weaknesses/"]),
                (int(max(1, round(batch_size * 0.10))), ["stockfish:endgames/"]),
                (batch_size - (int(max(1, round(batch_size * 0.30))) + int(max(1, round(batch_size * 0.30))) + 3*int(max(1, round(batch_size * 0.10))))), ["stockfish:positional/"],
            ]

            parts: List[Dict[str, np.ndarray]] = []
            for take, prefs in buckets:
                if take <= 0:
                    continue
                try:
                    gen = self.get_training_batch_by_source_prefixes(take, prefs)
                    batch = next(gen, None)
                    if batch is None:
                        continue
                    s, pi, z, lm = batch if len(batch) == 4 else (*batch, None)
                    d: Dict[str, np.ndarray] = {"s": s, "pi": pi, "z": z}
                    if lm is not None:
                        d["legal_mask"] = lm
                    parts.append(d)
                except Exception:
                    continue

            if not parts:
                return None

            # Concatenate parts
            combined: Dict[str, np.ndarray] = {}
            for key in ["s", "pi", "z"]:
                combined[key] = np.concatenate([p[key] for p in parts if key in p], axis=0)

            # Handle SSL targets
            ssl_keys_by_part = [[k for k in p.keys() if k.startswith('ssl_')] for p in parts]
            if all(ssl_keys_by_part) and len(set(tuple(k) for k in ssl_keys_by_part)) == 1:
                ssl_keys = ssl_keys_by_part[0]
                for ssl_key in ssl_keys:
                    try:
                        combined[ssl_key] = np.concatenate([p[ssl_key] for p in parts], axis=0)
                    except Exception as e:
                        logger.warning(f"Failed to concatenate SSL target {ssl_key}: {e}")

            if all("legal_mask" in p for p in parts):
                try:
                    combined["legal_mask"] = np.concatenate([p["legal_mask"] for p in parts], axis=0)
                except Exception:
                    combined.pop("legal_mask", None)

            # Shuffle
            idx = np.random.permutation(len(combined["s"]))
            for key in ["s", "pi", "z"]:
                combined[key] = combined[key][idx]
            if "legal_mask" in combined:
                combined["legal_mask"] = combined["legal_mask"][idx]
            # Shuffle SSL targets
            for key in list(combined.keys()):
                if key.startswith('ssl_'):
                    combined[key] = combined[key][idx]

            if not self._validate_shapes(combined['s'], combined['pi'], combined['z'], self.expected_planes, 'stockfish mixed'):
                return None
            return combined
        except Exception as e:
            logger.debug(f"Stockfish mixed batch failed: {e}")
            return None
    
    def get_external_data_stats(self) -> Dict[str, int]:
        """Report aggregate counts for non-self-play data sources."""

        stats = {
            'tactical_samples': 0,
            'openings_samples': 0,
            'stockfish_samples': 0,
            'teacher_samples': 0,
            'external_import_samples': 0,
            'external_total': 0,
        }

        # Legacy training bundles (data/training/*.npz)
        tactical_path = Path(self.base_dir) / "training" / "tactical_training_data.npz"
        if tactical_path.exists():
            try:
                with np.load(tactical_path) as data:
                    stats['tactical_samples'] = int(len(data['positions']))
            except Exception:
                logger.debug("Failed to read tactical_training_data.npz", exc_info=True)

        openings_path = Path(self.base_dir) / "training" / "openings_training_data.npz"
        if openings_path.exists():
            try:
                with np.load(openings_path) as data:
                    stats['openings_samples'] = int(len(data['positions']))
            except Exception:
                logger.debug("Failed to read openings_training_data.npz", exc_info=True)

        # Database registered shards (Stockfish, teacher, external imports)
        for shard in self._get_all_shards():
            source = (shard.source or "").lower()
            if source.startswith('stockfish'):
                stats['stockfish_samples'] += int(shard.sample_count)
            elif source.startswith('teacher:'):
                stats['teacher_samples'] += int(shard.sample_count)
            elif source.startswith('external'):
                stats['external_import_samples'] += int(shard.sample_count)

        stats['external_total'] = (
            stats['tactical_samples'] + stats['openings_samples'] +
            stats['stockfish_samples'] + stats['teacher_samples'] +
            stats['external_import_samples']
        )
        return stats
    
    def cleanup_old_shards(self, keep_recent: int = 64):
        """Remove old shards to maintain storage limits."""
        shards = self._get_all_shards()

        # Only prune replay-buffer shards inside data/replays and never external/Stockfish-tagged ones
        replays_root = self.replays_dir.resolve()
        eligible: List[DataShard] = []
        for s in shards:
            try:
                p = Path(s.path).resolve()
                src = (s.source or "")
                if str(p).startswith(str(replays_root)) and not src.startswith("stockfish:"):
                    eligible.append(s)
            except Exception:
                continue

        if len(eligible) <= keep_recent:
            return

        # Sort by creation time, keep most recent within eligible set
        eligible.sort(key=lambda x: x.created_at, reverse=True)
        to_remove = eligible[keep_recent:]

        for shard in to_remove:
            try:
                Path(shard.path).unlink(missing_ok=True)
                self._remove_shard_record(shard.path)
                logger.info(f"Removed old shard: {shard.path}")
            except Exception as e:
                # Many shards may already be removed by compaction; avoid noisy errors
                logger.debug(f"Shard already removed or missing: {shard.path} ({e})")
    
    def validate_data_integrity(self) -> Tuple[int, int]:
        """Validate all shards and return (valid, corrupted) counts."""
        shards = self._get_all_shards()
        valid_count = 0
        corrupted_count = 0
        
        for shard in shards:
            try:
                if self._validate_shard(shard):
                    valid_count += 1
                else:
                    corrupted_count += 1
                    self._mark_shard_corrupted(shard.path)
            except Exception as e:
                logger.error(f"Error validating shard {shard.path}: {e}")
                corrupted_count += 1
                self._mark_shard_corrupted(shard.path)
        
        return valid_count, corrupted_count

    def quarantine_corrupted_shards(self) -> int:
        """Moves corrupted shards to a quarantine directory."""
        quarantine_dir = self.backups_dir / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        shards = self._get_all_shards()
        quarantined_count = 0
        
        for shard in shards:
            if shard.corrupted:
                try:
                    shard_path = Path(shard.path)
                    if shard_path.exists():
                        new_path = quarantine_dir / shard_path.name
                        shard_path.rename(new_path)
                        logger.info(f"Quarantined corrupted shard: {shard.path} to {new_path}")
                        quarantined_count += 1
                except Exception as e:
                    logger.error(f"Error quarantining shard {shard.path}: {e}")
        
        return quarantined_count

    def get_stats(self) -> DataStats:
        """Get current data pipeline statistics."""
        shards = self._get_all_shards()
        
        total_shards = len(shards)
        total_samples = sum(s.sample_count for s in shards)
        total_size_gb = sum(s.size_bytes for s in shards) / (1024**3)
        corrupted_shards = sum(1 for s in shards if s.corrupted)
        
        return DataStats(
            total_shards=total_shards,
            total_samples=total_samples,
            total_size_gb=total_size_gb,
            corrupted_shards=corrupted_shards,
            last_updated=datetime.now().isoformat()
        )
    
    def create_backup(self) -> str:
        """Create a backup of the current data state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backups_dir / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        # Copy database
        import shutil
        shutil.copy2(self.db_path, backup_dir / "data_metadata.db")
        
        # Create manifest
        manifest = {
            "timestamp": timestamp,
            "version": self.version,
            "stats": asdict(self.get_stats()),
            "shards": [asdict(s) for s in self._get_all_shards()]
        }
        
        with open(backup_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created backup: {backup_dir}")
        return str(backup_dir)

    def compact_selfplay_to_replay(self):
        """Compacts self-play games into larger replay shards."""
        sp_files = sorted([x for x in self.selfplay_dir.glob("*.npz") if x.is_file()])
        if not sp_files:
            return

        # Determine next shard index
        existing = sorted(self.replays_dir.glob("shard_*.npz"))
        next_idx = 0
        if existing:
            last = existing[-1]
            try:
                next_idx = int(last.stem.split("_")[-1]) + 1
            except Exception:
                next_idx = len(existing)

        buf_s, buf_pi, buf_z = [], [], []
        buf_lm = []
        # Aggregates for dashboard
        games_count = 0
        sum_moves = 0
        sum_entropy = 0.0
        sum_avg_sims = 0.0
        resigned_count = 0
        draw_count = 0
        count = 0
        for f in sp_files:
            try:
                with np.load(f) as data:
                    s, pi, z = data["s"], data["pi"], data["z"]
                    lm = data.get("legal_mask")
                    # Collect per-game metadata if present (each file represents one game)
                    games_count += 1
                    try:
                        sum_moves += int(data.get("meta_moves", np.array([len(s)], dtype=np.int32))[0])
                    except Exception:
                        sum_moves += int(len(s))
                    try:
                        resigned_count += int(data.get("meta_resigned", np.array([0], dtype=np.int8))[0])
                    except Exception:
                        pass
                    try:
                        draw_count += int(data.get("meta_draw", np.array([0], dtype=np.int8))[0])
                    except Exception:
                        pass
                    try:
                        sum_entropy += float(data.get("meta_avg_policy_entropy", np.array([0.0], dtype=np.float32))[0])
                    except Exception:
                        pass
                    try:
                        sum_avg_sims += float(data.get("meta_avg_sims", np.array([0.0], dtype=np.float32))[0])
                    except Exception:
                        pass
                buf_s.append(s)
                buf_pi.append(pi)
                buf_z.append(z)
                if lm is not None:
                    buf_lm.append(lm)
                count += s.shape[0]
                # Move source file to backup instead of deleting
                backup_path = self.backups_dir / f.name
                backup_path.write_bytes(f.read_bytes())
                f.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error processing selfplay file {f}: {e}")
                self._mark_shard_corrupted(str(f))
                continue

            # Flush if buffer is large
            while count >= self.shard_size:
                take = self.shard_size
                s_cat = np.concatenate(buf_s, axis=0)
                pi_cat = np.concatenate(buf_pi, axis=0)
                z_cat = np.concatenate(buf_z, axis=0)
                lm_cat = None
                if buf_lm:
                    try:
                        lm_cat = np.concatenate(buf_lm, axis=0)
                    except Exception:
                        lm_cat = None
                shard_s, shard_pi, shard_z = s_cat[:take], pi_cat[:take], z_cat[:take]
                shard_lm = lm_cat[:take] if lm_cat is not None else None
                # Keep leftovers
                buf_s = [s_cat[take:]]
                buf_pi = [pi_cat[take:]]
                buf_z = [z_cat[take:]]
                if lm_cat is not None:
                    buf_lm = [lm_cat[take:]]
                else:
                    buf_lm = []
                count = s_cat.shape[0] - take
                
                shard_payload = {"s": shard_s, "pi": shard_pi, "z": shard_z}
                if shard_lm is not None:
                    shard_payload["legal_mask"] = shard_lm.astype(np.uint8, copy=False)
                self.add_training_data(shard_payload, next_idx)
                next_idx += 1

        # Flush tail
        if count > 0:
            s_cat = np.concatenate(buf_s, axis=0)
            pi_cat = np.concatenate(buf_pi, axis=0)
            z_cat = np.concatenate(buf_z, axis=0)
            lm_cat = None
            if buf_lm:
                try:
                    lm_cat = np.concatenate(buf_lm, axis=0)
                except Exception:
                    lm_cat = None
            payload = {"s": s_cat, "pi": pi_cat, "z": z_cat}
            if lm_cat is not None:
                payload["legal_mask"] = lm_cat.astype(np.uint8, copy=False)
            self.add_training_data(payload, next_idx)

        # Enforce max shards (keep most recent)
        self.cleanup_old_shards(keep_recent=self.max_shards)

        # Write dashboard CSV summary
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            csv_path = logs_dir / "selfplay_summary.csv"
            header = (
                "timestamp,processed_files,games,avg_moves,draw_rate,resign_rate,avg_policy_entropy,avg_sims\n"
            )
            if not csv_path.exists():
                csv_path.write_text(header)
            processed_files = len(sp_files)
            if games_count > 0:
                avg_moves = sum_moves / float(games_count)
                draw_rate = draw_count / float(games_count)
                resign_rate = resigned_count / float(games_count)
                avg_entropy = sum_entropy / float(games_count)
                avg_sims = sum_avg_sims / float(games_count)
            else:
                avg_moves = 0.0
                draw_rate = 0.0
                resign_rate = 0.0
                avg_entropy = 0.0
                avg_sims = 0.0
            line = f"{datetime.now().isoformat()},{processed_files},{games_count},{avg_moves:.3f},{draw_rate:.3f},{resign_rate:.3f},{avg_entropy:.3f},{avg_sims:.3f}\n"
            with csv_path.open('a') as fcsv:
                fcsv.write(line)
            logger.info(
                f"Self-play summary: files={processed_files} games={games_count} avg_moves={avg_moves:.1f} "
                f"draw={draw_rate:.1%} resign={resign_rate:.1%} entropy={avg_entropy:.3f} avg_sims={avg_sims:.1f}"
            )
        except Exception as e:
            logger.warning(f"Failed to write selfplay summary: {e}")

    def import_replay_dir(self, src_dir: str, source: str = "external", move_files: bool = False) -> int:
        """Import existing NPZ shards from a directory into the replay buffer and DB.

        If move_files is True, files are moved into the managed replay directory; otherwise
        they are left in place and simply recorded in the DB so training can read them.

        Returns number of files imported.
        """
        src = Path(src_dir)
        if not src.exists():
            return 0
        count = 0
        for f in sorted(src.glob('*.npz')):
            try:
                dest_path = f
                if move_files:
                    dest_path = self.replays_dir / f.name
                    if dest_path.resolve() != f.resolve():
                        dest_path.write_bytes(f.read_bytes())
                        f.unlink(missing_ok=True)
                # Load to get sample count quickly (supports aliased fields)
                with np.load(dest_path, mmap_mode='r') as data:
                    sample_count = int(self._infer_sample_count(data))
                size_bytes = dest_path.stat().st_size
                ts = datetime.now().isoformat()
                checksum = self._calculate_checksum(dest_path)
                self._record_shard(str(dest_path), size_bytes, sample_count, ts, checksum, source=source)
                count += 1
            except Exception as e:
                logger.error(f"Failed to import shard {f}: {e}")
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_field(self, data: "np.lib.npyio.NpzFile", field: str) -> np.ndarray:
        aliases = self._field_aliases.get(field, (field,))
        available = set(data.files) if hasattr(data, 'files') else set(data.keys())
        for alias in aliases:
            if alias in available:
                return data[alias]
        raise KeyError(f"Field '{field}' not found in shard (aliases tried: {aliases})")

    def _infer_sample_count(self, data: Dict[str, np.ndarray]) -> int:
        try:
            if hasattr(data, 'files'):
                available = set(data.files)
                for alias in self._field_aliases['s']:
                    if alias in available:
                        return int(data[alias].shape[0])
            else:
                for alias in self._field_aliases['s']:
                    if alias in data:
                        return int(data[alias].shape[0])
        except Exception:
            pass
        return 0

    def _extract_training_arrays(self, data: "np.lib.npyio.NpzFile") -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        states = self._resolve_field(data, 's')
        policies = self._resolve_field(data, 'pi')
        values = self._resolve_field(data, 'z')
        legal_mask = data.get('legal_mask', None)
        ssl_targets = {key: data[key] for key in data.files if key.startswith('ssl_')}
        return states, policies, values, legal_mask, ssl_targets

    def import_stockfish_tree(self, root_dir: str, move_files: bool = False) -> int:
        """Recursively import NPZ files from a Stockfish-generated tree, tagging sources.

        Expected layout: <root>/<domain>/<subcategory>/*.npz
        Records source as "stockfish:<domain>/<subcategory>" for filtering.
        """
        root = Path(root_dir)
        if not root.exists():
            logger.warning("Stockfish root does not exist: %s", root)
            return 0
        imported = 0
        for npz_path in root.rglob('*.npz'):
            try:
                rel = npz_path.relative_to(root)
                parts = rel.parts
                if len(parts) < 3:
                    # domain/subcategory/file.npz
                    source_tag = "stockfish:unknown"
                else:
                    domain = parts[0]
                    subcat = parts[1]
                    source_tag = f"stockfish:{domain}/{subcat}"
                imported += self.import_replay_dir(str(npz_path.parent), source=source_tag, move_files=move_files)
            except Exception as e:
                logger.error("Failed to import %s: %s", npz_path, e)
        return imported

    def _record_shard(self, path: str, size_bytes: int, sample_count: int, 
                      created_at: str, checksum: str, source: str = "selfplay"):
        """Record shard metadata in database."""
        conn = self._connect()
        cursor = conn.cursor()
        def _do():
            cursor.execute(
                """
                INSERT OR REPLACE INTO shards 
                (path, size_bytes, sample_count, created_at, checksum, version, source, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (path, size_bytes, sample_count, created_at, checksum, self.version, source, created_at)
            )
            conn.commit()
        self._with_retry(_do)
        conn.close()
    
    def _get_valid_shard_paths(self) -> List[str]:
        """Get paths of valid (non-corrupted) shards."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM shards WHERE corrupted = FALSE ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        paths = [row[0] for row in rows]
        # Filter out paths that don't exist on disk
        valid_paths = []
        for path in paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                self._mark_shard_corrupted(path)
        return valid_paths

    def _get_valid_shard_paths_by_source_prefixes(self, prefixes: List[str]) -> List[str]:
        """Get valid shard paths whose source starts with any of the given prefixes."""
        if not prefixes:
            return self._get_valid_shard_paths()
        conn = self._connect()
        cursor = conn.cursor()
        # Fetch path and source to filter in Python for portability
        cursor.execute("SELECT path, source FROM shards WHERE corrupted = FALSE ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        valid = []
        for path, source in rows:
            try:
                if not Path(path).exists():
                    self._mark_shard_corrupted(path)
                    continue
                src = source or ""
                if any(src.startswith(pref) for pref in prefixes):
                    valid.append(path)
            except Exception:
                continue
        return valid

    def get_training_batch_by_source_prefixes(self, batch_size: int, prefixes: List[str]) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """Yield training batches constrained to shards whose source matches any prefix.

        Prefix example: ["stockfish:openings/", "stockfish:king_safety/"]

        Note: SSL targets are loaded when present but returned as separate dicts for compatibility.
        """
        shard_paths = self._get_valid_shard_paths_by_source_prefixes(prefixes)
        if not shard_paths:
            raise RuntimeError("No valid training data available for requested sources")
        np.random.shuffle(shard_paths)
        for shard_path in shard_paths:
            try:
                with np.load(shard_path, mmap_mode='r') as data:
                    states, policies, values, legal_mask_all, ssl_targets_all = self._extract_training_arrays(data)

                    # Check if this is SSL-enabled data
                    ssl_keys = list(ssl_targets_all.keys())
                    # Note: SSL filtering will be handled at a higher level if needed

                    if values.ndim == 2 and values.shape[1] == 1:
                        values = values.reshape(values.shape[0])
                    if not self._validate_shapes(states, policies, values, self.expected_planes, shard_path):
                        self._mark_shard_corrupted(shard_path)
                        continue
                    if legal_mask_all is not None:
                        try:
                            if legal_mask_all.ndim > 2:
                                legal_mask_all = legal_mask_all.reshape(legal_mask_all.shape[0], -1)
                            if legal_mask_all.dtype != np.uint8:
                                legal_mask_all = legal_mask_all.astype(np.uint8, copy=False)
                        except Exception:
                            legal_mask_all = None

                    # Load SSL targets if present
                    ssl_targets = {}
                    for ssl_key in ssl_keys:
                        try:
                            ssl_data = ssl_targets_all[ssl_key]

                            # Convert control task from single-channel to 3-channel format
                            if ssl_key == 'ssl_control':
                                # Convert from (batch, 8, 8) with values [-1, 0, 1]
                                # to (batch, 3, 8, 8) with one-hot encoding
                                batch_size = ssl_data.shape[0]
                                control_3d = np.zeros((batch_size, 3, 8, 8), dtype=np.float32)

                                # Channel 0: black control (where data == -1)
                                control_3d[:, 0, :, :] = (ssl_data == -1).astype(np.float32)
                                # Channel 1: neutral (where data == 0)
                                control_3d[:, 1, :, :] = (ssl_data == 0).astype(np.float32)
                                # Channel 2: white control (where data == 1)
                                control_3d[:, 2, :, :] = (ssl_data == 1).astype(np.float32)

                                ssl_targets[ssl_key] = control_3d
                            else:
                                ssl_targets[ssl_key] = ssl_data

                        except Exception as e:
                            logger.warning(f"Failed to load SSL target {ssl_key}: {e}")

                    indices = np.random.permutation(len(states))
                    states = states[indices]
                    policies = policies[indices]
                    values = values[indices]
                    if legal_mask_all is not None:
                        legal_mask_all = legal_mask_all[indices]

                    # Shuffle SSL targets
                    for ssl_key in ssl_targets:
                        ssl_targets[ssl_key] = ssl_targets[ssl_key][indices]

                    for i in range(0, len(states), batch_size):
                        batch_states = states[i:i+batch_size]
                        batch_policies = policies[i:i+batch_size]
                        batch_values = values[i:i+batch_size]
                        batch_legal = None
                        if legal_mask_all is not None:
                            batch_legal = legal_mask_all[i:i+batch_size]
                            try:
                                target_mask = (batch_policies > 0)
                                if batch_legal.shape == target_mask.shape:
                                    batch_legal = np.logical_or(batch_legal.astype(np.uint8), target_mask.astype(np.uint8)).astype(np.uint8)
                            except Exception:
                                pass

                        # Extract batch SSL targets and store in class variable for retrieval
                        batch_ssl_targets = {}
                        for ssl_key in ssl_targets:
                            batch_ssl_targets[ssl_key] = ssl_targets[ssl_key][i:i+batch_size]

                        # Store SSL targets in class variable for the calling code to retrieve
                        self._current_ssl_targets = batch_ssl_targets

                        if not batch_states.flags['C_CONTIGUOUS']:
                            batch_states = np.ascontiguousarray(batch_states)
                        if not batch_policies.flags['C_CONTIGUOUS']:
                            batch_policies = np.ascontiguousarray(batch_policies)
                        if not batch_values.flags['C_CONTIGUOUS']:
                            batch_values = np.ascontiguousarray(batch_values)
                        if batch_states.dtype != np.float32:
                            batch_states = batch_states.astype(np.float32, copy=False)
                        if batch_policies.dtype != np.float32:
                            batch_policies = batch_policies.astype(np.float32, copy=False)
                        if batch_values.dtype != np.float32:
                            batch_values = batch_values.astype(np.float32, copy=False)
                        if len(batch_states) == batch_size:
                            yield batch_states, batch_policies, batch_values, batch_legal
            except Exception as e:
                logger.error(f"Error loading shard {shard_path}: {e}", exc_info=True)
                self._mark_shard_corrupted(shard_path)
                continue
    
    def _get_all_shards(self) -> List[DataShard]:
        """Get all shards with metadata."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT path, size_bytes, sample_count, created_at, checksum, version, source, corrupted
            FROM shards ORDER BY created_at DESC
        """)
        
        shards = []
        for row in cursor.fetchall():
            shards.append(DataShard(
                path=row[0],
                size_bytes=row[1],
                sample_count=row[2],
                created_at=row[3],
                checksum=row[4],
                version=row[5],
                source=row[6],
                corrupted=bool(row[7])
            ))
        
        conn.close()
        return shards
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _validate_shard(self, shard: DataShard) -> bool:
        """Validate a single shard's integrity."""
        try:
            filepath = Path(shard.path)
            if not filepath.exists():
                return False
            
            # Check file size
            if filepath.stat().st_size != shard.size_bytes:
                return False
            
            # Check checksum
            current_checksum = self._calculate_checksum(filepath)
            if current_checksum != shard.checksum:
                return False
            
            # Try to load data
            with np.load(filepath) as data:
                required_keys = ['s', 'pi', 'z']
                if not all(key in data for key in required_keys):
                    return False
                
                # Check sample count
                if len(data['s']) != shard.sample_count:
                    return False
            
            return True
            
        except Exception:
            return False

    def _validate_shapes(self,
                         states: np.ndarray,
                         policies: np.ndarray,
                         values: np.ndarray,
                         expected_planes: int,
                         source: str = "") -> bool:
        """Ensure training data has expected shapes.

        Expected shapes:
            states: (N, ``expected_planes``, 8, 8)
            policies: (N, 4672)
            values: (N,) or (N,1)

        Args:
            states: State tensor
            policies: Policy tensor
            values: Value tensor
            expected_planes: Number of input planes expected in ``states``
            source: Optional identifier for logging

        Returns:
            True if shapes match expectations, False otherwise.
        """
        n = states.shape[0]
        ok_states = (states.shape == (n, expected_planes, 8, 8))
        # Accept legacy 4672 and future 1858 policy sizes
        ok_policies = (policies.shape[0] == n and policies.ndim == 2 and policies.shape[1] in (4672, 1858))
        ok_values = (values.shape == (n,)) or (values.ndim == 2 and values.shape == (n, 1))
        if not (ok_states and ok_policies and ok_values):
            logger.warning(
                f"Shape mismatch in {source}: states {states.shape}, policies {policies.shape}, values {values.shape}"
            )
            return False
        return True

    def _mark_shard_corrupted(self, path: str):
        """Mark a shard as corrupted in the database."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("UPDATE shards SET corrupted = TRUE WHERE path = ?", (path,))
        
        conn.commit()
        conn.close()
    
    def _remove_shard_record(self, path: str):
        """Remove a shard record from the database."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM shards WHERE path = ?", (path,))
        
        conn.commit()
        conn.close()

    def quarantine_corrupted_shards(self, quarantine_dir: str | None = None) -> int:
        """Move corrupted shards to a quarantine directory and remove their DB records.

        Returns number of files quarantined.
        """
        qdir = Path(quarantine_dir or (self.backups_dir / "quarantine"))
        qdir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM shards WHERE corrupted = TRUE")
        rows = cursor.fetchall()
        count = 0
        for (pstr,) in rows:
            p = Path(pstr)
            try:
                if p.exists():
                    p.rename(qdir / p.name)
                self._remove_shard_record(pstr)
                count += 1
            except Exception:
                continue
        conn.close()
        return count

    def validate_all_data_sources(self) -> Dict[str, Dict[str, any]]:
        """Validate all data sources and return comprehensive status."""
        validation_results = {}
        
        # Validate selfplay data
        try:
            selfplay_files = list(self.selfplay_dir.glob("*.npz"))
            validation_results['selfplay'] = {
                'count': len(selfplay_files),
                'total_size_mb': sum(f.stat().st_size for f in selfplay_files) / (1024 * 1024),
                'status': 'ok' if selfplay_files else 'empty'
            }
        except Exception as e:
            validation_results['selfplay'] = {'status': 'error', 'error': str(e)}
        
        # Validate replay data
        try:
            replay_files = list(self.replays_dir.glob("*.npz"))
            validation_results['replays'] = {
                'count': len(replay_files),
                'total_size_mb': sum(f.stat().st_size for f in replay_files) / (1024 * 1024),
                'status': 'ok' if replay_files else 'empty'
            }
        except Exception as e:
            validation_results['replays'] = {'status': 'error', 'error': str(e)}
        
        # Validate tactical data
        try:
            tactical_path = Path(self.base_dir) / "training" / "tactical_training_data.npz"
            if tactical_path.exists():
                with np.load(tactical_path) as data:
                    validation_results['tactical'] = {
                        'count': len(data['positions']),
                        'size_mb': tactical_path.stat().st_size / (1024 * 1024),
                        'keys': list(data.keys()),
                        'status': 'ok'
                    }
            else:
                validation_results['tactical'] = {'status': 'missing'}
        except Exception as e:
            validation_results['tactical'] = {'status': 'error', 'error': str(e)}
        
        # Validate openings data
        try:
            openings_path = Path(self.base_dir) / "training" / "openings_training_data.npz"
            if openings_path.exists():
                with np.load(openings_path) as data:
                    validation_results['openings'] = {
                        'count': len(data['positions']),
                        'size_mb': openings_path.stat().st_size / (1024 * 1024),
                        'keys': list(data.keys()),
                        'status': 'ok'
                    }
            else:
                validation_results['openings'] = {'status': 'missing'}
        except Exception as e:
            validation_results['openings'] = {'status': 'error', 'error': str(e)}
        
        # Validate lichess data
        try:
            lichess_dir = Path(self.base_dir) / "lichess"
            lichess_files = list(lichess_dir.glob("*.npz"))
            if lichess_files:
                # Sample one file to check format
                sample_data = np.load(lichess_files[0])
                validation_results['lichess'] = {
                    'count': len(lichess_files),
                    'total_size_mb': sum(f.stat().st_size for f in lichess_files) / (1024 * 1024),
                    'sample_keys': list(sample_data.keys()),
                    'status': 'ok'
                }
            else:
                validation_results['lichess'] = {'status': 'empty'}
        except Exception as e:
            validation_results['lichess'] = {'status': 'error', 'error': str(e)}
        
        return validation_results

    def get_data_summary(self) -> str:
        """Get a human-readable summary of all data sources."""
        validation = self.validate_all_data_sources()
        summary = []
        
        for source, info in validation.items():
            if info['status'] == 'ok':
                if 'count' in info:
                    summary.append(f"{source}: {info['count']} samples")
                else:
                    summary.append(f"{source}: available")
            elif info['status'] == 'empty':
                summary.append(f"{source}: empty")
            elif info['status'] == 'missing':
                summary.append(f"{source}: missing")
            else:
                summary.append(f"{source}: error - {info.get('error', 'unknown')}")
        
        return " | ".join(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix0 Data Manager CLI")
    parser.add_argument("--base-dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--action", type=str, required=True,
                        choices=["stats", "validate", "quarantine", "compact", "backup", "import"],
                        help="Action to perform")
    parser.add_argument("--path", type=str, default=None, help="Path for import action")
    args = parser.parse_args()

    dm = DataManager(base_dir=args.base_dir)

    if args.action == "stats":
        stats = dm.get_stats()
        print(json.dumps(asdict(stats), indent=2))
    elif args.action == "validate":
        valid, corrupted = dm.validate_data_integrity()
        print(f"Validation complete. Valid shards: {valid}, Corrupted shards: {corrupted}")
    elif args.action == "quarantine":
        count = dm.quarantine_corrupted_shards()
        print(f"Quarantined {count} corrupted shards.")
    elif args.action == "compact":
        dm.compact_selfplay_to_replay()
        stats = dm.get_stats()
        print(json.dumps(asdict(stats), indent=2))
    elif args.action == "backup":
        path = dm.create_backup()
        print(json.dumps({"backup": path}, indent=2))
    elif args.action == "import":
        if not args.path:
            raise SystemExit("--path required for import action")
        n = dm.import_replay_dir(args.path, source="external", move_files=False)
        print(json.dumps({"imported": n, "from": args.path}, indent=2))
