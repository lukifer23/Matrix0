from __future__ import annotations

import os
import json
import hashlib
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from datetime import datetime
import numpy as np
import logging
import argparse
import time

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
    corrupted: bool = False


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
    
    def __init__(self, base_dir: str = "data", max_shards: int = 128, shard_size: int = 16384):
        self.base_dir = Path(base_dir)
        self.max_shards = max_shards
        self.shard_size = shard_size
        
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
        
        # Initialize database for tracking
        self.db_path = self.base_dir / "data_metadata.db"
        self._init_database()
        
        # Version tracking
        self.version = "1.0.0"
        
    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection with WAL and busy timeout enabled."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        c = conn.cursor()
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
        except Exception:
            pass
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
                except Exception:
                    pass
            if migrations:
                conn.commit()
        except Exception:
            pass
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
    
    def add_selfplay_data(self, data: Dict[str, np.ndarray], worker_id: int, game_id: int) -> str:
        """Add self-play data to the buffer."""
        # Use filesystem-safe timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selfplay_w{worker_id}_g{game_id}_{timestamp}.npz"
        # Ensure directory exists (defensive; created in __init__ but workers may run in fresh procs)
        self.selfplay_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.selfplay_dir / filename
        
        # Save atomically with retries: write to temp then replace
        attempts = 0
        while True:
            try:
                # Create temporary file in the same directory and write to its handle
                import tempfile, os
                with tempfile.NamedTemporaryFile(dir=str(self.selfplay_dir), suffix='.npz.tmp', delete=False) as tf:
                    tmp_path = Path(tf.name)
                    np.savez_compressed(tf, **data)  # write to the open handle to avoid suffix issues
                os.replace(tmp_path, filepath)
                break
            except Exception as e:
                attempts += 1
                if attempts >= 3:
                    logger.error(f"Failed to save selfplay data after {attempts} attempts: {e}")
                    raise
                logger.warning(f"Save attempt {attempts} failed, retrying: {e}")
                time.sleep(0.1 * attempts)
                # Clean up failed temp file if it exists
                try:
                    if 'tmp_path' in locals() and Path(tmp_path).exists():
                        Path(tmp_path).unlink()
                except Exception:
                    pass
        
        # Calculate metadata
        file_size = filepath.stat().st_size
        sample_count = len(data.get('s', []))
        checksum = self._calculate_checksum(filepath)
        
        # Record in database
        self._record_shard(str(filepath), file_size, sample_count, timestamp, checksum, source="selfplay")
        
        logger.info(f"Added self-play data: {filename} ({sample_count} samples, {file_size/1024:.1f}KB)")
        return str(filepath)
    
    def add_training_data(self, data: Dict[str, np.ndarray], shard_id: int, source: str = "selfplay") -> str:
        """Add processed training data to replay buffer."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"replay_{shard_id:06d}_{timestamp}.npz"
        filepath = self.replays_dir / filename
        
        # Save atomically with retries
        attempts = 0
        while True:
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(dir=str(self.replays_dir), suffix='.npz.tmp', delete=False) as tf:
                    tmp_path = Path(tf.name)
                    np.savez_compressed(tf, **data)
                os.replace(tmp_path, filepath)
                break
            except Exception as e:
                attempts += 1
                if attempts >= 3:
                    logger.error(f"Failed to save training data after {attempts} attempts: {e}")
                    raise
                logger.warning(f"Save attempt {attempts} failed, retrying: {e}")
                time.sleep(0.1 * attempts)
                try:
                    if 'tmp_path' in locals() and Path(tmp_path).exists():
                        Path(tmp_path).unlink()
                except Exception:
                    pass
        
        # Calculate metadata
        file_size = filepath.stat().st_size
        sample_count = len(data.get('s', []))
        checksum = self._calculate_checksum(filepath)
        
        # Record in database
        self._record_shard(str(filepath), file_size, sample_count, timestamp, checksum, source=source)
        
        logger.info(f"Added training data: {filename} ({sample_count} samples, {file_size/1024:.1f}KB)")
        return str(filepath)
    
    def get_training_batch(self, batch_size: int, device: str = "cpu") -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get training batches from replay buffer."""
        shard_paths = self._get_valid_shard_paths()
        
        if not shard_paths:
            raise RuntimeError("No valid training data available")
        
        # Randomly sample from shards
        np.random.shuffle(shard_paths)
        
        for shard_path in shard_paths:
            try:
                # Memory-map to reduce RSS and speed IO
                with np.load(shard_path, mmap_mode='r') as data:
                    states = data['s']
                    policies = data['pi']
                    values = data['z']
                    
                    # Shuffle within shard
                    indices = np.random.permutation(len(states))
                    states = states[indices]
                    policies = policies[indices]
                    values = values[indices]
                    
                    # Yield batches
                    for i in range(0, len(states), batch_size):
                        batch_states = states[i:i+batch_size]
                        batch_policies = policies[i:i+batch_size]
                        batch_values = values[i:i+batch_size]
                        
                        if len(batch_states) == batch_size:
                            yield batch_states, batch_policies, batch_values
                            
            except Exception as e:
                logger.error(f"Error loading shard {shard_path}: {e}")
                self._mark_shard_corrupted(shard_path)
                continue
    
    def get_external_training_batch(self, batch_size: int, source: str = "mixed") -> Optional[Dict[str, np.ndarray]]:
        """Get training batches from external training data sources.
        
        Args:
            batch_size: Size of training batch
            source: Data source ("tactical", "openings", "mixed")
            
        Returns:
            Training batch dict with keys 's', 'pi', 'z' or None if no data
        """
        try:
            if source == "tactical":
                return self._get_tactical_batch(batch_size)
            elif source == "openings":
                return self._get_openings_batch(batch_size)
            elif source == "mixed":
                return self._get_mixed_external_batch(batch_size)
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
        tactical_path = Path(self.base_dir) / "training" / "tactical_training_data.npz"
        if not tactical_path.exists():
            logger.warning("Tactical training data not found")
            return None
        
        try:
            with np.load(tactical_path) as data:
                indices = np.random.choice(len(data['positions']), batch_size, replace=False)
                return {
                    's': data['positions'][indices],
                    'pi': data['policy_targets'][indices],
                    'z': data['value_targets'][indices]
                }
        except Exception as e:
            logger.error(f"Error loading tactical data: {e}")
            return None
    
    def _get_openings_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get batch from openings training data."""
        openings_path = Path(self.base_dir) / "training" / "openings_training_data.npz"
        if not openings_path.exists():
            logger.warning("Openings training data not found")
            return None
        
        try:
            with np.load(openings_path) as data:
                indices = np.random.choice(len(data['positions']), batch_size, replace=False)
                return {
                    's': data['positions'][indices],
                    'pi': data['policy_targets'][indices],
                    'z': data['value_targets'][indices]
                }
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
        combined_batch = {}
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = np.concatenate([
                tactical_batch[key], openings_batch[key]
            ], axis=0)
        
        # Shuffle the combined batch
        indices = np.random.permutation(len(combined_batch['s']))
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = combined_batch[key][indices]
        
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
        combined_batch = {}
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = np.concatenate([
                openings_batch[key], tactical_batch[key]
            ], axis=0)
        
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
        combined_batch = {}
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = np.concatenate([
                tactical_batch[key], openings_batch[key]
            ], axis=0)
        
        return combined_batch
    
    def _get_curriculum_mixed_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get curriculum batch with balanced mix and self-play data."""
        # Try to get external data first
        external_batch = self._get_mixed_external_batch(batch_size // 2)
        
        # Try to get self-play data for the other half
        try:
            # Get a batch from the regular replay buffer
            sp_generator = self.get_training_batch(batch_size // 2, "cpu")
            sp_batch = next(sp_generator)
            sp_dict = {
                's': sp_batch[0],
                'pi': sp_batch[1], 
                'z': sp_batch[2]
            }
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
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = np.concatenate([
                external_batch[key], sp_dict[key]
            ], axis=0)
        
        # Shuffle the combined batch
        indices = np.random.permutation(len(combined_batch['s']))
        for key in ['s', 'pi', 'z']:
            combined_batch[key] = combined_batch[key][indices]
        
        return combined_batch
    
    def get_external_data_stats(self) -> Dict[str, int]:
        """Get statistics about external training data availability."""
        stats = {
            'tactical_samples': 0,
            'openings_samples': 0,
            'external_total': 0
        }
        
        # Check tactical data
        tactical_path = Path(self.base_dir) / "training" / "tactical_training_data.npz"
        if tactical_path.exists():
            try:
                with np.load(tactical_path) as data:
                    stats['tactical_samples'] = len(data['positions'])
            except Exception:
                pass
        
        # Check openings data
        openings_path = Path(self.base_dir) / "training" / "openings_training_data.npz"
        if openings_path.exists():
            try:
                with np.load(openings_path) as data:
                    stats['openings_samples'] = len(data['positions'])
            except Exception:
                pass
        
        stats['external_total'] = stats['tactical_samples'] + stats['openings_samples']
        return stats
    
    def cleanup_old_shards(self, keep_recent: int = 64):
        """Remove old shards to maintain storage limits."""
        shards = self._get_all_shards()
        
        if len(shards) <= keep_recent:
            return
        
        # Sort by creation time, keep most recent
        shards.sort(key=lambda x: x.created_at, reverse=True)
        to_remove = shards[keep_recent:]
        
        for shard in to_remove:
            try:
                Path(shard.path).unlink()
                self._remove_shard_record(shard.path)
                logger.info(f"Removed old shard: {shard.path}")
            except Exception as e:
                logger.error(f"Error removing shard {shard.path}: {e}")
    
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
                shard_s, shard_pi, shard_z = s_cat[:take], pi_cat[:take], z_cat[:take]
                # Keep leftovers
                buf_s = [s_cat[take:]]
                buf_pi = [pi_cat[take:]]
                buf_z = [z_cat[take:]]
                count = s_cat.shape[0] - take
                
                self.add_training_data({"s": shard_s, "pi": shard_pi, "z": shard_z}, next_idx)
                next_idx += 1

        # Flush tail
        if count > 0:
            s_cat = np.concatenate(buf_s, axis=0)
            pi_cat = np.concatenate(buf_pi, axis=0)
            z_cat = np.concatenate(buf_z, axis=0)
            self.add_training_data({"s": s_cat, "pi": pi_cat, "z": z_cat}, next_idx)

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
                # Load to get sample count quickly
                with np.load(dest_path, mmap_mode='r') as data:
                    s = data['s']
                    sample_count = int(s.shape[0])
                size_bytes = dest_path.stat().st_size
                ts = datetime.now().isoformat()
                checksum = self._calculate_checksum(dest_path)
                self._record_shard(str(dest_path), size_bytes, sample_count, ts, checksum, source=source)
                count += 1
            except Exception as e:
                logger.error(f"Failed to import shard {f}: {e}")
        return count

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
        paths = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Filter out paths that don't exist on disk
        valid_paths = []
        for path in paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                # Mark as corrupted if file doesn't exist
                self._mark_shard_corrupted(path)
        
        return valid_paths
    
    def _get_all_shards(self) -> List[DataShard]:
        """Get all shards with metadata."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT path, size_bytes, sample_count, created_at, checksum, version, corrupted
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
                corrupted=bool(row[6])
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
