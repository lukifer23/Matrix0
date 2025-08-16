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
        
    def _init_database(self):
        """Initialize SQLite database for metadata tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shards (
                path TEXT PRIMARY KEY,
                size_bytes INTEGER,
                sample_count INTEGER,
                created_at TEXT,
                checksum TEXT,
                version TEXT,
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
        conn.close()
    
    def add_selfplay_data(self, data: Dict[str, np.ndarray], worker_id: int, game_id: int) -> str:
        """Add self-play data to the buffer."""
        timestamp = datetime.now().isoformat()
        filename = f"selfplay_w{worker_id}_g{game_id}_{timestamp}.npz"
        filepath = self.selfplay_dir / filename
        
        # Save data
        np.savez_compressed(filepath, **data)
        
        # Calculate metadata
        file_size = filepath.stat().st_size
        sample_count = len(data.get('s', []))
        checksum = self._calculate_checksum(filepath)
        
        # Record in database
        self._record_shard(str(filepath), file_size, sample_count, timestamp, checksum)
        
        logger.info(f"Added self-play data: {filename} ({sample_count} samples, {file_size/1024:.1f}KB)")
        return str(filepath)
    
    def add_training_data(self, data: Dict[str, np.ndarray], shard_id: int) -> str:
        """Add processed training data to replay buffer."""
        timestamp = datetime.now().isoformat()
        filename = f"replay_{shard_id:06d}_{timestamp}.npz"
        filepath = self.replays_dir / filename
        
        # Save data
        np.savez_compressed(filepath, **data)
        
        # Calculate metadata
        file_size = filepath.stat().st_size
        sample_count = len(data.get('s', []))
        checksum = self._calculate_checksum(filepath)
        
        # Record in database
        self._record_shard(str(filepath), file_size, sample_count, timestamp, checksum)
        
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
                with np.load(shard_path) as data:
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
        count = 0
        for f in sp_files:
            try:
                with np.load(f) as data:
                    s, pi, z = data["s"], data["pi"], data["z"]
                buf_s.append(s)
                buf_pi.append(pi)
                buf_z.append(z)
                count += s.shape[0]
                # Move source file to backup or remove
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

    def _record_shard(self, path: str, size_bytes: int, sample_count: int, 
                      created_at: str, checksum: str):
        """Record shard metadata in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO shards 
            (path, size_bytes, sample_count, created_at, checksum, version, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (path, size_bytes, sample_count, created_at, checksum, self.version, created_at))
        
        conn.commit()
        conn.close()
    
    def _get_valid_shard_paths(self) -> List[str]:
        """Get paths of valid (non-corrupted) shards."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT path FROM shards WHERE corrupted = FALSE ORDER BY created_at DESC")
        paths = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return paths
    
    def _get_all_shards(self) -> List[DataShard]:
        """Get all shards with metadata."""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE shards SET corrupted = TRUE WHERE path = ?", (path,))
        
        conn.commit()
        conn.close()
    
    def _remove_shard_record(self, path: str):
        """Remove a shard record from the database."""
        conn = sqlite3.connect(self.db_path)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix0 Data Manager CLI")
    parser.add_argument("--action", type=str, required=True, choices=["stats", "validate", "quarantine"], help="Action to perform")
    args = parser.parse_args()

    dm = DataManager()

    if args.action == "stats":
        stats = dm.get_stats()
        print(json.dumps(asdict(stats), indent=2))
    elif args.action == "validate":
        valid, corrupted = dm.validate_data_integrity()
        print(f"Validation complete. Valid shards: {valid}, Corrupted shards: {corrupted}")
    elif args.action == "quarantine":
        count = dm.quarantine_corrupted_shards()
        print(f"Quarantined {count} corrupted shards.")
