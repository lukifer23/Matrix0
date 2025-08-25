#!/usr/bin/env python3
"""Clean up stray temporary NPZ files.

Recursively search a data directory for ``*.npz.tmp`` or ``*.npz.tmp.npz`` files.
If a matching final ``.npz`` file does not exist, remove the temporary file or
optionally move it to a quarantine directory.  A dry-run mode is available to
preview actions without modifying the filesystem.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from azchess.logging_utils import setup_logging


def find_temp_files(root: Path):
    """Yield temporary NPZ files under ``root``."""
    patterns = ["*.npz.tmp", "*.npz.tmp.npz"]
    for pattern in patterns:
        yield from root.rglob(pattern)


def resolve_final_path(temp_path: Path) -> Path:
    """Return the expected final ``.npz`` path for ``temp_path``."""
    if temp_path.name.endswith(".npz.tmp"):
        return Path(str(temp_path)[:-4])  # strip ".tmp"
    if temp_path.name.endswith(".npz.tmp.npz"):
        return Path(str(temp_path)[:-8])  # strip ".tmp.npz"
    return temp_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Directory to scan recursively for temporary NPZ files.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        help="If set, move files here instead of deleting them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without deleting or moving files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    args = parser.parse_args()

    logger = setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    root = args.data_dir
    if not root.exists():
        logger.error("Data directory %s does not exist", root)
        return 1

    quarantine_dir = args.quarantine_dir
    if quarantine_dir and not quarantine_dir.exists():
        if args.dry_run:
            logger.info(
                "[dry-run] Would create quarantine directory %s", quarantine_dir
            )
        else:
            quarantine_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for temp_path in find_temp_files(root):
        final_path = resolve_final_path(temp_path)
        if final_path.exists():
            logger.debug("Skipping %s because %s exists", temp_path, final_path)
            continue

        if args.dry_run:
            logger.info("[dry-run] Removing %s", temp_path)
            processed += 1
            continue

        try:
            if quarantine_dir:
                dest = quarantine_dir / temp_path.name
                shutil.move(str(temp_path), dest)
                logger.info("Moved %s to %s", temp_path, dest)
            else:
                temp_path.unlink()
                logger.info("Removed %s", temp_path)
            processed += 1
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to process %s: %s", temp_path, exc)

    logger.info("Processed %d temporary files", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
