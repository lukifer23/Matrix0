from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


class JSONLHandler(logging.Handler):
    def __init__(self, path: Path, level=logging.INFO, max_bytes: int = 5_000_000, backup_count: int = 5):
        super().__init__(level)
        self.path = path
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate(self) -> None:
        try:
            if not self.path.exists():
                return
            size = self.path.stat().st_size
            if size <= self.max_bytes:
                return
            # Shift older backups
            for i in range(self.backup_count - 1, 0, -1):
                src = self.path.with_suffix(self.path.suffix + f".{i}")
                dst = self.path.with_suffix(self.path.suffix + f".{i+1}")
                if src.exists():
                    try:
                        if dst.exists():
                            dst.unlink()
                    except Exception:
                        pass
                    src.rename(dst)
            # Move current to .1 and recreate file
            first_backup = self.path.with_suffix(self.path.suffix + ".1")
            try:
                if first_backup.exists():
                    first_backup.unlink()
            except Exception:
                pass
            self.path.rename(first_backup)
        except Exception:
            # Ignore rotation errors to avoid breaking logging
            pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "ts": record.created,
                "level": record.levelname,
                "msg": record.getMessage(),
                "name": record.name,
                "module": record.module,
                "func": record.funcName,
                "line": record.lineno,
            }
            # Rotate if file too large
            self._rotate()
            with self.path.open("a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            self.handleError(record)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    console = RichHandler(rich_tracebacks=False, markup=True)
    console.setLevel(level)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(Path(log_dir) / "matrix0.log", maxBytes=5_000_000, backupCount=5)
    file_handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    jsonl_handler = JSONLHandler(Path(log_dir) / "structured.jsonl", level=level, max_bytes=5_000_000, backup_count=5)
    logger.addHandler(jsonl_handler)

    logger.propagate = False
    return logger
