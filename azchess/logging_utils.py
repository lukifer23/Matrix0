from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


class JSONLHandler(logging.Handler):
    def __init__(self, path: Path, level=logging.INFO):
        super().__init__(level)
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

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

    jsonl_handler = JSONLHandler(Path(log_dir) / "structured.jsonl", level=level)
    logger.addHandler(jsonl_handler)

    logger.propagate = False
    return logger
