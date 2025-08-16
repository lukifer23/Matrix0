from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DirStats:
    path: str
    files: int
    bytes: int


def dir_stats(path: str) -> DirStats:
    p = Path(path)
    files = 0
    total = 0
    if not p.exists():
        return DirStats(path=str(p), files=0, bytes=0)
    for f in p.rglob("*"):
        if f.is_file():
            files += 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return DirStats(path=str(p), files=files, bytes=total)


def disk_free(path: str) -> Optional[int]:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except Exception:
        return None


def memory_usage_bytes() -> Optional[int]:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss
    except Exception:
        return None

