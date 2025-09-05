#!/usr/bin/env python3
"""
Fetch Syzygy WDL tablebases for Matrix0 self-play endgame adjudication.

Defaults:
- Downloads 3-4-5 men WDL (.rtbw) from the Lichess mirror (fast, public).
- Destination: data/syzygy

Usage examples:
  python tools/fetch_syzygy.py                          # 3-4-5 WDL to data/syzygy
  python tools/fetch_syzygy.py --sets 6-wdl             # 6-men WDL (large)
  python tools/fetch_syzygy.py --dest /path/to/tb       # custom destination
  python tools/fetch_syzygy.py --limit 10               # fetch only 10 files to test
  python tools/fetch_syzygy.py --continue               # skip files that already exist

Notes:
- WDL (.rtbw) is sufficient for our use (probe_wdl).
- 3-4-5 WDL ~1–2GB. 6-men WDL ~20–30GB. 7-men is very large — not recommended.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen, Request, urlretrieve


MIRRORS = {
    # Lichess standard chess Syzygy mirrors
    "345-wdl": "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/",
    "6-wdl": "https://tablebase.lichess.ovh/tables/standard/6-wdl/",
}


def list_rtbw_urls(index_url: str) -> list[str]:
    req = Request(index_url, headers={"User-Agent": "Matrix0-fetcher/1.0"})
    with urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    hrefs = re.findall(r'href=\"([^\"]+)\"', html)
    files = [h for h in hrefs if h.endswith('.rtbw')]
    return [urljoin(index_url, f) for f in files]


def human(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def fetch(urls: list[str], dest: Path, limit: int = 0, cont: bool = True) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for url in urls:
        if limit and count >= limit:
            break
        fname = url.split('/')[-1]
        out = dest / fname
        if cont and out.exists() and out.stat().st_size > 0:
            print(f"✅ Skip (exists): {fname}")
            count += 1
            continue
        try:
            t0 = time.time()
            print(f"⬇️  Fetching {fname} …")
            tmp = str(out) + ".part"
            urlretrieve(url, tmp)
            os.replace(tmp, out)
            dt = time.time() - t0
            sz = out.stat().st_size
            print(f"✅ Done: {fname} ({human(sz)}) in {dt:.1f}s")
            count += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            print(f"⚠️  Failed: {fname} ({e})")
            try:
                if Path(tmp).exists():
                    Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Syzygy WDL tablebases (rtbw)")
    ap.add_argument("--dest", type=str, default="data/syzygy", help="Destination directory")
    ap.add_argument("--sets", type=str, nargs="+", default=["345-wdl"], choices=["345-wdl", "6-wdl"], help="Which sets to fetch")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = all)")
    ap.add_argument("--continue", dest="cont", action="store_true", help="Skip already downloaded files")
    args = ap.parse_args()

    dest = Path(args.dest)
    all_urls: list[str] = []
    for s in args.sets:
        idx = MIRRORS[s]
        print(f"Listing {s} from {idx}")
        urls = list_rtbw_urls(idx)
        urls.sort()
        print(f"Found {len(urls)} files in {s}")
        all_urls.extend(urls)

    if not all_urls:
        print("No files to download.")
        sys.exit(1)

    fetch(all_urls, dest, limit=int(args.limit or 0), cont=bool(args.cont))
    print("\nAll requested downloads processed.")


if __name__ == "__main__":
    main()
