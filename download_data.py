#!/usr/bin/env python3
"""Download the public nflverse and ffopportunity files needed to rebuild.

Raw play-by-play parquet files are large and are gitignored. Re-run this
script from the repo root whenever you want a fresh snapshot.

Sources (all public GitHub releases):
  - nflverse/nflverse-data  — play-by-play, rosters, timestamps (CC BY 4.0)
  - ffverse/ffopportunity   — expected fantasy points used only as a
    comparison / validation baseline
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
FFOPP = "https://github.com/ffverse/ffopportunity/releases/download/latest-data"

FILES = [
    (f"{NFLVERSE}/pbp/timestamp.json", DATA / "pbp_timestamp.json"),
    (f"{FFOPP}/timestamp.txt", DATA / "ffopp_timestamp.txt"),
    (f"{FFOPP}/ep_weekly_2026.parquet", DATA / "ep_weekly_2026.parquet"),
    (f"{NFLVERSE}/rosters/roster_2025.parquet", DATA / "roster_2025.parquet"),
    (f"{NFLVERSE}/rosters/roster_2026.parquet", DATA / "roster_2026.parquet"),
]

for year in (2023, 2024, 2025, 2026):
    FILES.append(
        (
            f"{NFLVERSE}/pbp/play_by_play_{year}.parquet",
            DATA / f"play_by_play_{year}.parquet",
        )
    )


def _progress(url: str):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = min(downloaded, total_size) / 1e6
            total_mb = total_size / 1e6
            sys.stdout.write(f"\r  {mb:6.1f}/{total_mb:6.1f} MB ({pct:5.1f}%)  {url.split('/')[-1]}")
        else:
            sys.stdout.write(f"\r  {downloaded/1e6:6.1f} MB  {url.split('/')[-1]}")
        sys.stdout.flush()
    return hook


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"skip (exists): {dest.name}")
        return
    print(f"download {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress(url))
        print()
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> None:
    force = "--force" in sys.argv
    for url, dest in FILES:
        download(url, dest, force=force)
    print(f"\nData is in {DATA}")
    print("Next: python xtd_first_pass.py && python xfp_model.py && python build_explorer.py")


if __name__ == "__main__":
    main()
