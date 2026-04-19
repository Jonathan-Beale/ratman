"""
utils_io.py

Filesystem helpers, logging, and input/output discovery.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
import shutil
import sys

from config import Settings


def log(settings: Settings, *args) -> None:
    print("[pipeline]", *args)
    sys.stdout.flush()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_pose_videos(settings: Settings) -> List[Path]:
    if not settings.pose_input_dir.exists():
        return []
    return sorted(
        p for p in settings.pose_input_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm")
    )


def list_ref_images(settings: Settings) -> List[Path]:
    if not settings.ref_input_dir.exists():
        return []
    return sorted(
        p for p in settings.ref_input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )


def make_pose_cache_dir(settings: Settings, pose_video: Path) -> Path:
    base = settings.output_dir / "_pose_cache" / pose_video.stem
    ensure_dir(base)
    return base


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_timestamped_run_dir(root_dir: Path, pose_video: Path, ref_image: Path, stamp: str, suffix: str) -> Path:
    base = root_dir / pose_video.stem / f"{ref_image.stem}_{stamp}_{suffix}"
    ensure_dir(base)
    return base


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)