"""
video_frames.py

Video to frame utils that base output resolution on input pose resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import subprocess

import cv2
import numpy as np

from config import Settings
from utils_io import ensure_dir, log


# -------------------------
# Basic video info
# -------------------------

def get_video_info(video_path: Path) -> Tuple[float, int, int]:
    """
    Returns (fps, width, height) from the source video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or fps != fps:
        fps = 0.0

    return fps, w, h


# -------------------------
# Resolution helpers
# -------------------------

def _round_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 1:
        return x
    return max(multiple, int(round(x / multiple)) * multiple)


def compute_working_size(
    src_w: int,
    src_h: int,
    scale: float,
    multiple: int = 8,
    max_long_side: int | None = None,
) -> Tuple[int, int]:
    """
    Compute a working (generation) resolution from the source video resolution.

    - Preserves aspect ratio
    - Scales both dimensions by `scale`
    - Optionally caps the longest side to `max_long_side`
    - Rounds to nearest `multiple` (e.g. 8) for diffusion friendliness

    Example:
      src=436x536, scale=1.2, max_long_side=640 -> ~520x640
      src=1920x1080, scale=1.2, max_long_side=640 -> ~640x360
    """
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0")

    tw = max(1, int(round(src_w * scale)))
    th = max(1, int(round(src_h * scale)))

    if max_long_side is not None and max_long_side > 0:
        current_long = max(tw, th)
        if current_long > max_long_side:
            shrink = float(max_long_side) / float(current_long)
            tw = max(1, int(round(tw * shrink)))
            th = max(1, int(round(th * shrink)))

    tw = _round_to_multiple(tw, multiple)
    th = _round_to_multiple(th, multiple)
    return tw, th


def resize_letterbox(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """
    Resize with preserved aspect ratio and letterbox padding to exactly (target_w, target_h).
    """
    if target_w <= 0 or target_h <= 0:
        return frame

    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def resize_and_pad(
    frame: np.ndarray,
    target_size: int,
    pad_to_square: bool,
) -> np.ndarray:
    """
    Resize so longer side == target_size and optionally pad to square.

    Kind of deprecated.
    """
    if target_size <= 0:
        return frame

    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if not pad_to_square:
        return resized

    canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# -------------------------
# Video decoding
# -------------------------

def extract_frames(
    settings: Settings,
    video_path: Path,
    out_dir: Path,
    *,
    target_size: Optional[Tuple[int, int]] = None,  # (w, h) for aspect-preserving letterbox
) -> Tuple[List[Path], float]:
    """
    Decode a video into frames.

    Returns:
      - list of frame image paths
      - fps used
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN-safe
        fps = settings.default_fps

    frame_paths: List[Path] = []
    idx = 0

    # Read and write frames
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if target_size is not None:
            tw, th = target_size
            frame = resize_letterbox(frame, tw, th)
        else:
            frame = resize_and_pad(
                frame,
                target_size=getattr(settings, "target_size", 0),
                pad_to_square=getattr(settings, "pad_to_square", False),
            )

        frame_path = out_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)

        idx += 1
        if settings.max_frames > 0 and idx >= settings.max_frames:
            break

    cap.release()

    log(settings, f"Extracted {len(frame_paths)} frames from", video_path.name)
    return frame_paths, fps


# -------------------------
# Video encoding
# -------------------------

def write_video(
    settings: Settings,
    frames: List[Path],
    out_video: Path,
    fps: float,
) -> None:
    """
    Encode a list of frame images into a video.
    """
    if not frames:
        raise ValueError("No frames provided for video encoding")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read frame: {frames[0]}")

    h, w = first.shape[:2]
    ensure_dir(out_video.parent)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            # Safety: enforce consistent frame size
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)

    writer.release()
    log(settings, "Wrote video:", out_video.name)


# -------------------------
# Audio handling (NOT WORKING)
# -------------------------

def ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def copy_audio(
    settings: Settings,
    src_video: Path,
    dst_video: Path,
    out_video: Path,
) -> bool:
    """
    Copy audio stream from src_video onto dst_video.
    Writes to out_video.

    Returns True if successful.
    """
    if not ffmpeg_available():
        log(settings, "ffmpeg not available; skipping audio copy")
        return False

    ensure_dir(out_video.parent)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(dst_video),
        "-i", str(src_video),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(out_video),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log(settings, "Copied audio into:", out_video.name)
        return True
    except subprocess.CalledProcessError:
        log(settings, "Failed to copy audio")
        return False
