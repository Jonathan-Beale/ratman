"""
ref_preprocess.py

Reference image preprocessing.

Updated for the new pose-video-driven workflow:
- Reference image should be resized to the SAME working resolution as:
    * extracted pose frames
    * diffusion generation frames
- We use letterbox resizing (preserve aspect ratio, pad to target size).
  This avoids stretching the character.

API:
- preprocess_reference(settings, ref_image_path, out_dir, target_size=(w,h)) -> Path to saved ref.png
- load_reference(ref_path) -> np.ndarray (BGR)

This fixes:
TypeError: preprocess_reference() got an unexpected keyword argument 'target_size'
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

from config import Settings
from utils_io import ensure_dir, log
from video_frames import resize_letterbox, resize_and_pad


def preprocess_reference(
    settings: Settings,
    ref_image_path: Path,
    out_dir: Path,
    target_size: Optional[Tuple[int, int]] = None,  # (w, h)
) -> Path:
    """
    Preprocess a reference image and save it to out_dir.

    Args:
      target_size:
        If provided, the reference is resized via letterboxing to (w,h),
        matching the working resolution derived from the pose video.
        If None, falls back to the older square-ish behavior.

    Returns:
      Path to the processed reference image (ref.png).
    """
    ensure_dir(out_dir)

    img = cv2.imread(str(ref_image_path))
    if img is None:
        raise RuntimeError(f"Could not read reference image: {ref_image_path}")

    if target_size is not None:
        tw, th = target_size
        img = resize_letterbox(img, tw, th)
    else:
        # Backward-compatible fallback
        img = resize_and_pad(
            img,
            target_size=getattr(settings, "target_size", 0),
            pad_to_square=getattr(settings, "pad_to_square", False),
        )

    out_path = out_dir / "ref.png"
    cv2.imwrite(str(out_path), img)

    log(settings, "Preprocessed reference:", ref_image_path.name)
    return out_path


def load_reference(ref_path: Path) -> np.ndarray:
    """
    Load a preprocessed reference image (BGR).
    """
    img = cv2.imread(str(ref_path))
    if img is None:
        raise RuntimeError(f"Could not load reference image: {ref_path}")
    return img
