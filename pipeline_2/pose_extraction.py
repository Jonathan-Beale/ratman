"""
pose_extraction.py

Pose extraction + rendering for ControlNet OpenPose conditioning.

This version produces ControlNet-compatible OpenPose pose maps using:
  controlnet_aux.OpenposeDetector

Files produced (per pose video cache):
  frames/frame_000000.png ...
  OpenPose maps: poses/pose_000000.png ...
  old MediaPipe debug output: poses.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json

import cv2
import mediapipe as mp
from PIL import Image

from config import Settings
from utils_io import ensure_dir, log


# -------------------------
# MediaPipe Tasks PoseLandmarker -> poses.json (debug)
# -------------------------

def _require_model_file(settings: Settings) -> Path:
    rel = getattr(settings, "pose_landmarker_model_path", "").strip()
    if not rel:
        raise RuntimeError(
            "Missing Settings.pose_landmarker_model_path. "
            "Set it to something like 'models/pose_landmarker_full.task'."
        )
    model_path = Path(settings.project_root) / rel
    if not model_path.exists():
        raise FileNotFoundError(
            f"PoseLandmarker model not found at: {model_path}\n"
            "Download a Pose Landmarker .task model (e.g. pose_landmarker_full.task)\n"
            "and place it there (or change pose_landmarker_model_path)."
        )
    return model_path


def extract_pose_keypoints(
    settings: Settings,
    frame_paths: List[Path],
    out_json: Path,
    fps: Optional[float] = None,
) -> List[Optional[Dict[str, List[float]]]]:
    """
    Extract pose keypoints using MediaPipe Tasks PoseLandmarker (VIDEO mode).

    Output is written to poses.json and returned as a list.
    This is mainly for debugging/inspection; ControlNet conditioning uses OpenposeDetector maps.
    """
    model_path = _require_model_file(settings)

    use_fps = float(fps) if (fps is not None and fps > 0) else float(settings.default_fps)
    ms_per_frame = 1000.0 / max(use_fps, 1e-6)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=float(settings.mp_min_detection_confidence),
        min_tracking_confidence=float(settings.mp_min_tracking_confidence),
        min_pose_presence_confidence=float(settings.mp_min_detection_confidence),
    )

    results_all: List[Optional[Dict[str, List[float]]]] = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame_path in enumerate(frame_paths):
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                results_all.append(None)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(round(i * ms_per_frame))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                results_all.append(None)
                continue

            pose_lms = result.pose_landmarks[0]
            kp: Dict[str, List[float]] = {}
            for idx, lm in enumerate(pose_lms):
                vis = float(getattr(lm, "visibility", 1.0))
                kp[str(idx)] = [float(lm.x), float(lm.y), vis]
            results_all.append(kp)

    ensure_dir(out_json.parent)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)

    log(settings, f"Extracted poses for {len(frame_paths)} frames (PoseLandmarker VIDEO @ {use_fps:.3f} fps)")
    return results_all


# -------------------------
# OpenposeDetector -> ControlNet pose maps
# -------------------------

_OPENPOSE = None


def _get_openpose_detector(settings: Settings):
    """
    Lazy-load OpenposeDetector from controlnet_aux.
    """
    global _OPENPOSE
    if _OPENPOSE is not None:
        return _OPENPOSE

    try:
        from controlnet_aux.open_pose import OpenposeDetector
    except Exception as e:
        raise RuntimeError(
            "OpenposeDetector requires `controlnet-aux`.\n"
            "Install it with: pip install -U controlnet-aux\n"
            f"Original import error: {e}"
        )

    repo = getattr(settings, "openpose_detector_repo", "lllyasviel/Annotators")
    log(settings, f"Loading OpenposeDetector from: {repo}")
    _OPENPOSE = OpenposeDetector.from_pretrained(repo)
    return _OPENPOSE


def _frame_to_pil_rgb(frame_path: Path) -> Image.Image:
    bgr = cv2.imread(str(frame_path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {frame_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def render_pose_sequence(
    settings: Settings,
    frame_paths: List[Path],
    keypoints_all: List[Optional[Dict[str, List[float]]]],  # kept for API compatibility; not used
    out_dir: Path,
) -> List[Path]:
    """
    Render ControlNet OpenPose conditioning frames using OpenposeDetector.

    Output images are saved as:
      out_dir/pose_000000.png, ...

    These are the pose maps you pass to ControlNet OpenPose.
    """
    ensure_dir(out_dir)

    openpose = _get_openpose_detector(settings)

    include_hands = bool(getattr(settings, "openpose_include_hands", False))
    include_face = bool(getattr(settings, "openpose_include_face", False))
    detect_resolution = int(getattr(settings, "openpose_detect_resolution", 512))

    pose_paths: List[Path] = []

    for idx, frame_path in enumerate(frame_paths):
        out_path = out_dir / f"pose_{idx:06d}.png"
        if out_path.exists():
            pose_paths.append(out_path)
            continue

        pil = _frame_to_pil_rgb(frame_path)
        w, h = pil.size

        # Newer controlnet_aux supports these args; older versions may not.
        try:
            pose_img = openpose(
                pil,
                detect_resolution=detect_resolution,
                image_resolution=max(w, h),
                hand_and_face=(include_hands or include_face),
            )
        except TypeError:
            # Fallback for older signatures
            pose_img = openpose(pil)

        # Ensure the conditioning map matches the working frame size exactly
        if pose_img.size != (w, h):
            pose_img = pose_img.resize((w, h), resample=Image.BICUBIC)

        pose_img.save(out_path)
        pose_paths.append(out_path)

    log(settings, f"Rendered {len(pose_paths)} OpenPose conditioning frames")
    return pose_paths
