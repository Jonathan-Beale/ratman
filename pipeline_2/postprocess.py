"""
postprocess.py

Post-processing utilities for generated videos.

Responsibilities:
- Encode generated frames into a video
- (Optional) Temporal denoise / deflicker using FFmpeg filters
- (Optional) Copy audio from the original pose video
- Be robust on Windows (file locks) by using safe replace + retries

This file intentionally reads postprocess toggles from environment variables so
you don't have to change config.py to experiment:

Environment toggles (all optional):
- PIPELINE_POST_ENABLE=1              -> enable FFmpeg postprocessing (default: 0)
- PIPELINE_POST_DENOISE=1             -> enable hqdn3d denoise (default: 1 if POST_ENABLE=1)
- PIPELINE_POST_DEFLICKER=1           -> enable deflicker filter (default: 1 if POST_ENABLE=1)
- PIPELINE_POST_TEMPMIX=1             -> enable tmix temporal averaging (default: 0)
- PIPELINE_POST_FFMPEG_PATH=<path>    -> path to ffmpeg.exe (default: auto-detect via PATH)
- PIPELINE_POST_CRF=18                -> H.264 CRF (lower=better, bigger file) default 18
- PIPELINE_POST_PRESET=slow           -> x264 preset (default: slow)
- PIPELINE_COPY_AUDIO=1               -> overrides settings.copy_audio if Settings lacks it

Notes:
- Temporal denoise/deflicker is often the biggest improvement for AnimateDiff outputs.
- These filters are conservative; you can tune them later.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from config import Settings
from utils_io import ensure_dir, log
from video_frames import write_video, copy_audio


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    return v if v != "" else default


def _find_ffmpeg(explicit_path: Optional[str] = None) -> Optional[str]:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return str(p)
    return shutil.which("ffmpeg")


def _safe_replace(src: Path, dst: Path, attempts: int = 10, delay_s: float = 0.4) -> None:
    """
    Windows can keep mp4 files locked (e.g., if a player or antivirus scans it).
    This retries os.replace a few times.
    """
    last_err = None
    for _ in range(attempts):
        try:
            # os.replace works cross-platform and overwrites if exists
            os.replace(str(src), str(dst))
            return
        except OSError as e:
            last_err = e
            time.sleep(delay_s)
    if last_err:
        raise last_err


def _run_ffmpeg(settings: Settings, ffmpeg: str, args: List[str]) -> bool:
    """
    Run ffmpeg with args (excluding ffmpeg itself). Returns True on success.
    """
    cmd = [ffmpeg] + args
    log(settings, "[post] ffmpeg:", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            log(settings, "[post] ffmpeg failed (see output below):")
            # avoid spamming huge logs; print last chunk
            out = proc.stdout or ""
            log(settings, out[-4000:] if len(out) > 4000 else out)
            return False
        return True
    except FileNotFoundError:
        log(settings, "[post] ffmpeg not found.")
        return False


def _build_post_filter_chain(enable_denoise: bool, enable_deflicker: bool, enable_tmix: bool) -> str:
    """
    Conservative temporal cleanup filter chain.

    - hqdn3d: spatial+temporal denoise (good for speckle/grain)
    - deflicker: reduces brightness flicker between frames
    - tmix: temporal averaging (can reduce noise but may introduce ghosting)
    """
    filters: List[str] = []

    if enable_denoise:
        # hqdn3d params: luma_spatial:chroma_spatial:luma_tmp:chroma_tmp
        # Conservative defaults; increase last two for stronger temporal denoise.
        filters.append("hqdn3d=1.5:1.2:6.0:6.0")

    if enable_deflicker:
        # deflicker modes: 'am' (average/median) can blur; 'pm' (percentile/median-ish) often stable
        # size: number of frames window (odd preferred). 10 is reasonable.
        filters.append("deflicker=mode=pm:size=10")

    if enable_tmix:
        # Mix neighboring frames (can stabilize noise) but may ghost fast motion.
        # weights sum to 1.0 (approx). Here: current + neighbors.
        filters.append("tmix=frames=3:weights='1 2 1'")

    # If nothing enabled, return empty
    return ",".join(filters)


def _postprocess_video_ffmpeg(
    settings: Settings,
    input_video: Path,
    output_video: Path,
) -> Path:
    """
    Apply FFmpeg temporal cleanup filters and re-encode.
    Returns output_video on success, else returns input_video.
    """
    enable_post = _env_bool("PIPELINE_POST_ENABLE", default=False)
    if not enable_post:
        return input_video

    ffmpeg_path = _env_str("PIPELINE_POST_FFMPEG_PATH", "")
    ffmpeg = _find_ffmpeg(ffmpeg_path)
    if not ffmpeg:
        log(settings, "[post] PIPELINE_POST_ENABLE=1 but ffmpeg not found in PATH. Skipping postprocess.")
        return input_video

    enable_denoise = _env_bool("PIPELINE_POST_DENOISE", default=True)
    enable_deflicker = _env_bool("PIPELINE_POST_DEFLICKER", default=True)
    enable_tmix = _env_bool("PIPELINE_POST_TEMPMIX", default=False)

    vf = _build_post_filter_chain(enable_denoise, enable_deflicker, enable_tmix)
    if vf.strip() == "":
        return input_video

    crf = _env_str("PIPELINE_POST_CRF", "18")
    preset = _env_str("PIPELINE_POST_PRESET", "slow")

    # Write to a temp file first to avoid half-written output if ffmpeg fails
    tmp_out = output_video.with_suffix(".tmp.mp4")
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except OSError:
            pass

    ok = _run_ffmpeg(
        settings,
        ffmpeg,
        [
            "-y",
            "-i",
            str(input_video),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_out),
        ],
    )
    if not ok or not tmp_out.exists():
        log(settings, "[post] Postprocess failed; keeping unprocessed video.")
        return input_video

    # Replace output_video atomically-ish
    if output_video.exists():
        try:
            output_video.unlink()
        except OSError:
            pass
    _safe_replace(tmp_out, output_video)
    log(settings, "[post] Postprocessed video:", output_video.name)
    return output_video


def finalize_video(
    settings: Settings,
    generated_frames: List[Path],
    fps: float,
    out_video_dir: Path,
    pose_video_path: Path,
) -> Path:
    """
    Turn generated frames into a final video.

    Returns the path to the final video file.
    """
    ensure_dir(out_video_dir)

    # Stage 1: write raw video (no audio)
    raw_video = out_video_dir / "video_no_audio.mp4"
    final_video = out_video_dir / "final.mp4"
    post_video = out_video_dir / "video_post.mp4"

    write_video(
        settings=settings,
        frames=generated_frames,
        out_video=raw_video,
        fps=fps,
    )

    # Stage 2: optional temporal postprocess via FFmpeg
    processed_video = _postprocess_video_ffmpeg(settings, raw_video, post_video)

    # Decide whether to copy audio
    # - Prefer settings.copy_audio if it exists
    # - Allow env override PIPELINE_COPY_AUDIO=1/0
    copy_audio_env = os.environ.get("PIPELINE_COPY_AUDIO", "")
    if copy_audio_env != "":
        do_copy_audio = copy_audio_env.strip().lower() in ("1", "true", "yes", "y", "on")
    else:
        do_copy_audio = bool(getattr(settings, "copy_audio", False))

    # Stage 3: optional audio copy (from original pose video)
    if do_copy_audio:
        ok = copy_audio(
            settings=settings,
            src_video=pose_video_path,
            dst_video=processed_video,
            out_video=final_video,
        )
        if ok:
            # Best-effort cleanup
            for p in [raw_video, post_video]:
                if p.exists() and p != final_video:
                    try:
                        p.unlink()
                    except OSError:
                        pass
            return final_video

    # Fallback: no audio. Move processed_video (or raw_video) to final.mp4
    log(settings, "Final video (no audio):", final_video.name)

    # If processed_video is already the raw video path, we still want final.mp4
    if processed_video.resolve() == final_video.resolve():
        return final_video

    # Ensure target doesn't exist
    if final_video.exists():
        try:
            final_video.unlink()
        except OSError:
            pass

    # Use safe replace to avoid WinError 32 issues
    _safe_replace(processed_video, final_video)

    # If we used post_video -> final, remove raw if still there
    if raw_video.exists() and raw_video != final_video:
        try:
            raw_video.unlink()
        except OSError:
            pass

    return final_video
