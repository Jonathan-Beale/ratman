# pipeline.py

"""
pipeline.py

End-to-end pipeline orchestration.

This version uses:
- AnimateDiff T2V for the first window
- AnimateDiff V2V + ControlNet continuation for later windows
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
from typing import List, Tuple

import cv2
from PIL import Image

from config import load_settings, Settings
from utils_io import (
    list_pose_videos,
    list_ref_images,
    make_pose_cache_dir,
    make_timestamped_run_dir,
    timestamp_slug,
    log,
    ensure_dir,
)
from video_frames import extract_frames, compute_working_size
from pose_extraction import extract_pose_keypoints, render_pose_sequence
from ref_preprocess import preprocess_reference
from character_video_gen import (
    generate_character_video_t2v,
    generate_character_video_v2v,
)
from refine_video_frames import refine_video_frames
from postprocess import finalize_video


def _truncate(items: List[Path], max_n: int) -> List[Path]:
    if max_n and max_n > 0:
        return items[:max_n]
    return items


def _get_video_info(video_path: Path) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return w, h, fps


def _read_first_png_size(dir_path: Path, pattern: str) -> Tuple[int, int] | None:
    files = sorted(dir_path.glob(pattern))
    if not files:
        return None
    with Image.open(files[0]) as im:
        return im.size


def _invalidate_pose_cache_if_needed(
    settings: Settings,
    pose_video: Path,
    pose_cache: Path,
) -> None:
    frame_dir = pose_cache / "frames"
    pose_json = pose_cache / "poses.json"

    if not pose_json.exists():
        return

    cached_size = _read_first_png_size(frame_dir, "frame_*.png")
    if cached_size is None:
        log(settings, "Pose cache incomplete; regenerating.")
        shutil.rmtree(pose_cache, ignore_errors=True)
        return

    src_w, src_h, _fps = _get_video_info(pose_video)
    if src_w <= 0 or src_h <= 0:
        log(settings, "WARNING: Could not read video size; keeping pose cache.")
        return

    exp_w, exp_h = compute_working_size(
        src_w,
        src_h,
        scale=settings.output_scale,
        multiple=settings.output_multiple,
        max_long_side=settings.output_max_long_side,
    )

    cw, ch = cached_size
    if (cw, ch) != (exp_w, exp_h):
        log(
            settings,
            f"Pose cache resolution mismatch: cached={cw}x{ch}, expected={exp_w}x{exp_h}. Regenerating cache."
        )
        shutil.rmtree(pose_cache, ignore_errors=True)


def _stamp_video_file(final_path: Path, stamped_name: str) -> Path:
    stamped_path = final_path.parent / stamped_name
    if final_path.exists() and final_path != stamped_path:
        os.replace(str(final_path), str(stamped_path))
    return stamped_path


def _chunk_ranges(total_frames: int, window_size: int, overlap: int) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        return []

    window_size = max(1, min(window_size, 32))
    overlap = max(0, overlap)
    if overlap >= window_size:
        overlap = window_size - 1

    step = window_size - overlap
    ranges: List[Tuple[int, int]] = []

    start = 0
    while start < total_frames:
        end = min(start + window_size, total_frames)
        ranges.append((start, end))
        if end >= total_frames:
            break
        start += step

    return ranges


def _copy_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.save(dst)


def _build_v2v_init_video(
    assembled_paths: List[Path],
    start: int,
    end: int,
    overlap: int,
) -> List[Path]:
    """
    Build the init video for the next window.

    Only anchor the overlap region to previously assembled frames.
    For the new tail, use the corresponding source pose video frames as a weak-motion prior
    instead of repeating the last generated frame.
    """
    if not assembled_paths:
        raise ValueError("Cannot build V2V init video without assembled prior frames")

    # overlap portion comes from previous generated result
    init_paths: List[Path] = []
    for timeline_idx in range(start, min(start + overlap, end)):
        if 0 <= timeline_idx < len(assembled_paths):
            init_paths.append(assembled_paths[timeline_idx])

    return init_paths


def _merge_window_frames(
    assembled_dir: Path,
    chunk_frames: List[Path],
    global_start: int,
    overlap_count: int,
    assembled_paths: List[Path],
    *,
    keep_previous_overlap: bool,
) -> List[Path]:
    """
    Merge chunk output into one timeline.

    In strong continuation mode, keep the already-assembled overlap frames and only
    append the newly generated tail. This avoids visible seam blending artifacts.
    """
    ensure_dir(assembled_dir)

    if global_start == 0:
        for local_idx, chunk_path in enumerate(chunk_frames):
            out_path = assembled_dir / f"gen_{local_idx:06d}.png"
            _copy_frame(chunk_path, out_path)
            assembled_paths.append(out_path)
        return assembled_paths

    actual_overlap = min(overlap_count, len(chunk_frames), len(assembled_paths) - global_start)
    actual_overlap = max(0, actual_overlap)

    if not keep_previous_overlap:
        for j in range(actual_overlap):
            timeline_idx = global_start + j
            old_path = assembled_paths[timeline_idx]
            new_path = chunk_frames[j]
            with Image.open(old_path).convert("RGB") as a, Image.open(new_path).convert("RGB") as b:
                blended = Image.blend(a, b, 0.25)
                blended.save(old_path)

    for local_idx in range(actual_overlap, len(chunk_frames)):
        timeline_idx = global_start + local_idx
        out_path = assembled_dir / f"gen_{timeline_idx:06d}.png"
        _copy_frame(chunk_frames[local_idx], out_path)
        if timeline_idx < len(assembled_paths):
            assembled_paths[timeline_idx] = out_path
        else:
            assembled_paths.append(out_path)

    return assembled_paths


def _windowed_generate_character_video(
    settings: Settings,
    ref_image_path: Path,
    pose_frames: List[Path],
    out_frames_dir: Path,
    width: int,
    height: int,
) -> List[Path]:
    ensure_dir(out_frames_dir)

    total = len(pose_frames)
    if total == 0:
        return []

    window_size = min(max(1, settings.animate_window_size), 32)
    overlap = max(0, settings.animate_window_overlap)
    if overlap >= window_size:
        overlap = window_size - 1

    if total <= window_size:
        return generate_character_video_t2v(
            settings=settings,
            ref_image_path=ref_image_path,
            pose_frames=pose_frames,
            out_frames_dir=out_frames_dir,
            width=width,
            height=height,
            seed=int(settings.seed),
        )

    ranges = _chunk_ranges(total, window_size, overlap)
    assembled_dir = out_frames_dir / "assembled"
    assembled_paths: List[Path] = []

    log(
        settings,
        f"Windowed generation: total_frames={total}, window_size={window_size}, overlap={overlap}, windows={len(ranges)}",
    )

    for win_idx, (start, end) in enumerate(ranges):
        chunk_pose = pose_frames[start:end]
        chunk_dir = out_frames_dir / f"chunk_{win_idx:03d}"
        ensure_dir(chunk_dir)

        log(settings, f"  Window {win_idx + 1}/{len(ranges)}: frames [{start}:{end})")

        if win_idx == 0 or not settings.animate_use_v2v_continuation:
            chunk_frames = generate_character_video_t2v(
                settings=settings,
                ref_image_path=ref_image_path,
                pose_frames=chunk_pose,
                out_frames_dir=chunk_dir,
                width=width,
                height=height,
                seed=int(settings.seed),
            )
        else:
            overlap_count = overlap if win_idx > 0 else 0

            init_video_frames = _build_v2v_init_video(
                assembled_paths=assembled_paths,
                start=start,
                end=end,
                overlap=overlap_count,
            )

            chunk_frames = generate_character_video_v2v(
                settings=settings,
                ref_image_path=ref_image_path,
                pose_frames=chunk_pose,
                init_video_frames=init_video_frames,
                out_frames_dir=chunk_dir,
                width=width,
                height=height,
                seed=int(settings.seed),
                overlap_count=overlap_count,
            )

        overlap_count = overlap if win_idx > 0 else 0
        assembled_paths = _merge_window_frames(
            assembled_dir=assembled_dir,
            chunk_frames=chunk_frames,
            global_start=start,
            overlap_count=overlap_count,
            assembled_paths=assembled_paths,
            keep_previous_overlap=bool(settings.animate_v2v_keep_overlap_from_previous),
        )

    return assembled_paths


def run_pipeline(
    settings: Settings,
    pose_filter: str | None = None,
    ref_filter: str | None = None,
) -> None:
    pose_videos = list_pose_videos(settings)
    ref_images = list_ref_images(settings)

    if pose_filter:
        pose_videos = [p for p in pose_videos if p.name == pose_filter]
    if ref_filter:
        ref_images = [r for r in ref_images if r.name == ref_filter]

    if not pose_videos:
        log(settings, "No pose videos found.")
        return
    if not ref_images:
        log(settings, "No reference images found.")
        return

    for pose_video in pose_videos:
        log(settings, "Processing pose video:", pose_video.name)

        src_w, src_h, fps = _get_video_info(pose_video)
        if fps <= 0:
            fps = settings.default_fps

        work_w, work_h = compute_working_size(
            src_w,
            src_h,
            scale=settings.output_scale,
            multiple=settings.output_multiple,
            max_long_side=settings.output_max_long_side,
        )

        pose_cache = make_pose_cache_dir(settings, pose_video)
        frame_dir = pose_cache / "frames"
        pose_img_dir = pose_cache / "poses"
        pose_json = pose_cache / "poses.json"

        _invalidate_pose_cache_if_needed(settings, pose_video, pose_cache)

        if not pose_json.exists():
            frame_paths, fps = extract_frames(
                settings=settings,
                video_path=pose_video,
                out_dir=frame_dir,
                target_size=(work_w, work_h),
            )
            frame_paths = _truncate(frame_paths, settings.max_frames)

            keypoints_all = extract_pose_keypoints(
                settings=settings,
                frame_paths=frame_paths,
                out_json=pose_json,
                fps=fps,
            )

            pose_frames = render_pose_sequence(
                settings=settings,
                frame_paths=frame_paths,
                keypoints_all=keypoints_all,
                out_dir=pose_img_dir,
            )
            pose_frames = _truncate(pose_frames, settings.max_frames)
        else:
            frame_paths = sorted(frame_dir.glob("frame_*.png"))
            pose_frames = sorted(pose_img_dir.glob("pose_*.png"))
            fps = settings.default_fps if fps <= 0 else fps

            frame_paths = _truncate(frame_paths, settings.max_frames)
            pose_frames = _truncate(pose_frames, settings.max_frames)

        if not pose_frames:
            log(settings, "No pose frames available, skipping:", pose_video.name)
            continue

        for ref_image in ref_images:
            stamp = timestamp_slug()
            log(settings, "  Using reference:", ref_image.name)

            generated_run_dir = make_timestamped_run_dir(
                settings.generated_output_dir, pose_video, ref_image, stamp, "generated"
            )
            gen_ref_dir = generated_run_dir / "ref"
            gen_frames_dir = generated_run_dir / "frames"
            gen_video_dir = generated_run_dir / "video"

            ref_path = preprocess_reference(
                settings=settings,
                ref_image_path=ref_image,
                out_dir=gen_ref_dir,
                target_size=(work_w, work_h),
            )

            gen_frames = _windowed_generate_character_video(
                settings=settings,
                ref_image_path=ref_path,
                pose_frames=pose_frames,
                out_frames_dir=gen_frames_dir,
                width=work_w,
                height=work_h,
            )

            generated_video = finalize_video(
                settings=settings,
                generated_frames=gen_frames,
                fps=fps,
                out_video_dir=gen_video_dir,
                pose_video_path=pose_video,
            )

            generated_video = _stamp_video_file(
                generated_video,
                f"{pose_video.stem}__{ref_image.stem}__{stamp}__generated.mp4",
            )
            log(settings, "  Generated video:", generated_video)

            if settings.refine_enable:
                refined_run_dir = make_timestamped_run_dir(
                    settings.refined_output_dir, pose_video, ref_image, stamp, "refined"
                )
                refined_ref_dir = refined_run_dir / "ref"
                refined_frames_dir = refined_run_dir / "frames"
                refined_video_dir = refined_run_dir / "video"

                preprocess_reference(
                    settings=settings,
                    ref_image_path=ref_image,
                    out_dir=refined_ref_dir,
                    target_size=(work_w, work_h),
                )

                refined_frames = refine_video_frames(
                    settings=settings,
                    generated_frames=gen_frames,
                    pose_frames=pose_frames[:len(gen_frames)],
                    ref_image_path=ref_path,
                    out_frames_dir=refined_frames_dir,
                    width=work_w,
                    height=work_h,
                )

                refined_video = finalize_video(
                    settings=settings,
                    generated_frames=refined_frames,
                    fps=fps,
                    out_video_dir=refined_video_dir,
                    pose_video_path=pose_video,
                )

                refined_video = _stamp_video_file(
                    refined_video,
                    f"{pose_video.stem}__{ref_image.stem}__{stamp}__refined.mp4",
                )
                log(settings, "  Refined video:", refined_video)


if __name__ == "__main__":
    settings = load_settings()
    run_pipeline(settings)