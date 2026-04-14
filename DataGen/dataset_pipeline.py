import subprocess
import shutil
from pathlib import Path
import sys

BLENDER_EXE = "/mnt/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
BLENDER_SCRIPT = "/root/ratman/DataGen/blender_render_job.py"

SCENES_DIR = Path("/root/ratman/DataGen/Assets/Scenes")
MODELS_DIR = Path("/root/ratman/DataGen/Assets/Models")
ANIMS_DIR = Path("/root/ratman/DataGen/Assets/Animations")
OUTPUT_DIR = Path("/root/ratman/DataGen/Renders")

FFMPEG_EXE = "/mnt/c/ffmpeg/bin/ffmpeg.exe"

# Delete source PNG frames after successfully compiling each camera's videos.
# Saves significant disk space — MusePose only needs the .mp4 files.
DELETE_FRAMES_AFTER_COMPILE = True


def to_win_path(path) -> str:
    """Convert a WSL Linux path to a Windows path using wslpath."""
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()

SUPPORTED_SCENE_EXTS = {".blend"}
SUPPORTED_MODEL_EXTS = {".fbx", ".glb", ".gltf", ".blend"}
SUPPORTED_ANIM_EXTS = {".fbx", ".bvh"}

# Number of actors to render simultaneously per job.
# 1 = single-actor (original behavior). 2+ = multi-actor combos.
NUM_ACTORS = 1


def ensure_executable(path_str: str, name: str) -> str:
    if path_str and Path(path_str).exists():
        return path_str
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Could not find {name}. Set an explicit path.")


def compile_video(ffmpeg_exe: str, frames_dir: Path, pattern: str, output_name: str, fps: int = 30) -> bool:
    """Returns True if a video was compiled, False if skipped (no matching frames)."""
    pngs = sorted(frames_dir.glob(pattern.replace("%04d", "*")))
    if not pngs:
        print(f"Skipping video compile for {frames_dir} ({pattern} not found)")
        return False

    output_mp4 = frames_dir.parent / output_name

    cmd = [
        ffmpeg_exe,
        "-y",
        "-framerate", str(fps),
        "-i", to_win_path(frames_dir / pattern),
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        to_win_path(output_mp4),
    ]

    print(f"Compiling video from {frames_dir / pattern} -> {output_mp4}")
    subprocess.run(cmd, check=True)
    return True


def _delete_frames(cam_dir: Path, patterns: list[str]):
    """Delete PNG frames matching the given glob patterns from a camera directory."""
    deleted = 0
    for pattern in patterns:
        for f in cam_dir.glob(pattern):
            f.unlink()
            deleted += 1
    print(f"Deleted {deleted} PNG frames from {cam_dir}")


def render_job(blender_exe: str, blend_file: Path, actor_pairs: list):
    """
    actor_pairs: list of (model_path, anim_path) tuples, one per actor.
    Single-actor: actor_pairs = [(model, anim)]
    Multi-actor:  actor_pairs = [(model1, anim1), (model2, anim2), ...]
    """
    scene_name = blend_file.stem
    num_actors = len(actor_pairs)

    if num_actors == 1:
        model, anim = actor_pairs[0]
        job_root = OUTPUT_DIR / model.stem / anim.stem / scene_name
        job_name = f"{model.stem}/{anim.stem}/{scene_name}"
    else:
        models_tag = "+".join(m.stem for m, _ in actor_pairs)
        anims_tag  = "+".join(a.stem for _, a in actor_pairs)
        job_root = OUTPUT_DIR / f"multi_{models_tag}" / anims_tag / scene_name
        job_name = f"multi[{models_tag}]/{anims_tag}/{scene_name}"

    job_root.mkdir(parents=True, exist_ok=True)

    actor_args = []
    for model, anim in actor_pairs:
        actor_args.extend([to_win_path(model), to_win_path(anim)])

    cmd = [
        blender_exe,
        "-b",
        to_win_path(blend_file),
        "-P",
        to_win_path(BLENDER_SCRIPT),
        "--",
        to_win_path(job_root),
        str(num_actors),
    ] + actor_args

    print(f"Running Blender job: {job_name}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"Blender job failed for scene={blend_file.name}, "
            f"actors={[f'{m.name}+{a.name}' for m,a in actor_pairs]}, "
            f"returncode={result.returncode}"
        )

    return job_root


def _compile_camera_videos(ffmpeg_exe: str, job_root: Path, num_actors: int):
    """Compile all video sequences found under job_root camera subdirs."""
    camera_dirs = [
        p for p in job_root.iterdir()
        if p.is_dir() and any(p.glob("frame_*.png"))
    ]
    print(f"Found render directories: {[d.name for d in camera_dirs]}")
    for cam_dir in camera_dirs:
        compiled_patterns = []

        if compile_video(ffmpeg_exe, cam_dir, "frame_%04d.png",   f"{cam_dir.name}.mp4"):
            compiled_patterns.append("frame_*.png")
        if compile_video(ffmpeg_exe, cam_dir, "overlay_%04d.png", f"{cam_dir.name}_overlay.mp4"):
            compiled_patterns.append("overlay_*.png")
        if compile_video(ffmpeg_exe, cam_dir, "dwpose_%04d.png",  f"{cam_dir.name}_dwpose.mp4"):
            compiled_patterns.append("dwpose_*.png")
        if num_actors == 1:
            if compile_video(ffmpeg_exe, cam_dir, "dwpose_rainbow_%04d.png", f"{cam_dir.name}_dwpose_rainbow.mp4"):
                compiled_patterns.append("dwpose_rainbow_*.png")
        else:
            if compile_video(
                ffmpeg_exe, cam_dir,
                "dwpose_multiactor_%04d.png",
                f"{cam_dir.name}_dwpose_multiactor.mp4",
            ):
                compiled_patterns.append("dwpose_multiactor_*.png")

        if DELETE_FRAMES_AFTER_COMPILE and compiled_patterns:
            _delete_frames(cam_dir, compiled_patterns)


def main():
    blender_exe = ensure_executable(BLENDER_EXE, "blender")
    ffmpeg_exe = ensure_executable(FFMPEG_EXE, "ffmpeg")

    scenes = [
        p for p in sorted(SCENES_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SCENE_EXTS
    ]
    models = [
        p for p in sorted(MODELS_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_MODEL_EXTS
    ]
    anims = [
        p for p in sorted(ANIMS_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_ANIM_EXTS
    ]

    if not scenes:
        raise RuntimeError(f"No scenes found in {SCENES_DIR}")
    if not models:
        raise RuntimeError(f"No models found in {MODELS_DIR}")
    if not anims:
        raise RuntimeError(f"No animations found in {ANIMS_DIR}")

    failures = []

    if NUM_ACTORS == 1:
        # Single-actor: one model + one animation per job (original behavior)
        jobs = [
            [(model, anim)]
            for model in models
            for anim in anims
        ]
    else:
        # Multi-actor: generate all combinations of NUM_ACTORS distinct (model, anim) pairs.
        # Uses itertools.product to pair each actor independently (different models/anims allowed).
        import itertools
        single_pairs = [(m, a) for m in models for a in anims]
        jobs = [
            list(combo)
            for combo in itertools.combinations(single_pairs, NUM_ACTORS)
            if len(set(combo)) == NUM_ACTORS  # skip if any pair repeated
        ]
        print(f"Generated {len(jobs)} multi-actor ({NUM_ACTORS}-actor) job combinations")

    for scene_file in scenes:
        for actor_pairs in jobs:
            label = " + ".join(f"{m.name}/{a.name}" for m, a in actor_pairs)
            try:
                print(f"\n=== scene={scene_file.name}  actors=[{label}] ===")
                job_root = render_job(blender_exe, scene_file, actor_pairs)
                _compile_camera_videos(ffmpeg_exe, job_root, len(actor_pairs))

            except Exception as e:
                print(f"FAILED: {scene_file.name} + [{label}]")
                print(e)
                failures.append((scene_file.name, label, str(e)))

    print("\nBatch complete.")
    if failures:
        print("\nFailures:")
        for scene_name, label, err in failures:
            print(f"- {scene_name} + [{label}]: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()