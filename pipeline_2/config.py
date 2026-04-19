# config.py

"""
config.py

Central configuration for the pipeline.

Values can be overridden via environment variables using the PIPELINE_* prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except Exception:
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except Exception:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, None)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class Settings:
    # Project structure
    project_root: Path
    pose_input_dir: Path
    ref_input_dir: Path
    output_dir: Path
    generated_output_dir: Path
    refined_output_dir: Path

    # Video sizing
    output_scale: float
    output_multiple: int
    output_max_long_side: int
    default_fps: float
    max_frames: int

    # AnimateDiff windowing
    animate_window_size: int
    animate_window_overlap: int

    # Strong continuation mode
    animate_use_v2v_continuation: bool
    animate_v2v_strength: float
    animate_v2v_keep_overlap_from_previous: bool

    # Device / performance
    device: str
    sd_enable_xformers: bool
    sd_enable_attention_slicing: bool
    sd_enable_cpu_offload: bool

    # Pose extraction
    pose_landmarker_model_path: str
    mp_min_detection_confidence: float
    mp_min_tracking_confidence: float

    # Misc
    copy_audio: bool

    # OpenPose detector
    openpose_detector_repo: str
    openpose_include_hands: bool
    openpose_include_face: bool
    openpose_detect_resolution: int

    # Backend selection
    generator_backend: str

    # SD / AnimateDiff models
    sd_base_model: str
    sd_motion_module: str
    sd_controlnet_openpose_model: str

    # Prompting
    seed: int
    sd_prompt: str
    sd_negative_prompt: str

    # Sampling params (first pass)
    sd_num_inference_steps: int
    sd_guidance_scale: float
    sd_controlnet_conditioning_scale: float

    # IP-Adapter (first pass)
    ip_adapter_enable: bool
    ip_adapter_model: str
    ip_adapter_subfolder: str
    ip_adapter_weight_name: str
    ip_adapter_scale: float

    # LoRA
    lora_enable: bool
    lora_path: str
    lora_scale: float

    # Refinement pass
    refine_enable: bool
    refine_prompt: str
    refine_negative_prompt: str
    refine_num_inference_steps: int
    refine_guidance_scale: float
    refine_strength: float
    refine_controlnet_conditioning_scale: float
    refine_ip_adapter_enable: bool
    refine_ip_adapter_scale: float


def load_settings() -> Settings:
    root = Path(os.getcwd())

    pose_input = root / "PoseInput"
    ref_input = root / "RefInput"
    cache_dir = root / "Output"
    generated_dir = root / "generatedoutput"
    refined_dir = root / "refinedoutput"

    return Settings(
        project_root=root,
        pose_input_dir=Path(_get_env("PIPELINE_POSE_INPUT", str(pose_input))),
        ref_input_dir=Path(_get_env("PIPELINE_REF_INPUT", str(ref_input))),
        output_dir=Path(_get_env("PIPELINE_OUTPUT_DIR", str(cache_dir))),
        generated_output_dir=Path(_get_env("PIPELINE_GENERATED_OUTPUT_DIR", str(generated_dir))),
        refined_output_dir=Path(_get_env("PIPELINE_REFINED_OUTPUT_DIR", str(refined_dir))),

        # Video sizing
        output_scale=_get_env_float("PIPELINE_OUTPUT_SCALE", 1.2),
        output_multiple=_get_env_int("PIPELINE_OUTPUT_MULTIPLE", 8),
        output_max_long_side=_get_env_int("PIPELINE_OUTPUT_MAX_LONG_SIDE", 512),
        default_fps=_get_env_float("PIPELINE_DEFAULT_FPS", 25.0),
        max_frames=_get_env_int("PIPELINE_MAX_FRAMES", 128),
        copy_audio=_get_env_bool("PIPELINE_COPY_AUDIO", True),

        # AnimateDiff windowing
        animate_window_size=_get_env_int("PIPELINE_ANIMATE_WINDOW_SIZE", 32),
        animate_window_overlap=_get_env_int("PIPELINE_ANIMATE_WINDOW_OVERLAP", 2),

        # Strong continuation mode
        animate_use_v2v_continuation=_get_env_bool("PIPELINE_ANIMATE_USE_V2V_CONTINUATION", True),
        animate_v2v_strength=_get_env_float("PIPELINE_ANIMATE_V2V_STRENGTH", 0.60),
        animate_v2v_keep_overlap_from_previous=_get_env_bool("PIPELINE_ANIMATE_KEEP_PREV_OVERLAP", True),

        # Device / perf
        device=_get_env("PIPELINE_DEVICE", "cuda"),
        sd_enable_xformers=_get_env_bool("PIPELINE_SD_XFORMERS", False),
        sd_enable_attention_slicing=_get_env_bool("PIPELINE_SD_ATTENTION_SLICING", True),
        sd_enable_cpu_offload=_get_env_bool("PIPELINE_SD_CPU_OFFLOAD", False),

        # MediaPipe
        pose_landmarker_model_path=_get_env(
            "PIPELINE_POSE_TASK_MODEL",
            "models/pose_landmarker_full.task",
        ),
        mp_min_detection_confidence=_get_env_float("PIPELINE_MP_MIN_DET", 0.5),
        mp_min_tracking_confidence=_get_env_float("PIPELINE_MP_MIN_TRACK", 0.5),

        # OpenPose
        openpose_detector_repo=_get_env("PIPELINE_OPENPOSE_REPO", "lllyasviel/Annotators"),
        openpose_include_hands=_get_env_bool("PIPELINE_OPENPOSE_HANDS", False),
        openpose_include_face=_get_env_bool("PIPELINE_OPENPOSE_FACE", False),
        openpose_detect_resolution=_get_env_int("PIPELINE_OPENPOSE_DETECT_RES", 512),

        # Backend
        generator_backend=_get_env("PIPELINE_BACKEND", "animatediff_controlnet"),

        # Models
        sd_base_model=_get_env("PIPELINE_SD_BASE", "runwayml/stable-diffusion-v1-5"),
        sd_motion_module=_get_env("PIPELINE_SD_MOTION", "guoyww/animatediff-motion-adapter-v1-5-2"),
        sd_controlnet_openpose_model=_get_env(
            "PIPELINE_SD_CONTROLNET",
            "lllyasviel/control_v11p_sd15_openpose",
        ),

        # Prompting
        seed=_get_env_int("PIPELINE_SEED", 1234),
        sd_prompt=_get_env(
            "PIPELINE_SD_PROMPT",
            "rtmn vigilante, full body, high quality, cinematic lighting",
        ),
        sd_negative_prompt=_get_env(
            "PIPELINE_SD_NEGATIVE_PROMPT",
            "lowres, blurry, bad anatomy, extra limbs, deformed, watermark, text",
        ),

        # First pass sampling
        sd_num_inference_steps=_get_env_int("PIPELINE_SD_STEPS", 30),
        sd_guidance_scale=_get_env_float("PIPELINE_SD_GUIDANCE", 6.5),
        sd_controlnet_conditioning_scale=_get_env_float("PIPELINE_SD_CONTROL_SCALE", 1.4),

        # IP-Adapter
        ip_adapter_enable=_get_env_bool("PIPELINE_IPA_ENABLE", True),
        ip_adapter_model=_get_env("PIPELINE_IPA_MODEL", "h94/IP-Adapter"),
        ip_adapter_subfolder=_get_env("PIPELINE_IPA_SUBFOLDER", "models"),
        ip_adapter_weight_name=_get_env("PIPELINE_IPA_WEIGHT", "ip-adapter_sd15.bin"),
        ip_adapter_scale=_get_env_float("PIPELINE_IPA_SCALE", 1.0),

        # LoRA
        lora_enable=_get_env_bool("PIPELINE_LORA_ENABLE", True),
        lora_path=_get_env("PIPELINE_LORA_PATH", str(root / "Output" / "loras" / "ratman_sd15")),
        lora_scale=_get_env_float("PIPELINE_LORA_SCALE", 1.2),

        # Refinement pass
        refine_enable=_get_env_bool("PIPELINE_REFINE_ENABLE", False),
        refine_prompt=_get_env("PIPELINE_REFINE_PROMPT", ""),
        refine_negative_prompt=_get_env("PIPELINE_REFINE_NEGATIVE_PROMPT", ""),
        refine_num_inference_steps=_get_env_int("PIPELINE_REFINE_STEPS", 20),
        refine_guidance_scale=_get_env_float("PIPELINE_REFINE_GUIDANCE", 5.5),
        refine_strength=_get_env_float("PIPELINE_REFINE_STRENGTH", 0.35),
        refine_controlnet_conditioning_scale=_get_env_float("PIPELINE_REFINE_CONTROL_SCALE", 0.9),
        refine_ip_adapter_enable=_get_env_bool("PIPELINE_REFINE_IPA_ENABLE", True),
        refine_ip_adapter_scale=_get_env_float("PIPELINE_REFINE_IPA_SCALE", 0.35),
    )