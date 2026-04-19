# character_video_gen.py

"""
character_video_gen.py

AnimateDiff + ControlNet(OpenPose) + optional IP-Adapter + optional LoRA video generation.

This version supports:
- T2V first window via AnimateDiffControlNetPipeline
- V2V continuation windows via AnimateDiffVideoToVideoControlNetPipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence
import inspect

import torch
from PIL import Image

from diffusers import (
    AnimateDiffControlNetPipeline,
    AnimateDiffVideoToVideoControlNetPipeline,
    ControlNetModel,
    MotionAdapter,
    DPMSolverMultistepScheduler,
)

from config import Settings
from utils_io import ensure_dir, log


_T2V_PIPE = None
_T2V_PIPE_SIGNATURE = None

_V2V_PIPE = None
_V2V_PIPE_SIGNATURE = None


def _get_device(settings: Settings) -> str:
    if settings.device.lower() == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pil_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _choose_call_keys(pipe) -> Dict[str, Optional[str]]:
    sig = inspect.signature(pipe.__call__)
    params = set(sig.parameters.keys())

    control_key = None
    for k in ("control_image", "conditioning_frames", "controlnet_image", "controlnet_conditioning_image"):
        if k in params:
            control_key = k
            break

    frames_key = None
    for k in ("num_frames", "video_length", "num_frames_per_prompt"):
        if k in params:
            frames_key = k
            break

    video_key = "video" if "video" in params else None
    ip_image_key = "ip_adapter_image" if "ip_adapter_image" in params else None
    ip_embeds_key = "ip_adapter_image_embeds" if "ip_adapter_image_embeds" in params else None

    return {
        "control_key": control_key,
        "frames_key": frames_key,
        "video_key": video_key,
        "ip_image_key": ip_image_key,
        "ip_embeds_key": ip_embeds_key,
    }


def _maybe_enable_xformers(settings: Settings, pipe) -> None:
    if not settings.sd_enable_xformers:
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log(settings, "Enabled xFormers attention")
    except Exception as e:
        log(settings, "xFormers not available:", e)


def _maybe_enable_memory_savers(pipe) -> None:
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass


def _enable_cpu_offload(settings: Settings, pipe) -> None:
    if not settings.sd_enable_cpu_offload:
        return

    if hasattr(pipe, "enable_sequential_cpu_offload"):
        try:
            pipe.enable_sequential_cpu_offload()
            log(settings, "Enabled sequential CPU offload")
            return
        except Exception as e:
            log(settings, "Could not enable sequential CPU offload:", e)

    if hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
            log(settings, "Enabled model CPU offload")
            return
        except Exception as e:
            log(settings, "Could not enable model CPU offload:", e)


def _load_lora_into_pipe(pipe, settings: Settings) -> None:
    if not settings.lora_enable:
        return

    lora_path = Path(settings.lora_path).expanduser()
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

    adapter_name = "ratman_lora"
    log(settings, f"Loading LoRA from: {lora_path}")

    if lora_path.is_file():
        parent = lora_path.parent
        weight_name = lora_path.name
        try:
            pipe.load_lora_weights(str(parent), weight_name=weight_name, adapter_name=adapter_name)
        except TypeError:
            pipe.load_lora_weights(str(parent), weight_name=weight_name)
    else:
        weight_name = None
        for candidate in ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"):
            candidate_path = lora_path / candidate
            if candidate_path.exists():
                weight_name = candidate
                break

        if weight_name is not None:
            try:
                pipe.load_lora_weights(str(lora_path), weight_name=weight_name, adapter_name=adapter_name)
            except TypeError:
                pipe.load_lora_weights(str(lora_path), weight_name=weight_name)
        else:
            try:
                pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
            except TypeError:
                pipe.load_lora_weights(str(lora_path))

    applied_scale = False

    if hasattr(pipe, "set_adapters"):
        for name in (adapter_name, "default_0", "default"):
            try:
                pipe.set_adapters(name, adapter_weights=[float(settings.lora_scale)])
                applied_scale = True
                log(settings, f"Set LoRA adapter '{name}' scale to {settings.lora_scale}")
                break
            except Exception:
                pass

    if not applied_scale and hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora(lora_scale=float(settings.lora_scale))
            applied_scale = True
            log(settings, f"Fused LoRA at scale {settings.lora_scale}")
        except Exception as e:
            log(settings, "WARNING: Could not fuse LoRA:", e)

    if not applied_scale:
        log(settings, "WARNING: Loaded LoRA but could not explicitly set scale; default scale will be used.")


def _build_common_components(settings: Settings, dtype: torch.dtype):
    controlnet = ControlNetModel.from_pretrained(
        settings.sd_controlnet_openpose_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    motion_adapter = MotionAdapter.from_pretrained(
        settings.sd_motion_module,
        torch_dtype=dtype,
    )
    return controlnet, motion_adapter


def _configure_loaded_pipe(settings: Settings, pipe, device: str, dtype: torch.dtype) -> None:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe.to("cuda")
        if settings.sd_enable_attention_slicing:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        _maybe_enable_xformers(settings, pipe)
        _maybe_enable_memory_savers(pipe)
        _enable_cpu_offload(settings, pipe)
    else:
        pipe.to("cpu")

    if settings.ip_adapter_enable:
        if not hasattr(pipe, "load_ip_adapter"):
            raise RuntimeError(
                "This diffusers pipeline does not support IP-Adapter loading (missing load_ip_adapter). "
                "Upgrade diffusers."
            )

        pipe.load_ip_adapter(
            settings.ip_adapter_model,
            subfolder=settings.ip_adapter_subfolder,
            weight_name=settings.ip_adapter_weight_name,
        )

        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(settings.ip_adapter_scale)

        if device == "cuda" and hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
            try:
                pipe.image_encoder.to(device="cuda", dtype=dtype)
            except Exception as e:
                log(settings, "WARNING: Could not move image_encoder to CUDA:", e)

    if settings.lora_enable:
        _load_lora_into_pipe(pipe, settings)


def _load_t2v_pipeline(settings: Settings):
    global _T2V_PIPE, _T2V_PIPE_SIGNATURE

    device = _get_device(settings)
    dtype = torch.float16 if device == "cuda" else torch.float32

    sig = (
        "t2v",
        settings.sd_base_model,
        settings.sd_motion_module,
        settings.sd_controlnet_openpose_model,
        settings.ip_adapter_enable,
        settings.ip_adapter_model,
        settings.ip_adapter_weight_name,
        settings.ip_adapter_scale,
        settings.lora_enable,
        settings.lora_path,
        settings.lora_scale,
        device,
        bool(settings.sd_enable_cpu_offload),
    )
    if _T2V_PIPE is not None and _T2V_PIPE_SIGNATURE == sig:
        return _T2V_PIPE

    log(settings, "Loading AnimateDiff T2V ControlNet pipeline")
    controlnet, motion_adapter = _build_common_components(settings, dtype)

    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        settings.sd_base_model,
        controlnet=controlnet,
        motion_adapter=motion_adapter,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    _configure_loaded_pipe(settings, pipe, device, dtype)

    _T2V_PIPE = pipe
    _T2V_PIPE_SIGNATURE = sig
    return pipe


def _load_v2v_pipeline(settings: Settings):
    global _V2V_PIPE, _V2V_PIPE_SIGNATURE

    device = _get_device(settings)
    dtype = torch.float16 if device == "cuda" else torch.float32

    sig = (
        "v2v",
        settings.sd_base_model,
        settings.sd_motion_module,
        settings.sd_controlnet_openpose_model,
        settings.ip_adapter_enable,
        settings.ip_adapter_model,
        settings.ip_adapter_weight_name,
        settings.ip_adapter_scale,
        settings.lora_enable,
        settings.lora_path,
        settings.lora_scale,
        device,
        bool(settings.sd_enable_cpu_offload),
    )
    if _V2V_PIPE is not None and _V2V_PIPE_SIGNATURE == sig:
        return _V2V_PIPE

    log(settings, "Loading AnimateDiff V2V ControlNet pipeline")
    controlnet, motion_adapter = _build_common_components(settings, dtype)

    pipe = AnimateDiffVideoToVideoControlNetPipeline.from_pretrained(
        settings.sd_base_model,
        controlnet=controlnet,
        motion_adapter=motion_adapter,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    _configure_loaded_pipe(settings, pipe, device, dtype)

    _V2V_PIPE = pipe
    _V2V_PIPE_SIGNATURE = sig
    return pipe


def _prepare_ip_adapter_embeds_if_needed(
    pipe,
    ref_image: Image.Image,
    device: str,
    do_cfg: bool,
) -> List[torch.Tensor]:
    for helper_name in ("prepare_ip_adapter_image_embeds", "_prepare_ip_adapter_image_embeds"):
        if hasattr(pipe, helper_name):
            fn = getattr(pipe, helper_name)
            return fn(
                ip_adapter_image=[ref_image],
                device=torch.device(device),
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
            )

    raise RuntimeError(
        "Pipeline accepts ip_adapter_image_embeds but no helper found to build embeds. "
        "Upgrade diffusers."
    )


def _build_generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _save_result_frames(result, out_frames_dir: Path) -> List[Path]:
    ensure_dir(out_frames_dir)

    frames = None
    if hasattr(result, "frames"):
        frames = result.frames
    elif hasattr(result, "images"):
        frames = result.images

    if frames is None:
        raise RuntimeError("Pipeline result has neither `.frames` nor `.images`")

    if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], list):
        frames = frames[0]

    out_paths: List[Path] = []
    for idx, img in enumerate(frames):
        out_path = out_frames_dir / f"gen_{idx:06d}.png"
        img.save(out_path)
        out_paths.append(out_path)
    return out_paths


def generate_character_video_t2v(
    settings: Settings,
    ref_image_path: Path,
    pose_frames: List[Path],
    out_frames_dir: Path,
    width: int,
    height: int,
    *,
    seed: Optional[int] = None,
) -> List[Path]:
    ensure_dir(out_frames_dir)

    device = _get_device(settings)
    pipe = _load_t2v_pipeline(settings)

    control_images = [_pil_rgb(p) for p in pose_frames]
    num_frames = len(control_images)
    ref_img = _pil_rgb(ref_image_path)

    keys = _choose_call_keys(pipe)
    control_key = keys["control_key"]
    frames_key = keys["frames_key"]
    ip_image_key = keys["ip_image_key"]
    ip_embeds_key = keys["ip_embeds_key"]

    if control_key is None:
        raise TypeError("AnimateDiffControlNetPipeline call signature missing control image arg.")

    do_cfg = float(settings.sd_guidance_scale) > 1.0
    use_seed = int(settings.seed if seed is None else seed)

    call_kwargs: Dict[str, Any] = dict(
        prompt=settings.sd_prompt,
        negative_prompt=settings.sd_negative_prompt,
        num_inference_steps=int(settings.sd_num_inference_steps),
        guidance_scale=float(settings.sd_guidance_scale),
        controlnet_conditioning_scale=float(settings.sd_controlnet_conditioning_scale),
        generator=_build_generator(device, use_seed),
    )

    call_kwargs[control_key] = control_images

    if frames_key is not None:
        call_kwargs[frames_key] = int(num_frames)

    if settings.ip_adapter_enable:
        if ip_image_key is not None:
            call_kwargs[ip_image_key] = [ref_img]
        elif ip_embeds_key is not None:
            call_kwargs[ip_embeds_key] = _prepare_ip_adapter_embeds_if_needed(
                pipe=pipe, ref_image=ref_img, device=device, do_cfg=do_cfg
            )
        else:
            raise TypeError("IP-Adapter enabled but pipeline call doesn't accept image or embeds.")

    sig = inspect.signature(pipe.__call__).parameters
    if "height" in sig:
        call_kwargs["height"] = int(height)
    if "width" in sig:
        call_kwargs["width"] = int(width)

    ctx = torch.autocast("cuda") if device == "cuda" else torch.no_grad()
    with ctx:
        result = pipe(**call_kwargs)

    out_paths = _save_result_frames(result, out_frames_dir)
    log(settings, f"[t2v] Generated {len(out_paths)} frames")
    return out_paths


def generate_character_video_v2v(
    settings: Settings,
    ref_image_path: Path,
    pose_frames: List[Path],
    init_video_frames: Sequence[Path],
    out_frames_dir: Path,
    width: int,
    height: int,
    *,
    seed: Optional[int] = None,
    overlap_count: int,
) -> List[Path]:
    """
    Continue a window using V2V only on the overlap anchor frames, then append fresh T2V
    frames for the tail of the window.

    This avoids freezing caused by repeating the last frame across the unseen tail.
    """
    ensure_dir(out_frames_dir)

    if overlap_count <= 0:
        return generate_character_video_t2v(
            settings=settings,
            ref_image_path=ref_image_path,
            pose_frames=pose_frames,
            out_frames_dir=out_frames_dir,
            width=width,
            height=height,
            seed=seed,
        )

    if len(init_video_frames) != overlap_count:
        raise ValueError("init_video_frames must match overlap_count")

    if overlap_count >= len(pose_frames):
        overlap_count = len(pose_frames) - 1

    device = _get_device(settings)
    pipe = _load_v2v_pipeline(settings)

    overlap_pose = pose_frames[:overlap_count]
    tail_pose = pose_frames[overlap_count:]

    control_images = [_pil_rgb(p) for p in overlap_pose]
    init_video = [_pil_rgb(p) for p in init_video_frames]
    ref_img = _pil_rgb(ref_image_path)

    keys = _choose_call_keys(pipe)
    control_key = keys["control_key"]
    video_key = keys["video_key"]
    ip_image_key = keys["ip_image_key"]
    ip_embeds_key = keys["ip_embeds_key"]

    if control_key is None:
        raise TypeError("AnimateDiffVideoToVideoControlNetPipeline call signature missing control image arg.")
    if video_key is None:
        raise TypeError("AnimateDiffVideoToVideoControlNetPipeline call signature missing video arg.")

    do_cfg = float(settings.sd_guidance_scale) > 1.0
    use_seed = int(settings.seed if seed is None else seed)

    call_kwargs: Dict[str, Any] = dict(
        prompt=settings.sd_prompt,
        negative_prompt=settings.sd_negative_prompt,
        num_inference_steps=int(settings.sd_num_inference_steps),
        guidance_scale=float(settings.sd_guidance_scale),
        controlnet_conditioning_scale=float(settings.sd_controlnet_conditioning_scale),
        strength=float(settings.animate_v2v_strength),
        generator=_build_generator(device, use_seed),
    )

    call_kwargs[video_key] = init_video
    call_kwargs[control_key] = control_images

    if settings.ip_adapter_enable:
        if ip_image_key is not None:
            call_kwargs[ip_image_key] = [ref_img]
        elif ip_embeds_key is not None:
            call_kwargs[ip_embeds_key] = _prepare_ip_adapter_embeds_if_needed(
                pipe=pipe, ref_image=ref_img, device=device, do_cfg=do_cfg
            )
        else:
            raise TypeError("IP-Adapter enabled but pipeline call doesn't accept image or embeds.")

    sig = inspect.signature(pipe.__call__).parameters
    if "height" in sig:
        call_kwargs["height"] = int(height)
    if "width" in sig:
        call_kwargs["width"] = int(width)

    ctx = torch.autocast("cuda") if device == "cuda" else torch.no_grad()
    with ctx:
        result = pipe(**call_kwargs)

    overlap_out = _save_result_frames(result, out_frames_dir / "overlap")

    # Fresh T2V for the new tail
    tail_out: List[Path] = []
    if tail_pose:
        tail_out = generate_character_video_t2v(
            settings=settings,
            ref_image_path=ref_image_path,
            pose_frames=tail_pose,
            out_frames_dir=out_frames_dir / "tail",
            width=width,
            height=height,
            seed=use_seed,
        )

    # Reindex into one output sequence for this window
    out_paths: List[Path] = []
    merged = overlap_out + tail_out
    for idx, src in enumerate(merged):
        out_path = out_frames_dir / f"gen_{idx:06d}.png"
        _pil_rgb(src).save(out_path)
        out_paths.append(out_path)

    log(settings, f"[v2v] Generated {len(out_paths)} frames ({len(overlap_out)} overlap-continuation + {len(tail_out)} fresh tail)")
    return out_paths