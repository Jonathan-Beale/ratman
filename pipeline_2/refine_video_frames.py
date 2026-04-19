"""
refine_video_frames.py

Second-pass Ratman refinement using Stable Diffusion img2img + ControlNet OpenPose
+ optional IP-Adapter + optional LoRA.

Kinda mid for temporal preservation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import inspect

import torch
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

from config import Settings
from utils_io import ensure_dir, log


_REFINE_PIPE = None
_REFINE_PIPE_SIGNATURE = None


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
    for k in ("control_image", "controlnet_conditioning_image", "controlnet_image"):
        if k in params:
            control_key = k
            break

    ip_image_key = "ip_adapter_image" if "ip_adapter_image" in params else None
    ip_embeds_key = "ip_adapter_image_embeds" if "ip_adapter_image_embeds" in params else None

    return {
        "control_key": control_key,
        "ip_image_key": ip_image_key,
        "ip_embeds_key": ip_embeds_key,
    }


def _maybe_enable_xformers(settings: Settings, pipe) -> None:
    if not settings.sd_enable_xformers:
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log(settings, "[refine] Enabled xFormers attention")
    except Exception as e:
        log(settings, "[refine] xFormers not available:", e)


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
            log(settings, "[refine] Enabled sequential CPU offload")
            return
        except Exception as e:
            log(settings, "[refine] Could not enable sequential CPU offload:", e)

    if hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
            log(settings, "[refine] Enabled model CPU offload")
            return
        except Exception as e:
            log(settings, "[refine] Could not enable model CPU offload:", e)


def _load_lora_into_pipe(pipe, settings: Settings) -> None:
    if not settings.lora_enable:
        return

    lora_path = Path(settings.lora_path).expanduser()
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

    adapter_name = "ratman_lora_refine"
    log(settings, f"[refine] Loading LoRA from: {lora_path}")

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
                log(settings, f"[refine] Set LoRA adapter '{name}' scale to {settings.lora_scale}")
                break
            except Exception:
                pass

    if not applied_scale and hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora(lora_scale=float(settings.lora_scale))
            applied_scale = True
            log(settings, f"[refine] Fused LoRA at scale {settings.lora_scale}")
        except Exception as e:
            log(settings, "[refine] WARNING: Could not fuse LoRA:", e)

    if not applied_scale:
        log(settings, "[refine] WARNING: Loaded LoRA but could not explicitly set scale.")


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
        "Refine pipeline accepts ip_adapter_image_embeds but no helper was found. "
        "Upgrade diffusers."
    )


def _load_refine_pipeline(settings: Settings):
    global _REFINE_PIPE, _REFINE_PIPE_SIGNATURE

    device = _get_device(settings)
    dtype = torch.float16 if device == "cuda" else torch.float32

    sig = (
        settings.sd_base_model,
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
    if _REFINE_PIPE is not None and _REFINE_PIPE_SIGNATURE == sig:
        return _REFINE_PIPE

    log(settings, "[refine] Loading ControlNet:", settings.sd_controlnet_openpose_model)
    controlnet = ControlNetModel.from_pretrained(
        settings.sd_controlnet_openpose_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    log(settings, "[refine] Loading base SD model:", settings.sd_base_model)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        settings.sd_base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe.to("cuda")
        _maybe_enable_xformers(settings, pipe)
        _maybe_enable_memory_savers(pipe)
    else:
        pipe.to("cpu")

    if settings.refine_ip_adapter_enable:
        if not hasattr(pipe, "load_ip_adapter"):
            raise RuntimeError(
                "This diffusers pipeline does not support IP-Adapter loading for refinement. "
                "Upgrade diffusers."
            )

        log(settings, "[refine] Loading IP-Adapter:", settings.ip_adapter_model)
        pipe.load_ip_adapter(
            settings.ip_adapter_model,
            subfolder=settings.ip_adapter_subfolder,
            weight_name=settings.ip_adapter_weight_name,
        )

        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(settings.refine_ip_adapter_scale)

        if device == "cuda" and hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
            try:
                pipe.image_encoder.to(device="cuda", dtype=dtype)
                log(settings, "[refine] Moved IP-Adapter image_encoder to CUDA")
            except Exception as e:
                log(settings, "[refine] WARNING: Could not move image_encoder to CUDA:", e)

    if settings.lora_enable:
        _load_lora_into_pipe(pipe, settings)

    _REFINE_PIPE = pipe
    _REFINE_PIPE_SIGNATURE = sig
    return pipe

    if device == "cuda":
        if settings.sd_enable_attention_slicing and not settings.refine_ip_adapter_enable:
            pipe.enable_attention_slicing()

        _enable_cpu_offload(settings, pipe)


def refine_video_frames(
    settings: Settings,
    generated_frames: List[Path],
    pose_frames: List[Path],
    ref_image_path: Path,
    out_frames_dir: Path,
    width: int,
    height: int,
) -> List[Path]:
    """
    Refine generated frames one-by-one with img2img.
    """
    ensure_dir(out_frames_dir)

    if not generated_frames:
        return []
    if not pose_frames:
        raise ValueError("No pose frames provided to refinement")

    n = min(len(generated_frames), len(pose_frames))
    generated_frames = generated_frames[:n]
    pose_frames = pose_frames[:n]

    device = _get_device(settings)
    pipe = _load_refine_pipeline(settings)

    keys = _choose_call_keys(pipe)
    control_key = keys["control_key"]
    ip_image_key = keys["ip_image_key"]
    ip_embeds_key = keys["ip_embeds_key"]

    if control_key is None:
        raise TypeError(
            "StableDiffusionControlNetImg2ImgPipeline does not expose a recognized control image argument."
        )

    ref_img = _pil_rgb(ref_image_path)
    do_cfg = float(settings.refine_guidance_scale) > 1.0

    prompt = settings.refine_prompt.strip() or settings.sd_prompt
    negative_prompt = settings.refine_negative_prompt.strip() or settings.sd_negative_prompt

    out_paths: List[Path] = []

    log(settings, f"[refine] Refining {n} frames with img2img + ControlNet"
                  f"{' + IP-Adapter' if settings.refine_ip_adapter_enable else ''}"
                  f"{' + LoRA' if settings.lora_enable else ''}")

    call_sig = inspect.signature(pipe.__call__)
    params = call_sig.parameters

    for idx, (gen_path, pose_path) in enumerate(zip(generated_frames, pose_frames)):
        init_img = _pil_rgb(gen_path)
        control_img = _pil_rgb(pose_path)

        generator = torch.Generator(device=device).manual_seed(int(settings.seed) + idx)

        call_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_img,
            num_inference_steps=int(settings.refine_num_inference_steps),
            guidance_scale=float(settings.refine_guidance_scale),
            strength=float(settings.refine_strength),
            controlnet_conditioning_scale=float(settings.refine_controlnet_conditioning_scale),
            generator=generator,
        )

        call_kwargs[control_key] = control_img

        if settings.refine_ip_adapter_enable:
            if ip_image_key is not None:
                call_kwargs[ip_image_key] = [ref_img]
            elif ip_embeds_key is not None:
                call_kwargs[ip_embeds_key] = _prepare_ip_adapter_embeds_if_needed(
                    pipe=pipe,
                    ref_image=ref_img,
                    device=device,
                    do_cfg=do_cfg,
                )

        if "ip_adapter_scale" in params and settings.refine_ip_adapter_enable:
            call_kwargs["ip_adapter_scale"] = float(settings.refine_ip_adapter_scale)

        if "height" in params:
            call_kwargs["height"] = int(height)
        if "width" in params:
            call_kwargs["width"] = int(width)

        ctx = torch.autocast("cuda") if device == "cuda" else torch.no_grad()
        with ctx:
            result = pipe(**call_kwargs)

        img = result.images[0]
        out_path = out_frames_dir / f"refined_{idx:06d}.png"
        img.save(out_path)
        out_paths.append(out_path)

    log(settings, f"[refine] Wrote {len(out_paths)} refined frames")
    return out_paths