"""
train_ratman_lora.py

Trains a Ratman subject LoRA for Stable Diffusion 1.5 using the official
Diffusers DreamBooth LoRA training script.

Usage:
    python train_ratman_lora.py ^
      --dataset-dir RatmanLoraInput ^
      --output-dir Output/loras/ratman_sd15 ^
      --instance-token rtmn ^
      --class-noun vigilante
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def find_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def locate_diffusers_train_script(user_supplied: str | None) -> Path:
    """
    Find train_dreambooth_lora.py.

    Search order:
    1) --train-script
    2) DIFFUSERS_TRAIN_DREAMBOOTH_LORA env var
    3) ./diffusers/examples/dreambooth/train_dreambooth_lora.py
    """
    candidates: List[Path] = []

    if user_supplied:
        candidates.append(Path(user_supplied))

    env_path = os.environ.get("DIFFUSERS_TRAIN_DREAMBOOTH_LORA", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path("diffusers/examples/dreambooth/train_dreambooth_lora.py"))

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not find train_dreambooth_lora.py.\n"
        "Provide --train-script or set DIFFUSERS_TRAIN_DREAMBOOTH_LORA, or clone diffusers so that\n"
        "diffusers/examples/dreambooth/train_dreambooth_lora.py exists."
    )


def ensure_accelerate_installed() -> None:
    try:
        import accelerate  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "The `accelerate` package is required.\n"
            "Install it with: pip install -U accelerate"
        ) from e


def accelerate_config_exists() -> bool:
    home = Path.home()
    cfg1 = home / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
    cfg2 = home / ".cache" / "huggingface" / "accelerate" / "default_config.yml"
    return cfg1.exists() or cfg2.exists()


def write_env_hint(output_dir: Path, lora_name: str, instance_token: str) -> Path:
    """
    Helper env file.
    """
    env_path = output_dir / "ratman_lora.env"
    content = f"""# Ratman LoRA helper env

PIPELINE_LORA_ENABLE=1
PIPELINE_LORA_PATH={output_dir.as_posix()}
PIPELINE_LORA_SCALE=0.9
PIPELINE_LORA_NAME={lora_name}

# Recommended trigger token in prompt:
# {instance_token}
"""
    env_path.write_text(content, encoding="utf-8")
    return env_path


def build_command(args: argparse.Namespace, train_script: Path, dataset_dir: Path, output_dir: Path) -> List[str]:
    """
    Build the accelerate command for the official Diffusers DreamBooth LoRA trainer.
    """
    instance_prompt = f"a photo of {args.instance_token} {args.class_noun}"
    class_prompt = f"a photo of a {args.class_noun}"

    cmd: List[str] = [
        "accelerate",
        "launch",
        str(train_script),
        "--pretrained_model_name_or_path", args.base_model,
        "--instance_data_dir", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--instance_prompt", instance_prompt,
        "--class_prompt", class_prompt,
        "--resolution", str(args.resolution),
        "--train_batch_size", str(args.train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", args.lr_scheduler,
        "--lr_warmup_steps", str(args.lr_warmup_steps),
        "--max_train_steps", str(args.max_train_steps),
        "--checkpointing_steps", str(args.checkpointing_steps),
        "--seed", str(args.seed),
        "--validation_prompt", f"{args.instance_token} {args.class_noun}, full body, standing, high quality",
        "--validation_epochs", str(args.validation_epochs),
        "--rank", str(args.rank),
    ]

    if args.mixed_precision:
        cmd += ["--mixed_precision", args.mixed_precision]

    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    if args.use_8bit_adam:
        cmd.append("--use_8bit_adam")

    if args.enable_xformers_memory_efficient_attention:
        cmd.append("--enable_xformers_memory_efficient_attention")

    if args.train_text_encoder:
        cmd.append("--train_text_encoder")

    # Optional prior preservation if the user wants it
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("--with-prior-preservation requires --class-data-dir")
        class_data_dir = Path(args.class_data_dir)
        class_data_dir.mkdir(parents=True, exist_ok=True)
        cmd += [
            "--with_prior_preservation",
            "--class_data_dir", str(class_data_dir),
            "--num_class_images", str(args.num_class_images),
            "--prior_loss_weight", str(args.prior_loss_weight),
        ]

    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Ratman LoRA for SD 1.5 using Diffusers DreamBooth LoRA.")

    p.add_argument("--dataset-dir", type=str, required=True, help="Folder containing Ratman training images.")
    p.add_argument("--output-dir", type=str, default="Output/loras/ratman_sd15", help="Where to save LoRA weights.")
    p.add_argument("--train-script", type=str, default=None, help="Path to train_dreambooth_lora.py")
    p.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--instance-token", type=str, default="rtmn", help="Unique subject token, e.g. rtmn")
    p.add_argument("--class-noun", type=str, default="vigilante", help="Class noun, e.g. vigilante or superhero")

    # Training hyperparams
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", type=str, default="constant")
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--max-train-steps", type=int, default=1500)
    p.add_argument("--checkpointing-steps", type=int, default=250)
    p.add_argument("--validation-epochs", type=int, default=50)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)

    # Memory / speed
    p.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--use-8bit-adam", action="store_true", default=False)
    p.add_argument("--no-use-8bit-adam", action="store_false", dest="use_8bit_adam")
    p.add_argument("--enable-xformers-memory-efficient-attention", action="store_true", default=False)

    # Quality / subject fidelity
    p.add_argument("--train-text-encoder", action="store_true", default=False)

    # Optional prior preservation
    p.add_argument("--with-prior-preservation", action="store_true", default=False)
    p.add_argument("--class-data-dir", type=str, default=None)
    p.add_argument("--num-class-images", type=int, default=100)
    p.add_argument("--prior-loss-weight", type=float, default=1.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    images = find_images(dataset_dir)
    if len(images) == 0:
        raise RuntimeError(f"No training images found in {dataset_dir}")
    if len(images) < 8:
        print(
            f"[warn] Only found {len(images)} images. DreamBooth LoRA usually works better with ~10-30 subject images.",
            file=sys.stderr,
        )
    else:
        print(f"[info] Found {len(images)} subject images in {dataset_dir}")

    ensure_accelerate_installed()

    if not accelerate_config_exists():
        print(
            "[warn] No Accelerate default config detected. If `accelerate launch` fails, run:\n"
            "       accelerate config default\n",
            file=sys.stderr,
        )

    train_script = locate_diffusers_train_script(args.train_script)
    require_file(train_script, "Diffusers DreamBooth LoRA trainer")

    # Training manifest copy for reproducibility
    manifest_path = output_dir / "training_manifest.txt"
    manifest_lines = [
        f"dataset_dir={dataset_dir}",
        f"base_model={args.base_model}",
        f"instance_token={args.instance_token}",
        f"class_noun={args.class_noun}",
        f"num_images={len(images)}",
        f"resolution={args.resolution}",
        f"max_train_steps={args.max_train_steps}",
        f"rank={args.rank}",
        "",
        "images:",
        *[str(p) for p in images],
        "",
    ]
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    cmd = build_command(args, train_script, dataset_dir, output_dir)

    print("[info] Launching LoRA training...")
    print("[info] Command:")
    print(" ".join(cmd))
    print()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LoRA training failed with exit code {e.returncode}") from e

    env_hint = write_env_hint(
        output_dir=output_dir,
        lora_name=output_dir.name,
        instance_token=args.instance_token,
    )

    print()
    print("[done] Ratman LoRA training finished.")
    print(f"[done] Output dir: {output_dir}")
    print(f"[done] Manifest:   {manifest_path}")
    print(f"[done] Env hint:   {env_hint}")


if __name__ == "__main__":
    main()