"""
DreamBooth LoRA training script for character fine-tuning.

Trains a LoRA on top of DreamShaper 8 using a small set of character images
(15-20 is enough). The trained weights are saved as pytorch_lora_weights.safetensors
and can be loaded at inference time via ratman_pipeline.py --lora_weights.

Usage:
    python3 lora_train.py \
        --instance_data_dir training_data/ \
        --output_dir lora_weights/ \
        --num_train_steps 1000

Estimated time: ~3-6 hours on GTX 1660 Super, ~4-8 hours on Apple Silicon MPS.
"""

import argparse
import os
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import save_file

_HERE = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = "Lykon/dreamshaper-8"
RESOLUTION = 512


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class InstanceDataset(Dataset):
    def __init__(self, data_dir, prompt, tokenizer, size=RESOLUTION):
        self.paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        if not self.paths:
            raise RuntimeError(f"No images found in {data_dir}")
        print(f"Found {len(self.paths)} training images.")

        self.prompt = prompt
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Pre-tokenise the prompt once
        self.input_ids = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx % len(self.paths)]).convert("RGB")
        return {"pixel_values": self.transform(img), "input_ids": self.input_ids}


# ---------------------------------------------------------------------------
# LoRA injection helpers
# ---------------------------------------------------------------------------

class LoRALinear(torch.nn.Module):
    """Wrap an existing Linear layer with a LoRA delta (A * B)."""

    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = torch.nn.Linear(in_f, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, out_f, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora(unet, rank: int, alpha: float):
    """Replace attention projection layers in the UNet with LoRA-wrapped versions."""
    lora_layers = {}
    for name, module in unet.named_modules():
        if module.__class__.__name__ == "Attention":
            for proj_name in ("to_q", "to_k", "to_v", "to_out"):
                attr = getattr(module, proj_name, None)
                if attr is None:
                    continue
                # to_out is a ModuleList — take index 0
                if isinstance(attr, torch.nn.ModuleList):
                    linear = attr[0]
                    key = f"{name}.{proj_name}.0"
                else:
                    linear = attr
                    key = f"{name}.{proj_name}"
                if not isinstance(linear, torch.nn.Linear):
                    continue
                wrapped = LoRALinear(linear, rank, alpha)
                if isinstance(attr, torch.nn.ModuleList):
                    attr[0] = wrapped
                else:
                    setattr(module, proj_name, wrapped)
                lora_layers[key] = wrapped
    return lora_layers


def lora_parameters(lora_layers):
    params = []
    for layer in lora_layers.values():
        params += list(layer.lora_A.parameters())
        params += list(layer.lora_B.parameters())
    return params


def save_lora_weights(lora_layers, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tensors = {}
    for key, layer in lora_layers.items():
        safe_key = key.replace(".", "_")
        tensors[f"{safe_key}.lora_A.weight"] = layer.lora_A.weight.detach().cpu()
        tensors[f"{safe_key}.lora_B.weight"] = layer.lora_B.weight.detach().cpu()
    out_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    save_file(tensors, out_path)
    print(f"LoRA weights saved to: {out_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    print(f"Training on device: {device}")

    # Load models
    print("Loading tokenizer and text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder").to(device)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").to(device)

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    # Freeze everything
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Inject LoRA into UNet attention layers
    print(f"Injecting LoRA (rank={args.rank}, alpha={args.rank})...")
    lora_layers = inject_lora(unet, rank=args.rank, alpha=float(args.rank))
    trainable_params = lora_parameters(lora_layers)
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")

    # Dataset + dataloader (loops infinitely via sampler)
    dataset = InstanceDataset(args.instance_data_dir, args.instance_prompt, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    unet.train()
    global_step = 0
    data_iter = iter(dataloader)

    print(f"Starting training for {args.num_train_steps} steps...")
    while global_step < args.num_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
        input_ids = batch["input_ids"].to(device)

        # Encode image to latents
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode text
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        global_step += 1

        if global_step % 50 == 0 or global_step == 1:
            print(f"  step {global_step}/{args.num_train_steps}  loss={loss.item():.4f}")

        # Save checkpoint every 500 steps
        if global_step % 500 == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_lora_weights(lora_layers, ckpt_dir)

    save_lora_weights(lora_layers, args.output_dir)
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DreamBooth LoRA training for character fine-tuning")
    parser.add_argument("--instance_data_dir", required=True,
                        help="Folder containing training images (15-20 images recommended)")
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "lora_weights"),
                        help="Where to save the trained LoRA weights")
    parser.add_argument("--instance_prompt", default="batman, dark superhero suit, yellow utility belt, masked hero",
                        help="Caption applied to all training images")
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank — higher = more parameters, more expressive (8-16 recommended)")
    args = parser.parse_args()

    if not os.path.isdir(args.instance_data_dir):
        raise FileNotFoundError(f"Training data directory not found: {args.instance_data_dir}")

    train(args)


if __name__ == "__main__":
    main()
