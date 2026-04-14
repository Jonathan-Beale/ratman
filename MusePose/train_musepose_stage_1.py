#!/usr/bin/env python3
"""
train_musepose_stage_1.py

MusePose Stage 1 Training Script (student-teacher distillation)

This script is adapted from train_pose_guider_unet_distill.py for the MusePose pipeline.

Teacher (frozen): PoseGuider + denoising UNet, using teacher skeleton videos (dwpose)
Student (trainable): PoseGuider, using student videos (custom)

Data:
	- Teacher skeleton videos: /root/RATMAN_Model/datasets/synthetic/synthetic_single_dwpose/<seq>/*.mp4
	- Student videos: /root/RATMAN_Model/datasets/synthetic/custom/<seq>/*.mp4

Usage:
	python MusePose/train_musepose_stage_1.py --steps 20000 --lr 5e-5 --out_dir exp_output/stage_1
"""


import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from musepose.models.pose_guider import PoseGuider
from musepose.models.unet_3d import UNet3DConditionModel

IMG_SIZE = 256

TEACHER_ROOT = Path("/root/RATMAN_Model/datasets/synthetic/synthetic_single_dwpose")
STUDENT_ROOT = Path("/root/RATMAN_Model/datasets/synthetic/synthetic_single_dwpose_custom")
MUSEPOSE_DIR = Path("/root/RATMAN_Model/MusePose")
WEIGHTS_DIR  = Path("/root/RATMAN_Model/MusePose/pretrained_weights/MusePose")

# --- Dataset for paired teacher/student videos ---
class PairedVideoDataset:
	def __init__(self, teacher_root, student_root):
		self.pairs = []
		# Pair .mp4 files by filename in flat directories
		teacher_vids = {p.name: p for p in teacher_root.glob("*.mp4")}
		student_vids = {p.name: p for p in student_root.glob("*.mp4")}
		common = sorted(set(teacher_vids.keys()) & set(student_vids.keys()))
		for name in common:
			tvid = teacher_vids[name]
			svid = student_vids[name]
			n_frames = min(_count_video_frames(tvid), _count_video_frames(svid))
			if n_frames == 0:
				continue
			self.pairs.append((tvid, svid, n_frames))
		print(f"PairedVideoDataset: {len(self.pairs)} pairs found.")
		if not self.pairs:
			raise RuntimeError("No valid teacher/student video pairs found.")

	def stream(self):
		while True:
			random.shuffle(self.pairs)
			for tvid, svid, n_frames in self.pairs:
				indices = list(range(n_frames))
				random.shuffle(indices)
				for i in indices:
					teacher_pil = _read_frame_at(tvid, i)
					student_pil = _read_frame_at(svid, i)
					if teacher_pil is None or student_pil is None:
						continue
					yield student_pil, teacher_pil

def _count_video_frames(path: Path) -> int:
	cap = cv2.VideoCapture(str(path))
	n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()
	return n

def _read_frame_at(path: Path, idx: int, flip: bool = False):
	cap = cv2.VideoCapture(str(path))
	cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
	ret, bgr = cap.read()
	cap.release()
	if not ret:
		return None
	if flip:
		bgr = cv2.flip(bgr, 1)
	return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def pil_to_tensor5d(pil_img, device, dtype):
	img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).convert("RGB")
	arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
	t = torch.from_numpy(arr).permute(2, 0, 1)
	return t.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)

def train(args):
	device      = "cuda" if torch.cuda.is_available() else "cpu"
	inf_dtype   = torch.float16
	train_dtype = torch.float32

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# --- Load frozen UNet ---
	infer_cfg = OmegaConf.load(MUSEPOSE_DIR / "configs" / "inference_v2.yaml")
	unet = UNet3DConditionModel.from_pretrained_2d(
		str(WEIGHTS_DIR.parent / "sd-image-variations-diffusers"),
		str(WEIGHTS_DIR / "motion_module.pth"),
		subfolder="unet",
		unet_additional_kwargs=infer_cfg.unet_additional_kwargs,
	)
	unet.load_state_dict(
		torch.load(str(WEIGHTS_DIR / "denoising_unet.pth"), map_location="cpu"),
		strict=False,
	)
	unet.to(device=device, dtype=inf_dtype).eval()
	for p in unet.parameters():
		p.requires_grad_(False)
	print("UNet loaded and frozen.")

	# --- Frozen teacher PoseGuider ---
	teacher = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(device, dtype=inf_dtype)
	teacher.load_state_dict(
		torch.load(str(WEIGHTS_DIR / "pose_guider.pth"), map_location="cpu")
	)
	teacher.eval()
	for p in teacher.parameters():
		p.requires_grad_(False)
	print("Teacher PoseGuider loaded and frozen.")

	# --- Trainable student PoseGuider ---
	student = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(device, dtype=train_dtype)
	start_step = 0
	if args.resume_ckpt and Path(args.resume_ckpt).exists():
		ckpt = torch.load(args.resume_ckpt, map_location="cpu")
		if isinstance(ckpt, dict) and "model" in ckpt:
			student.load_state_dict(ckpt["model"])
			start_step = ckpt.get("step", 0)
		else:
			student.load_state_dict(ckpt)
		print(f"Student resumed from: {args.resume_ckpt}  (step {start_step})")
	else:
		student.load_state_dict(
			torch.load(str(WEIGHTS_DIR / "pose_guider.pth"), map_location="cpu")
		)
		print("Student initialised from teacher weights.")

	optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

	enc_hs = torch.zeros(1, 1, 768, device=device, dtype=inf_dtype)

	# --- Dataset ---
	dataset  = PairedVideoDataset(TEACHER_ROOT, STUDENT_ROOT)
	data_gen = dataset.stream()

	step      = start_step
	best_loss = float("inf")
	running_loss = 0.0

	pbar = tqdm(total=args.steps, initial=start_step, desc="MusePose Stage 1")

	while step < args.steps:
		student_pil, teacher_pil = next(data_gen)

		student_t = pil_to_tensor5d(student_pil, device, train_dtype)
		teacher_t = pil_to_tensor5d(teacher_pil, device, inf_dtype)

		lat_h, lat_w = IMG_SIZE // 8, IMG_SIZE // 8
		noise = torch.randn(1, 4, 1, lat_h, lat_w, device=device, dtype=inf_dtype)
		t = torch.randint(0, 1000, (1,), device=device).long()

		with torch.no_grad():
			teacher_pose = teacher(teacher_t)
			teacher_pred = unet(
				sample=noise,
				timestep=t,
				encoder_hidden_states=enc_hs,
				pose_cond_fea=teacher_pose,
				return_dict=False,
			)[0]

		student_pose = student(student_t)
		student_pred = unet(
			sample=noise,
			timestep=t,
			encoder_hidden_states=enc_hs,
			pose_cond_fea=student_pose.to(inf_dtype),
			return_dict=False,
		)[0]
		loss = F.mse_loss(student_pred.float(), teacher_pred.float())

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
		optimizer.step()
		scheduler.step()

		running_loss += loss.item()
		step += 1
		pbar.update(1)
		pbar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

		if step % args.log_every == 0:
			avg = running_loss / args.log_every
			running_loss = 0.0
			print(f"\nStep {step}/{args.steps}  loss={avg:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

			if avg < best_loss:
				best_loss = avg
				torch.save({"model": student.state_dict(), "step": step},
						   out_dir / "student_pose_guider_best.pt")
				print(f"  ★ New best: {best_loss:.5f}")

		if step % args.save_every == 0:
			torch.save({"model": student.state_dict(), "step": step},
					   out_dir / f"student_pose_guider_step{step:06d}.pt")
			print(f"  Checkpoint saved at step {step}")

	torch.save({"model": student.state_dict(), "step": step},
			   out_dir / "student_pose_guider_final.pt")
	pbar.close()
	print(f"\nDone. Best loss: {best_loss:.5f}")

def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--resume_ckpt",  type=str, default=None,
				   help="Student checkpoint to resume from")
	p.add_argument("--out_dir",      type=str, default="exp_output/stage_1")
	p.add_argument("--steps",        type=int,   default=20000)
	p.add_argument("--lr",           type=float, default=5e-5)
	p.add_argument("--log_file",     type=str,   default=None)
	p.add_argument("--log_every",    type=int,   default=100)
	p.add_argument("--save_every",   type=int,   default=2000)
	return p.parse_args()

if __name__ == "__main__":
	train(parse_args())
