import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time

import os.path as osp
import argparse
import logging
import math
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory



import diffusers
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from musepose.models.mutual_self_attention import ReferenceAttentionControl

# ---- Inlined HumanDanceDataset ----
import json
import torchvision.transforms as transforms
from decord import VideoReader
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

class HumanDanceDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.img_size,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.img_size,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)


    def __getitem__(self, index, retry_count=0):
        start_time = time.time()
        print(f"[DIAG] Enter __getitem__ index={index} retry_count={retry_count}")
        DATASET_ROOT = os.path.abspath(os.path.join(os.getcwd(), 'datasets'))
        def resolve_path(p):
            if not p:
                return p
            if os.path.isabs(p):
                return p
            abs_path = os.path.join(DATASET_ROOT, p)
            return abs_path

        def timeout_handler(signum, frame):
            raise TimeoutError("Timed out loading file")

        import sys
        import traceback
        import signal
        import os
        MAX_RETRIES = 10

        video_meta = self.vid_meta[index]
        print(f"[DIAG] video_meta: {video_meta}")
        video_path = resolve_path(video_meta.get("video_path", ""))
        kps_path = resolve_path(video_meta["kps_path"])
        kps_path_custom = resolve_path(video_meta.get("kps_path_custom", None))
        print(f"[DIAG] video_path: {video_path}, kps_path: {kps_path}, kps_path_custom: {kps_path_custom}")

        # Debug: print absolute paths and existence
        try:
            print(f"[DIAG] Attempting to open video/kps files at index {index}")
            if video_path:
                print(f"[DEBUG] video_path: {video_path} | exists: {os.path.exists(video_path)}")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)  # 10 second timeout
                video_reader = VideoReader(video_path)
                signal.alarm(0)
            else:
                video_reader = None
        except Exception as e:
            print(f"[ERROR] Skipping video_path {video_path}: {e}")
            traceback.print_exc(file=sys.stdout)
            video_reader = None
        try:
            print(f"[DEBUG] kps_path: {kps_path} | exists: {os.path.exists(kps_path)}")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            kps_reader = VideoReader(kps_path)
            signal.alarm(0)
        except Exception as e:
            print(f"[ERROR] Skipping kps_path {kps_path}: {e}")
            traceback.print_exc(file=sys.stdout)
            kps_reader = None
        kps_custom_reader = None
        if kps_path_custom:
            try:
                print(f"[DEBUG] kps_path_custom: {kps_path_custom} | exists: {os.path.exists(kps_path_custom)}")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                kps_custom_reader = VideoReader(kps_path_custom)
                signal.alarm(0)
            except Exception as e:
                print(f"[ERROR] Skipping kps_path_custom {kps_path_custom}: {e}")
                traceback.print_exc(file=sys.stdout)
                kps_custom_reader = None

        # Use skeleton length if video is missing
        try:
            print(f"[DIAG] Checking video/kps lengths at index {index}")
            if video_reader is not None and kps_reader is not None:
                assert len(video_reader) == len(kps_reader), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"
                if kps_custom_reader:
                    assert (
                        len(video_reader) == len(kps_custom_reader)
                    ), f"{len(video_reader) = } != {len(kps_custom_reader) = } in {video_path}"
                video_length = len(video_reader)
            elif kps_reader is not None:
                if kps_custom_reader:
                    assert len(kps_reader) == len(kps_custom_reader), f"{len(kps_reader) = } != {len(kps_custom_reader) = } in {kps_path}"
                video_length = len(kps_reader)
            else:
                raise RuntimeError("No valid video or kps reader")
        except Exception as e:
            print(f"[ERROR] Skipping index {index} due to length assertion: {e}")
            traceback.print_exc(file=sys.stdout)
            if retry_count < MAX_RETRIES:
                return self.__getitem__((index + 1) % len(self.vid_meta), retry_count=retry_count+1)
            else:
                print(f"[FATAL] Max retries ({MAX_RETRIES}) exceeded at index {index}. Raising error.")
                raise

        margin = min(self.sample_margin, video_length)

        try:
            print(f"[DIAG] Sampling indices and loading frames at index {index}")
            ref_img_idx = random.randint(0, video_length - 1)
            if ref_img_idx + margin < video_length:
                tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
            elif ref_img_idx - margin > 0:
                tgt_img_idx = random.randint(0, ref_img_idx - margin)
            else:
                tgt_img_idx = random.randint(0, video_length - 1)

            if video_reader is not None:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                ref_img = video_reader[ref_img_idx]
                ref_img_pil = Image.fromarray(ref_img.asnumpy())
                tgt_img = video_reader[tgt_img_idx]
                tgt_img_pil = Image.fromarray(tgt_img.asnumpy())
                signal.alarm(0)
            else:
                ref_img_pil = Image.new('RGB', self.img_size, (0, 0, 0))
                tgt_img_pil = Image.new('RGB', self.img_size, (0, 0, 0))

            if kps_reader is not None:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                tgt_pose_teacher = kps_reader[tgt_img_idx]
                tgt_pose_teacher_pil = Image.fromarray(tgt_pose_teacher.asnumpy())
                signal.alarm(0)
            else:
                raise RuntimeError("No valid kps_reader for pose teacher")

            tgt_pose_student_img = None
            if kps_custom_reader:
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(10)
                    tgt_pose_student = kps_custom_reader[tgt_img_idx]
                    tgt_pose_student_pil = Image.fromarray(tgt_pose_student.asnumpy())
                    tgt_pose_student_img = self.augmentation(
                        tgt_pose_student_pil, self.cond_transform, torch.get_rng_state()
                    )
                    signal.alarm(0)
                except Exception as e:
                    print(f"[ERROR] Skipping custom pose for idx {index}: {e}")
                    traceback.print_exc(file=sys.stdout)
                    tgt_pose_student_img = None

            state = torch.get_rng_state()
            tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
            tgt_pose_teacher_img = self.augmentation(
                tgt_pose_teacher_pil, self.cond_transform, state
            )
            ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
            clip_image = self.clip_image_processor(
                images=ref_img_pil, return_tensors="pt"
            ).pixel_values[0]

            sample = dict(
                video_dir=video_path,
                img=tgt_img,
                tgt_pose_teacher=tgt_pose_teacher_img,
                ref_img=ref_img_vae,
                clip_images=clip_image,
            )
            if tgt_pose_student_img is not None:
                sample["tgt_pose_student"] = tgt_pose_student_img

            print(f"[DIAG] Successfully loaded sample at index {index} in {time.time() - start_time:.2f}s")
            return sample
        except Exception as e:
            print(f"[ERROR] Skipping index {index} due to frame read or transform: {e}")
            traceback.print_exc(file=sys.stdout)
            if retry_count < MAX_RETRIES:
                return self.__getitem__((index + 1) % len(self.vid_meta), retry_count=retry_count+1)
            else:
                print(f"[FATAL] Max retries ({MAX_RETRIES}) exceeded at index {index}. Raising error.")
                raise

    def __len__(self):
        return len(self.vid_meta)
# ---- End Inlined HumanDanceDataset ----
from musepose.models.pose_guider import PoseGuider
from musepose.models.unet_2d_condition import UNet2DConditionModel
from musepose.models.unet_3d import UNet3DConditionModel
from musepose.pipelines.pipeline_pose2img import Pose2ImagePipeline
from musepose.utils.util import delete_additional_ckpt, import_filename, seed_everything

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr



def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    if accelerator.is_local_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter(log_dir=f"{cfg.output_dir}/{cfg.exp_name}/loss")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path, local_files_only=True).to(
        "cuda", dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
        local_files_only=True
    ).to(dtype=weight_dtype, device="cuda")

    if cfg.pose_guider_pretrain:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda")
        # load pretrained controlnet-openpose params for pose_guider
        controlnet_openpose_state_dict = torch.load(cfg.controlnet_openpose_path)
        state_dict_to_load = {}
        for k in controlnet_openpose_state_dict.keys():
            if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
                new_k = k.replace("controlnet_cond_embedding.", "")
                state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
        miss, _ = pose_guider.load_state_dict(state_dict_to_load, strict=False)
        logger.info(f"Missing key for pose guider: {len(miss)}")
    else:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
        ).to(device="cuda")

    # load pretrained weights
    if cfg.load_pth:
        denoising_unet.load_state_dict(
            torch.load(config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
        )
        pose_guider.load_state_dict(
            torch.load(config.pose_guider_path, map_location="cpu"),
        )



    # Freeze and train
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    denoising_unet.requires_grad_(True)
    reference_unet.requires_grad_(True)
    pose_guider.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = HumanDanceDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        img_scale=(0.9, 1.0),
        data_meta_paths=cfg.data.meta_paths,
        sample_margin=cfg.data.sample_margin,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=cfg.data.train_bs, drop_last=True
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values = batch["img"].to(weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                tgt_pose_img = batch["tgt_pose"]
                tgt_pose_img = tgt_pose_img.unsqueeze(2)  # (bs, 3, 1, 512, 512)

                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                for batch_idx, (ref_img, clip_img) in enumerate(
                    zip(
                        batch["ref_img"],
                        batch["clip_images"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    tgt_pose_img,
                    uncond_fwd,
                )

                # Clear reference controls after each forward to prevent graph reuse issues
                reference_control_reader.clear()
                reference_control_writer.clear()

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    tensorboard_writer.add_scalar( 'train_loss' , train_loss, global_step)
                train_loss = 0.0


            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # save model after save_model_epoch_interval
        if (
            epoch + 1
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=20,
            )
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=20,
            )
            save_checkpoint(
                unwrap_net.pose_guider,
                save_dir,
                "pose_guider",
                global_step,
                total_limit=20,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_stage_1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    

    main(config)
