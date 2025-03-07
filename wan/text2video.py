# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

class WanT2V:

    def __init__(self, config, checkpoint_dir, device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir).to(self.device)
        self.model.eval().requires_grad_(False)
        # self.model = torch.compile(
        #     self.model,
        #     backend="inductor",
        #     mode="default",
        # )

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import (usp_attn_forward, usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self, input_prompt, size=(1280, 720), frame_num=81, shift=5.0, sample_solver='unipc', sampling_steps=50, guide_scale=5.0, n_prompt="", seed=-1, offload_model=True):
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1, size[1] // self.vae_stride[1], size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.patch_size[1] * self.patch_size[2]) * target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Only move to device once and avoid switching back and forth
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)

        noise = [torch.randn(target_shape[0], target_shape[1], target_shape[2], target_shape[3], dtype=torch.float32, device=self.device, generator=seed_g)]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            timestep_tensor = torch.zeros(1, dtype=torch.long, device=self.device)  # Pre-allocate a tensor for timestep

            for t in tqdm(timesteps):
                timestep_tensor[0] = t  # Reuse pre-allocated tensor
                latent_model_input = latents

                # Model inference
                self.model.to(self.device)

                noise_pred_cond = self.model(latent_model_input, t=timestep_tensor, **arg_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep_tensor, **arg_null)[0]

                # Noise prediction and sampling
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                # In-place operation to reduce memory usage
                temp_x0 = sample_scheduler.step(noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g)[0]
                latents[0].copy_(temp_x0.squeeze(0))  # In-place copy

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        return videos[0] if self.rank == 0 else None
