import gc
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from functools import partial
import types

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm
from safetensors.torch import load_file

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanI2V:
    """Optimized WanI2V model for faster image-to-video generation"""
    
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        """
        Initializes the image-to-video generation model components with optimized setup.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        # Configure model loading with optimized setup
        shard_fn = partial(shard_model, device_id=device_id)
        
        # Initialize T5 encoder with optimal placement
        t5_device = torch.device('cpu') if t5_cpu else self.device
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=t5_device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        # Initialize configuration parameters
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        
        # Initialize VAE with device placement
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        # Initialize CLIP model
        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        # Load and configure diffusion model
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        # Handle distributed training setup
        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        # Configure USP (unified sequence parallel) if enabled
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import (usp_attn_forward, usp_dit_forward)
            
            # Apply USP optimization
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        # Handle model distribution
        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        # Set default negative prompt
        self.sample_neg_prompt = config.sample_neg_prompt
        
        # Pre-allocate buffers for generation
        self._init_generation_buffers()
    
    def _init_generation_buffers(self):
        """Pre-allocate buffers for faster generation"""
        # These will be resized as needed during generation
        self._noise_buffer = None
        self._latent_buffer = None
        self._mask_buffer = None
        
    def _prepare_generation_buffers(self, frame_num, lat_h, lat_w):
        """Prepare buffers with correct dimensions for current generation task"""
        device = self.device
        
        # Allocate or resize noise buffer
        if (self._noise_buffer is None or 
            self._noise_buffer.shape != (16, int((frame_num - 1)/4 + 1), lat_h, lat_w)):
            self._noise_buffer = torch.empty(
                16, 21, lat_h, lat_w, 
                dtype=torch.float32, 
                device=device)
            
        # Allocate or resize mask buffer
        if (self._mask_buffer is None or
            self._mask_buffer.shape != (1, frame_num, lat_h, lat_w)):
            self._mask_buffer = torch.ones(
                1, frame_num, lat_h, lat_w, 
                device=device)
            # First frame is 1, rest are 0
            self._mask_buffer[:, 1:] = 0

    # def apply_lora(self, lora_path, alpha=1.0, target_replace_module=None):
    #     """
    #     Apply LoRA weights to the model.

    #     Args:
    #         lora_path (str): Path to the LoRA weights in safetensors format
    #         alpha (float): Scaling factor for LoRA weights
    #         target_replace_module (dict, optional): Dictionary mapping module names to replace with LoRA
    #     """
    #     if not os.path.exists(lora_path):
    #         logging.warning(f"LoRA file not found: {lora_path}")
    #         return

    #     logging.info(f"Loading LoRA weights from {lora_path}")
    #     lora_state_dict = load_file(lora_path)

    #     # Remove "diffusion_model." prefix from keys
    #     clean_state_dict = {}
    #     for key in lora_state_dict:
    #         if key.startswith("diffusion_model."):
    #             clean_key = key[len("diffusion_model."):]
    #             clean_state_dict[clean_key] = lora_state_dict[key]
    #         else:
    #             clean_state_dict[key] = lora_state_dict[key]

    #     # Apply LoRA weights
    #     torch_dtype = self.model.device.type
    #     self.model.to(self.device)  # Ensure model is on the correct device

    #     # Create sets to track LoRA modules we've processed
    #     processed_modules = set()

    #     for name, value in self.model.named_parameters():
    #         # Check if this is a base weight with a corresponding LoRA weight
    #         if ".weight" in name:
    #             lora_a_name = name.replace(".weight", ".lora_A.weight")
    #             lora_b_name = name.replace(".weight", ".lora_B.weight")

    #             if lora_a_name in clean_state_dict and lora_b_name in clean_state_dict:
    #                 module_name = name[:name.rfind(".weight")]

    #                 # Skip if we've already processed this module
    #                 if module_name in processed_modules:
    #                     continue

    #                 # Skip if there's a target module filter and this module doesn't match
    #                 if target_replace_module is not None and not any(
    #                     x in module_name for x in target_replace_module
    #                 ):
    #                     continue

    #                 # Get the base weight and LoRA weights
    #                 lora_a = clean_state_dict[lora_a_name].to(value.device, value.dtype)
    #                 lora_b = clean_state_dict[lora_b_name].to(value.device, value.dtype)

    #                 # Apply LoRA: W = W + alpha * (A · B)
    #                 delta = torch.mm(lora_b, lora_a) * alpha
    #                 value.data += delta

    #                 processed_modules.add(module_name)
    #                 logging.info(f"Applied LoRA to {module_name}")

    #     torch.cuda.empty_cache()
    #     logging.info(f"Applied {len(processed_modules)} LoRA modules to model")

    def apply_lora(self, lora_path, alpha=1.0, target_replace_module=None):
        """
        Apply LoRA weights to the model
        """
        import time
        from contextlib import nullcontext

        start_time = time.time()

        if not os.path.exists(lora_path):
            logging.warning(f"LoRA file not found: {lora_path}")
            return 0

        # Load LoRA state dict with minimal logging
        logging.info(f"Loading LoRA weights from {lora_path}")
        lora_state_dict = load_file(lora_path)

        # Pre-process state dict to remove prefixes - do this once upfront
        clean_state_dict = {}
        for key, tensor in lora_state_dict.items():
            if key.startswith("diffusion_model."):
                key = key[len("diffusion_model."):]
            clean_state_dict[key] = tensor

        # Move model to target device once
        self.model.to(self.device)

        # Get parameter dictionary for O(1) lookups
        param_dict = dict(self.model.named_parameters())

        # Identify all valid LoRA pairs upfront
        lora_pairs = {}
        for key in clean_state_dict:
            if ".lora_A.weight" in key:
                base_name = key.replace(".lora_A.weight", "")
                lora_b_key = f"{base_name}.lora_B.weight"
                weight_name = f"{base_name}.weight"

                # Skip if not a valid LoRA pair
                if lora_b_key not in clean_state_dict or weight_name not in param_dict:
                    continue

                # Apply target module filter if provided
                if target_replace_module is not None and not any(x in base_name for x in target_replace_module):
                    continue

                # Store for processing
                weight = param_dict[weight_name]
                lora_pairs[base_name] = (
                    clean_state_dict[key],
                    clean_state_dict[lora_b_key],
                    weight
                )

        # Set up CUDA streams for parallel processing if available
        streams = {}
        if torch.cuda.is_available():
            num_streams = min(4, len(lora_pairs))
            for i in range(num_streams):
                streams[i] = torch.cuda.Stream()

        # Process LoRA pairs efficiently with streams
        processed_count = 0
        for i, (base_name, (lora_a, lora_b, weight)) in enumerate(lora_pairs.items()):
            # Determine which stream to use for parallel execution
            stream_idx = i % len(streams) if streams else None
            stream = streams.get(stream_idx) if stream_idx is not None else None

            # Use appropriate context manager
            ctx = torch.cuda.stream(stream) if stream else nullcontext()

            with ctx:
                # Move tensors to device efficiently with non-blocking transfers
                lora_a = lora_a.to(weight.device, weight.dtype, non_blocking=True)
                lora_b = lora_b.to(weight.device, weight.dtype, non_blocking=True)

                # Apply LoRA update in-place with optimized computation
                with torch.no_grad():  # Prevent unnecessary gradient tracking
                    delta = torch.mm(lora_b, lora_a) * alpha

                    weight.data.add_(delta)  # In-place addition

                processed_count += 1

        # Synchronize all streams when done
        if streams:
            torch.cuda.synchronize()

        # Clean up GPU memory only if needed
        if processed_count > 0:
            torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time
        logging.info(f"Applied {processed_count} LoRA modules in {elapsed_time:.2f}s")

        return processed_count

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        """
        Optimized video frame generation from input image and text prompt.
        """
        # Convert input image to tensor and normalize
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        # Calculate dimensions 
        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        
        # Calculate latent dimensions based on max_area constraint
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        
        # Calculate output dimensions
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        # Calculate sequence length for transformer
        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # Set up random seed for reproducibility
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        # Prepare generation buffers
        self._prepare_generation_buffers(frame_num, lat_h, lat_w)
        
        # Generate initial noise
        torch.randn(
            16, int((frame_num - 1)/4 + 1), lat_h, lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
            out=self._noise_buffer
        )
        noise = self._noise_buffer  # Reference, not copy
        
        # Prepare mask for conditioning
        msk = self._mask_buffer  # Reference to buffer
        msk = torch.cat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # Use default negative prompt if not provided
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Process text prompts with T5 encoder
        with torch.no_grad():
            if not self.t5_cpu:
                # Move T5 to device if not kept on CPU
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                # Process on CPU and move results to GPU
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]

            # Process image with CLIP
            self.clip.model.to(self.device)
            clip_context = self.clip.visual([img[:, None, :, :]])
            if offload_model:
                self.clip.model.cpu()

            # Encode image with VAE
            img_resized = torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1)
            
            # Create input for VAE
            vae_input = torch.cat([
                img_resized,
                torch.zeros(3, frame_num - 1, h, w, device=img_resized.device)
            ], dim=1).to(self.device)
            
            # Encode with VAE
            y = self.vae.encode([vae_input])[0]
            y = torch.cat([msk, y])

        # Define empty context manager if no_sync not available
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # Set up diffusion sampler
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Configure sampler based on selected solver
            if sample_solver == 'unipc':
                # UniPC sampler initialization
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                # DPM++ sampler initialization
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # Initialize latent with noise
            latent = noise

            # Prepare model arguments for conditional generation
            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            # Prepare model arguments for unconditional generation
            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            # Clear cache if offloading models
            if offload_model:
                torch.cuda.empty_cache()

            # Move model to device for sampling
            self.model.to(self.device)
            
            # Pre-allocate buffer for noise predictions to reduce memory allocations
            noise_pred_buffer = torch.zeros_like(latent)
            
            # Run sampling loop with optimized memory management
            for step_idx, t in enumerate(tqdm(timesteps)):
                # Prepare input latent
                latent_model_input = [latent.to(self.device)]
                timestep = torch.tensor([t], device=self.device)
                
                # Get conditional noise prediction
                with amp.autocast(dtype=torch.bfloat16):
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                
                # Move to CPU if offloading
                if offload_model:
                    noise_pred_cond = noise_pred_cond.to('cpu')
                    torch.cuda.empty_cache()
                
                # Get unconditional noise prediction    
                with amp.autocast(dtype=torch.bfloat16):
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]
                
                # Move to appropriate device
                if offload_model:
                    noise_pred_uncond = noise_pred_uncond.to('cpu')
                    noise_pred_cond = noise_pred_cond.to(self.device)
                    torch.cuda.empty_cache()
                
                # Apply classifier-free guidance
                if offload_model:
                    noise_pred_uncond = noise_pred_uncond.to(self.device)
                
                # Calculate guided noise prediction
                torch.add(
                    noise_pred_uncond,
                    torch.mul(torch.sub(noise_pred_cond, noise_pred_uncond), guide_scale),
                    out=noise_pred_buffer
                )

                # Move latent to appropriate device
                if offload_model and latent.device != self.device:
                    latent = latent.to(self.device)

                # Apply denoising step
                with amp.autocast(dtype=torch.float32):
                    temp_x0 = sample_scheduler.step(
                        noise_pred_buffer.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                
                # Update latent for next step
                latent = temp_x0.squeeze(0)
                
                # Clear temporary variables
                del latent_model_input, timestep

            # Decode the final latent to get the generated video
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            # Only decode on rank 0 to save computation in distributed settings
            if self.rank == 0:
                with amp.autocast(dtype=torch.float32):
                    videos = self.vae.decode([latent.to(self.device)])

        # Clean up resources
        del noise, latent
        del sample_scheduler
        
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
            
        if dist.is_initialized():
            dist.barrier()

        # Return the generated video from rank 0 only
        return videos[0] if self.rank == 0 else None