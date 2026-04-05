"""
Image generation engine with full parameter control.
Supports SD 1.5, SDXL, FLUX with LoRA loading.
Optimized for 8GB VRAM with configurable offloading.
"""
import gc
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _load_with_offline_fallback(pipeline_cls, model_path: str, **kwargs):
    """Try loading a diffusers pipeline normally; on network error, retry with
    ``local_files_only=True`` so cached models work without internet."""
    try:
        return pipeline_cls.from_pretrained(model_path, **kwargs)
    except (OSError, ConnectionError, Exception) as first_err:
        err_str = str(first_err).lower()
        if any(tok in err_str for tok in ("connection", "resolve", "timeout",
                                           "offline", "404", "urlopen")):
            logger.warning(
                f"Online load failed ({first_err}); retrying from local cache..."
            )
            try:
                return pipeline_cls.from_pretrained(
                    model_path, local_files_only=True, **kwargs
                )
            except Exception:
                pass  # fall through to re-raise original
        raise


SAMPLERS = {
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "dpm++_2m": "DPMSolverMultistepScheduler",
    "dpm++_2m_karras": "DPMSolverMultistepScheduler",
    "dpm++_sde": "DPMSolverSDEScheduler",
    "dpm++_sde_karras": "DPMSolverSDEScheduler",
    "ddim": "DDIMScheduler",
    "ddpm": "DDPMScheduler",
    "uni_pc": "UniPCMultistepScheduler",
    "lms": "LMSDiscreteScheduler",
    "heun": "HeunDiscreteScheduler",
    "pndm": "PNDMScheduler",
}


class ImageGenerator:
    """Full-featured image generation with LoRA support and VRAM management."""

    def __init__(self, hardware_config=None):
        from core.config import ConfigManager
        cfg = ConfigManager()
        self.hardware = hardware_config or cfg.config.hardware
        self.pipe = None
        self.current_model = None
        self.current_lora = None
        self.model_type = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str, model_type: str = "sdxl",
                   on_progress: Callable = None):
        """Load a base model with memory optimizations.
        Auto-downloads from Hugging Face if the model is not available locally."""
        from core.model_downloader import ensure_model_available, resolve_model_path

        resolved = resolve_model_path(model_path, model_type)
        if self.current_model == resolved:
            return

        self.unload()

        if on_progress:
            on_progress("Checking model availability...")

        try:
            # Auto-download if needed
            resolved = ensure_model_available(
                model_path, model_type,
                on_status=lambda msg: on_progress(msg) if on_progress else None,
            )

            if on_progress:
                on_progress("Loading model into memory...")

            if model_type == "sd15":
                self._load_sd15(resolved)
            elif model_type == "sdxl":
                self._load_sdxl(resolved)
            elif model_type == "flux":
                self._load_flux(resolved)
            else:
                self._load_sdxl(resolved)

            self.current_model = resolved
            self.model_type = model_type
            self._apply_optimizations()

            logger.info(f"Model loaded: {resolved} ({model_type})")
            if on_progress:
                on_progress("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_sd15(self, model_path: str):
        from diffusers import StableDiffusionPipeline
        self.pipe = _load_with_offline_fallback(
            StableDiffusionPipeline, model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

    def _load_sdxl(self, model_path: str):
        from diffusers import StableDiffusionXLPipeline

        kwargs = {
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "use_safetensors": True,
        }

        # Check if it's a local safetensors file
        if model_path.endswith(".safetensors"):
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, **kwargs
            )
        else:
            self.pipe = _load_with_offline_fallback(
                StableDiffusionXLPipeline, model_path, **kwargs
            )

    def _load_flux(self, model_path: str):
        from diffusers import FluxPipeline

        self.pipe = _load_with_offline_fallback(
            FluxPipeline, model_path,
            torch_dtype=torch.bfloat16,
        )

    def _apply_optimizations(self):
        """Apply VRAM optimizations based on hardware config."""
        if self.pipe is None:
            return

        hw = self.hardware

        # Offloading
        if hw.offload_mode == "aggressive" or hw.sequential_offload:
            self.pipe.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offload")
        elif hw.cpu_offload:
            self.pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")
        else:
            self.pipe = self.pipe.to(self._device)

        # Memory optimizations
        if hw.attention_slicing:
            self.pipe.enable_attention_slicing(1)
        if hw.vae_slicing:
            self.pipe.enable_vae_slicing()
        if hw.vae_tiling:
            self.pipe.enable_vae_tiling()
        if hw.xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers enabled")
            except Exception:
                logger.info("xformers not available, using default attention")

    def load_lora(self, lora_path: str, weight: float = 0.8):
        """Load a LoRA adapter onto the current model."""
        if self.pipe is None:
            raise RuntimeError("No base model loaded")

        try:
            lora_dir = Path(lora_path)

            if lora_dir.is_dir():
                # PEFT format
                self.pipe.load_lora_weights(str(lora_dir))
            elif lora_path.endswith(".safetensors"):
                # Single file
                self.pipe.load_lora_weights(
                    str(lora_dir.parent),
                    weight_name=lora_dir.name,
                )
            else:
                self.pipe.load_lora_weights(lora_path)

            # Set LoRA scale
            self.pipe.fuse_lora(lora_scale=weight)
            self.current_lora = lora_path
            logger.info(f"LoRA loaded: {lora_path} (weight={weight})")

        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            raise

    def unload_lora(self):
        """Remove current LoRA."""
        if self.pipe and self.current_lora:
            try:
                self.pipe.unfuse_lora()
                self.pipe.unload_lora_weights()
                self.current_lora = None
                logger.info("LoRA unloaded")
            except Exception as e:
                logger.warning(f"Error unloading LoRA: {e}")

    def set_scheduler(self, sampler_name: str):
        """Change the sampling scheduler."""
        if self.pipe is None:
            return

        from diffusers import (
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSDEScheduler,
            DDIMScheduler,
            DDPMScheduler,
            UniPCMultistepScheduler,
            LMSDiscreteScheduler,
            HeunDiscreteScheduler,
            PNDMScheduler,
        )

        scheduler_map = {
            "euler": EulerDiscreteScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "dpm++_2m": DPMSolverMultistepScheduler,
            "dpm++_2m_karras": lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True),
            "dpm++_sde": DPMSolverSDEScheduler,
            "dpm++_sde_karras": lambda cfg: DPMSolverSDEScheduler.from_config(cfg, use_karras_sigmas=True),
            "ddim": DDIMScheduler,
            "ddpm": DDPMScheduler,
            "uni_pc": UniPCMultistepScheduler,
            "lms": LMSDiscreteScheduler,
            "heun": HeunDiscreteScheduler,
            "pndm": PNDMScheduler,
        }

        sched_cls = scheduler_map.get(sampler_name)
        if sched_cls:
            if callable(sched_cls) and not isinstance(sched_cls, type):
                self.pipe.scheduler = sched_cls(self.pipe.scheduler.config)
            else:
                self.pipe.scheduler = sched_cls.from_config(self.pipe.scheduler.config)
            logger.info(f"Scheduler set to: {sampler_name}")

    def generate(self, prompt: str, negative_prompt: str = "",
                 width: int = 768, height: int = 768,
                 steps: int = 30, cfg_scale: float = 7.5,
                 seed: int = -1, batch_size: int = 1,
                 sampler: str = "euler_a", clip_skip: int = 2,
                 hires_fix: bool = False, hires_scale: float = 1.5,
                 hires_steps: int = 15, hires_denoising: float = 0.55,
                 callback: Callable = None) -> List[Image.Image]:
        """Generate images with full parameter control."""
        if self.pipe is None:
            raise RuntimeError("No model loaded")

        # Set scheduler
        self.set_scheduler(sampler)

        # Handle seed
        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())

        generator = torch.Generator(device=self._device).manual_seed(seed)

        # Round dimensions to 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info(f"Generating: {width}x{height}, steps={steps}, cfg={cfg_scale}, seed={seed}")

        try:
            # Build generation kwargs
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "generator": generator,
                "num_images_per_prompt": batch_size,
            }

            # FLUX doesn't use negative prompt or cfg in the same way
            if self.model_type == "flux":
                gen_kwargs.pop("negative_prompt", None)
                gen_kwargs.pop("guidance_scale", None)

            # Clip skip for SD models
            if self.model_type in ("sd15", "sdxl") and clip_skip > 1:
                gen_kwargs["clip_skip"] = clip_skip

            # Progress callback
            if callback:
                def step_callback(pipe, step_index, timestep, callback_kwargs):
                    callback(step_index, steps)
                    return callback_kwargs
                gen_kwargs["callback_on_step_end"] = step_callback

            # Generate
            output = self.pipe(**gen_kwargs)
            images = output.images

            # Hi-res fix (upscale + img2img)
            if hires_fix and self.model_type in ("sd15", "sdxl"):
                images = self._apply_hires_fix(
                    images, prompt, negative_prompt,
                    hires_scale, hires_steps, hires_denoising,
                    generator
                )

            return images

        except torch.cuda.OutOfMemoryError:
            logger.error("VRAM OOM during generation — attempting recovery...")
            gc.collect()
            torch.cuda.empty_cache()

            # Auto-retry at 75% resolution if there's room to shrink
            reduced_w = ((int(width * 0.75)) // 8) * 8
            reduced_h = ((int(height * 0.75)) // 8) * 8
            if reduced_w >= 256 and reduced_h >= 256 and (reduced_w < width or reduced_h < height):
                logger.info(
                    f"Retrying at reduced resolution: {reduced_w}x{reduced_h} "
                    f"(was {width}x{height})"
                )
                try:
                    gen_kwargs["width"] = reduced_w
                    gen_kwargs["height"] = reduced_h
                    output = self.pipe(**gen_kwargs)
                    return output.images
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()

            raise RuntimeError(
                f"Out of VRAM at {width}x{height} (and {reduced_w}x{reduced_h}). Try:\n"
                f"- Reducing resolution further\n"
                f"- Enabling more aggressive offloading\n"
                f"- Reducing batch size\n"
                f"- Closing other GPU applications"
            )

    def _apply_hires_fix(self, images, prompt, negative_prompt,
                         scale, steps, denoising, generator):
        """Apply hi-res fix: upscale then img2img."""
        try:
            from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline

            upscaled = []
            for img in images:
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8

                # Lanczos upscale
                img_up = img.resize((new_w, new_h), Image.LANCZOS)

                # img2img pass
                if self.model_type == "sdxl":
                    i2i_pipe = StableDiffusionXLImg2ImgPipeline(
                        **self.pipe.components
                    )
                else:
                    i2i_pipe = StableDiffusionImg2ImgPipeline(
                        **self.pipe.components
                    )

                result = i2i_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=img_up,
                    strength=denoising,
                    num_inference_steps=steps,
                    generator=generator,
                ).images[0]

                upscaled.append(result)
                del i2i_pipe

            return upscaled

        except Exception as e:
            logger.warning(f"Hi-res fix failed: {e}, returning original images")
            return images

    def save_images(self, images: List[Image.Image], output_dir: str,
                    prefix: str = "gen", format: str = "png",
                    metadata: Dict = None) -> List[str]:
        """Save generated images with metadata."""
        saved_paths = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, img in enumerate(images):
            filename = f"{prefix}_{timestamp}_{i:03d}.{format}"
            filepath = os.path.join(output_dir, filename)

            # Save with metadata in PNG
            if format == "png" and metadata:
                from PIL.PngImagePlugin import PngInfo
                pnginfo = PngInfo()
                for k, v in metadata.items():
                    pnginfo.add_text(k, str(v))
                img.save(filepath, pnginfo=pnginfo)
            else:
                save_kwargs = {}
                if format in ("jpg", "jpeg"):
                    save_kwargs["quality"] = 95
                elif format == "webp":
                    save_kwargs["quality"] = 95
                img.save(filepath, **save_kwargs)

            saved_paths.append(filepath)
            logger.info(f"Saved: {filepath}")

        return saved_paths

    def unload(self):
        """Free all GPU memory using the hardened cleanup sequence."""
        from core.gpu_utils import deep_cleanup_pipeline
        if self.pipe is not None:
            deep_cleanup_pipeline(self.pipe, label=f"ImageGen({self.current_model})")
            self.pipe = None
        self.current_model = None
        self.current_lora = None

    def get_vram_usage(self) -> Dict:
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1),
        }
