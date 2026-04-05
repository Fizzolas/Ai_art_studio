"""
Image generation engine with full parameter control.
Supports SD 1.5, SDXL, FLUX with LoRA loading.
Optimized for 8GB VRAM with configurable offloading.
"""
import gc
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

import torch
import numpy as np
from PIL import Image

from core.logger import get_logger
logger = get_logger(__name__)

from generation.utils import _load_with_offline_fallback  # noqa: E402
from core.gpu_utils import flush_gpu_memory  # noqa: E402


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


FALLBACK_STEPS = [0.75, 0.50, None]  # None = absolute minimum 256x256


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
        self._oom_callback = None
        self._controlnet_pipe = None
        self._ip_adapter_repo = "h94/IP-Adapter"
        self._ip_adapter_subfolder = "models"
        self._ip_adapter_weight_name = "ip-adapter_sd15.bin"

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
        self._apply_optimizations_to(self.pipe)

    def _apply_optimizations_to(self, pipe):
        """Apply VRAM optimizations to any pipeline."""
        if pipe is None:
            return

        hw = self.hardware

        # Offloading
        if hw.offload_mode == "aggressive" or hw.sequential_offload:
            pipe.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offload")
        elif hw.cpu_offload:
            pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")
        else:
            pipe = pipe.to(self._device)

        # Memory optimizations
        if hw.attention_slicing:
            pipe.enable_attention_slicing(1)
        if hw.vae_slicing:
            pipe.enable_vae_slicing()
        if hw.vae_tiling:
            pipe.enable_vae_tiling()
        if hw.xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
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
            flush_gpu_memory()

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
                flush_gpu_memory()
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
                 callback: Callable = None,
                 init_image=None, strength: float = 0.75,
                 controlnet_enabled: bool = False,
                 controlnet_model_id: str = "",
                 controlnet_preprocessor: str = "canny",
                 controlnet_input_image: str = "",
                 controlnet_strength: float = 1.0,
                 controlnet_guidance_start: float = 0.0,
                 controlnet_guidance_end: float = 1.0,
                 ip_adapter_enabled: bool = False,
                 ip_adapter_image: str = "",
                 ip_adapter_scale: float = 0.6) -> List[Image.Image]:
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

        # Determine which pipeline to use
        active_pipe = self.pipe

        # ControlNet setup
        if controlnet_enabled and controlnet_input_image and controlnet_model_id:
            if self._controlnet_pipe is None or getattr(self, '_cn_model_id', '') != controlnet_model_id:
                self._load_controlnet_pipeline(controlnet_model_id)
                self._cn_model_id = controlnet_model_id
            if self._controlnet_pipe is not None:
                active_pipe = self._controlnet_pipe

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

            # img2img mode
            if init_image is not None:
                if isinstance(init_image, str):
                    init_image = Image.open(init_image).convert("RGB")
                init_image = init_image.resize((width, height), Image.LANCZOS)
                gen_kwargs["image"] = init_image
                gen_kwargs["strength"] = strength
                gen_kwargs.pop("width", None)
                gen_kwargs.pop("height", None)

            # ControlNet conditioning
            if controlnet_enabled and controlnet_input_image and self._controlnet_pipe is not None:
                ctrl_img = self._preprocess_controlnet_image(
                    controlnet_input_image, controlnet_preprocessor, width, height)
                gen_kwargs["image"] = ctrl_img
                gen_kwargs["controlnet_conditioning_scale"] = controlnet_strength
                gen_kwargs["control_guidance_start"] = controlnet_guidance_start
                gen_kwargs["control_guidance_end"] = controlnet_guidance_end

            # IP-Adapter style reference
            if ip_adapter_enabled and ip_adapter_image:
                ref_img = self._apply_ip_adapter(active_pipe, ip_adapter_image, ip_adapter_scale)
                if ref_img is not None:
                    gen_kwargs["ip_adapter_image"] = ref_img

            # Generate with stepped OOM fallback
            output = self._generate_with_fallback(gen_kwargs, width, height, pipeline=active_pipe)
            images = output.images

            # Hi-res fix (upscale + img2img)
            if hires_fix and self.model_type in ("sd15", "sdxl"):
                images = self._apply_hires_fix(
                    images, prompt, negative_prompt,
                    hires_scale, hires_steps, hires_denoising,
                    generator
                )

            return images

        except RuntimeError as e:
            if "VRAM" in str(e) or "OutOfMemory" in str(e):
                raise
            raise

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

    def _write_error_report(self, error: Exception, gen_kwargs: dict, tb: str):
        from core.logger import get_log_dir
        import json
        from datetime import datetime
        error_dir = get_log_dir() / "errors"
        error_dir.mkdir(exist_ok=True)
        report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": tb,
            "parameters": {k: str(v) for k, v in gen_kwargs.items()
                          if k not in ("pipeline", "model")},
        }
        fname = error_dir / f"gen_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fname.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ── ControlNet ──────────────────────────────────────────────────────

    def _load_controlnet_pipeline(self, controlnet_model_id: str):
        """Load a ControlNet-aware pipeline variant."""
        try:
            from diffusers import ControlNetModel
            cn_model = ControlNetModel.from_pretrained(
                controlnet_model_id, torch_dtype=torch.float16)
            if self.model_type == "sdxl":
                from diffusers import StableDiffusionXLControlNetPipeline
                self._controlnet_pipe = StableDiffusionXLControlNetPipeline(
                    **self.pipe.components, controlnet=cn_model)
            else:
                from diffusers import StableDiffusionControlNetPipeline
                self._controlnet_pipe = StableDiffusionControlNetPipeline(
                    **self.pipe.components, controlnet=cn_model)
            self._apply_optimizations_to(self._controlnet_pipe)
            logger.info(f"ControlNet pipeline loaded: {controlnet_model_id}")
        except Exception as e:
            logger.warning(f"ControlNet load failed: {e}")
            self._controlnet_pipe = None

    def _preprocess_controlnet_image(self, image_path: str, preprocessor: str,
                                      width: int, height: int):
        """Preprocess a control image based on the selected preprocessor."""
        img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
        if preprocessor == "canny":
            try:
                import cv2
                import numpy as np
                arr = np.array(img)
                edges = cv2.Canny(arr, 100, 200)
                return Image.fromarray(edges).convert("RGB")
            except ImportError:
                logger.warning("OpenCV not installed for canny preprocessing")
                return img
        elif preprocessor == "depth":
            try:
                from transformers import pipeline as hf_pipeline
                depth_pipe = hf_pipeline("depth-estimation")
                result = depth_pipe(img)
                return result["depth"].convert("RGB").resize((width, height))
            except Exception as e:
                logger.warning(f"Depth preprocessing failed: {e}")
                return img
        elif preprocessor == "openpose":
            try:
                from controlnet_aux import OpenposeDetector
                detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                return detector(img)
            except ImportError:
                logger.warning("Install controlnet-aux for OpenPose support")
                return img
            except Exception as e:
                logger.warning(f"OpenPose failed: {e}")
                return img
        return img  # "none" — pass as-is

    # ── IP-Adapter ────────────────────────────────────────────────────

    def _apply_ip_adapter(self, pipeline, image_path: str, scale: float):
        """Load IP-Adapter and return the reference image for gen_kwargs."""
        try:
            pipeline.load_ip_adapter(
                self._ip_adapter_repo,
                subfolder=self._ip_adapter_subfolder,
                weight_name=self._ip_adapter_weight_name,
            )
            pipeline.set_ip_adapter_scale(scale)
            ref_img = Image.open(image_path).convert("RGB")
            return ref_img
        except Exception as e:
            logger.warning(f"IP-Adapter load failed: {e}")
            return None

    # ── OOM Fallback ──────────────────────────────────────────────────

    def _generate_with_fallback(self, gen_kwargs: dict, original_w: int,
                                original_h: int, pipeline=None):
        """Try generation, stepping down resolution on OOM before giving up."""
        pipe = pipeline or self.pipe
        for step in FALLBACK_STEPS:
            try:
                return pipe(**gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if step is None:
                    gen_kwargs["width"] = 256
                    gen_kwargs["height"] = 256
                    logger.warning("OOM: falling back to 256x256 minimum resolution")
                else:
                    new_w = max(256, (int(original_w * step) // 8) * 8)
                    new_h = max(256, (int(original_h * step) // 8) * 8)
                    gen_kwargs["width"] = new_w
                    gen_kwargs["height"] = new_h
                    logger.warning(f"OOM: retrying at {new_w}x{new_h} ({int(step*100)}%)")
                    self._bump_offload_mode()
                try:
                    return pipe(**gen_kwargs)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
        raise RuntimeError("Generation failed: insufficient VRAM even at minimum resolution")

    def _bump_offload_mode(self):
        """Escalate offload mode one step and emit a notification."""
        try:
            from core.config import ConfigManager
            cfg = ConfigManager()
            order = ["none", "balanced", "aggressive", "cpu_only"]
            current = cfg.config.hardware.offload_mode
            idx = order.index(current) if current in order else 0
            if idx < len(order) - 1:
                new_mode = order[idx + 1]
                cfg.update_and_save("hardware", "offload_mode", new_mode)
                logger.warning(f"OOM: auto-bumped offload mode to '{new_mode}'")
                if self._oom_callback:
                    self._oom_callback(new_mode)
        except Exception as e:
            logger.warning(f"Failed to bump offload mode: {e}")

    def unload(self):
        """Free all GPU memory using the hardened cleanup sequence."""
        from core.gpu_utils import deep_cleanup_pipeline
        if self._controlnet_pipe is not None:
            deep_cleanup_pipeline(self._controlnet_pipe, label="ControlNetPipe")
            self._controlnet_pipe = None
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
