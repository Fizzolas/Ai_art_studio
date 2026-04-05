"""
Video generation engine.
Supports WAN2.1, AnimateDiff, and other video diffusion models.
Handles VRAM constraints with aggressive offloading.
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
                pass
        raise


class VideoGenerator:
    """Video generation with full parameter control and VRAM management."""

    def __init__(self, hardware_config=None):
        from core.config import ConfigManager
        cfg = ConfigManager()
        self.hardware = hardware_config or cfg.config.hardware
        self.pipe = None
        self.current_model = None
        self.model_type = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str, model_type: str = "wan21",
                   on_progress: Callable = None):
        """Load a video generation model.
        Auto-downloads from Hugging Face if not available locally."""
        from core.model_downloader import ensure_model_available

        self.unload()

        if on_progress:
            on_progress("Checking video model availability...")

        try:
            # Auto-download if needed
            resolved = ensure_model_available(
                model_path, model_type,
                on_status=lambda msg: on_progress(msg) if on_progress else None,
            )

            if on_progress:
                on_progress("Loading video model into memory...")

            if model_type == "wan21":
                self._load_wan21(resolved)
            elif model_type == "animatediff":
                self._load_animatediff(resolved)
            else:
                self._load_wan21(resolved)

            self.current_model = resolved
            self.model_type = model_type
            self._apply_optimizations()

            logger.info(f"Video model loaded: {resolved}")
            if on_progress:
                on_progress("Video model loaded")

        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
            raise

    def _load_wan21(self, model_path: str):
        """Load WAN 2.1 video generation model."""
        try:
            from diffusers import WanPipeline

            self.pipe = WanPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )
        except ImportError:
            logger.warning("WAN pipeline not available in this diffusers version")
            raise ImportError(
                "WAN 2.1 requires diffusers >= 0.31. "
                "Install with: pip install diffusers>=0.31"
            )

    def _load_animatediff(self, model_path: str):
        """Load AnimateDiff pipeline (SD 1.5 based video)."""
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-3",
            torch_dtype=torch.float16,
        )

        self.pipe = AnimateDiffPipeline.from_pretrained(
            model_path,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

    def _apply_optimizations(self):
        """Apply VRAM optimizations for video generation."""
        if self.pipe is None:
            return

        hw = self.hardware

        # Video generation is very VRAM hungry - always use offloading
        if hw.offload_mode in ("aggressive", "cpu_only") or hw.sequential_offload:
            self.pipe.enable_sequential_cpu_offload()
            logger.info("Video: sequential CPU offload enabled")
        else:
            self.pipe.enable_model_cpu_offload()
            logger.info("Video: model CPU offload enabled")

        # VAE optimizations are critical for video
        if hw.vae_slicing:
            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass
        if hw.vae_tiling:
            try:
                self.pipe.enable_vae_tiling()
            except Exception:
                pass

    def generate(self, prompt: str, negative_prompt: str = "",
                 width: int = 512, height: int = 512,
                 num_frames: int = 49, fps: int = 16,
                 steps: int = 30, cfg_scale: float = 6.0,
                 seed: int = -1, flow_shift: float = 3.0,
                 callback: Callable = None) -> List[Any]:
        """Generate a video clip."""
        if self.pipe is None:
            raise RuntimeError("No video model loaded")

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Round to 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Ensure frame count is valid (WAN uses 4n+1 frames)
        if self.model_type == "wan21":
            num_frames = ((num_frames - 1) // 4) * 4 + 1

        logger.info(f"Generating video: {width}x{height}, {num_frames} frames, {steps} steps")

        try:
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "generator": generator,
            }

            if callback:
                def step_cb(pipe, step_index, timestep, cb_kwargs):
                    callback(step_index, steps)
                    return cb_kwargs
                gen_kwargs["callback_on_step_end"] = step_cb

            output = self.pipe(**gen_kwargs)
            frames = output.frames[0] if hasattr(output, "frames") else output.images

            return frames

        except torch.cuda.OutOfMemoryError:
            logger.error("VRAM OOM during video generation!")
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Out of VRAM for video generation. Try:\n"
                "- Reducing resolution (try 480x320)\n"
                "- Reducing frame count\n"
                "- Enabling aggressive offloading\n"
                "- Closing other GPU applications"
            )

    def save_video(self, frames, output_dir: str, fps: int = 16,
                   format: str = "mp4", prefix: str = "vid",
                   metadata: Dict = None) -> str:
        """Save generated frames as video."""
        from diffusers.utils import export_to_video, export_to_gif

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.{format}"
        filepath = os.path.join(output_dir, filename)

        if format == "gif":
            export_to_gif(frames, filepath)
        else:
            export_to_video(frames, filepath, fps=fps)

        logger.info(f"Video saved: {filepath}")
        return filepath

    def unload(self):
        """Free all GPU memory using the hardened cleanup sequence."""
        from core.gpu_utils import deep_cleanup_pipeline
        if self.pipe is not None:
            deep_cleanup_pipeline(self.pipe, label=f"VideoGen({self.current_model})")
            self.pipe = None
        self.current_model = None
