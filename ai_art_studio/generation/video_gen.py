"""
Video generation engine.
Supports WAN2.1, AnimateDiff, and other video diffusion models.
Handles VRAM constraints with aggressive offloading.
"""
import gc
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from core.logger import get_logger
logger = get_logger(__name__)


class VideoGenerator:
    """Video generation with full parameter control and VRAM management."""

    def __init__(self, hardware_config=None):
        import torch
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
        import torch
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
        import torch
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
        import torch
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

    def generate_long_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        frames_per_clip: int = 49,
        clip_count: int = 3,
        overlap_frames: int = 4,
        fps: int = 16,
        steps: int = 30,
        cfg_scale: float = 6.0,
        seed: int = -1,
        flow_shift: float = 3.0,
        output_dir: str = "",
        callback=None,
        caption_clips: bool = True,
        caption_sample_frames: int = 5,
    ) -> str:
        """Generate a long video by stitching multiple clips together.

        When caption_clips is True, each clip is captioned after generation and
        the description is fed into the next clip's prompt as continuation context.
        """
        import torch
        import gc
        if self.pipe is None:
            raise RuntimeError("No video model loaded")

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())

        if self.model_type == "wan21":
            frames_per_clip = ((frames_per_clip - 1) // 4) * 4 + 1

        all_clips = []
        prev_last_frame = None
        prev_clip_context = ""
        total_ops = clip_count * steps
        ops_done = 0

        for clip_idx in range(clip_count):
            logger.info(f"Generating clip {clip_idx + 1}/{clip_count} "
                        f"(seed={seed + clip_idx})")

            if callback:
                callback(ops_done, total_ops,
                        f"Clip {clip_idx + 1}/{clip_count}")

            # Build effective prompt with context from previous clip
            if clip_idx > 0 and prev_clip_context:
                effective_prompt = f"{prompt}, continuing from: {prev_clip_context}"
            else:
                effective_prompt = prompt

            clip_seed = seed + clip_idx
            generator = torch.Generator(device="cpu").manual_seed(clip_seed)

            gen_kwargs = {
                "prompt": effective_prompt,
                "negative_prompt": negative_prompt or None,
                "width": width,
                "height": height,
                "num_frames": frames_per_clip,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "generator": generator,
            }

            if prev_last_frame is not None:
                gen_kwargs = self._blend_last_frame(
                    gen_kwargs, prev_last_frame, clip_seed)

            def _step_cb(pipe, step_idx, timestep, cb_kwargs):
                nonlocal ops_done
                ops_done += 1
                if callback:
                    callback(ops_done, total_ops,
                            f"Clip {clip_idx+1}/{clip_count} — step {step_idx+1}/{steps}")
                return cb_kwargs

            if callback:
                gen_kwargs["callback_on_step_end"] = _step_cb

            try:
                output = self.pipe(**gen_kwargs)
                frames = output.frames[0] if hasattr(output, "frames") else output.images
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    f"OOM generating clip {clip_idx + 1}. "
                    "Try reducing frames_per_clip or resolution."
                )

            # Caption this clip's events for use as context in the next clip
            clip_context = ""
            if caption_clips:
                if callback:
                    callback(ops_done, total_ops,
                            f"Clip {clip_idx+1}/{clip_count} — captioning for context...")
                clip_context = self._caption_clip(frames, caption_sample_frames)
                if clip_context:
                    logger.info(f"Clip {clip_idx+1} context: {clip_context[:120]}...")

            all_clips.append(frames)
            prev_last_frame = frames[-1]
            prev_clip_context = clip_context

            torch.cuda.empty_cache()
            gc.collect()

        stitched = self._stitch_clips(all_clips, overlap_frames)

        output_path = self.save_video(
            stitched, output_dir, fps=fps,
            format="mp4", prefix="long_vid"
        )
        logger.info(f"Long video saved: {output_path} "
                    f"({len(stitched)} total frames from {clip_count} clips)")
        return output_path

    def _blend_last_frame(self, gen_kwargs: dict, last_frame, seed: int) -> dict:
        """Inject last frame as a soft conditioning hint for the next clip."""
        try:
            from PIL import Image
            import numpy as np

            if isinstance(last_frame, np.ndarray):
                frame_pil = Image.fromarray(last_frame)
            elif hasattr(last_frame, 'convert'):
                frame_pil = last_frame
            else:
                return gen_kwargs

            w = gen_kwargs.get("width", 512)
            h = gen_kwargs.get("height", 512)
            frame_pil = frame_pil.resize((w, h), Image.LANCZOS)

            if hasattr(self.pipe, "image_encoder") or hasattr(self.pipe, "vae"):
                gen_kwargs["image"] = frame_pil
                gen_kwargs["strength"] = 0.75
                logger.debug("Clip stitching: injecting last frame as img conditioning")
        except Exception as e:
            logger.debug(f"Frame blending skipped: {e}")

        return gen_kwargs

    def _stitch_clips(self, clips: list, overlap_frames: int) -> list:
        """Concatenate clips with linear crossfade at overlap regions."""
        if len(clips) == 1:
            return clips[0]

        import numpy as np
        from PIL import Image

        def to_array(frame):
            if isinstance(frame, np.ndarray):
                return frame
            return np.array(frame)

        def to_pil(arr):
            return Image.fromarray(arr.astype(np.uint8))

        result = list(clips[0])

        for clip in clips[1:]:
            if overlap_frames <= 0 or len(result) < overlap_frames or len(clip) < overlap_frames:
                result.extend(clip)
                continue

            tail = [to_array(f) for f in result[-overlap_frames:]]
            result = result[:-overlap_frames]
            head = [to_array(f) for f in clip[:overlap_frames]]
            rest = clip[overlap_frames:]

            for i, (a, b) in enumerate(zip(tail, head)):
                alpha = (i + 1) / (overlap_frames + 1)
                blended = (a * (1.0 - alpha) + b * alpha).clip(0, 255)
                result.append(to_pil(blended))

            result.extend(rest)

        return result

    # ── Clip context captioning ──────────────────────────────────────

    def _caption_clip(self, frames: list, sample_count: int = 5) -> str:
        """Caption the sequence of events across a full clip.

        Samples `sample_count` evenly-spaced frames, captions each with BLIP-2
        or Florence-2 (whichever is available), then summarises the sequence
        into a single natural-language description of what happens in the clip.

        Falls back gracefully:
        - If no captioning model is available -> returns ""
        - If only one frame captions successfully -> uses that alone
        - Never raises -- a failure here must not abort video generation
        """
        if not frames:
            return ""

        try:
            import numpy as np
            from PIL import Image as _PILImage

            # Sample frames evenly across the clip
            indices = [int(i * (len(frames) - 1) / max(sample_count - 1, 1))
                       for i in range(sample_count)]
            sampled = []
            for idx in indices:
                f = frames[min(idx, len(frames) - 1)]
                if isinstance(f, np.ndarray):
                    f = _PILImage.fromarray(f)
                sampled.append(f)

            # Caption each sampled frame
            frame_captions = []
            for img in sampled:
                cap = self._caption_single_frame(img)
                if cap:
                    frame_captions.append(cap)

            if not frame_captions:
                return ""

            # Summarise the sequence into a flowing description
            return self._summarise_sequence(frame_captions)

        except Exception as e:
            logger.warning(f"Clip captioning failed (non-fatal): {e}")
            return ""

    def _caption_single_frame(self, image) -> str:
        """Caption one frame using the best available model."""
        # Try BLIP-2 first (best quality)
        try:
            cap = self._caption_with_blip2(image)
            if cap and len(cap.strip()) > 5:
                return cap.strip()
        except Exception:
            pass

        # Try Florence-2
        try:
            cap = self._caption_with_florence(image)
            if cap and len(cap.strip()) > 5:
                return cap.strip()
        except Exception:
            pass

        # Try WD tagger (returns tags, not prose -- still useful)
        try:
            tags = self._caption_with_wd(image)
            if tags:
                return tags
        except Exception:
            pass

        return ""

    def _caption_with_blip2(self, image) -> str:
        """Caption a frame using BLIP-2 if loaded."""
        # Use the app's existing AutoCaptioner if available
        try:
            from captioning.auto_caption import AutoCaptioner
            from core.config import ConfigManager
            cfg = ConfigManager()
            if not hasattr(self, '_clip_captioner'):
                self._clip_captioner = AutoCaptioner(cfg.config.captioning)
            result = self._clip_captioner._caption_image_blip2(image)
            return result or ""
        except Exception:
            # Try a lightweight direct call
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                import torch
                if not hasattr(self, '_blip_proc'):
                    self._blip_proc = BlipProcessor.from_pretrained(
                        "Salesforce/blip-image-captioning-base")
                    self._blip_model = BlipForConditionalGeneration.from_pretrained(
                        "Salesforce/blip-image-captioning-base",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._blip_model = self._blip_model.to(device)
                device = next(self._blip_model.parameters()).device
                inputs = self._blip_proc(image, return_tensors="pt").to(device)
                with torch.inference_mode():
                    out = self._blip_model.generate(**inputs, max_new_tokens=60)
                return self._blip_proc.decode(out[0], skip_special_tokens=True)
            except Exception as e:
                logger.debug(f"BLIP captioning failed: {e}")
                return ""

    def _caption_with_florence(self, image) -> str:
        """Caption a frame using Florence-2 if loaded."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            if not hasattr(self, '_florence_proc'):
                model_id = "microsoft/Florence-2-base"
                self._florence_proc = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True)
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self._florence_model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=dtype, trust_remote_code=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._florence_model = self._florence_model.to(device)
            device = next(self._florence_model.parameters()).device
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self._florence_proc(
                text=prompt, images=image, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = self._florence_model.generate(
                    **inputs, max_new_tokens=80,
                    do_sample=False, num_beams=3)
            result = self._florence_proc.batch_decode(
                out, skip_special_tokens=False)[0]
            import re
            match = re.search(r'<MORE_DETAILED_CAPTION>(.*?)</MORE_DETAILED_CAPTION>',
                              result, re.DOTALL)
            if match:
                return match.group(1).strip()
            return result.strip()
        except Exception as e:
            logger.debug(f"Florence-2 captioning failed: {e}")
            return ""

    def _caption_with_wd(self, image) -> str:
        """Get WD tagger tags for a frame as a fallback caption."""
        try:
            from captioning.auto_caption import AutoCaptioner
            from core.config import ConfigManager
            cfg = ConfigManager()
            if not hasattr(self, '_clip_captioner'):
                self._clip_captioner = AutoCaptioner(cfg.config.captioning)
            tags = self._clip_captioner._get_wd_tags(image)
            return ", ".join(tags[:20]) if tags else ""
        except Exception:
            return ""

    def _summarise_sequence(self, frame_captions: list) -> str:
        """Convert a list of per-frame captions into a flowing sequence description."""
        if len(frame_captions) == 1:
            return frame_captions[0]

        # Strategy 1: try LLM summarisation (tiny model, fast)
        try:
            summary = self._llm_summarise(frame_captions)
            if summary and len(summary) > 10:
                return summary
        except Exception:
            pass

        # Strategy 2: deterministic template join
        return self._template_summarise(frame_captions)

    def _llm_summarise(self, captions: list) -> str:
        """Use a small text-generation model to summarise frame captions."""
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            if not hasattr(self, '_summariser'):
                self._summariser = hf_pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )

            numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(captions))
            prompt = (
                f"These are descriptions of frames from a video clip in order:\n"
                f"{numbered}\n\n"
                f"Write one fluent sentence describing the sequence of events "
                f"in this clip, from start to finish:"
            )
            result = self._summariser(prompt, max_new_tokens=80, do_sample=False)
            return result[0]["generated_text"].strip()
        except Exception as e:
            logger.debug(f"LLM summarise failed: {e}")
            return ""

    def _template_summarise(self, captions: list) -> str:
        """Join captions with temporal connectives into a flowing description."""
        if len(captions) == 2:
            return f"{captions[0]}, transitioning to {captions[1]}"

        connectives = ["beginning with", "then", "then", "then", "ending with"]
        parts = []
        for i, cap in enumerate(captions):
            conn = connectives[min(i, len(connectives) - 1)]
            parts.append(f"{conn} {cap}")
        return ", ".join(parts)

    def unload(self):
        """Free all GPU memory using the hardened cleanup sequence."""
        from core.gpu_utils import deep_cleanup_pipeline
        if self.pipe is not None:
            deep_cleanup_pipeline(self.pipe, label=f"VideoGen({self.current_model})")
            self.pipe = None
        self.current_model = None
