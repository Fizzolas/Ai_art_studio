"""
Audio generation engine — scaffold for future implementation.

This module defines the AudioGenerator interface. Currently no audio model
is loaded — all methods return informative stubs. When an audio diffusion
model is added (e.g. AudioLDM2, MusicGen, Stable Audio), the implementation
goes here without touching any other files.

Designed to sync generated audio with video output for video model workflows.
"""
from core.logger import get_logger
logger = get_logger(__name__)


class AudioGenerator:
    """
    Scaffold for audio generation, ready for a future model backend.

    Planned backends:
    - AudioLDM 2 (CVSSP/audioldm2) — text-to-audio/music
    - MusicGen (facebook/musicgen-small/medium/large) — text-to-music
    - Stable Audio (stabilityai/stable-audio-open-1.0) — text-to-audio
    - Bark (suno-ai/bark) — text-to-speech + music

    Video sync strategy:
    - Generate audio matching the duration of a video (fps * frames)
    - Embed generated audio track into the video container using ffmpeg
    """

    def __init__(self, config=None):
        from core.config import ConfigManager
        cfg = ConfigManager()
        self.config = config or cfg.config.audio
        self.pipe = None
        self.current_model = None
        self._available = False
        self._check_availability()

    def _check_availability(self):
        """Check if any supported audio generation library is installed."""
        try:
            import transformers
            from transformers import AutoProcessor
            self._available = True
            logger.info("Audio: transformers available — MusicGen scaffold ready")
        except ImportError:
            logger.info("Audio: no supported audio library found (scaffold mode)")

    def is_available(self) -> bool:
        """Return True if an audio generation backend is installed."""
        return self._available

    def list_supported_models(self) -> list:
        """Return list of supported model IDs."""
        return [
            "facebook/musicgen-small",
            "facebook/musicgen-medium",
            "facebook/musicgen-large",
            "CVSSP/audioldm2",
            "CVSSP/audioldm2-music",
            "stabilityai/stable-audio-open-1.0",
        ]

    def load_model(self, model_id: str, on_progress=None):
        """Load an audio generation model."""
        if on_progress:
            on_progress(f"Loading audio model: {model_id}...")

        try:
            if "musicgen" in model_id.lower():
                self._load_musicgen(model_id, on_progress)
            elif "audioldm" in model_id.lower():
                self._load_audioldm2(model_id, on_progress)
            else:
                raise NotImplementedError(
                    f"Audio model '{model_id}' not yet implemented. "
                    f"Supported: {self.list_supported_models()}"
                )
            self.current_model = model_id
            logger.info(f"Audio model loaded: {model_id}")
        except NotImplementedError:
            raise
        except Exception as e:
            logger.error(f"Audio model load failed: {e}")
            raise

    def _load_musicgen(self, model_id: str, on_progress=None):
        """Load Meta MusicGen model via transformers."""
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(device)
            self.pipe = "musicgen"
            if on_progress:
                on_progress("MusicGen loaded")
        except ImportError:
            raise ImportError(
                "MusicGen requires transformers>=4.31: "
                "pip install transformers>=4.31"
            )

    def _load_audioldm2(self, model_id: str, on_progress=None):
        """Load AudioLDM 2 via diffusers."""
        try:
            from diffusers import AudioLDM2Pipeline
            import torch
            self.pipe = AudioLDM2Pipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            if on_progress:
                on_progress("AudioLDM2 loaded")
        except ImportError:
            raise ImportError(
                "AudioLDM2 requires diffusers>=0.21: "
                "pip install diffusers>=0.21"
            )

    def generate(
        self,
        prompt: str,
        duration_seconds: float = 10.0,
        sample_rate: int = 32000,
        guidance_scale: float = 3.5,
        seed: int = -1,
        callback=None,
    ) -> tuple:
        """Generate audio from a text prompt. Returns (audio_array, sample_rate)."""
        import numpy as np
        import torch

        if self.pipe is None:
            logger.warning("AudioGenerator.generate() called with no model — returning silence")
            samples = int(duration_seconds * sample_rate)
            return np.zeros(samples, dtype=np.float32), sample_rate

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        generator = torch.Generator().manual_seed(seed)

        try:
            if self.pipe == "musicgen":
                return self._generate_musicgen(
                    prompt, duration_seconds, guidance_scale, generator, callback)
            elif hasattr(self.pipe, "unet"):
                return self._generate_audioldm2(
                    prompt, duration_seconds, guidance_scale, sample_rate,
                    generator, callback)
        except Exception as e:
            logger.error(f"Audio generation failed: {e}", exc_info=True)
            raise

        # Fallback
        samples = int(duration_seconds * sample_rate)
        return np.zeros(samples, dtype=np.float32), sample_rate

    def _generate_musicgen(self, prompt, duration, guidance_scale, generator, callback):
        import torch
        inputs = self._processor(text=[prompt], padding=True, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_tokens = int(duration * 50)

        with torch.inference_mode():
            audio_values = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                guidance_scale=guidance_scale,
            )

        audio = audio_values[0, 0].cpu().numpy()
        sr = self._model.config.audio_encoder.sampling_rate
        return audio, sr

    def _generate_audioldm2(self, prompt, duration, guidance_scale,
                             sample_rate, generator, callback):
        output = self.pipe(
            prompt,
            num_inference_steps=200,
            audio_length_in_s=duration,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return output.audios[0], sample_rate

    def generate_for_video(
        self,
        prompt: str,
        video_path: str,
        guidance_scale: float = 3.5,
        seed: int = -1,
    ) -> tuple:
        """Generate audio matched to a video's duration."""
        duration = self._get_video_duration(video_path)
        logger.info(f"Generating {duration:.1f}s audio for video: {video_path}")
        return self.generate(prompt, duration_seconds=duration,
                            guidance_scale=guidance_scale, seed=seed)

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frames / fps
        except Exception:
            return 10.0

    def embed_audio_in_video(
        self,
        audio_array,
        sample_rate: int,
        video_path: str,
        output_path: str,
    ) -> str:
        """Embed audio into a video file using ffmpeg."""
        import subprocess
        import tempfile
        import os
        import numpy as np

        try:
            import soundfile as sf
        except ImportError:
            try:
                from scipy.io import wavfile
                sf = None
            except ImportError:
                raise ImportError(
                    "Install soundfile or scipy to embed audio: "
                    "pip install soundfile"
                )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio = tmp.name

        try:
            if sf is not None:
                import soundfile
                soundfile.write(tmp_audio, audio_array, sample_rate)
            else:
                from scipy.io import wavfile
                audio_int16 = (audio_array * 32767).astype(np.int16)
                wavfile.write(tmp_audio, sample_rate, audio_int16)

            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", tmp_audio,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            logger.info(f"Audio embedded in video: {output_path}")
            return output_path
        finally:
            try:
                os.unlink(tmp_audio)
            except OSError:
                pass

    def save_audio(
        self,
        audio_array,
        sample_rate: int,
        output_dir: str,
        format: str = "wav",
        prefix: str = "audio",
    ) -> str:
        """Save generated audio to file."""
        import os
        from pathlib import Path
        from datetime import datetime

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.{format}"
        filepath = os.path.join(output_dir, filename)

        try:
            import soundfile
            soundfile.write(filepath, audio_array, sample_rate)
        except ImportError:
            try:
                from scipy.io import wavfile
                import numpy as np
                audio_int16 = (audio_array * 32767).astype(np.int16)
                wavfile.write(filepath, sample_rate, audio_int16)
            except ImportError:
                logger.error("No audio write library available (soundfile or scipy)")
                raise

        logger.info(f"Audio saved: {filepath}")
        return filepath

    def unload(self):
        """Free GPU memory."""
        import gc
        import torch
        self.pipe = None
        self.current_model = None
        if hasattr(self, '_model'):
            del self._model
        if hasattr(self, '_processor'):
            del self._processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Audio model unloaded")
