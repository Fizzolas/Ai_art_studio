"""
Central configuration manager with auto-save.
All settings persist to JSON and reload on startup.
"""
import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import threading


CONFIG_DIR = Path.home() / ".ai_art_studio"
CONFIG_FILE = CONFIG_DIR / "settings.json"
MODELS_DIR = CONFIG_DIR / "models"
CACHE_DIR = CONFIG_DIR / "cache"
OUTPUTS_DIR = CONFIG_DIR / "outputs"
DATASETS_DIR = CONFIG_DIR / "datasets"
CAPTIONS_DIR = CONFIG_DIR / "captions"
LOGS_DIR = CONFIG_DIR / "logs"


def ensure_dirs():
    for d in [CONFIG_DIR, MODELS_DIR, CACHE_DIR, OUTPUTS_DIR, DATASETS_DIR, CAPTIONS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()


@dataclass
class HardwareProfile:
    gpu_name: str = "NVIDIA GeForce RTX 4070 Laptop GPU"
    vram_total_mb: int = 8188
    ram_total_gb: float = 31.64
    cpu_cores: int = 16
    cuda_version: str = "12.4"

    # Offloading config
    offload_mode: str = "balanced"  # "none", "balanced", "aggressive", "cpu_only"
    transformer_offload_pct: int = 50
    text_encoder_offload_pct: int = 100
    vae_offload: bool = True
    sequential_offload: bool = False
    cpu_offload: bool = True
    attention_slicing: bool = True
    vae_slicing: bool = True
    vae_tiling: bool = True
    xformers: bool = True
    use_fp8: bool = True
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    max_vram_usage_mb: int = 7500
    gpu_device_index: int = 0


@dataclass
class TrainingConfig:
    # Base model
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_type: str = "sdxl"  # "sd15", "sdxl", "flux", "wan21"
    training_type: str = "lora"  # "lora", "dreambooth_lora", "textual_inversion"

    # LoRA params
    lora_rank: int = 32
    lora_alpha: int = 16
    network_module: str = "networks.lora"

    # Training hyperparams
    learning_rate: float = 0.0003
    unet_lr: float = 0.0004
    text_encoder_lr: float = 0.0003
    batch_size: int = 1
    epochs: int = 10
    max_train_steps: int = 2000
    save_every_n_epochs: int = 1
    save_every_n_steps: int = 500
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer: str = "Adafactor"  # "AdamW8bit", "Adafactor", "Prodigy", "Lion8bit"
    optimizer_args: str = "scale_parameter=False relative_step=False warmup_init=False"
    lr_scheduler: str = "constant"  # "constant", "cosine", "cosine_with_restarts", "polynomial"
    lr_warmup_steps: int = 0

    # Resolution
    resolution: int = 512
    enable_bucket: bool = True
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64

    # Memory optimization
    mixed_precision: str = "bf16"
    save_precision: str = "bf16"
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    cache_text_encoder_outputs: bool = True
    cache_text_encoder_outputs_to_disk: bool = True
    fp8_base: bool = True
    full_bf16: bool = True
    gradient_checkpointing: bool = True
    split_mode: bool = False

    # Regularization
    noise_offset: float = 0.05
    max_grad_norm: float = 1.0
    prior_loss_weight: float = 1.0
    min_snr_gamma: float = 5.0

    # Data augmentation
    flip_aug: bool = False
    color_aug: bool = False
    random_crop: bool = False

    # Dataset
    dataset_dir: str = ""
    reg_dir: str = ""
    instance_prompt: str = ""
    class_prompt: str = ""
    caption_extension: str = ".txt"
    num_repeats: int = 10

    # Output
    output_dir: str = str(MODELS_DIR)
    output_name: str = "my_lora"
    log_dir: str = str(LOGS_DIR)


@dataclass
class GenerationConfig:
    # Image generation
    img_width: int = 768
    img_height: int = 768
    img_steps: int = 30
    img_cfg_scale: float = 7.5
    img_sampler: str = "euler_a"  # "euler", "euler_a", "dpm++_2m", "dpm++_2m_karras", "ddim", "uni_pc"
    img_seed: int = -1
    img_batch_count: int = 1
    img_batch_size: int = 1
    img_clip_skip: int = 2
    img_hires_fix: bool = False
    img_hires_scale: float = 1.5
    img_hires_steps: int = 15
    img_hires_denoising: float = 0.55

    # Negative prompt defaults
    img_negative_prompt: str = "worst quality, low quality, blurry, deformed, bad anatomy"

    # LoRA loading
    lora_path: str = ""
    lora_weight: float = 0.8

    # Video generation
    vid_width: int = 512
    vid_height: int = 512
    vid_frames: int = 49
    vid_fps: int = 16
    vid_steps: int = 30
    vid_cfg_scale: float = 6.0
    vid_seed: int = -1
    vid_sampler: str = "euler"
    vid_flow_shift: float = 3.0

    # ControlNet
    controlnet_enabled: bool = False
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_canny"
    controlnet_preprocessor: str = "canny"   # "canny", "depth", "openpose", "none"
    controlnet_strength: float = 1.0
    controlnet_guidance_start: float = 0.0
    controlnet_guidance_end: float = 1.0
    controlnet_input_image: str = ""

    # IP-Adapter
    ip_adapter_enabled: bool = False
    ip_adapter_model: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"
    ip_adapter_scale: float = 0.6
    ip_adapter_image: str = ""

    # Advanced image generation
    img_eta: float = 0.0
    img_denoising_start: float = 0.0
    img_denoising_end: float = 1.0
    img_tiling: bool = False
    img_karras_sigmas: bool = True
    img_rescale_cfg: float = 0.0
    img_aesthetic_score: float = 6.0
    img_negative_aesthetic_score: float = 2.5

    # Advanced video generation
    vid_motion_bucket_id: int = 127
    vid_noise_aug_strength: float = 0.02
    vid_decode_chunk_size: int = 8
    vid_overlap_frames: int = 4
    vid_clip_count: int = 1
    vid_sample_method: str = "euler"
    vid_negative_prompt: str = ""
    vid_tiling: bool = False
    vid_format: str = "mp4"
    vid_caption_clips: bool = True
    vid_caption_sample_frames: int = 5

    # Output
    output_dir: str = str(OUTPUTS_DIR)
    auto_save: bool = True
    save_format: str = "png"  # "png", "jpg", "webp" for images; "mp4", "gif" for video


@dataclass
class CaptioningConfig:
    method: str = "combined"  # "wd_tagger", "blip2", "florence2", "combined"
    wd_model: str = "SmilingWolf/wd-swinv2-tagger-v3"
    wd_threshold: float = 0.35
    blip2_model: str = "Salesforce/blip2-opt-2.7b"
    florence_model: str = "microsoft/Florence-2-base"
    include_nsfw_tags: bool = True
    max_tags: int = 75
    prepend_trigger_word: bool = True
    trigger_word: str = ""
    caption_format: str = "tags_and_natural"  # "tags_only", "natural_only", "tags_and_natural"
    overwrite_existing: bool = False
    # Verbose natural-language description: tag-aware paragraph that
    # expands detected anatomy, actions, and positions into full prose.
    verbose_description: bool = True
    # Video frame extraction: extract every Nth frame for individual
    # captioning.  Each extracted frame becomes its own training image
    # with its own caption file.  0 = auto (every 5th frame, max 60).
    video_frame_interval: int = 5
    video_max_frames: int = 60
    # Pipeline mode mirror — set by GUI from AppConfig.pipeline_mode
    # so the captioner knows whether to split videos or keep them whole.
    pipeline_mode: str = "image"  # "image" or "video"
    keep_models_in_memory: bool = False


@dataclass
class AudioConfig:
    """Configuration for audio generation (scaffold for future implementation)."""
    enabled: bool = False
    model: str = ""
    prompt: str = ""
    duration_seconds: float = 10.0
    sample_rate: int = 44100
    guidance_scale: float = 3.5
    seed: int = -1
    output_format: str = "wav"
    output_dir: str = ""
    sync_to_video: bool = False
    video_path: str = ""


@dataclass
class AppConfig:
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    theme: str = "dark"
    last_dataset_dir: str = ""
    recent_models: List[str] = field(default_factory=list)
    window_geometry: str = ""
    # Pipeline mode: determines how videos are treated throughout the
    # entire dataset → captioning → training pipeline.
    #   "image" — train an image model; videos are split into frames
    #   "video" — train a video model; videos are kept as video input
    pipeline_mode: str = "image"  # "image" or "video"


class ConfigManager:
    """Thread-safe auto-saving configuration manager."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._save_lock = threading.Lock()
        self.config = AppConfig()
        self.load()

    @classmethod
    def reset(cls):
        """Reset the singleton instance. Useful for testing."""
        with cls._lock:
            cls._instance = None

    def config_file_exists(self) -> bool:
        """Check if the config JSON file exists on disk (for first-run detection)."""
        return CONFIG_FILE.exists()

    def load(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                self._apply_dict(data)
            except Exception as e:
                print(f"[Config] Failed to load config: {e}, trying backup...")
                bak = Path(str(CONFIG_FILE) + ".bak")
                if bak.exists():
                    try:
                        with open(bak, "r") as f:
                            data = json.load(f)
                        self._apply_dict(data)
                        print("[Config] Restored settings from backup")
                    except Exception as e2:
                        print(f"[Config] Backup also failed: {e2}, using defaults")
                else:
                    print("[Config] No backup available, using defaults")

    def save(self):
        """Atomic save: write to temp file, then rename.
        Prevents config corruption if the app crashes mid-write (#12)."""
        with self._save_lock:
            try:
                # Back up current config before overwriting
                import shutil
                if CONFIG_FILE.exists():
                    shutil.copy2(str(CONFIG_FILE), str(CONFIG_FILE) + ".bak")

                data = self._to_dict()
                # Write to temp file in same directory, then atomic rename
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(CONFIG_DIR), suffix=".tmp", prefix="settings_")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                    # Atomic rename (on same filesystem)
                    os.replace(tmp_path, str(CONFIG_FILE))
                except Exception:
                    # Clean up temp file on failure
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            except Exception as e:
                print(f"[Config] Failed to save config: {e}")

    def _to_dict(self) -> dict:
        return {
            "hardware": asdict(self.config.hardware),
            "training": asdict(self.config.training),
            "generation": asdict(self.config.generation),
            "captioning": asdict(self.config.captioning),
            "audio": asdict(self.config.audio),
            "theme": self.config.theme,
            "last_dataset_dir": self.config.last_dataset_dir,
            "recent_models": self.config.recent_models,
            "window_geometry": self.config.window_geometry,
            "pipeline_mode": self.config.pipeline_mode,
        }

    def _apply_dict(self, data: dict):
        if "hardware" in data:
            for k, v in data["hardware"].items():
                if hasattr(self.config.hardware, k):
                    setattr(self.config.hardware, k, v)
        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(self.config.training, k):
                    setattr(self.config.training, k, v)
        if "generation" in data:
            for k, v in data["generation"].items():
                if hasattr(self.config.generation, k):
                    setattr(self.config.generation, k, v)
        if "captioning" in data:
            for k, v in data["captioning"].items():
                if hasattr(self.config.captioning, k):
                    setattr(self.config.captioning, k, v)
        if "audio" in data:
            for k, v in data["audio"].items():
                if hasattr(self.config.audio, k):
                    setattr(self.config.audio, k, v)
        for key in ["theme", "last_dataset_dir", "recent_models",
                   "window_geometry", "pipeline_mode"]:
            if key in data:
                setattr(self.config, key, data[key])

    def update_and_save(self, section: str, key: str, value: Any):
        """Update a single setting and auto-save.

        For top-level AppConfig fields, pass section="app".
        """
        if section == "app" and hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save()
            return
        section_obj = getattr(self.config, section, None)
        if section_obj and hasattr(section_obj, key):
            setattr(section_obj, key, value)
            self.save()

    def get_offload_preset(self) -> Dict[str, Any]:
        """Return offloading settings based on current mode."""
        hw = self.config.hardware
        mode = hw.offload_mode

        if mode == "none":
            return {
                "cpu_offload": False, "sequential_offload": False,
                "attention_slicing": False, "vae_slicing": False,
                "vae_tiling": False, "transformer_offload_pct": 0,
                "text_encoder_offload_pct": 0, "vae_offload": False,
            }
        elif mode == "balanced":
            return {
                "cpu_offload": True, "sequential_offload": False,
                "attention_slicing": True, "vae_slicing": True,
                "vae_tiling": True, "transformer_offload_pct": 50,
                "text_encoder_offload_pct": 100, "vae_offload": True,
            }
        elif mode == "aggressive":
            return {
                "cpu_offload": True, "sequential_offload": True,
                "attention_slicing": True, "vae_slicing": True,
                "vae_tiling": True, "transformer_offload_pct": 100,
                "text_encoder_offload_pct": 100, "vae_offload": True,
            }
        elif mode == "cpu_only":
            return {
                "cpu_offload": True, "sequential_offload": True,
                "attention_slicing": True, "vae_slicing": True,
                "vae_tiling": True, "transformer_offload_pct": 100,
                "text_encoder_offload_pct": 100, "vae_offload": True,
            }
        return {}
