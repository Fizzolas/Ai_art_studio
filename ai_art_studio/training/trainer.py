"""
Training engine for LoRA fine-tuning on 8GB VRAM.

Two training backends:
  1. kohya-ss/sd-scripts (external) — launched via `accelerate` CLI.
     Used when sd-scripts is installed and detected.
  2. DiffusersTrainer (built-in) — pure diffusers + PEFT.
     Always available, no external scripts needed.

Optimized for RTX 4070 Laptop GPU (8 GB VRAM).
"""
import os
import gc
import sys
import json
import time
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import asdict

import torch

from core.logger import get_logger
logger = get_logger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────

def _find_accelerate() -> Optional[str]:
    """Return the full path to the ``accelerate`` CLI script, or None."""
    acc = shutil.which("accelerate")
    if acc:
        return acc

    # Inside a venv on Windows the script may live next to python.exe
    venv_bin = Path(sys.executable).parent
    for name in ("accelerate", "accelerate.exe"):
        candidate = venv_bin / name
        if candidate.is_file():
            return str(candidate)

    return None


def _has_kohya_scripts() -> bool:
    """Check whether kohya-ss sd-scripts are reachable."""
    project_root = Path(__file__).resolve().parent.parent
    sd_scripts = project_root / "sd-scripts"
    return sd_scripts.is_dir() and (sd_scripts / "sdxl_train_network.py").is_file()


# ── VRAM monitor ─────────────────────────────────────────────────────────

class VRAMMonitor:
    """Real-time VRAM monitoring during training."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._running = False
        self._thread = None
        self.peak_vram_mb = 0
        self.current_vram_mb = 0
        self.history: List[Dict] = []

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self):
        while self._running:
            try:
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / (1024 ** 2)
                    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    self.current_vram_mb = mem
                    self.peak_vram_mb = max(self.peak_vram_mb, mem)
                    self.history.append({
                        "time": time.time(),
                        "allocated_mb": round(mem, 1),
                        "reserved_mb": round(reserved, 1),
                    })
                    if len(self.history) > 1000:
                        self.history = self.history[-500:]
            except Exception:
                pass
            time.sleep(self.interval)

    def get_stats(self) -> Dict:
        return {
            "current_mb": round(self.current_vram_mb, 1),
            "peak_mb": round(self.peak_vram_mb, 1),
            "available_mb": round(
                (torch.cuda.get_device_properties(0).total_memory / (1024**2))
                - self.current_vram_mb, 1
            ) if torch.cuda.is_available() else 0,
        }


# ══════════════════════════════════════════════════════════════════════════
#  TrainingJob  (subprocess-based — kohya-ss or accelerate)
# ══════════════════════════════════════════════════════════════════════════

class TrainingJob:
    """Manages a single training job with progress tracking."""

    def __init__(self, config, on_progress=None, on_log=None, on_complete=None, on_error=None):
        from core.config import TrainingConfig
        self.config: TrainingConfig = config
        self.on_progress = on_progress
        self.on_log = on_log
        self.on_complete = on_complete
        self.on_error = on_error

        self.process: Optional[subprocess.Popen] = None
        self.vram_monitor = VRAMMonitor()
        self.is_running = False
        self.is_cancelled = False
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.current_lr = 0.0
        self.start_time = 0.0
        self.elapsed_seconds = 0.0

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """Find the latest checkpoint in the output directory."""
        from pathlib import Path
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
        return str(checkpoints[-1]) if checkpoints else None

    def _write_error_report(self, error: Exception, tb: str):
        from core.logger import get_log_dir
        import json as _json
        from datetime import datetime
        error_dir = get_log_dir() / "errors"
        error_dir.mkdir(exist_ok=True)
        report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": tb,
            "step": self.current_step,
            "total_steps": self.total_steps,
            "loss": self.current_loss,
            "model_type": self.config.model_type,
            "training_type": self.config.training_type,
        }
        fname = error_dir / f"train_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fname.write_text(_json.dumps(report, indent=2), encoding="utf-8")

    # ── command builders ─────────────────────────────────────────────────

    def build_training_command(self, hardware) -> List[str]:
        c = self.config
        if c.model_type in ("sd15", "sdxl"):
            return self._build_kohya_command(hardware)
        elif c.model_type == "flux":
            return self._build_flux_command(hardware)
        else:
            return self._build_kohya_command(hardware)

    def _accelerate_prefix(self) -> List[str]:
        """Return the accelerate launch prefix.  Uses the CLI binary when
        found, otherwise falls back to ``sys.executable -m accelerate``."""
        acc = _find_accelerate()
        if acc:
            return [acc, "launch",
                    "--num_cpu_threads_per_process", "2",
                    "--mixed_precision", self.config.mixed_precision,
                    "--num_processes", "1",
                    "--num_machines", "1"]
        # Fallback — this only works when the accelerate *package* exposes __main__
        return [sys.executable, "-m", "accelerate", "launch",
                "--num_cpu_threads_per_process", "2",
                "--mixed_precision", self.config.mixed_precision,
                "--num_processes", "1",
                "--num_machines", "1"]

    def _build_kohya_command(self, hardware) -> List[str]:
        c = self.config
        script = "sdxl_train_network.py" if c.model_type == "sdxl" else "train_network.py"

        # Resolve script path inside sd-scripts
        project_root = Path(__file__).resolve().parent.parent
        sd_scripts = project_root / "sd-scripts"
        script_path = sd_scripts / script
        if not script_path.is_file():
            # If exact script not found, try library_scripts subdir
            script_path = sd_scripts / "library" / script
        if not script_path.is_file():
            raise FileNotFoundError(
                f"kohya-ss script not found: {script}\n"
                f"Looked in: {sd_scripts}\n"
                f"Run setup.py to install sd-scripts, or use the built-in trainer."
            )

        cmd = self._accelerate_prefix()
        cmd.extend([
            str(script_path),
            "--pretrained_model_name_or_path", c.base_model,
            "--train_data_dir", c.dataset_dir,
            "--output_dir", c.output_dir,
            "--output_name", c.output_name,
            "--logging_dir", c.log_dir,
            "--save_model_as", "safetensors",
            "--network_module", c.network_module,
            "--network_dim", str(c.lora_rank),
            "--network_alpha", str(c.lora_alpha),
            "--learning_rate", str(c.learning_rate),
            "--unet_lr", str(c.unet_lr),
            "--text_encoder_lr", str(c.text_encoder_lr),
            "--train_batch_size", str(c.batch_size),
            "--max_train_steps", str(c.max_train_steps),
            "--save_every_n_epochs", str(c.save_every_n_epochs),
            "--optimizer_type", c.optimizer,
            "--lr_scheduler", c.lr_scheduler,
            "--resolution", f"{c.resolution},{c.resolution}",
            "--mixed_precision", c.mixed_precision,
            "--save_precision", c.save_precision,
            "--noise_offset", str(c.noise_offset),
            "--max_grad_norm", str(c.max_grad_norm),
            "--prior_loss_weight", str(c.prior_loss_weight),
            "--caption_extension", c.caption_extension,
        ])

        if c.cache_latents:
            cmd.append("--cache_latents")
        if c.cache_latents_to_disk:
            cmd.append("--cache_latents_to_disk")
        if c.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if c.fp8_base:
            cmd.append("--fp8_base")
        if c.full_bf16:
            cmd.append("--full_bf16")
        if c.enable_bucket:
            cmd.extend([
                "--enable_bucket",
                "--min_bucket_reso", str(c.min_bucket_reso),
                "--max_bucket_reso", str(c.max_bucket_reso),
                "--bucket_reso_steps", str(c.bucket_reso_steps),
            ])
        if hardware.xformers:
            cmd.append("--xformers")
        if c.optimizer_args:
            cmd.extend(["--optimizer_args", c.optimizer_args])
        if c.flip_aug:
            cmd.append("--flip_aug")
        if c.color_aug:
            cmd.append("--color_aug")
        if c.random_crop:
            cmd.append("--random_crop")
        if c.model_type == "sdxl":
            cmd.append("--no_half_vae")
            cmd.extend(["--max_data_loader_n_workers", "0"])
            cmd.append("--bucket_no_upscale")

        return cmd

    def _build_flux_command(self, hardware) -> List[str]:
        c = self.config
        project_root = Path(__file__).resolve().parent.parent
        sd_scripts = project_root / "sd-scripts"
        script_path = sd_scripts / "flux_train_network.py"
        if not script_path.is_file():
            raise FileNotFoundError(
                "flux_train_network.py not found in sd-scripts.\n"
                "Run setup.py to install sd-scripts."
            )

        cmd = self._accelerate_prefix()
        cmd.extend([
            str(script_path),
            "--pretrained_model_name_or_path", c.base_model,
            "--train_data_dir", c.dataset_dir,
            "--output_dir", c.output_dir,
            "--output_name", c.output_name,
            "--logging_dir", c.log_dir,
            "--save_model_as", "safetensors",
            "--network_module", "networks.lora_flux",
            "--network_dim", str(c.lora_rank),
            "--network_alpha", str(c.lora_alpha),
            "--network_args", "train_blocks=single",
            "--unet_lr", str(c.unet_lr),
            "--network_train_unet_only",
            "--train_batch_size", str(c.batch_size),
            "--max_train_steps", str(c.max_train_steps),
            "--save_every_n_steps", str(c.save_every_n_steps),
            "--optimizer_type", "AdamW8bit",
            "--lr_scheduler", c.lr_scheduler,
            "--resolution", f"{c.resolution},{c.resolution}",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--full_bf16",
            "--fp8_base",
            "--cache_latents",
            "--cache_latents_to_disk",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--gradient_checkpointing",
            "--split_mode",
            "--model_prediction_type", "raw",
            "--discrete_flow_shift", "3.1582",
            "--timestep_sampling", "shift",
            "--guidance_scale", "1.0",
            "--t5xxl_max_token_length", "512",
            "--apply_t5_attn_mask",
            "--enable_bucket",
            "--min_bucket_reso", "256",
            "--max_bucket_reso", "1024",
            "--bucket_reso_steps", "64",
            "--sdpa",
            "--noise_offset", str(c.noise_offset),
            "--max_grad_norm", str(c.max_grad_norm),
            "--caption_extension", c.caption_extension,
            "--max_data_loader_n_workers", "0",
        ])
        return cmd

    # ── launch ───────────────────────────────────────────────────────────

    def start(self, hardware):
        """Start the training process."""
        if self.is_running:
            return

        self.is_running = True
        self.is_cancelled = False
        self.start_time = time.time()
        self.total_steps = self.config.max_train_steps
        self.vram_monitor.start()

        # Video model training (wan21) uses a separate pipeline
        if self.config.model_type == "wan21":
            self._log("Starting training: Wan2.1 video LoRA  [diffusers backend]")
            self._log(f"Output: {self.config.output_dir}/{self.config.output_name}")
            self._log(f"Steps: {self.config.max_train_steps}")
            thread = threading.Thread(
                target=self._run_video_training, args=(hardware,), daemon=True
            )
            thread.start()
            return

        # Decide backend for image models
        use_kohya = _has_kohya_scripts() and _find_accelerate() is not None

        if use_kohya:
            cmd = self.build_training_command(hardware)
            self._log(f"Starting training: {self.config.model_type} LoRA  [kohya-ss backend]")
            self._log(f"Output: {self.config.output_dir}/{self.config.output_name}")
            self._log(f"Steps: {self.config.max_train_steps}")
            thread = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
            thread.start()
        else:
            # Fall back to built-in diffusers trainer
            if not _has_kohya_scripts():
                self._log("kohya-ss sd-scripts not found — using built-in diffusers trainer.")
            elif not _find_accelerate():
                self._log("accelerate CLI not found — using built-in diffusers trainer.")
            self._log(f"Starting training: {self.config.model_type} LoRA  [diffusers backend]")
            self._log(f"Output: {self.config.output_dir}/{self.config.output_name}")
            self._log(f"Steps: {self.config.max_train_steps}")
            thread = threading.Thread(
                target=self._run_diffusers_training, args=(hardware,), daemon=True
            )
            thread.start()

    # ── subprocess backend (kohya-ss) ────────────────────────────────────

    def _run_subprocess(self, cmd: List[str]):
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Propagate CUDA fragmentation prevention to subprocess
            env.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "max_split_size_mb:128,garbage_collection_threshold:0.6",
            )

            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, bufsize=1,
            )

            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                self._log(line)
                self._parse_progress(line)
                if self.is_cancelled:
                    self.process.terminate()
                    self._log("Training cancelled by user.")
                    break

            self.process.wait()

            if not self.is_cancelled:
                if self.process.returncode == 0:
                    output_path = os.path.join(
                        self.config.output_dir,
                        f"{self.config.output_name}.safetensors"
                    )
                    self._log(f"Training complete! Model saved to: {output_path}")
                    if self.on_complete:
                        self.on_complete(output_path)
                else:
                    self._log(f"Training failed with exit code {self.process.returncode}")
                    if self.on_error:
                        self.on_error(f"Exit code: {self.process.returncode}")

        except Exception as e:
            import traceback as _tb
            self._write_error_report(e, _tb.format_exc())
            self._log(f"Training error: {e}")
            if self.on_error:
                self.on_error(str(e))
        finally:
            self.is_running = False
            self.vram_monitor.stop()
            self.elapsed_seconds = time.time() - self.start_time

    # ── diffusers backend (built-in) ─────────────────────────────────────

    def _run_diffusers_training(self, hardware):
        """Train LoRA using diffusers + PEFT directly — no external scripts."""
        try:
            from core.model_downloader import ensure_model_available

            c = self.config

            # Auto-download base model if needed
            self._log("Checking base model availability...")
            resolved_model = ensure_model_available(
                c.base_model, c.model_type,
                on_status=lambda msg: self._log(msg),
            )

            self._log("Loading base model for training...")

            from diffusers import (
                StableDiffusionPipeline, StableDiffusionXLPipeline,
                DDPMScheduler,
            )
            from peft import LoraConfig, get_peft_model
            from torchvision import transforms

            # Load pipeline
            if c.model_type == "sdxl":
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    resolved_model,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    resolved_model,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                )

            # Memory optimizations
            if hardware.attention_slicing:
                pipe.enable_attention_slicing()
            if hardware.vae_slicing:
                pipe.enable_vae_slicing()
            if hardware.xformers:
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

            # Configure LoRA on UNet
            unet = pipe.unet
            lora_config = LoraConfig(
                r=c.lora_rank,
                lora_alpha=c.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            unet = get_peft_model(unet, lora_config)
            unet.to("cuda", dtype=torch.float16)

            if c.gradient_checkpointing:
                unet.enable_gradient_checkpointing()

            trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
            total = sum(p.numel() for p in unet.parameters())
            self._log(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

            # Gather training images
            dataset_dir = c.dataset_dir
            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(Path(dataset_dir).rglob(f"*{ext}"))
            image_paths = sorted(image_paths)

            if not image_paths:
                raise FileNotFoundError(f"No training images found in {dataset_dir}")

            # ── Video group awareness ──
            # If a video_groups.json manifest exists, order training data
            # so frames from the same source video are trained
            # consecutively.  This teaches the model that those frames
            # share a continuous scene and shouldn't be treated as
            # independent images.
            video_groups = {}  # group_id -> [image_path, ...]
            manifest_candidates = [
                Path(dataset_dir) / "video_groups.json",
                Path(dataset_dir) / "captions" / "video_groups.json",
            ]
            # Also check the work_dir / captions folder (where captioner writes)
            home_cache = Path.home() / ".ai_art_studio" / "cache" / "datasets"
            if home_cache.is_dir():
                for d in home_cache.iterdir():
                    cand = d / "captions" / "video_groups.json"
                    manifest_candidates.append(cand)

            for mpath in manifest_candidates:
                if mpath.is_file():
                    try:
                        raw = json.loads(mpath.read_text(encoding="utf-8"))
                        for gid, gdata in raw.items():
                            frame_files = []
                            for fr in gdata.get("frames", []):
                                fp = Path(fr["image"])
                                if fp.is_file():
                                    frame_files.append(fp)
                            if frame_files:
                                video_groups[gid] = sorted(
                                    frame_files,
                                    key=lambda p: p.stem)
                        self._log(
                            f"Loaded video group manifest: "
                            f"{len(video_groups)} groups from {mpath}")
                    except Exception as e:
                        self._log(f"Warning: could not parse {mpath}: {e}")
                    break  # use first found manifest

            # Re-order image_paths: video-grouped frames in consecutive
            # blocks, then standalone images.  Within each epoch the
            # model sees all frames of one video together before moving
            # on to the next.
            grouped_frame_set = set()
            ordered_paths = []
            resolved_images = {p.resolve() for p in image_paths}
            for gid, gframes in video_groups.items():
                for fp in gframes:
                    if fp.resolve() in resolved_images:
                        ordered_paths.append(fp)
                        grouped_frame_set.add(fp.resolve())

            # Append standalone (non-video) images after grouped blocks
            for ip in image_paths:
                if ip.resolve() not in grouped_frame_set:
                    ordered_paths.append(ip)

            image_paths = ordered_paths if ordered_paths else image_paths

            n_vid_frames = len(grouped_frame_set)
            n_standalone = len(image_paths) - n_vid_frames
            self._log(
                f"Found {len(image_paths)} training images "
                f"({n_vid_frames} video frames in "
                f"{len(video_groups)} group(s), "
                f"{n_standalone} standalone)")

            # ── Optimizer selection (#11) ──
            # Respect the user's config.optimizer choice instead of
            # always using AdamW.
            optimizer = self._create_optimizer(unet.parameters(), c)

            noise_scheduler = DDPMScheduler.from_pretrained(
                resolved_model, subfolder="scheduler"
            )
            transform = transforms.Compose([
                transforms.Resize((c.resolution, c.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

            vae = pipe.vae.to("cuda", dtype=torch.float16)
            text_encoder = pipe.text_encoder.to("cuda", dtype=torch.float16)
            tokenizer = pipe.tokenizer

            # VAE scaling factor: read from model config, not hardcoded (#5)
            vae_scaling_factor = getattr(
                vae.config, "scaling_factor", 0.18215)
            self._log(f"VAE scaling factor: {vae_scaling_factor}")

            # ── Latent caching (#16) ──
            # Pre-encode images through VAE so we don't run it every step
            latent_cache = {}
            if c.cache_latents:
                latent_cache = self._cache_latents(
                    image_paths, vae, transform, vae_scaling_factor)

            # GradScaler for stable fp16 training (#3)
            use_amp = c.mixed_precision == "fp16"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            # Gradient accumulation (#8)
            grad_accum_steps = max(1, c.gradient_accumulation_steps)
            if grad_accum_steps > 1:
                self._log(f"Gradient accumulation: {grad_accum_steps} steps")

            # LR warmup (#9)
            lr_scheduler = None
            if c.lr_warmup_steps > 0:
                from torch.optim.lr_scheduler import LambdaLR
                warmup_steps = c.lr_warmup_steps
                def warmup_fn(current_step):
                    if current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    return 1.0
                lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
                self._log(f"LR warmup: {warmup_steps} steps")

            total_steps = c.max_train_steps
            step = 0
            accum_loss = 0.0

            self._log("Training started...")

            while step < total_steps and not self.is_cancelled:
                for img_path in image_paths:
                    if self.is_cancelled or step >= total_steps:
                        break

                    try:
                        # Load caption
                        caption_path = img_path.with_suffix(".txt")
                        if caption_path.exists():
                            caption = caption_path.read_text(encoding="utf-8").strip()
                        else:
                            caption = c.instance_prompt or "a character"

                        # Encode image → latents (use cache if available)
                        cached_pt = latent_cache.get(img_path)
                        if cached_pt and Path(cached_pt).is_file():
                            latents = torch.load(str(cached_pt), weights_only=True).to(
                                "cuda", dtype=torch.float16)
                        else:
                            from PIL import Image
                            img = Image.open(img_path).convert("RGB")
                            pixel_values = transform(img).unsqueeze(0).to(
                                "cuda", dtype=torch.float16)
                            with torch.no_grad():
                                latents = (vae.encode(pixel_values).latent_dist.sample()
                                           * vae_scaling_factor)

                        with torch.no_grad():
                            tokens = tokenizer(
                                caption, padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt",
                            ).input_ids.to("cuda")
                            encoder_hidden_states = text_encoder(tokens)[0]

                        # Forward with autocast for fp16 stability (#3)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            noise = torch.randn_like(latents)
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (1,), device="cuda",
                            ).long()
                            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                            loss = torch.nn.functional.mse_loss(noise_pred, noise)
                            # Scale loss for gradient accumulation
                            if grad_accum_steps > 1:
                                loss = loss / grad_accum_steps

                        scaler.scale(loss).backward()
                        accum_loss += loss.item()

                        step += 1
                        self.current_step = step

                        # Optimizer step on accumulation boundary (#2, #8)
                        if step % grad_accum_steps == 0:
                            if c.max_grad_norm > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    unet.parameters(), c.max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)  # (#2) saves memory
                            if lr_scheduler is not None:
                                lr_scheduler.step()

                        self.current_loss = accum_loss if step % grad_accum_steps == 0 else accum_loss
                        self.current_lr = (
                            lr_scheduler.get_last_lr()[0]
                            if lr_scheduler else c.learning_rate)

                        if step % grad_accum_steps == 0:
                            accum_loss = 0.0

                        if step % 10 == 0:
                            self._log(f"Step {step}/{total_steps}  |  Loss: {self.current_loss:.4f}")
                            if self.on_progress:
                                self.on_progress(step, total_steps, self.current_loss, self.current_lr)

                        # Checkpoint
                        if step % c.save_every_n_steps == 0:
                            save_path = Path(c.output_dir) / f"{c.output_name}_step{step}"
                            save_path.mkdir(parents=True, exist_ok=True)
                            unet.save_pretrained(str(save_path))
                            self._log(f"Checkpoint saved: {save_path}")

                    except torch.cuda.OutOfMemoryError:
                        self._log("VRAM OOM — clearing cache and continuing...")
                        gc.collect()
                        torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()  # reset scaler state after OOM
                        continue

            # Final save
            if not self.is_cancelled:
                final_path = Path(c.output_dir) / c.output_name
                final_path.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(str(final_path))
                safetensors_path = str(final_path)
                self._log(f"Training complete! LoRA saved to: {safetensors_path}")
                if self.on_complete:
                    self.on_complete(safetensors_path)
            else:
                # Save a "last_good" checkpoint so work is not lost
                try:
                    last_good = Path(c.output_dir) / f"{c.output_name}_last_good_step{step}"
                    last_good.mkdir(parents=True, exist_ok=True)
                    unet.save_pretrained(str(last_good))
                    self._log(f"Training cancelled — last-good checkpoint saved: {last_good}")
                except Exception as save_err:
                    self._log(f"Training cancelled — failed to save last-good checkpoint: {save_err}")

        except Exception as e:
            import traceback as _tb
            self._write_error_report(e, _tb.format_exc())
            self._log(f"Training error: {e}")
            if self.on_error:
                self.on_error(str(e))
        finally:
            self.is_running = False
            self.vram_monitor.stop()
            self.elapsed_seconds = time.time() - self.start_time
            # Hardened GPU cleanup after training ends
            from core.gpu_utils import flush_gpu_memory, log_vram_snapshot
            try:
                # Explicitly free large training tensors
                for name in ("unet", "vae", "text_encoder", "pipe",
                             "optimizer", "noise_scheduler"):
                    obj = locals().get(name)
                    if obj is not None:
                        try:
                            if hasattr(obj, "to"):
                                obj.to("cpu")
                        except Exception:
                            pass
                        del obj
            except Exception:
                pass
            flush_gpu_memory()
            log_vram_snapshot("post-training")

    # ── video training backend (Wan2.1 etc.) ─────────────────────────────

    def _run_video_training(self, hardware):
        """Train LoRA on a video model (Wan2.1) using video clips.

        Video-mode datasets contain original video files with .txt
        caption files alongside them.  The trainer loads short clips,
        encodes them through the video VAE, and trains a LoRA on
        the 3D UNet / transformer.
        """
        try:
            from core.model_downloader import ensure_model_available
            c = self.config

            self._log("Checking Wan2.1 model availability…")
            resolved_model = ensure_model_available(
                c.base_model, c.model_type,
                on_status=lambda msg: self._log(msg),
            )

            self._log("Loading Wan2.1 pipeline for video LoRA training…")

            # Gather video files from dataset
            dataset_dir = c.dataset_dir
            video_extensions = {
                ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
                ".webm", ".m4v", ".mpg", ".mpeg",
            }
            video_paths = []
            for ext in video_extensions:
                video_paths.extend(Path(dataset_dir).rglob(f"*{ext}"))
            video_paths = sorted(video_paths)

            # Also gather any image files (they can augment video
            # training alongside clips)
            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(Path(dataset_dir).rglob(f"*{ext}"))
            image_paths = sorted(image_paths)

            total_items = len(video_paths) + len(image_paths)
            if total_items == 0:
                raise FileNotFoundError(
                    f"No video or image files found in {dataset_dir}")

            self._log(
                f"Found {len(video_paths)} video(s) and "
                f"{len(image_paths)} image(s) for video LoRA training")

            # Try to load Wan2.1 pipeline
            try:
                from diffusers import WanPipeline
                pipe = WanPipeline.from_pretrained(
                    resolved_model,
                    torch_dtype=torch.bfloat16,
                )
            except ImportError:
                self._log(
                    "WanPipeline not available in this diffusers version. "
                    "Falling back to generic DiffusionPipeline…")
                try:
                    from diffusers import DiffusionPipeline
                    pipe = DiffusionPipeline.from_pretrained(
                        resolved_model,
                        torch_dtype=torch.bfloat16,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Cannot load video model {resolved_model}: {e}\n"
                        f"Make sure you have diffusers >= 0.31 with Wan2.1 "
                        f"support installed."
                    )

            # Memory optimizations
            if hardware.cpu_offload:
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            if hardware.vae_slicing:
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass

            # Configure LoRA on the transformer / unet
            from peft import LoraConfig, get_peft_model

            target_model = getattr(pipe, "transformer", None)
            if target_model is None:
                target_model = getattr(pipe, "unet", None)
            if target_model is None:
                raise RuntimeError(
                    "Cannot find transformer or UNet in the video pipeline")

            lora_config = LoraConfig(
                r=c.lora_rank,
                lora_alpha=c.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            target_model = get_peft_model(target_model, lora_config)
            target_model.to("cuda", dtype=torch.bfloat16)

            if c.gradient_checkpointing:
                target_model.enable_gradient_checkpointing()

            trainable = sum(
                p.numel() for p in target_model.parameters()
                if p.requires_grad)
            total_p = sum(p.numel() for p in target_model.parameters())
            self._log(
                f"Trainable parameters: {trainable:,} / {total_p:,} "
                f"({100*trainable/total_p:.1f}%)")

            # Optimizer: respect user selection (#11)
            optimizer = self._create_optimizer(
                target_model.parameters(), c)

            # Proper noise scheduler (#4)
            try:
                from diffusers import DDPMScheduler
                noise_scheduler = DDPMScheduler.from_pretrained(
                    resolved_model, subfolder="scheduler")
                self._log("Using DDPMScheduler from model config")
            except Exception:
                from diffusers import DDPMScheduler
                noise_scheduler = DDPMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="scaled_linear")
                self._log("Using default DDPMScheduler")

            total_steps = c.max_train_steps
            step = 0

            self._log("Video LoRA training started…")
            self._log(
                "Each step processes one video clip through the video "
                "VAE and 3D transformer.")

            import cv2
            from PIL import Image
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize((c.resolution, c.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

            while step < total_steps and not self.is_cancelled:
                for vid_path in video_paths:
                    if self.is_cancelled or step >= total_steps:
                        break

                    try:
                        # Read a short clip
                        cap = cv2.VideoCapture(
                            str(vid_path), cv2.CAP_FFMPEG)
                        if not cap.isOpened():
                            cap = cv2.VideoCapture(str(vid_path))
                        if not cap.isOpened():
                            continue

                        frames = []
                        max_clip = min(49, 16 * 3)  # ~3s at 16 fps
                        fi = 0
                        while len(frames) < max_clip:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if fi % 2 == 0:  # subsample for memory
                                rgb = cv2.cvtColor(
                                    frame, cv2.COLOR_BGR2RGB)
                                pil = Image.fromarray(rgb)
                                frames.append(transform(pil))
                            fi += 1
                        cap.release()

                        if len(frames) < 2:
                            continue

                        clip = torch.stack(frames).unsqueeze(0)
                        clip = clip.to("cuda", dtype=torch.bfloat16)

                        # Load caption
                        caption_path = vid_path.with_suffix(".txt")
                        if caption_path.exists():
                            caption = caption_path.read_text(
                                encoding="utf-8").strip()
                        else:
                            caption = c.instance_prompt or "a video"

                        # Encode caption
                        tokenizer = getattr(pipe, "tokenizer", None)
                        text_enc = getattr(pipe, "text_encoder", None)
                        if tokenizer and text_enc:
                            with torch.no_grad():
                                tokens = tokenizer(
                                    caption, padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt",
                                ).input_ids.to("cuda")
                                hidden = text_enc(tokens)[0]
                        else:
                            hidden = None

                        # Encode clip through video VAE
                        vae = getattr(pipe, "vae", None)
                        if vae:
                            with torch.no_grad():
                                vid_in = clip.permute(0, 2, 1, 3, 4)
                                latents = vae.encode(
                                    vid_in).latent_dist.sample()
                                sf = getattr(
                                    vae.config, "scaling_factor", 0.18215)
                                latents = latents * sf
                        else:
                            latents = clip

                        # Noise + predict (#4: proper scheduler)
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (1,), device="cuda").long()

                        noisy = noise_scheduler.add_noise(
                            latents, noise, timesteps)

                        kw = {"encoder_hidden_states": hidden} \
                            if hidden is not None else {}
                        pred = target_model(
                            noisy, timesteps, **kw).sample

                        loss = torch.nn.functional.mse_loss(
                            pred, noise)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)  # (#2)

                        step += 1
                        self.current_step = step
                        self.current_loss = loss.item()
                        self.current_lr = c.learning_rate

                        if step % 5 == 0:
                            self._log(
                                f"Step {step}/{total_steps}  |  "
                                f"Loss: {loss.item():.4f}  |  "
                                f"Clip: {vid_path.name} "
                                f"({len(frames)} frames)")
                            if self.on_progress:
                                self.on_progress(
                                    step, total_steps,
                                    loss.item(), c.learning_rate)

                        if step % c.save_every_n_steps == 0:
                            save_path = Path(
                                c.output_dir
                            ) / f"{c.output_name}_step{step}"
                            save_path.mkdir(parents=True, exist_ok=True)
                            target_model.save_pretrained(
                                str(save_path))
                            self._log(
                                f"Checkpoint saved: {save_path}")

                    except torch.cuda.OutOfMemoryError:
                        self._log(
                            "VRAM OOM on video clip — clearing cache…")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    except Exception as e:
                        self._log(
                            f"Error processing {vid_path.name}: {e}")
                        continue

            # Final save
            if not self.is_cancelled:
                final_path = Path(c.output_dir) / c.output_name
                final_path.mkdir(parents=True, exist_ok=True)
                target_model.save_pretrained(str(final_path))
                self._log(
                    f"Video LoRA training complete! "
                    f"Saved to: {final_path}")
                if self.on_complete:
                    self.on_complete(str(final_path))
            else:
                self._log("Training cancelled by user.")

        except Exception as e:
            import traceback as _tb
            self._write_error_report(e, _tb.format_exc())
            self._log(f"Video training error: {e}")
            if self.on_error:
                self.on_error(str(e))
        finally:
            self.is_running = False
            self.vram_monitor.stop()
            self.elapsed_seconds = time.time() - self.start_time
            from core.gpu_utils import flush_gpu_memory, log_vram_snapshot
            try:
                for name in ("target_model", "vae", "text_enc",
                             "pipe", "optimizer"):
                    obj = locals().get(name)
                    if obj is not None:
                        try:
                            if hasattr(obj, "to"):
                                obj.to("cpu")
                        except Exception:
                            pass
                        del obj
            except Exception:
                pass
            flush_gpu_memory()
            log_vram_snapshot("post-video-training")

    # ── progress parsing (kohya-ss output) ───────────────────────────────

    def _parse_progress(self, line: str):
        try:
            if "it," in line and ("loss" in line or "lr" in line):
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        step_part = parts[1].strip()
                        if "/" in step_part:
                            current = step_part.split("/")[0].strip()
                            total = step_part.split("/")[1].split()[0]
                            self.current_step = int(current)
                            self.total_steps = int(total)

                if "loss=" in line or "loss:" in line:
                    import re
                    m = re.search(r'loss[=:]\s*([\d.e-]+)', line)
                    if m:
                        self.current_loss = float(m.group(1))

                if "lr=" in line or "lr:" in line:
                    import re
                    m = re.search(r'lr[=:]\s*([\d.e-]+)', line)
                    if m:
                        self.current_lr = float(m.group(1))

                if self.on_progress:
                    self.on_progress(
                        self.current_step, self.total_steps,
                        self.current_loss, self.current_lr,
                    )

            elif "epoch" in line.lower() and "/" in line:
                import re
                m = re.search(r'epoch\s*(\d+)/(\d+)', line, re.IGNORECASE)
                if m:
                    self._log(f"Epoch {m.group(1)}/{m.group(2)}")
        except Exception:
            pass

    # ── control ──────────────────────────────────────────────────────────

    def cancel(self):
        self.is_cancelled = True
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass

    # ── latent caching (#16) ─────────────────────────────────────────

    def _cache_latents(self, image_paths, vae, transform, vae_scaling_factor):
        """Pre-encode all training images through the VAE and cache as .pt files.
        Returns a dict mapping image_path -> cached .pt path."""
        cache_dir = Path(self.config.dataset_dir) / "latents_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = {}
        self._log(f"Caching latents for {len(image_paths)} images...")

        for i, img_path in enumerate(image_paths):
            pt_name = f"{img_path.stem}.pt"
            pt_path = cache_dir / pt_name
            if pt_path.is_file():
                cached[img_path] = pt_path
                continue
            try:
                from PIL import Image as _PILImage
                img = _PILImage.open(img_path).convert("RGB")
                pixel_values = transform(img).unsqueeze(0).to(
                    "cuda", dtype=torch.float16)
                with torch.no_grad():
                    latents = (vae.encode(pixel_values).latent_dist.sample()
                               * vae_scaling_factor)
                torch.save(latents.cpu(), str(pt_path))
                cached[img_path] = pt_path
            except Exception as e:
                self._log(f"Warning: could not cache latents for {img_path}: {e}")

            if (i + 1) % 50 == 0:
                self._log(f"  Cached {i + 1}/{len(image_paths)} latents")

        self._log(f"Latent caching complete: {len(cached)}/{len(image_paths)} cached")
        return cached

    # ── optimizer factory (#11) ───────────────────────────────────────

    @staticmethod
    def _create_optimizer(params, config):
        """Create optimizer matching the user's config.optimizer choice.
        Falls back to AdamW if the requested optimizer isn't available."""
        name = config.optimizer.lower().strip()
        lr = config.learning_rate

        if name in ("adamw", "adamw_torch"):
            return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        if name == "adamw8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-4)
            except ImportError:
                logger.warning("bitsandbytes not installed — falling back to AdamW")
                return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        if name == "adafactor":
            try:
                from transformers.optimization import Adafactor
                return Adafactor(
                    params, lr=lr, scale_parameter=False,
                    relative_step=False, warmup_init=False)
            except ImportError:
                logger.warning("transformers not installed — falling back to AdamW")
                return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        if name == "prodigy":
            try:
                from prodigyopt import Prodigy
                return Prodigy(params, lr=1.0, weight_decay=1e-4)
            except ImportError:
                logger.warning("prodigyopt not installed — falling back to AdamW")
                return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        if name == "lion8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.Lion8bit(params, lr=lr, weight_decay=1e-4)
            except ImportError:
                logger.warning("bitsandbytes not installed — falling back to AdamW")
                return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

        # Unknown optimizer name — default to AdamW
        logger.warning(f"Unknown optimizer '{config.optimizer}' — using AdamW")
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    def _log(self, message: str):
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def get_eta(self) -> str:
        if self.current_step == 0 or not self.is_running:
            return "Calculating..."
        elapsed = time.time() - self.start_time
        steps_per_sec = self.current_step / elapsed
        remaining_steps = self.total_steps - self.current_step
        if steps_per_sec > 0:
            remaining_sec = remaining_steps / steps_per_sec
            hours = int(remaining_sec // 3600)
            minutes = int((remaining_sec % 3600) // 60)
            return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        return "Calculating..."
