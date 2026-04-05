# AI Art Studio — Developer Context
**Version:** 0.6.0  
**Last updated:** 2026-04-05  
**Repo:** https://github.com/Fizzolas/Ai_art_studio  
**Install target:** `D:\ai_art_studio\`  
**Platform:** Windows 11 primary, Linux forward-compatible  
**Hardware target:** RTX 4070 Laptop 8GB VRAM (scales to 4GB, CPU fallback)

---

## Project Layout

```
ai_art_studio/
├── main.py                      # Entry point — QApplication, first-run wizard, global exception hook, DPI awareness
├── setup.py                     # Auto-installer (CUDA detection, venv, selective optional components)
├── run.bat                      # Launcher (venv check, activate, log redirect, crash reporter)
├── requirements.txt             # Full dependency list (all new deps are optional)
├── VERSION                      # Plain-text version string (0.6.0)
├── CHANGELOG.md                 # Full version history
├── README.md                    # User-facing setup + usage guide
│
├── core/
│   ├── config.py                # All dataclasses + ConfigManager singleton
│   ├── dataset.py               # DatasetManager: scan, validate/convert, prepare_training_dir
│   ├── generation_queue.py      # Sequential generation job queue (GenerationQueue, GenerationJob)
│   ├── gpu_utils.py             # VRAM queries, flush_gpu_memory, disk space, hardware detection
│   ├── logger.py                # Centralized rotating logger
│   ├── model_downloader.py      # HuggingFace snapshot_download + ModelDownloadWorker
│   ├── video_utils.py           # UNIFIED video frame extraction (used by captioner + processor)
│   └── __init__.py
│
├── captioning/
│   ├── auto_caption.py          # AutoCaptioner: BLIP-2, WD Tagger, Florence-2, combined. Lazy-loads.
│   └── __init__.py
│
├── training/
│   ├── trainer.py               # DiffusersTrainer + KohyaTrainer, checkpoint resume, latent caching
│   └── __init__.py
│
├── generation/
│   ├── image_gen.py             # ImageGenerator: txt2img, img2img, LoRA, ControlNet, IP-Adapter, advanced params
│   ├── video_gen.py             # VideoGenerator: single-clip + long-video stitching, clip context captioning
│   ├── audio_gen.py             # AudioGenerator: MusicGen + AudioLDM2 scaffold, video sync, ffmpeg embed
│   ├── upscaler.py              # Real-ESRGAN wrapper with LANCZOS fallback
│   ├── utils.py                 # Shared: _load_with_offline_fallback
│   └── __init__.py
│
├── gui/
│   ├── main_window.py           # MainWindow (~3500+ lines) — all tabs, workers, signals
│   ├── setup_wizard.py          # First-run QWizard (3 pages: Welcome/hardware → Paths → Models)
│   ├── theme.py                 # DARK_THEME + LIGHT_THEME QSS stylesheets
│   ├── widgets.py               # LabeledSlider, LabeledCombo, LabeledCheck, PathSelector,
│   │                            #   CollapsibleSection, DropZone, make_scroll_panel, StatusCard
│   └── __init__.py
│
└── configs/
    ├── presets.py               # Generation and training presets
    ├── prompt_presets.py        # Style preset save/load (~/.ai_art_studio/prompt_presets.json)
    └── __init__.py
```

---

## Configuration System (`core/config.py`)

**Singleton:** `ConfigManager()` — thread-safe, auto-saves on every `update_and_save()`.  
**Config file:** `~/.ai_art_studio/settings.json` with atomic write + `.bak` fallback.

### Dataclasses

#### `HardwareProfile`
| Field | Default | Notes |
|---|---|---|
| `gpu_name` | RTX 4070 Laptop | Auto-detected on first run |
| `vram_total_mb` | 8188 | Auto-detected |
| `offload_mode` | "balanced" | "none"/"balanced"/"aggressive"/"cpu_only" |
| `gpu_device_index` | 0 | Multi-GPU index |
| `xformers`, `bf16`, `fp8` | True | Memory optimisations |
| `attention_slicing`, `vae_slicing`, `vae_tiling` | True | Auto-enabled if VRAM < 6GB |
| `sequential_offload` | False | |
| `max_vram_usage_mb` | 7500 | |

#### `TrainingConfig`
`base_model`, `model_type` ("sd15"/"sdxl"/"flux"/"wan21"), `training_type` ("lora"/"dreambooth_lora"/"textual_inversion"), `lora_rank/alpha`, `learning_rate`, `batch_size`, `max_train_steps`, `optimizer` (AdamW8bit/Adafactor/Prodigy/Lion8bit), `mixed_precision` ("bf16"), `cache_latents`, `cache_latents_to_disk`, `fp8_base`, `gradient_checkpointing`, `enable_bucket`, resolution bucketing fields, `dataset_dir`, `output_dir`, `output_name`

#### `GenerationConfig`
Standard: `img_width/height/steps/cfg_scale/sampler/seed/batch_count/batch_size/clip_skip/hires_fix*`  
Advanced image: `img_eta`, `img_tiling`, `img_karras_sigmas`, `img_rescale_cfg`, `img_aesthetic_score`, `img_negative_aesthetic_score`, `img_denoising_start/end`  
ControlNet: `controlnet_enabled`, `controlnet_model_id`, `controlnet_preprocessor`, `controlnet_strength`, `controlnet_guidance_start/end`, `controlnet_input_image`  
IP-Adapter: `ip_adapter_enabled`, `ip_adapter_model`, `ip_adapter_subfolder`, `ip_adapter_weight_name`, `ip_adapter_scale`, `ip_adapter_image`  
Video basic: `vid_width/height/frames/fps/steps/cfg_scale/seed/flow_shift/negative_prompt/format`  
Video advanced: `vid_motion_bucket_id`, `vid_noise_aug_strength`, `vid_decode_chunk_size`, `vid_overlap_frames`, `vid_clip_count`, `vid_sample_method`, `vid_tiling`  
Video stitching: `vid_caption_clips` (True), `vid_caption_sample_frames` (5)  
Output: `output_dir`, `auto_save`, `save_format`

#### `CaptioningConfig`
`method` ("wd_tagger"/"blip2"/"florence2"/"combined"), `wd_model`, `wd_threshold`, `blip2_model`, `florence_model`, `include_nsfw_tags`, `max_tags`, `caption_format`, `overwrite_existing`, `verbose_description`, `video_frame_interval`, `video_max_frames`, `pipeline_mode`, `keep_models_in_memory`

#### `AudioConfig` *(scaffold)*
`enabled`, `model`, `duration_seconds`, `sample_rate`, `guidance_scale`, `seed`, `output_format`, `output_dir`, `sync_to_video`, `video_path`

#### `AppConfig`
`hardware`, `training`, `generation`, `captioning`, `audio`, `theme` ("dark"/"light"), `pipeline_mode` ("image"/"video"), `last_dataset_dir`, `recent_models`, `window_geometry`

### Key methods
- `update_and_save(section, key, value)` — use `section="app"` for top-level AppConfig fields
- `config_file_exists()` — used for first-run wizard detection
- `get_offload_preset()` — returns dict for current offload mode
- `reset()` — destroys singleton (testing only)

---

## Logging System (`core/logger.py`)

```python
from core.logger import get_logger
logger = get_logger(__name__)
```

- Rotating file: max 10MB, 7-day retention → `logs/app_YYYYMMDD.log`
- Console: INFO+ only
- `set_log_level(level)` — live change from GUI log viewer
- `get_log_file()` / `get_log_dir()`
- Startup banner: Python, PyTorch, CUDA, GPU, VRAM logged on every launch
- Global `sys.excepthook` in `main.py` logs full traceback + shows dialog
- Structured error reports: `logs/errors/gen_error_*.json`, `train_error_*.json`

---

## GUI Architecture (`gui/main_window.py`)

### Tabs (in order)
| # | Tab | Key content |
|---|---|---|
| 0 | Dataset | Scan, convert/validate, caption, review captions, deduplicate |
| 1 | Training | LoRA training, presets, sample generation, checkpoint resume |
| 2 | Generate | txt2img, img2img CollapsibleSection (also standalone tab), LoRA stack, ControlNet, IP-Adapter, queue, A/B compare, advanced settings |
| 3 | img2img | Dedicated img2img tab with full params |
| 4 | Settings | Hardware/offload, model manager, log viewer, theme, VRAM budget |
| 5 | Gallery | Grid thumbnails, sort, filter |
| 6 | Models | Model browser — 12 curated models with download/status |

### Widget conventions (CRITICAL — never get these wrong)
| Widget | Signal | Value method |
|---|---|---|
| `LabeledCombo` | `currentTextChanged` (str) | `.currentText()` |
| `LabeledCheck` | `toggled` (bool) | `.isChecked()` |
| `LabeledSlider` | `valueChanged` | `.value()` |
| `PathSelector` | — | `.path()` |
| `CollapsibleSection` | — | `.addWidget(w)` |

### Config auto-save wiring
Every setting widget connects to `self.cfg.update_and_save(section, key, value)`. No explicit Save button needed for any setting.

### Workers (all QThread-based, all have `error = pyqtSignal(str)`, all `run()` wrapped in try/except)
| Worker | File | Purpose |
|---|---|---|
| `DatasetWorker` | main_window | scan / convert / caption |
| `ModelLoadWorker` | main_window | load generation pipeline |
| `ModelDownloadWorker` | model_downloader | HuggingFace download with progress |
| `GenerationWorker` | main_window | image or video generation (single or long-video) |
| `TrainingWorker` | main_window | training loop |
| `CaptionWorker` | main_window | auto-captioning pipeline |

### Status bar (4 zones)
`status_activity` (left, stretches) | `status_progress` (inline bar, hidden when idle) | `status_vram` | `status_temp`  
Use `self._set_status(msg, progress)` — `progress=-1` hides bar, `0` = indeterminate, `1-100` = determinate.

### Toast notifications
`self._notify(message, level)` — level: "info"/"success"/"warning"/"error"  
Fade-out overlay in bottom-right. Use instead of QMessageBox for non-blocking feedback.

### Undo/redo
Ctrl+Z / Ctrl+Y, 50-item stack. Register widgets in `self._param_widgets` dict.

### Key `__init__` state
- `self._last_video_path = ""` — set in `_on_gen_complete` when a video is saved
- `self._review_items = []`, `self._review_index = 0` — caption review state
- `self._prompt_queue_list = []`, `self._prompt_queue_index = 0` — prompt queue state
- `self.audio_generator` — NOT created in `__init__`, created lazily in `_load_audio_model()`; always guard with `hasattr(self, 'audio_generator')`
- `self.image_generator`, `self.video_generator` — created in `__init__`

---

## Dataset Pipeline (`core/dataset.py`)

### `DatasetItem` fields
`original_path` (never modified), `converted_path`, `caption_text`, `file_type` ("image"/"video"/"animated"), `is_valid`

### Scan
`scan_directory(path, progress_callback=None)` — emits every 100 files: `callback(count, 0, filename)`

### Supported formats
Images: PNG, JPG, JPEG, WebP, BMP, TIFF, HEIC, AVIF, PSD, TGA + all PIL types  
Animated: GIF, APNG, animated WebP  
Video: see `core/video_utils.VIDEO_EXTENSIONS` (.mp4, .mov, .avi, .mkv, .webm, .flv, .wmv, .m4v, .ts, .mts, .m2ts, .3gp, .ogv, .vob, .divx, .xvid)

### Convert & Validate
`validate_and_convert(max_resolution=1024)` — RGB, LANCZOS resize if > max_res, round to 8px, save PNG. Images: `ThreadPoolExecutor(4)`. Video: serial, calls `core.video_utils.extract_frames()`.

### Video Frame Sync (unified)
Both dataset processor and captioner call `core.video_utils.extract_frames()` with identical params → same output dir, same filename scheme (`{stem}_{md5hash8}_f{seq:05d}.png`) → captions always match training frames.

### Training Dir Preparation
`prepare_training_dir()` assembles kohya-ss folder. Matches captions from both `captions/` and `captions/video_frames/` by stem.

### Deduplication
`find_duplicates(threshold=8)` — pHash via `imagehash`. Graceful if not installed.

---

## Captioning (`captioning/auto_caption.py`)

**Lazy loading** — models only load on first `caption_dataset()` call via `_ensure_models_loaded()`.  
**Resume** — `caption_progress.json` tracks completed files.  
**CPU fallback** — `_create_onnx_session()` tries `CUDAExecutionProvider` → `CPUExecutionProvider`.  
**BLIP-2 sanity check** — `_is_degenerate_caption()` detects garbage, falls back to tags.  
**Model caching** — `keep_models_in_memory` config toggle.

---

## Training (`training/trainer.py`)

`DiffusersTrainer` (diffusers + PEFT) and `KohyaTrainer` (shells to kohya-ss sd-scripts).  
Features: latent caching to disk, checkpoint resume, cancel saves last-good checkpoint, VRAM suggestions before training, sample generation mid-run.

---

## Image Generation (`generation/image_gen.py`)

### `ImageGenerator.generate()` full signature
```python
generate(prompt, negative_prompt, width, height, steps, cfg_scale, sampler, seed,
         batch_size, batch_count, loras, init_image, strength,
         controlnet_model, controlnet_input, controlnet_strength,
         controlnet_guidance_start, controlnet_guidance_end, controlnet_preprocessor,
         ip_adapter_image, ip_adapter_scale,
         clip_skip, eta, tiling, karras_sigmas, rescale_cfg,
         aesthetic_score, negative_aesthetic_score, denoising_start, denoising_end,
         embeddings, callback)
```

Advanced params applied only when non-default (safe for existing callers with defaults).

### OOM stepped fallback
`_generate_with_fallback(gen_kwargs, w, h)` — tries 100% → 75% → 50% → 256×256.  
Each step calls `_bump_offload_mode()` (safe at "cpu_only" — no-op) and emits toast.

### ControlNet
Only loads when `controlnet_enabled=True` AND valid input image provided. Preprocessing: canny (cv2), depth (transformers), openpose (controlnet_aux optional), none.

### IP-Adapter
Only loads when `ip_adapter_enabled=True` AND `ip_adapter_image` exists.

---

## Video Generation (`generation/video_gen.py`)

### Single-clip: `generate()`
Standard params: prompt, negative_prompt, width, height, num_frames, fps, steps, cfg_scale, seed, flow_shift, callback.  
WAN frame count auto-corrected to 4n+1. OOM raises RuntimeError with actionable message.

### Long-video: `generate_long_video()`
```python
generate_long_video(prompt, negative_prompt, width, height,
                    frames_per_clip, clip_count, overlap_frames, fps,
                    steps, cfg_scale, seed, flow_shift, output_dir,
                    caption_clips=True, caption_sample_frames=5, callback)
```

**Clip loop:**
1. Build `effective_prompt` = original prompt + ", continuing from: {prev_clip_context}" for clips > 0
2. Blend last frame of previous clip into gen_kwargs via `_blend_last_frame()`
3. Generate clip, `torch.cuda.empty_cache()` between clips
4. Caption the clip via `_caption_clip(frames, sample_count)` → store as `prev_clip_context`

**Peak VRAM = one clip's worth, regardless of total clip count.**

### Clip context captioning (`_caption_clip()`)
Samples N evenly-spaced frames → captions each with best available model (BLIP-2 → Florence-2 → WD Tagger) → summarises sequence:
- Strategy 1: `flan-t5-small` LLM writes a coherent arc description
- Strategy 2: deterministic template ("beginning with X, then Y, ending with Z")

All captioning is try/except guarded — never aborts generation on failure.

### Clip stitching (`_stitch_clips()`)
Linear crossfade at overlap regions. Handles numpy arrays and PIL Images. Single-clip passthrough.

### Frame blending (`_blend_last_frame()`)
Encodes last frame as soft img conditioning for next clip. Broad except — returns gen_kwargs unchanged on any failure.

---

## Audio Generation (`generation/audio_gen.py`) — SCAFFOLD

**Status:** Architecture complete, backends wired, no audio model bundled. MusicGen + AudioLDM2 load on-demand if transformers/diffusers are installed. Returns silence stub if no model loaded.

### `AudioGenerator` API
```python
load_model(model_id, on_progress)   # loads MusicGen or AudioLDM2
generate(prompt, duration_seconds, sample_rate, guidance_scale, seed, callback)
    → (np.ndarray, int)              # audio array + sample rate
generate_for_video(prompt, video_path, ...)   # auto-matches video duration
embed_audio_in_video(audio, sr, video_path, output_path)   # ffmpeg mux
save_audio(audio, sr, output_dir, format, prefix)
unload()
```

Supported models: `facebook/musicgen-small/medium/large`, `CVSSP/audioldm2`, `CVSSP/audioldm2-music`, `stabilityai/stable-audio-open-1.0`

GUI section in generation tab: model selector, prompt, duration, CFG, sync-to-video toggle, format selector. Audio section is collapsible and scaffold-labelled.

---

## Upscaler (`generation/upscaler.py`)

`upscale_image(image, scale)` — tries Real-ESRGAN, falls back to LANCZOS.  
`is_realesrgan_available()` — bool check.  
GUI: "2x (AI)"/"4x (AI)"/"2x (Lanczos)"/"4x (Lanczos)" in upscale section.

---

## Generation Queue (`core/generation_queue.py`)

`GenerationQueue` with `GenerationJob` items. Thread-safe. `add_job()`, `cancel_job()`, `pause()`, `resume()`, `clear()`. GUI panel in generation tab with job list, pause/cancel per job.

---

## Prompt Style Presets (`configs/prompt_presets.py`)

Saved to `~/.ai_art_studio/prompt_presets.json`. 5 built-in defaults. GUI: combo above prompt box, apply/save/delete controls.

---

## Model Browser Tab (tab 6)

12 curated models: SD1.5, SDXL Base, SDXL VAE, FLUX.1-dev, FLUX.1-schnell, Wan 2.1, ControlNet Canny/Depth SD1.5, ControlNet Canny SDXL, IP-Adapter SD1.5, BLIP-2, WD Tagger.  
Filter by tag (Image/Video/ControlNet/Captioning). Status checked against HF cache.

---

## Model Manager (Settings Tab)

6 managed models with Download/Remove cards: BLIP-2, WD Tagger, Florence-2, SDXL Base, SDXL VAE, Wan 2.1. Per-model progress bars.

---

## First-Run Wizard (`gui/setup_wizard.py`)

Runs when `~/.ai_art_studio/settings.json` does not exist. 3 pages: Welcome (hardware detection + offload preset recommendation) → Paths → Model Selection. Hardware auto-sets offload mode and VAE tiling if VRAM < 6GB.

---

## Installation

### `setup.py`
1. Detect Python (warn < 3.10 or > 3.12), detect CUDA via nvidia-smi
2. Select torch variant: `+cu121` (CUDA 12.x), `+cu118` (CUDA 11.x), CPU
3. Create/reuse venv, install in order: torch → xformers → diffusers → rest
4. Interactive optional components menu:
   - [1] BLIP-2 captioning
   - [2] WD Tagger (ONNX)
   - [3] Florence-2
   - [4] Video (opencv)
   - [5] Deduplication (imagehash)
   - [6] All / [7] Skip
5. Logs to `setup_log.txt`

### `run.bat`
Checks venv, activates, reads VERSION, pipes to `logs/app.log`, prints last 20 lines on crash.

---

## Key Dependencies

| Package | Purpose | Required |
|---|---|---|
| torch + CUDA | Core ML | Yes |
| diffusers ≥ 0.31 | Pipelines (SDXL, Flux, WAN, AnimateDiff) | Yes |
| transformers ≥ 4.40 | BLIP-2, Florence-2, text encoders, MusicGen | Yes |
| peft ≥ 0.10 | LoRA via diffusers | Yes |
| PyQt6 ≥ 6.6 | GUI | Yes |
| Pillow ≥ 10 | Image I/O | Yes |
| xformers | Memory-efficient attention | Optional |
| opencv-python | Video frame extraction | Optional |
| onnxruntime-gpu | WD Tagger ONNX | Optional |
| imagehash | pHash deduplication | Optional |
| realesrgan | AI upscaling | Optional |
| controlnet_aux | OpenPose preprocessor | Optional |
| soundfile / scipy | Audio file I/O | Optional |
| safetensors ≥ 0.4 | Model weight format | Yes |
| psutil, GPUtil | System/GPU monitoring | Yes |
| pillow-heif | HEIC support | Optional |

---

## Runtime Folder Structure

```
D:\ai_art_studio\
├── venv\
├── logs\
│   ├── app_YYYYMMDD.log     # Rotating, 10MB max, 7-day retention
│   └── errors\              # gen_error_*.json, train_error_*.json
└── setup_log.txt

~\.ai_art_studio\            # User data
├── settings.json            # Auto-saved config
├── settings.json.bak        # Backup before each write
├── models\                  # LoRA outputs
├── outputs\                 # Generated images/videos/audio
├── datasets\                # Dataset work dirs
├── captions\
├── logs\
└── prompt_presets.json      # Style presets

~\.cache\huggingface\hub\    # HF model cache (set via HF_HOME)
```

---

## Known Limitations

- **ControlNet inference** — GUI + pipeline swap complete. ControlNet models must be downloaded separately via the Model Browser.
- **Real-ESRGAN** — requires `pip install realesrgan basicsr`. Falls back to LANCZOS if not installed.
- **OpenPose preprocessor** — requires `pip install controlnet-aux`. Falls back to "none" if not installed.
- **Audio generation** — scaffold only. MusicGen requires `transformers>=4.31`. AudioLDM2 requires `diffusers>=0.21`. No model is bundled.
- **Video LoRA training** — video generation works, video model LoRA training not yet implemented.
- **Multi-GPU** — `gpu_device_index` config field exists, no actual multi-GPU dispatch.
- **Wan 2.1 img2vid** — `_blend_last_frame` uses soft conditioning (not true img2vid). Native img2vid support pending diffusers update.
- **flan-t5-small** for clip summarisation — ~300MB, downloads on first long-video run if not cached. Falls back to template if unavailable.
