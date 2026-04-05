## [0.6.0] — 2026-04-05

### Full End-to-End Audit (Part A)
- Traced all user flows (startup, dataset, training, generation, settings, gallery) and fixed broken widget references
- Removed duplicate img2img CollapsibleSection from generation tab (kept standalone tab)
- Verified all worker error signals connected to visible handlers
- Confirmed _review_items and _review_index initialized in __init__

### Advanced Image Generation Settings (Part B)
- Added "Advanced Image Settings" CollapsibleSection with: DDIM Eta, Seamless Tiling, Karras Sigmas, CFG Rescale, Aesthetic Score (SDXL), Negative Aesthetic Score, Denoise Start/End
- All widgets wired to config auto-save
- Advanced params (eta, tiling, karras_sigmas, guidance_rescale, aesthetic_score) passed through to image_gen.generate()

### Advanced Video Generation Settings (Part C)
- Added video resolution preset buttons (480p, 720p, 512², 480²) with portrait variants
- Added video negative prompt text area and output format selector (mp4/gif/webm)
- Added "Advanced Video Settings" CollapsibleSection with: Flow Shift (WAN), Decode Chunk Size, Clip Overlap Frames, Clip Count, Spatial Tiling, Sampling Method
- All advanced video config fields added to GenerationConfig with auto-save wiring

### Long Video via Clip Stitching (Part D)
- `generate_long_video()` — generates N clips sequentially, each seeded from the last frame of the previous clip, with crossfade overlap stitching. Peak VRAM equals one clip's worth.
- `_blend_last_frame()` — injects last frame as soft conditioning hint via VAE latent blending
- `_stitch_clips()` — concatenates clips with linear crossfade at overlap regions
- GUI routing: clip_count > 1 automatically uses long-video generation path with per-clip progress reporting

### Audio Generation Scaffold (Part E)
- New `AudioConfig` dataclass in config.py (model, prompt, duration, format, video sync options)
- New `generation/audio_gen.py` with full AudioGenerator class — MusicGen and AudioLDM2 backends, video duration detection, ffmpeg audio embedding, save with soundfile/scipy fallback
- GUI scaffold: "Audio Generation (Experimental)" CollapsibleSection with model selector, prompt, duration, guidance scale, video sync toggle, format selector, generate button
- AudioGenerator exported from generation/__init__.py

---

## [0.5.0] — 2026-04-05

### New Features (Batch G)
- **ControlNet support** (Task 1) — Full ControlNet integration with canny/depth/openpose preprocessors, per-model selection (SD1.5 + SDXL), guidance start/end sliders, live conditioning preview button, and auto-save config wiring
- **IP-Adapter style reference** (Task 2) — Style-guided generation using h94/IP-Adapter with reference image selector, scale slider, thumbnail preview, and informational tooltip
- **img2img standalone tab** (Task 3) — Dedicated tab with input image selector + preview, denoising strength slider, full parameter controls, and independent output display
- **Hardware auto-detection** (Task 4) — First-run wizard now detects GPU, VRAM, CUDA version; recommends offload mode with "Use Recommended" / "Choose Manually" options
- **Tiled VAE auto-enable** (Task 5) — Automatically enables VAE tiling, slicing, and attention slicing on GPUs with <6GB VRAM at startup
- **Stepped OOM fallback** (Task 6) — Generation retries at 75%, 50%, and 256×256 minimum resolution on OOM; auto-bumps offload mode and notifies user via toast
- **Lazy model loading** (Task 7) — Captioning models load on first use instead of at init; video_gen.py imports torch/diffusers lazily; removed eager onnxruntime check from startup
- **CPU fallback for ONNX** (Task 8) — WD Tagger ONNX session now tries CUDAExecutionProvider first, falls back to CPUExecutionProvider with logged warnings
- **Generation queue system** (Task 9) — Thread-safe job queue (core/generation_queue.py) with add/cancel/pause/clear; GUI panel with status icons and right-click cancel
- **Prompt style presets** (Task 10) — Save/load named style strings (configs/prompt_presets.py); 5 built-in presets (Photorealistic, Anime, Oil Painting, Portal 2, Dark Fantasy); Apply/Save/Delete buttons in generation tab
- **Model browser tab** (Task 11) — Curated registry of 12 recommended HuggingFace models with search filter, tag chips, download status detection, one-click download via ModelDownloadWorker
- **Real-ESRGAN upscaler** (Task 12) — AI upscaling with Real-ESRGAN (generation/upscaler.py); falls back to LANCZOS if not installed; upscale factor combo shows "AI" vs "Lanczos" options; availability note in GUI

---

## [0.4.0] — 2026-04-04

### Auto-Installation (Batch D)
- **Rewritten setup.py** — auto-detects Python version and CUDA version, creates venv, installs torch with correct CUDA variant, interactive selective-install menu for optional components (BLIP-2, WD Tagger, Florence-2, video support, deduplication)
- **Model Manager** — new Settings tab section with download/remove controls for 6 managed models (BLIP-2, WD Tagger, Florence-2, SDXL Base, SDXL VAE, Wan Video), per-model progress bars and live status
- **First-run wizard** — 3-page QWizard on first launch: Welcome → Paths → Model Selection, pre-downloads chosen models automatically
- **Rewritten run.bat** — venv check, auto-activate, stdout+stderr redirected to logs/app.log, crash reporter showing last 20 log lines

### Logging & Error Documentation (Batch E)
- **Centralized logger** (core/logger.py) — rotating file handler (10MB, 7-day retention), console handler, all modules migrated to use `get_logger(__name__)`
- **Global exception hook** — uncaught exceptions logged with full traceback + user-facing error dialog with log file path
- **Worker exception handling** — all QThread workers wrap run() in try/except, emit error signal, get logged with full traceback
- **Log viewer panel** — live-updating log panel in Settings tab with level filter, refresh, open-folder, and clear buttons
- **Structured error reports** — generation and training failures write JSON error reports to logs/errors/ with parameters and traceback
- **Startup banner** — logs Python version, PyTorch version, CUDA version, GPU name and VRAM on every launch

### GUI Polish (Batch F)
- **CRITICAL: Video frame sync fix** — unified core/video_utils.py replaces two separate frame extraction implementations; captioner and processor now extract identical frames to the same folder with the same naming scheme, so per-frame captions correctly match training images
- **Status bar overhaul** — 4-zone status bar: activity label (with inline progress bar), VRAM meter, GPU temperature
- **Toast notifications** — fade-in/fade-out overlay notifications for success/info/warning/error events
- **Tab icons** — emoji icon prefixes and improved tab bar styling (height, padding, active underline)
- **Dataset UX** — post-scan summary label (image/video/animated counts), color-coded list items (grey/blue/yellow/green/red by state), right-click context menu
- **Generation UX** — seed lock toggle, quick resolution preset buttons (512²→1216×832), history list thumbnail icons
- **Training UX** — quick-start use-case presets (Quick Test / Character / Style / Concept), ETA countdown display
- **Keyboard shortcuts dialog** — proper table in Help menu with 9 shortcuts

---

# Changelog

## [0.3.0] — 2026-03-15

### New Features (Batch C)
- **img2img mode** — Full image-to-image generation with input image selector and denoising strength slider (Task 3.2)
- **Dark/light theme toggle** — Complete light theme stylesheet with toggle in Settings tab (Task 3.7)
- **Undo/redo for parameters** — Ctrl+Z/Ctrl+Y undo stack for slider and combo changes (Task 3.9)
- **VRAM budget indicator** — Visual estimate of VRAM usage per model type and offload mode (Task 3.11)
- **Sample generation during training** — Preview training quality mid-run with configurable interval (Task 3.13)
- **Prompt queue / batch generation** — Enter multiple prompts (one per line) and generate all sequentially (Task 6.1)
- **A/B comparison view** — Side-by-side comparison of two generated images (Task 6.2)
- **Dataset deduplication** — Perceptual hash (pHash) based duplicate detection with imagehash (Task 6.4)
- **ControlNet / IP-Adapter GUI** — Model selector, preprocessor, control image input, and strength slider (Task 6.5)
- **Multi-LoRA stacking** — Load multiple LoRA models with individual weight control (Task 6.6)
- **Textual inversion embedding manager** — Add/remove embeddings for generation (Task 6.7)
- **Caption review/editor panel** — Review mode with prev/next navigation and inline editing (Task 6.8)
- **Training resume / checkpoint continuation** — Detect and resume from existing checkpoints (Task 6.9)
- **Automatic VRAM-based parameter suggestions** — Warn and auto-adjust when settings exceed VRAM (Task 6.10)
- **Output metadata viewer** — View prompt, seed, sampler, and other metadata for generated images (Task 6.11)
- **Standalone image upscaling** — Upscale images 2x or 4x with LANCZOS resampling (Task 6.12)
- **Gallery view for outputs** — New Gallery tab with grid view, sorting, and filtering (Task 6.14)
- **Video frame viewer** — View extracted frames with captions for video dataset items (Task 6.15)

### Performance (Batch C)
- **Dataset scan incremental feedback** — Progress callbacks emitted every 100 files during scan (Task 2.3)
- **Captioning model caching toggle** — Option to keep captioning models in memory between runs (Task 2.4)

### Documentation (Batch C)
- **Inline docstrings** — Added comprehensive docstrings to all 7 complex GUI methods (Task 8.2)

---

## [0.2.0] — 2026-03-15

### Bug Fixes
- **scan_files clears items** — Added `add_files` method to `DatasetManager` that appends without clearing existing items (Task 1.10)
- **Animated frame extraction hardcoded** — `_extract_animated_frames` now accepts a configurable `max_frames` parameter, wired through `validate_and_convert` (Task 1.11)
- **_ignore_patterns_for excludes .bin globally** — `.bin` is now only excluded for model types with `.safetensors` alternatives (sd15, sdxl, flux), preserving wan21/animatediff compatibility (Task 1.12)
- **No GPU flush after LoRA load/unload** — `flush_gpu_memory()` is now called at the end of `load_lora` and `unload_lora` in image_gen.py (Task 1.4)
- **Captioning overwrite check misses work_dir captions** — Skip check now also looks in `caption_dir` for existing `.txt` files (Task 1.6)
- **OOM retry reuses stale gen_kwargs** — OOM retry in `generate()` now rebuilds kwargs fresh with reduced resolution and no stale callback references (Task 1.9)
- **BLIP-2 degenerate output** — Added `_is_degenerate_caption` sanity check; falls back to tags-only when NL caption is garbage (Task 5.6)
- **prepare_training_dir always wipes train directory** — Now only removes the specific repeat-folder, preserving other contents in train_dir (Task 4.8)
- **Windows path length** — Shortened work_dir hash from 12 to 8 characters (Task 7.4)

### Performance
- **validate_and_convert processes serially** — Image processing now uses `ThreadPoolExecutor(max_workers=4)` for parallel conversion; video items remain serial (Task 2.5)
- **No latent caching in built-in trainer** — Added `_cache_latents` method that pre-encodes images through VAE and saves as `.pt` files, loaded during training instead of re-encoding each step (Task 2.8)

### Architecture / Code Quality
- **_load_with_offline_fallback duplicated** — Extracted to `generation/utils.py` and imported in both `image_gen.py` and `video_gen.py` (Task 4.1)
- **ConfigManager singleton not reset-safe** — Added `reset()` classmethod for testing (Task 4.4)
- **Missing __init__.py files** — Created `__init__.py` in core/, gui/, generation/, captioning/, training/, configs/ (Task 4.7)

### Robustness / Error Handling
- **No graceful handling of corrupted config** — `save()` now creates `.bak` backup; `load()` falls back to backup on corruption (Task 5.1)
- **No timeout on model downloads** — Added `HF_HUB_DOWNLOAD_TIMEOUT=300` environment variable before `snapshot_download` (Task 5.2)
- **No disk space check** — Added `check_disk_space_mb()` and `warn_if_low_disk()` to `gpu_utils.py` (Task 5.3)
- **No captioning resume** — `caption_dataset` now saves `caption_progress.json` tracking completed files; resumes from previous run automatically (Task 5.7)
- **Training cancel doesn't guarantee clean state** — Cancel now saves a "last_good" checkpoint before stopping (Task 5.8)

### Platform Compatibility
- **No macOS/Apple Silicon support path** — Added `get_device()` utility with cuda/mps/cpu detection (Task 7.1)
- **onnxruntime-gpu CUDA provider check** — Startup now warns if CUDA provider is unavailable for WD Tagger (Task 7.2)
- **No multi-GPU support** — Added `gpu_device_index` to `HardwareProfile` config (Task 7.3)

### Documentation
- Created `VERSION` file (0.2.0)
- Created this `CHANGELOG.md`
- Updated `README.md` with prerequisites, setup, first-run, supported models, and known limitations
