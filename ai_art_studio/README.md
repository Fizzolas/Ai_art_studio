# AI Art Studio

Local AI image and video training + generation, optimized for your RTX 4070 Laptop GPU (8GB VRAM).

## Prerequisites

- **Python** 3.10–3.12 (3.13 has limited onnxruntime-gpu support)
- **NVIDIA GPU** with CUDA 11.8+ (8 GB VRAM minimum, 12 GB+ recommended)
  - Apple Silicon (MPS) is partially supported for generation; training requires CUDA
- **Git** (for cloning sd-scripts if using the kohya-ss training backend)
- **~20 GB disk space** for base models, plus your dataset and outputs

## Quick Start

```bash
# 1. Run automated setup (creates venv, installs everything)
python setup.py

# 2. Launch the app
run.bat
# or manually:
venv\Scripts\python.exe main.py
```

## First Run

On first launch, the application will:

1. Create the config directory at `~/.ai_art_studio/`
2. Open with default settings tuned for an 8 GB VRAM GPU
3. Prompt you to download base models when needed (automatic via Hugging Face Hub)

**Typical workflow:**
1. **Dataset tab** — Import your training images/videos
2. **Auto-caption** — Run captioning (WD Tagger + BLIP-2)
3. **Training tab** — Configure and start LoRA training
4. **Generation tab** — Load your LoRA and generate images/videos

## Architecture

```
ai_art_studio/
├── main.py              # Entry point
├── setup.py             # Automated installer
├── run.bat              # Windows launcher
├── requirements.txt     # Dependencies
├── core/
│   ├── config.py        # Central config with auto-save
│   └── dataset.py       # Dataset management (all formats)
├── captioning/
│   └── auto_caption.py  # WD Tagger + BLIP-2/Florence-2
├── training/
│   └── trainer.py       # LoRA training engine
├── generation/
│   ├── image_gen.py     # Image generation (SD1.5/SDXL/FLUX)
│   └── video_gen.py     # Video generation (WAN2.1/AnimateDiff)
└── gui/
    ├── main_window.py   # Main application window
    ├── widgets.py        # Custom reusable widgets
    └── theme.py          # Dark theme stylesheet
```

## Features

### Dataset Tab
- Drag folders or pick individual files
- Supports **every image format**: PNG, JPG, GIF, WebP, HEIC, AVIF, BMP, TIFF, PSD, DNG, CR2, NEF, TGA, EXR, HDR, JXL, and dozens more
- Supports **every video format**: MP4, AVI, MOV, MKV, WebM, FLV, MTS, and more
- Animated GIF/APNG/WebP frame extraction
- Video keyframe extraction
- Auto-converts everything to training-compatible format
- Preview with caption display

### Auto-Captioning (Dataset Coherence System)
This is what makes your diverse dataset work together:

1. **WD Tagger v3** — Identifies specific visual features, character details, poses, compositions using booru-style tags
2. **BLIP-2 / Florence-2** — Generates natural language descriptions for scene context
3. **Content Analyzer** — Categorizes images (solo, multi-character, comic, scene) for format-appropriate captions
4. **Combined Mode** — Merges tags + natural language so the model understands:
   - What characters look like
   - What's happening in the scene
   - How different images in your dataset relate to each other

### Training Tab
- **SD 1.5**: Full LoRA training, ~6GB VRAM
- **SDXL**: LoRA with fused backward pass, ~8-10GB VRAM
- **FLUX**: LoRA with split_mode + FP8, fits in 8GB at 512x512
- Real-time loss/LR/VRAM monitoring
- Full hyperparameter control
- Multiple optimizers: Adafactor, AdamW8bit, Prodigy, Lion
- Automatic checkpoint saving

### Generation Tab
- Image: SD 1.5, SDXL, FLUX
- Video: WAN 2.1, AnimateDiff
- 12+ samplers with Karras variants
- Hi-res fix (upscale + refine)
- Full LoRA loading with adjustable weight
- Generation history with thumbnails
- Auto-save with metadata

### Settings Tab
- **4 offloading presets** tuned for 8GB VRAM:
  - `none` — All on GPU (needs 12GB+)
  - `balanced` — Smart CPU offload (recommended)
  - `aggressive` — Sequential offload (slow but fits anything)
  - `cpu_only` — Maximum VRAM savings
- Granular transformer/text encoder offload percentages
- VAE slicing, tiling, attention slicing toggles
- FP8 base model support
- xformers toggle
- All settings auto-save on change

## Supported Models

| Type | Model | Training | Generation | Notes |
|------|-------|----------|------------|-------|
| Image | Stable Diffusion 1.5 | LoRA | Yes | ~6 GB VRAM |
| Image | SDXL | LoRA | Yes | ~8–10 GB VRAM |
| Image | FLUX | LoRA | Yes | FP8 + split_mode for 8 GB |
| Video | WAN 2.1 | — | Yes | Aggressive offload recommended |
| Video | AnimateDiff | — | Yes | SD 1.5–based motion |

**Captioning models** (auto-downloaded on first use):
- WD Tagger v3 (ONNX) — booru-style tag prediction
- BLIP-2 / Florence-2 — natural-language captioning

## Known Limitations

- **VRAM**: 8 GB is the practical minimum. Some model/resolution combinations may OOM; the app retries at reduced resolution automatically.
- **Training**: Only LoRA training is supported through the built-in trainer. Full fine-tuning requires external tools.
- **Apple Silicon**: MPS is supported for image generation only. Training requires an NVIDIA GPU with CUDA.
- **Python 3.13**: `onnxruntime-gpu` does not yet ship CUDA wheels for 3.13; WD Tagger will fall back to CPU.
- **Video generation**: High frame counts or resolutions require aggressive CPU offloading and are slow on 8 GB cards.
- **Windows path length**: Long dataset paths may hit the 260-character limit; the app shortens internal hashes to help, but keep dataset folders shallow.

## Hardware Optimization Notes (RTX 4070 Laptop 8GB)

Your setup: i7-13620H, 32GB DDR5 5600, RTX 4070 Laptop 8GB, CUDA 12.4

**Training recommendations:**
- SD 1.5 LoRA: 512x512, batch 1, Adafactor — fits easily
- SDXL LoRA: 512x512, batch 1, Adafactor, fp8_base, cache_latents — tight but works
- FLUX LoRA: 512x512, batch 1, split_mode + fp8 — ~7.5GB VRAM
- Always use `cache_latents_to_disk` and `cache_text_encoder_outputs_to_disk`
- 32GB RAM is plenty for aggressive CPU offloading

**Generation recommendations:**
- SDXL: 768x768 with balanced offloading works well
- FLUX: 512x512-768x768 with model CPU offload
- Video (WAN 2.1): 480x320, 49 frames with aggressive offload
- Use `balanced` offload mode as default, switch to `aggressive` if OOM

## Config Location

All settings, models, and outputs are stored in:
```
%USERPROFILE%\.ai_art_studio\
├── settings.json    # All app settings (auto-saved)
├── models/          # Trained LoRAs
├── outputs/         # Generated images/videos
├── datasets/        # Processed datasets
├── captions/        # Caption logs
├── cache/           # Latent cache
└── logs/            # Application logs
```
