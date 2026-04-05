"""
Model availability checker and auto-downloader.
Uses huggingface_hub to check cache and download models on demand.

Defensive features:
  - Automatic retry (3 attempts) with exponential back-off
  - Resume-friendly downloads (HF hub resumes by default)
  - Offline resilience: if the network is unreachable but the model is
    already cached, load silently from cache
  - Graceful error messages with manual-download instructions
"""
import os
import time
import shutil
from pathlib import Path
from typing import Optional, Callable, Dict
from core.logger import get_logger

logger = get_logger(__name__)

# Default model IDs for each architecture
DEFAULT_MODELS: Dict[str, str] = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-schnell",
    "wan21": "Wan-AI/Wan2.1-T2V-14B",
    "animatediff": "runwayml/stable-diffusion-v1-5",
}

# Approximate download sizes for user-facing messages
APPROX_SIZES: Dict[str, str] = {
    "sd15": "~2 GB",
    "sdxl": "~6.5 GB (fp16)",
    "flux": "~12 GB",
    "wan21": "~28 GB",
    "animatediff": "~2.5 GB",
}

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 10  # seconds


def is_local_path(path: str) -> bool:
    """Check if a path points to a local file or directory."""
    if not path:
        return False
    p = Path(path)
    return p.exists() or path.endswith((".safetensors", ".ckpt", ".bin", ".pt"))


def is_hf_repo_id(path: str) -> bool:
    """Check if a string looks like a Hugging Face repo ID (org/name)."""
    if not path:
        return False
    if os.sep in path or path.endswith((".safetensors", ".ckpt", ".bin", ".pt")):
        return False
    parts = path.strip("/").split("/")
    return len(parts) == 2 and all(p for p in parts)


def resolve_model_path(model_path: str, model_type: str) -> str:
    """Resolve a model path: use as-is if local, fill in default HF repo if empty."""
    if not model_path or model_path.strip() == "":
        return DEFAULT_MODELS.get(model_type, DEFAULT_MODELS["sdxl"])

    # Local file/directory — use as-is
    if Path(model_path).exists():
        return model_path

    # Looks like a HF repo ID — use as-is (diffusers will download)
    if is_hf_repo_id(model_path):
        return model_path

    # Could be a local path that doesn't exist yet
    return model_path


def check_model_cached(repo_id: str) -> bool:
    """Check if a HF model is already in the local cache."""
    if not is_hf_repo_id(repo_id):
        return Path(repo_id).exists()

    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo_info in cache_info.repos:
            if repo_info.repo_id == repo_id:
                return True
    except Exception:
        pass

    return False


def get_download_size_label(model_type: str) -> str:
    """Return a human-readable approximate download size."""
    return APPROX_SIZES.get(model_type, "unknown size")


def _is_network_available() -> bool:
    """Quick check: can we reach huggingface.co?"""
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except (OSError, socket.timeout):
        return False


def ensure_model_available(
    model_path: str,
    model_type: str,
    on_status: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> str:
    """
    Make sure a model is available locally or in HF cache.
    Downloads from Hugging Face if necessary, with retry + resume.

    Args:
        model_path:  HF repo ID or local path (empty = use default for model_type)
        model_type:  One of sd15, sdxl, flux, wan21, animatediff
        on_status:   Callback for status messages
        on_progress: Callback for download progress (0.0 - 1.0)

    Returns:
        The resolved model path/repo ID ready for diffusers to load.
    """
    resolved = resolve_model_path(model_path, model_type)

    def _status(msg):
        logger.info(msg)
        if on_status:
            on_status(msg)

    # ── Local file or directory ──────────────────────────────────────
    if Path(resolved).exists():
        _status(f"Using local model: {resolved}")
        return resolved

    # ── HF repo ──────────────────────────────────────────────────────
    if is_hf_repo_id(resolved):
        # Check cache first
        if check_model_cached(resolved):
            _status(f"Model found in cache: {resolved}")
            return resolved

        # Check network before attempting download
        if not _is_network_available():
            _status(
                f"No internet connection and model '{resolved}' is not cached.\n"
                f"Connect to the internet, or download it manually:\n"
                f"  huggingface-cli download {resolved}"
            )
            raise RuntimeError(
                f"Model '{resolved}' is not cached and the network is unreachable.\n"
                f"Download it manually with: huggingface-cli download {resolved}"
            )

        # ── Download with retry ──────────────────────────────────────
        size_label = get_download_size_label(model_type)
        _status(f"Downloading {resolved} ({size_label})... This may take a while.")

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                from huggingface_hub import snapshot_download

                os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
                snapshot_download(
                    repo_id=resolved,
                    repo_type="model",
                    ignore_patterns=_ignore_patterns_for(model_type),
                    # HF hub ≥ 0.21 always resumes; no flag needed
                )
                _status(f"Download complete: {resolved}")
                return resolved

            except KeyboardInterrupt:
                raise
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    _status(
                        f"Download attempt {attempt}/{MAX_RETRIES} failed: {e}\n"
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    _status(f"Download failed after {MAX_RETRIES} attempts: {e}")

        # All retries exhausted — check if partial download landed in cache
        if check_model_cached(resolved):
            _status(
                f"Download had errors but model files are in cache. "
                f"Attempting to load from cache..."
            )
            return resolved

        raise RuntimeError(
            f"Could not download model '{resolved}' after {MAX_RETRIES} attempts.\n"
            f"Last error: {last_error}\n\n"
            f"You can download it manually:\n"
            f"  huggingface-cli download {resolved}\n\n"
            f"Or point to a local model path instead."
        )

    # ── Not a HF repo and doesn't exist locally ─────────────────────
    _status(f"Model not found: {resolved}")
    raise FileNotFoundError(
        f"Model not found: {resolved}\n\n"
        f"Enter a Hugging Face repo ID (e.g., {DEFAULT_MODELS.get(model_type, 'org/model')})\n"
        f"or a local path to a model directory."
    )


def _ignore_patterns_for(model_type: str) -> list:
    """Return HF download ignore patterns to skip unnecessarily large files."""
    patterns = [
        "*.msgpack",        # Old format duplicates
        "*.h5",             # TensorFlow weights
        "*.ot",             # ONNX duplicates
        "flax_model*",      # Flax/JAX weights
        "tf_model*",        # TensorFlow weights
        "model.fp32*",      # Full-precision duplicates
        "*.onnx",           # ONNX exports
        "*.onnx_data",
    ]

    # Only exclude .bin for model types that have .safetensors alternatives
    if model_type in ("sd15", "sdxl", "flux"):
        patterns.append("*.bin")

    if model_type == "sdxl":
        # Skip the monolithic 6.5 GB checkpoint — we only need the diffusers split files
        patterns.extend([
            "sd_xl_base_1.0.safetensors",
            "sd_xl_base_1.0_0.9vae.safetensors",
        ])
    elif model_type == "sd15":
        patterns.extend([
            "v1-5-pruned*.safetensors",
            "v1-5-pruned*.ckpt",
        ])

    return patterns


# ── Managed Models for Model Manager UI ──────────────────────────────────

MANAGED_MODELS = {
    "blip2": {
        "label": "BLIP-2 (2.7B)",
        "repo": "Salesforce/blip2-opt-2.7b",
        "size_gb": 5.5,
        "type": "captioning",
        "required": False,
    },
    "wd_tagger": {
        "label": "WD SwinV2 Tagger",
        "repo": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "size_gb": 0.35,
        "type": "captioning",
        "required": False,
    },
    "florence2": {
        "label": "Florence-2 Large",
        "repo": "microsoft/Florence-2-large",
        "size_gb": 1.6,
        "type": "captioning",
        "required": False,
    },
    "sdxl_base": {
        "label": "SDXL Base 1.0",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "size_gb": 6.9,
        "type": "generation",
        "required": False,
    },
    "sdxl_vae": {
        "label": "SDXL VAE (fp16-fix)",
        "repo": "madebyollin/sdxl-vae-fp16-fix",
        "size_gb": 0.16,
        "type": "generation",
        "required": False,
    },
    "wan_video": {
        "label": "Wan 2.1 Video (1.3B)",
        "repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "size_gb": 11.0,
        "type": "video",
        "required": False,
    },
}


def get_hf_cache_dir() -> Path:
    """Return the HuggingFace Hub cache directory."""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub"


def get_model_cache_path(repo_id: str) -> Optional[Path]:
    """Return the cache folder for a specific repo, or None if not found."""
    cache_dir = get_hf_cache_dir()
    # HF hub stores models as models--org--name
    folder_name = "models--" + repo_id.replace("/", "--")
    model_path = cache_dir / folder_name
    if model_path.exists():
        return model_path
    return None


def remove_model_cache(repo_id: str) -> bool:
    """Delete the cached model folder. Returns True on success."""
    path = get_model_cache_path(repo_id)
    if path and path.exists():
        try:
            shutil.rmtree(path)
            logger.info("Removed model cache: %s", path)
            return True
        except Exception as e:
            logger.error("Failed to remove model cache %s: %s", path, e)
            return False
    return False
