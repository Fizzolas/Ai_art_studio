"""
GPU memory management utilities.

Centralises the cleanup sequence that prevents the known memory-leak
patterns in diffusers when switching or unloading pipelines:

  1. remove_all_hooks()        — detach accelerate CPU-offload hooks
  2. pipe.to("cpu")            — move all tensors off GPU
  3. torch.cuda.synchronize()  — wait for async CUDA ops to finish
  4. del pipe (+ components)   — drop Python references
  5. gc.collect()              — run Python garbage collector
  6. torch.cuda.empty_cache()  — release CUDA cache back to the driver

References:
  - https://github.com/huggingface/diffusers/discussions/10936
  - https://github.com/huggingface/diffusers/issues/2284
"""
import gc
from typing import Optional
from core.logger import get_logger

logger = get_logger(__name__)


def _safe_import_torch():
    """Import torch lazily so this module doesn't crash at import time."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def deep_cleanup_pipeline(pipe, label: str = "pipeline"):
    """
    Aggressively free a diffusers pipeline and all GPU memory it holds.

    This is the nuclear option — after calling this, ``pipe`` must not
    be referenced again.
    """
    torch = _safe_import_torch()
    if pipe is None:
        return

    logger.info(f"Deep cleanup: {label}")

    # 1. Remove accelerate hooks (cpu_offload / sequential_offload)
    try:
        if hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
            logger.debug(f"  removed accelerate hooks from {label}")
    except Exception as e:
        logger.debug(f"  remove_all_hooks failed (non-fatal): {e}")

    # 2. Move entire pipeline to CPU (frees CUDA tensors)
    try:
        pipe.to("cpu")
    except Exception:
        # Some pipelines (sequential offload) don't support .to()
        # Attempt component-by-component fallback
        for attr in ("unet", "vae", "text_encoder", "text_encoder_2",
                      "transformer", "decoder"):
            comp = getattr(pipe, attr, None)
            if comp is not None:
                try:
                    comp.to("cpu")
                except Exception:
                    pass

    # 3. Synchronize CUDA before deleting (prevents dangling async ops)
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    # 4. Explicitly delete known heavy sub-components
    for attr in ("unet", "vae", "text_encoder", "text_encoder_2",
                  "transformer", "decoder", "scheduler"):
        comp = getattr(pipe, attr, None)
        if comp is not None:
            try:
                delattr(pipe, attr)
            except Exception:
                pass
            del comp

    # 5. Delete the pipeline object itself (caller must also drop its ref)
    del pipe

    # 6. Python GC + CUDA cache
    flush_gpu_memory()


def flush_gpu_memory():
    """Run gc.collect() + torch.cuda.empty_cache() safely."""
    torch = _safe_import_torch()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_free_vram_mb() -> float:
    """Return approximate free VRAM in MiB, or 0 if CUDA unavailable."""
    torch = _safe_import_torch()
    if torch is None or not torch.cuda.is_available():
        return 0.0
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        return (total - allocated) / (1024 ** 2)
    except Exception:
        return 0.0


def log_vram_snapshot(tag: str = ""):
    """Log a one-line VRAM snapshot for debugging."""
    torch = _safe_import_torch()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        logger.info(
            f"[VRAM {tag}] allocated={alloc:.0f} MB  "
            f"reserved={reserved:.0f} MB  total={total:.0f} MB  "
            f"free≈{total - alloc:.0f} MB"
        )
    except Exception:
        pass


def get_device() -> str:
    """Return the best available device: cuda, mps, or cpu."""
    torch = _safe_import_torch()
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_disk_space_mb(path: str = ".") -> float:
    """Return available disk space in MiB for the given path."""
    import shutil
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 2)
    except Exception:
        return float("inf")


def warn_if_low_disk(path: str, min_mb: float = 5120.0) -> str:
    """Return a warning string if disk space is below threshold, else empty."""
    free = check_disk_space_mb(path)
    if free < min_mb:
        return (f"Low disk space: only {free:.0f} MB free. "
                f"At least {min_mb:.0f} MB recommended.")
    return ""


def detect_best_offload_mode() -> str:
    """Return the best offload_mode string based on available VRAM."""
    torch = _safe_import_torch()
    if torch is None or not torch.cuda.is_available():
        return "cpu_only"
    try:
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        if vram_mb >= 12000:
            return "none"
        elif vram_mb >= 8000:
            return "balanced"
        elif vram_mb >= 4000:
            return "aggressive"
        else:
            return "cpu_only"
    except Exception:
        return "balanced"


def detect_hardware_profile() -> dict:
    """Return dict of detected hardware info."""
    torch = _safe_import_torch()
    info = {
        "gpu_name": "None",
        "vram_total_mb": 0,
        "cuda_version": "N/A",
        "recommended_offload": "cpu_only",
    }
    if torch is not None and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["vram_total_mb"] = props.total_memory // (1024 * 1024)
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["recommended_offload"] = detect_best_offload_mode()
        except Exception:
            pass
    return info


def apply_low_vram_defaults(config) -> bool:
    """If VRAM < 6GB, enable tiling/slicing. Returns True if changes were made."""
    torch = _safe_import_torch()
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        if vram_mb < 6000:
            config.hardware.vae_tiling = True
            config.hardware.vae_slicing = True
            config.hardware.attention_slicing = True
            config.hardware.vae_offload = True
            return True
    except Exception:
        pass
    return False
