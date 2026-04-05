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
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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
