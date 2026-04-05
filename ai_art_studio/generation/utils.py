"""
Shared utilities for generation pipelines.
"""
from core.logger import get_logger

logger = get_logger(__name__)


def _load_with_offline_fallback(pipeline_cls, model_path: str, **kwargs):
    """Try loading a diffusers pipeline normally; on network error, retry with
    ``local_files_only=True`` so cached models work without internet."""
    try:
        return pipeline_cls.from_pretrained(model_path, **kwargs)
    except (OSError, ConnectionError, Exception) as first_err:
        err_str = str(first_err).lower()
        if any(tok in err_str for tok in ("connection", "resolve", "timeout",
                                           "offline", "404", "urlopen")):
            logger.warning(
                f"Online load failed ({first_err}); retrying from local cache..."
            )
            try:
                return pipeline_cls.from_pretrained(
                    model_path, local_files_only=True, **kwargs
                )
            except Exception:
                pass  # fall through to re-raise original
        raise
