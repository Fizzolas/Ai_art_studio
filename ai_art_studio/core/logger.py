"""
Centralized logging for AI Art Studio.

Usage:
    from core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.error("Something failed", exc_info=True)
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_LOG_FILE = _LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

_FMT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Root handler — rotating file, max 10MB, keep 7 days
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=7, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
_file_handler.setLevel(logging.DEBUG)

# Console handler — INFO+ only
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
_console_handler.setLevel(logging.INFO)

# Root logger config
_root = logging.getLogger("ai_art_studio")
_root.setLevel(logging.DEBUG)
if not _root.handlers:
    _root.addHandler(_file_handler)
    _root.addHandler(_console_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the ai_art_studio namespace."""
    if not name.startswith("ai_art_studio"):
        name = f"ai_art_studio.{name}"
    return logging.getLogger(name)


def set_log_level(level: str):
    """Dynamically change log level. level: 'debug', 'info', 'warning', 'error'"""
    lvl = getattr(logging, level.upper(), logging.INFO)
    _root.setLevel(lvl)
    _file_handler.setLevel(lvl)


def get_log_file() -> Path:
    return _LOG_FILE


def get_log_dir() -> Path:
    return _LOG_DIR
