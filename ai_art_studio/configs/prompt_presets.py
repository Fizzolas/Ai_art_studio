"""Prompt style presets — save/load named style strings."""
import json
from pathlib import Path

from core.logger import get_logger
logger = get_logger(__name__)

PRESETS_FILE = Path.home() / ".ai_art_studio" / "prompt_presets.json"

DEFAULT_PRESETS = {
    "Photorealistic": "photorealistic, 8k uhd, dslr, high quality, film grain",
    "Anime": "anime style, cel shaded, vibrant colors, studio ghibli",
    "Oil Painting": "oil painting, textured canvas, impressionist brushwork, warm palette",
    "Portal 2": "portal 2 aesthetic, clean white walls, orange-blue accent lighting, metal grating",
    "Dark Fantasy": "dark fantasy, dramatic lighting, cinematic, moody atmosphere, high contrast",
}


def load_presets() -> dict:
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(DEFAULT_PRESETS)


def save_presets(presets: dict):
    PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")
