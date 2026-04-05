"""
AI upscaling with Real-ESRGAN (optional). Falls back to LANCZOS if not installed.
"""
from core.logger import get_logger
logger = get_logger(__name__)


def upscale_image(image, scale: int = 4):
    """Upscale image using Real-ESRGAN if available, else LANCZOS.

    Args:
        image: PIL Image
        scale: 2 or 4
    Returns:
        Upscaled PIL Image
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import numpy as np
        import torch

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=scale)
        model_path = _get_realesrgan_weights(scale)
        if model_path is None:
            raise ImportError("Real-ESRGAN weights not found")

        upsampler = RealESRGANer(
            scale=scale, model_path=model_path, model=model,
            tile=512, tile_pad=10, pre_pad=0,
            half=torch.cuda.is_available()
        )
        img_array = np.array(image.convert("RGB"))
        output, _ = upsampler.enhance(img_array, outscale=scale)
        from PIL import Image
        return Image.fromarray(output)
    except ImportError:
        logger.info("Real-ESRGAN not installed — using LANCZOS upscaling")
        return _lanczos_upscale(image, scale)
    except Exception as e:
        logger.warning(f"Real-ESRGAN failed ({e}), falling back to LANCZOS")
        return _lanczos_upscale(image, scale)


def _lanczos_upscale(image, scale: int):
    from PIL import Image
    return image.resize(
        (image.width * scale, image.height * scale), Image.LANCZOS)


def _get_realesrgan_weights(scale: int):
    """Find or return path to Real-ESRGAN weights."""
    from pathlib import Path
    names = {2: "RealESRGAN_x2plus.pth", 4: "RealESRGAN_x4plus.pth"}
    name = names.get(scale, "RealESRGAN_x4plus.pth")
    candidates = [
        Path.home() / ".ai_art_studio" / "upscaler" / name,
        Path(__file__).parent.parent / "models" / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def is_realesrgan_available() -> bool:
    try:
        import realesrgan
        return True
    except ImportError:
        return False
