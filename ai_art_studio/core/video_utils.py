"""Shared video frame extraction used by both the dataset processor
and the captioning pipeline, so frames and captions always match."""

import hashlib
from pathlib import Path
from typing import List, Optional, Callable
from core.logger import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
    ".ts", ".mts", ".m2ts", ".3gp", ".ogv", ".vob", ".divx", ".xvid",
    ".mpg", ".mpeg", ".3g2", ".rm", ".rmvb", ".asf", ".f4v",
}


def extract_frames(
    video_path: str,
    out_dir: str,
    every_n_frames: int = 30,   # every Nth frame (1fps at 30fps video)
    max_frames: int = 60,
    max_resolution: int = 1024,
    on_progress: Optional[Callable] = None,
) -> List[str]:
    """Extract frames from a video file.

    Uses sequential reading (no random seek) to avoid h264 mmco decoder
    errors. Frames are saved as PNGs with a deterministic naming scheme:
        {stem}_{md5hash8}_f{seq:05d}.png

    Both the dataset processor and captioner call this function, so
    the exact same frames and filenames are used for both captions and
    training images.

    Args:
        video_path: Path to source video file.
        out_dir: Directory to save extracted frames.
        every_n_frames: Keep every Nth frame (default 30 = ~1fps at 30fps).
        max_frames: Maximum frames to extract.
        max_resolution: Resize if longest edge exceeds this.
        on_progress: Optional callback(saved_count, total_estimate, frame_path).

    Returns:
        List of paths to saved frame PNG files.
    """
    try:
        import cv2
    except ImportError:
        logger.error("cv2 not installed — cannot extract video frames")
        return []

    src = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = _open_capture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    base_hash = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:8]
    stem = src.stem

    estimate = min(total // max(every_n_frames, 1), max_frames) if total > 0 else max_frames
    logger.info(f"Extracting ~{estimate} frames from {src.name} "
                f"(total={total}, fps={fps:.1f}, every_n={every_n_frames})")

    saved: List[str] = []
    frame_idx = 0
    seq = 0
    fail_streak = 0

    while len(saved) < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            fail_streak += 1
            if fail_streak >= 30:
                break
            frame_idx += 1
            continue
        fail_streak = 0

        if frame_idx % every_n_frames == 0:
            h, w = frame.shape[:2]
            if max(w, h) > max_resolution:
                ratio = max_resolution / max(w, h)
                nw = (int(w * ratio) // 8) * 8
                nh = (int(h * ratio) // 8) * 8
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            fname = f"{stem}_{base_hash}_f{seq:05d}.png"
            fpath = out / fname
            cv2.imwrite(str(fpath), frame)
            saved.append(str(fpath))
            seq += 1
            if on_progress:
                on_progress(len(saved), estimate, str(fpath))

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(saved)} frames from {src.name}")
    return saved


def _open_capture(video_path: str):
    """Open VideoCapture with ffmpeg backend preferred."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        return cv2.VideoCapture(video_path)
    except Exception:
        import cv2
        return cv2.VideoCapture(video_path)


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS
