"""
Dataset manager: handles loading, converting, and organizing training data.
Supports ALL image formats (including obscure ones) and ALL video formats.
Uses PIL + OpenCV + imageio for maximum format coverage.
"""
import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from core.logger import get_logger

logger = get_logger(__name__)

# ── Exhaustive format support ──────────────────────────────────────────────

IMAGE_EXTENSIONS: Set[str] = {
    # Standard
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp",
    # Professional / RAW
    ".psd", ".psb", ".xcf", ".svg", ".eps", ".ai",
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf",
    ".srw", ".pef", ".raw",
    # HDR / Scientific
    ".exr", ".hdr", ".pfm", ".pbm", ".pgm", ".ppm", ".pnm",
    # Obscure / Legacy
    ".tga", ".ico", ".cur", ".icns", ".dds", ".pcx", ".sgi", ".rgb",
    ".rgba", ".bw", ".pict", ".pct", ".pic", ".jfif", ".jp2", ".j2k",
    ".jpf", ".jpx", ".jpm", ".mj2",
    # Modern
    ".avif", ".heic", ".heif", ".jxl", ".qoi", ".flif",
    # Animation
    ".apng",
}

VIDEO_EXTENSIONS: Set[str] = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".3gp", ".3g2", ".mts", ".m2ts", ".ts", ".vob",
    ".ogv", ".rm", ".rmvb", ".asf", ".divx", ".f4v",
}

# Extensions that are ALWAYS animated (no need to check)
_ALWAYS_ANIMATED: Set[str] = {".gif", ".apng"}
# Extensions that MAY be animated (need frame-count check)
_MAYBE_ANIMATED: Set[str] = {".webp"}
ANIMATED_IMAGE_EXTENSIONS: Set[str] = _ALWAYS_ANIMATED | _MAYBE_ANIMATED


def _is_actually_animated(path: Path) -> bool:
    """Return True only if the file contains more than one frame.
    Prevents static .webp from being treated as animated."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            try:
                img.seek(1)
                return True  # has at least 2 frames
            except EOFError:
                return False  # single frame
    except Exception:
        return False


@dataclass
class DatasetItem:
    original_path: str
    converted_path: str = ""
    caption_path: str = ""
    caption_text: str = ""
    media_type: str = "image"  # "image", "video", "animated"
    width: int = 0
    height: int = 0
    frame_count: int = 1
    file_size_mb: float = 0.0
    hash: str = ""
    is_valid: bool = True
    error: str = ""


@dataclass
class DatasetStats:
    total_files: int = 0
    valid_images: int = 0
    valid_videos: int = 0
    animated_images: int = 0
    invalid_files: int = 0
    captioned_files: int = 0
    uncaptioned_files: int = 0
    total_size_mb: float = 0.0
    avg_resolution: Tuple[int, int] = (0, 0)
    format_breakdown: Dict[str, int] = field(default_factory=dict)


class DatasetManager:
    """
    Manages dataset loading, conversion, validation, and organization.

    IMPORTANT: All processing (resizing, format conversion, frame extraction,
    captioning) writes to an isolated work_dir that lives OUTSIDE the user's
    original dataset folder.  Original files are NEVER modified.
    """

    def __init__(self, dataset_dir: str, work_dir: Optional[str] = None):
        self.dataset_dir = Path(dataset_dir)

        # Work directory lives under the app's config folder — NOT inside
        # the user's dataset directory — so originals are never touched.
        if work_dir:
            self.work_dir = Path(work_dir)
        else:
            from core.config import CACHE_DIR
            # Derive a unique subfolder from the dataset path so different
            # dataset dirs each get their own processing cache.
            dir_hash = hashlib.md5(str(self.dataset_dir.resolve()).encode()).hexdigest()[:8]
            safe_name = self.dataset_dir.name or "dataset"
            self.work_dir = CACHE_DIR / "datasets" / f"{safe_name}_{dir_hash}"

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.items: List[DatasetItem] = []
        self.stats = DatasetStats()
        self._manifest_path = self.work_dir / "manifest.json"

    def scan_directory(self, recursive: bool = True, progress_callback=None) -> List[DatasetItem]:
        """Scan dataset directory for all supported media files."""
        self.items.clear()
        all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

        pattern = "**/*" if recursive else "*"
        for i, fpath in enumerate(sorted(self.dataset_dir.glob(pattern))):
            if progress_callback and i % 100 == 0:
                progress_callback(i, 0, str(fpath))
            if fpath.is_file() and fpath.suffix.lower() in all_extensions:
                ext = fpath.suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    media_type = "video"
                elif ext in _ALWAYS_ANIMATED:
                    media_type = "animated"
                elif ext in _MAYBE_ANIMATED:
                    # .webp can be static or animated — check frame count
                    media_type = "animated" if _is_actually_animated(fpath) else "image"
                else:
                    media_type = "image"

                item = DatasetItem(
                    original_path=str(fpath),
                    media_type=media_type,
                    file_size_mb=fpath.stat().st_size / (1024 * 1024),
                )

                # Check for existing caption
                caption_path = fpath.with_suffix(".txt")
                if caption_path.exists():
                    item.caption_path = str(caption_path)
                    item.caption_text = caption_path.read_text(encoding="utf-8", errors="ignore").strip()

                self.items.append(item)

        self._update_stats()
        return self.items

    def scan_files(self, file_paths: List[str]) -> List[DatasetItem]:
        """Scan specific files (from file picker)."""
        self.items.clear()
        all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

        for fpath_str in file_paths:
            fpath = Path(fpath_str)
            if fpath.is_file() and fpath.suffix.lower() in all_extensions:
                ext = fpath.suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    media_type = "video"
                elif ext in _ALWAYS_ANIMATED:
                    media_type = "animated"
                elif ext in _MAYBE_ANIMATED:
                    media_type = "animated" if _is_actually_animated(fpath) else "image"
                else:
                    media_type = "image"

                item = DatasetItem(
                    original_path=str(fpath),
                    media_type=media_type,
                    file_size_mb=fpath.stat().st_size / (1024 * 1024),
                )
                self.items.append(item)

        self._update_stats()
        return self.items

    def add_files(self, file_paths: List[str]) -> List[DatasetItem]:
        """Add files without clearing existing items. Skips duplicates."""
        all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        existing_paths = {i.original_path for i in self.items}

        for fpath_str in file_paths:
            fpath = Path(fpath_str)
            if str(fpath) in existing_paths:
                continue
            if fpath.is_file() and fpath.suffix.lower() in all_extensions:
                ext = fpath.suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    media_type = "video"
                elif ext in _ALWAYS_ANIMATED:
                    media_type = "animated"
                elif ext in _MAYBE_ANIMATED:
                    media_type = "animated" if _is_actually_animated(fpath) else "image"
                else:
                    media_type = "image"

                item = DatasetItem(
                    original_path=str(fpath),
                    media_type=media_type,
                    file_size_mb=fpath.stat().st_size / (1024 * 1024),
                )
                self.items.append(item)

        self._update_stats()
        return self.items

    def validate_and_convert(self, target_format: str = "png",
                             max_resolution: int = 1024,
                             max_animated_frames: int = 20,
                             progress_callback=None) -> List[DatasetItem]:
        """
        Validate all items, convert unsupported formats to PNG/MP4.
        Extract frames from GIFs/animated images for image training.
        """
        import threading

        converted_dir = self.work_dir / "converted"
        converted_dir.mkdir(exist_ok=True)

        total = len(self.items)
        progress_lock = threading.Lock()
        completed_count = [0]  # mutable container for closure

        image_items = [(idx, item) for idx, item in enumerate(self.items)
                       if item.media_type in ("image", "animated")]
        video_items = [(idx, item) for idx, item in enumerate(self.items)
                       if item.media_type == "video"]

        def _process_image(item):
            try:
                self._convert_image(item, converted_dir, target_format, max_resolution, max_animated_frames)
            except Exception as e:
                item.is_valid = False
                item.error = str(e)
                logger.error(f"Failed to process {item.original_path}: {e}")
            if progress_callback:
                with progress_lock:
                    completed_count[0] += 1
                    progress_callback(completed_count[0], total, item.original_path)

        # Process image items in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_process_image, item) for _, item in image_items]
            for future in as_completed(futures):
                future.result()  # propagate any unexpected exceptions

        # Process video items serially (they use cv2 VideoCapture)
        for _, item in video_items:
            try:
                self._convert_video(item, converted_dir, max_resolution)
            except Exception as e:
                item.is_valid = False
                item.error = str(e)
                logger.error(f"Failed to process {item.original_path}: {e}")
            if progress_callback:
                with progress_lock:
                    completed_count[0] += 1
                    progress_callback(completed_count[0], total, item.original_path)

        self._update_stats()
        self._save_manifest()
        return self.items

    def _convert_image(self, item: DatasetItem, out_dir: Path,
                       target_format: str, max_res: int,
                       max_animated_frames: int = 20):
        """Convert any image format to training-compatible format."""
        try:
            from PIL import Image, ImageSequence
            import pillow_heif
        except ImportError:
            from PIL import Image, ImageSequence

        src = Path(item.original_path)
        ext = src.suffix.lower()

        # Handle animated images - extract frames
        if item.media_type == "animated":
            self._extract_animated_frames(item, out_dir, target_format, max_res, max_animated_frames)
            return

        # Try PIL first (handles most formats)
        try:
            img = Image.open(src)
            img = img.convert("RGB")

            # Resize if needed
            w, h = img.size
            if max(w, h) > max_res:
                ratio = max_res / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                # Round to nearest multiple of 8
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8
                img = img.resize((new_w, new_h), Image.LANCZOS)
                w, h = new_w, new_h

            item.width = w
            item.height = h

            out_name = f"{src.stem}_{hashlib.md5(str(src).encode()).hexdigest()[:8]}.{target_format}"
            out_path = out_dir / out_name
            img.save(out_path, quality=95)
            item.converted_path = str(out_path)
            item.is_valid = True
            return
        except Exception:
            pass

        # Fallback: OpenCV
        try:
            import cv2
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"OpenCV cannot read {src}")

            h, w = img.shape[:2]
            if max(w, h) > max_res:
                ratio = max_res / max(w, h)
                new_w = (int(w * ratio) // 8) * 8
                new_h = (int(h * ratio) // 8) * 8
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                w, h = new_w, new_h

            item.width = w
            item.height = h

            out_name = f"{src.stem}_{hashlib.md5(str(src).encode()).hexdigest()[:8]}.{target_format}"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), img)
            item.converted_path = str(out_path)
            item.is_valid = True
        except Exception as e:
            item.is_valid = False
            item.error = f"Cannot convert: {e}"

    def _extract_animated_frames(self, item: DatasetItem, out_dir: Path,
                                 target_format: str, max_res: int,
                                 max_frames: int = 20):
        """Extract key frames from animated images (GIF, APNG, animated WebP).

        Iterates frames one at a time instead of loading the entire
        sequence into RAM (avoids memory spikes on large GIFs).
        """
        from PIL import Image, ImageSequence

        src = Path(item.original_path)
        img = Image.open(src)

        # First pass: count frames without holding them all in memory
        total_frames = 0
        try:
            while True:
                img.seek(total_frames)
                total_frames += 1
        except EOFError:
            pass
        item.frame_count = total_frames

        step = max(1, total_frames // max_frames)

        # Second pass: extract selected frames one at a time
        base_hash = hashlib.md5(str(src).encode()).hexdigest()[:8]
        img.seek(0)
        saved = 0
        first_w, first_h = 0, 0

        for fi in range(total_frames):
            try:
                img.seek(fi)
            except EOFError:
                break

            if fi % step != 0:
                continue
            if saved >= max_frames:
                break

            frame_rgb = img.convert("RGB")
            w, h = frame_rgb.size
            if saved == 0:
                first_w, first_h = w, h
            if max(w, h) > max_res:
                ratio = max_res / max(w, h)
                new_w = (int(w * ratio) // 8) * 8
                new_h = (int(h * ratio) // 8) * 8
                frame_rgb = frame_rgb.resize((new_w, new_h), Image.LANCZOS)
                if saved == 0:
                    first_w, first_h = new_w, new_h

            out_name = f"{src.stem}_{base_hash}_f{saved:03d}.{target_format}"
            out_path = out_dir / out_name
            frame_rgb.save(out_path, quality=95)
            frame_rgb.close()
            saved += 1

        img.close()
        item.converted_path = str(out_dir)
        item.is_valid = saved > 0
        item.width = first_w
        item.height = first_h

    def _convert_video(self, item: DatasetItem, out_dir: Path, max_res: int):
        """Extract frames from video files for training.

        Delegates to the shared core.video_utils.extract_frames so that
        the captioner and processor produce identical frame filenames.
        """
        from core.video_utils import extract_frames

        src = Path(item.original_path)
        try:
            import cv2
            cap = cv2.VideoCapture(str(src))
            if cap.isOpened():
                item.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                item.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                item.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                cap.release()
            else:
                fps = 30
        except Exception:
            fps = 30

        video_frames_dir = out_dir / "video_frames"
        saved = extract_frames(
            video_path=str(src),
            out_dir=str(video_frames_dir),
            every_n_frames=max(1, int(fps)),   # ~1fps
            max_frames=60,
            max_resolution=max_res,
        )

        item.converted_path = str(video_frames_dir)
        item.is_valid = len(saved) > 0
        if not item.is_valid:
            item.error = "No frames extracted"
        else:
            logger.info(f"Extracted {len(saved)} frames from {src.name}")

    def prepare_training_dir(self, num_repeats: int = 10,
                             instance_prompt: str = "sks",
                             class_prompt: str = "character") -> str:
        """
        Organize converted images into kohya-ss compatible directory structure.
        All operations copy into our work_dir; originals are never touched.

        train_dir/
            {num_repeats}_{instance_prompt} {class_prompt}/
                image1.png
                image1.txt
                ...
        """
        train_dir = self.work_dir / "train"
        folder_name = f"{num_repeats}_{instance_prompt} {class_prompt}"
        img_dir = train_dir / folder_name

        # Only remove the specific repeat-folder, not the entire train_dir
        if img_dir.exists():
            shutil.rmtree(img_dir)

        img_dir.mkdir(parents=True, exist_ok=True)

        # Also look for captions that were written to our captions/ subdir
        captions_dir = self.work_dir / "captions"

        for item in self.items:
            if not item.is_valid:
                continue

            converted = Path(item.converted_path)
            if converted.is_file():
                dest = img_dir / converted.name
                shutil.copy2(converted, dest)

                # Copy caption: check item.caption_path, then captions/ dir
                caption_dest = dest.with_suffix(".txt")
                if item.caption_path and Path(item.caption_path).is_file():
                    shutil.copy2(item.caption_path, caption_dest)
                elif captions_dir.is_dir():
                    orig_stem = Path(item.original_path).stem
                    cap_candidate = captions_dir / f"{orig_stem}.txt"
                    if cap_candidate.is_file():
                        shutil.copy2(cap_candidate, caption_dest)
                elif item.caption_text:
                    caption_dest.write_text(item.caption_text, encoding="utf-8")
            elif converted.is_dir():
                # Animated/video frames
                for frame_file in sorted(converted.glob("*.png")):
                    if Path(item.original_path).stem in frame_file.stem:
                        dest = img_dir / frame_file.name
                        shutil.copy2(frame_file, dest)
                        # Check for per-frame caption from captioner
                        caption_dest = dest.with_suffix(".txt")
                        video_cap_dir = captions_dir / "video_frames" if captions_dir.is_dir() else None
                        per_frame_cap = video_cap_dir / f"{frame_file.stem}.txt" if video_cap_dir else None
                        if per_frame_cap and per_frame_cap.is_file():
                            shutil.copy2(per_frame_cap, caption_dest)
                        elif item.caption_text:
                            caption_dest.write_text(item.caption_text, encoding="utf-8")

        # Also copy captions from video_frames subfolder for any frames
        # that may have been matched by stem
        video_cap_dir = captions_dir / "video_frames" if captions_dir.is_dir() else None
        if video_cap_dir and video_cap_dir.exists():
            for cap_file in video_cap_dir.glob("*.txt"):
                matching_img = img_dir / f"{cap_file.stem}.png"
                if matching_img.exists():
                    shutil.copy2(cap_file, img_dir / cap_file.name)

        return str(train_dir)

    def _update_stats(self):
        s = self.stats
        s.total_files = len(self.items)
        s.valid_images = sum(1 for i in self.items if i.is_valid and i.media_type == "image")
        s.valid_videos = sum(1 for i in self.items if i.is_valid and i.media_type == "video")
        s.animated_images = sum(1 for i in self.items if i.is_valid and i.media_type == "animated")
        s.invalid_files = sum(1 for i in self.items if not i.is_valid)
        s.captioned_files = sum(1 for i in self.items if i.caption_text)
        s.uncaptioned_files = s.total_files - s.captioned_files
        s.total_size_mb = sum(i.file_size_mb for i in self.items)

        # Format breakdown
        s.format_breakdown = {}
        for item in self.items:
            ext = Path(item.original_path).suffix.lower()
            s.format_breakdown[ext] = s.format_breakdown.get(ext, 0) + 1

        # Average resolution
        valid_items = [i for i in self.items if i.width > 0 and i.height > 0]
        if valid_items:
            avg_w = sum(i.width for i in valid_items) // len(valid_items)
            avg_h = sum(i.height for i in valid_items) // len(valid_items)
            s.avg_resolution = (avg_w, avg_h)

    def _save_manifest(self):
        data = []
        for item in self.items:
            data.append({
                "original_path": item.original_path,
                "converted_path": item.converted_path,
                "caption_path": item.caption_path,
                "caption_text": item.caption_text,
                "media_type": item.media_type,
                "width": item.width,
                "height": item.height,
                "frame_count": item.frame_count,
                "is_valid": item.is_valid,
                "error": item.error,
            })
        with open(self._manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_manifest(self) -> bool:
        if not self._manifest_path.exists():
            return False
        try:
            with open(self._manifest_path, "r") as f:
                data = json.load(f)
            self.items = [DatasetItem(**d) for d in data]
            self._update_stats()
            return True
        except Exception:
            return False

    def find_duplicates(self, threshold: int = 8) -> List[tuple]:
        """Find near-duplicate images using perceptual hashing.
        Returns list of (item_a, item_b, distance) tuples.
        """
        try:
            from PIL import Image
            import imagehash
        except ImportError:
            return []

        hashes = {}
        duplicates = []
        for item in self.items:
            if item.media_type not in ("image",):
                continue
            try:
                img = Image.open(item.original_path)
                h = imagehash.phash(img)
                for other_path, other_hash in hashes.items():
                    dist = h - other_hash
                    if dist <= threshold:
                        duplicates.append((other_path, item.original_path, dist))
                hashes[item.original_path] = h
            except Exception:
                continue
        return duplicates

    def get_uncaptioned_items(self) -> List[DatasetItem]:
        return [i for i in self.items if i.is_valid and not i.caption_text]

    def get_all_valid_items(self) -> List[DatasetItem]:
        return [i for i in self.items if i.is_valid]
