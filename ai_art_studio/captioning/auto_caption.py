"""
Auto-captioning system for dataset coherence.
Combines multiple vision models to understand diverse image content:
- WD Tagger: Detailed booru-style tags (great for character features, poses, compositions)
- BLIP-2/Florence-2: Natural language descriptions for context
- Combined mode: Tags + natural language for maximum training coherence

Performance optimisations (RTX 4070 8 GB target):
- WD Tagger uses batched ONNX inference (4-8 images per GPU call)
- BLIP-2 runs with torch.inference_mode, flash-attention / SDPA when
  available, and reduced beam width for speed
- Images are pre-loaded & resized on CPU threads while the GPU runs
- Video frames are extracted in bulk via OpenCV, then batch-captioned

All captioner models are auto-downloaded from Hugging Face on first use.
"""
import os
import gc
import hashlib
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image

from core.logger import get_logger

logger = get_logger(__name__)


# ── Video frame extraction ───────────────────────────────────────────────

VIDEO_EXTENSIONS: Set[str] = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".3gp", ".3g2", ".mts", ".m2ts", ".ts", ".vob",
    ".ogv", ".rm", ".rmvb", ".asf", ".divx", ".f4v",
}


def _open_video_capture(video_path: str):
    """Open a video with ffmpeg backend preferred (avoids h264 mmco bugs
    in the default backend).  Falls back to the default if ffmpeg is
    unavailable."""
    import cv2
    # CAP_FFMPEG is more tolerant of h264 stream errors than the
    # default backend (which sometimes hangs on mmco: unref short).
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    # Fallback
    cap = cv2.VideoCapture(video_path)
    return cap


def _extract_video_frames(
    video_path: str,
    out_dir: str,
    every_n: int = 5,
    max_frames: int = 60,
    max_resolution: int = 1024,
    on_status: Callable = None,
) -> List[str]:
    """Extract every *every_n*-th frame using sequential reading.

    IMPORTANT: We do NOT use ``cap.set(CAP_PROP_POS_FRAMES, n)``
    because seeking in h264 streams without a keyframe index causes
    the decoder to walk every intermediate frame, triggering
    ``mmco: unref short failure`` warnings and extreme slowdowns.

    Instead we read sequentially and keep every Nth frame — this is
    dramatically faster and avoids h264 decoder errors entirely.
    """
    try:
        import cv2
    except ImportError:
        if on_status:
            on_status("OpenCV (cv2) not installed — cannot process videos.")
        return []

    cap = _open_video_capture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    if every_n <= 0:
        every_n = 5

    stem = Path(video_path).stem
    base_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    frame_idx = 0
    seq = 0
    consecutive_failures = 0
    max_consecutive_failures = 30  # give up if 30 reads in a row fail

    expected = min(total // max(every_n, 1), max_frames) if total > 0 else max_frames
    if on_status:
        on_status(
            f"Extracting ~{expected} frames from "
            f"{Path(video_path).name} "
            f"({total} total, {fps:.0f} fps)…")

    while len(saved) < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break  # corrupted tail or EOF
            frame_idx += 1
            continue

        consecutive_failures = 0  # reset on good read

        if frame_idx % every_n == 0:
            fh, fw = frame.shape[:2]
            if max(fw, fh) > max_resolution:
                ratio = max_resolution / max(fw, fh)
                new_w = (int(fw * ratio) // 8) * 8
                new_h = (int(fh * ratio) // 8) * 8
                frame = cv2.resize(frame, (new_w, new_h),
                                   interpolation=cv2.INTER_LANCZOS4)
            fname = f"{stem}_{base_hash}_vf{seq:04d}.png"
            fpath = out_path_dir / fname
            cv2.imwrite(str(fpath), frame)
            saved.append(str(fpath))
            seq += 1

        frame_idx += 1

    cap.release()
    if on_status:
        on_status(f"Extracted {len(saved)} frames from {Path(video_path).name}")
    return saved


def _extract_representative_frame(video_path: str) -> Optional[Image.Image]:
    """Quick single-frame extraction (~25% in, capped at 500 frames max).
    Uses sequential read to avoid h264 seek issues.  For very long
    videos the cap prevents reading tens of thousands of frames just
    to find one representative sample."""
    try:
        import cv2
    except ImportError:
        return None
    cap = _open_video_capture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    # Cap at 500 frames to avoid reading huge portions of long videos (#6)
    target = min(max(0, int(total * 0.25)), 500) if total > 1 else 0
    # Read sequentially up to the target frame
    frame = None
    for i in range(target + 1):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
        else:
            break
    cap.release()
    if frame is None:
        return None
    return Image.fromarray(frame[:, :, ::-1])


# ── Dependency helpers ────────────────────────────────────────────────────

def _check_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _check_captioning_deps(method: str) -> Tuple[bool, str]:
    missing = []
    if method in ("wd_tagger", "combined"):
        if not _check_import("onnxruntime"):
            missing.append("onnxruntime-gpu  (or onnxruntime)")
        if not _check_import("pandas"):
            missing.append("pandas")
    if method in ("blip2", "florence2", "combined"):
        if not _check_import("transformers"):
            missing.append("transformers")
    if method in ("blip2", "combined"):
        if not _check_import("bitsandbytes"):
            missing.append("bitsandbytes  (needed for 8-bit BLIP-2 loading)")
    if missing:
        names = ", ".join(missing)
        return False, (
            f"Missing packages for '{method}' captioning: {names}\n"
            f"Install with:  pip install {' '.join(m.split()[0] for m in missing)}"
        )
    return True, ""


def _ensure_hf_model_cached(repo_id: str, files: List[str] = None,
                             on_status: Callable = None) -> bool:
    def _s(msg):
        logger.info(msg)
        if on_status:
            on_status(msg)
    try:
        from huggingface_hub import scan_cache_dir
        for ri in scan_cache_dir().repos:
            if ri.repo_id == repo_id:
                _s(f"Model cached: {repo_id}")
                return True
    except Exception:
        pass
    _s(f"Downloading model: {repo_id} (first-time setup)…")
    try:
        if files:
            from huggingface_hub import hf_hub_download
            for fn in files:
                _s(f"  ↓ {fn}")
                hf_hub_download(repo_id, fn)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id, repo_type="model",
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*",
                                 "tf_model*", "*.onnx_data"],
            )
        _s(f"Download complete: {repo_id}")
        return True
    except Exception as e:
        _s(f"Failed to download {repo_id}: {e}")
        return False


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class CaptionResult:
    file_path: str
    tags: List[str] = field(default_factory=list)
    tag_confidences: Dict[str, float] = field(default_factory=dict)
    natural_description: str = ""
    verbose_description: str = ""
    combined_caption: str = ""
    content_type: str = ""
    # Video grouping — all frames from the same source video share a group ID
    video_group_id: str = ""        # e.g. "vid_<stem>_<hash>"
    video_source: str = ""          # original video path
    frame_index: int = -1           # sequential frame number within this video


# ── WD Tagger (with batch support) ───────────────────────────────────────

class WDTagger:
    """WD Tagger — batched ONNX inference for speed."""

    BATCH_SIZE = 4  # images per GPU call (tune vs VRAM)

    def __init__(self, model_name: str = "SmilingWolf/wd-swinv2-tagger-v3",
                 threshold: float = 0.35, device: str = "cuda"):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.model = None
        self.tag_names = None
        self.general_indices = None
        self.character_indices = None
        self.input_size = 448

    def load(self, on_status: Callable = None):
        if self.model is not None:
            return
        def _status(msg):
            logger.info(msg)
            if on_status:
                on_status(msg)
        if not _check_import("onnxruntime"):
            _status("onnxruntime not installed — WD Tagger unavailable.")
            self.model = "fallback"; return
        if not _check_import("pandas"):
            _status("pandas not installed — WD Tagger unavailable.")
            self.model = "fallback"; return
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd
            _status(f"Loading WD Tagger: {self.model_name}")
            model_path = hf_hub_download(self.model_name, "model.onnx")
            labels_path = hf_hub_download(self.model_name, "selected_tags.csv")
            df = pd.read_csv(labels_path)
            self.tag_names = df["name"].tolist()
            self.general_indices = list(np.where(df["category"] == 0)[0])
            self.character_indices = list(np.where(df["category"] == 4)[0])

            self.model = self._create_onnx_session(model_path)
            input_shape = self.model.get_inputs()[0].shape
            self.input_size = (input_shape[1]
                               if isinstance(input_shape[1], int)
                               else 448)
            _status("WD Tagger loaded (GPU-optimised)")
        except Exception as e:
            _status(f"Failed to load WD Tagger: {e}")
            logger.error(f"WD Tagger load error: {e}", exc_info=True)
            self.model = "fallback"

    def _create_onnx_session(self, model_path: str):
        providers_to_try = []
        try:
            import torch
            if torch.cuda.is_available():
                providers_to_try.append(
                    ("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo"})
                )
        except ImportError:
            pass
        providers_to_try.append("CPUExecutionProvider")

        last_error = None
        for provider in providers_to_try:
            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                provider_list = [provider] if isinstance(provider, str) else [provider]
                session = ort.InferenceSession(model_path, sess_options, providers=provider_list)
                used = session.get_providers()[0]
                logger.info(f"WD Tagger ONNX session created with provider: {used}")
                return session
            except Exception as e:
                last_error = e
                logger.warning(f"ONNX provider {provider} failed: {e}, trying next...")
        raise RuntimeError(f"Could not create ONNX session with any provider. Last error: {last_error}")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize + normalise a single image for the tagger."""
        img = image.convert("RGB").resize(
            (self.input_size, self.input_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)[:, :, ::-1]   # RGB→BGR
        return arr

    def predict(self, image: Image.Image) -> Tuple[List[str], Dict[str, float]]:
        """Tag a single image."""
        results = self.predict_batch([image])
        return results[0] if results else ([], {})

    def predict_batch(self, images: List[Image.Image]
                      ) -> List[Tuple[List[str], Dict[str, float]]]:
        """Tag a batch of images in one ONNX call — much faster."""
        if self.model is None:
            self.load()
        if self.model == "fallback":
            return [([], {})] * len(images)

        # Pre-process all images (CPU, can be threaded)
        arrays = [self._preprocess(im) for im in images]
        batch = np.stack(arrays, axis=0)   # (N, H, W, 3)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        all_probs = self.model.run([output_name], {input_name: batch})[0]

        results = []
        for probs in all_probs:
            tags = []
            confidences = {}
            for idx in self.general_indices + self.character_indices:
                if idx < len(probs) and probs[idx] >= self.threshold:
                    tag = self.tag_names[idx].replace("_", " ")
                    tags.append(tag)
                    confidences[tag] = float(probs[idx])
            tags.sort(key=lambda t: confidences.get(t, 0), reverse=True)
            results.append((tags, confidences))
        return results

    def unload(self):
        self.model = None
        gc.collect()


# ── Natural Language Captioner ────────────────────────────────────────────

class NaturalCaptioner:
    """BLIP-2 or Florence-2 — with speed optimisations."""

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b",
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._model_type = None

    def load(self, on_status: Callable = None):
        if self.model is not None:
            return
        def _status(msg):
            logger.info(msg)
            if on_status:
                on_status(msg)
        if not _check_import("transformers"):
            _status("transformers not installed — captioner unavailable.")
            self.model = "fallback"
            return
        _status(f"Loading captioner: {self.model_name}")
        try:
            if "florence" in self.model_name.lower():
                self._load_florence(_status)
            else:
                self._load_blip2(_status)
        except Exception as e:
            _status(f"Failed to load {self.model_name}: {e}")
            logger.error(f"Captioner load error: {e}", exc_info=True)
            self.model = "fallback"

    def _load_blip2(self, _status: Callable):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        use_8bit = _check_import("bitsandbytes")
        if not use_8bit:
            _status("bitsandbytes not found — loading in fp16 (uses more VRAM).")
        _ensure_hf_model_cached(self.model_name, on_status=_status)

        _status("Loading BLIP-2 processor…")
        self.processor = Blip2Processor.from_pretrained(self.model_name)

        _status("Loading BLIP-2 model (this may take a minute)…")
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        if use_8bit:
            load_kwargs["load_in_8bit"] = True

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )

        # ── Speed: enable SDPA / Flash-Attention if available ──
        try:
            self.model = self.model.to_bettertransformer()
            _status("BLIP-2: BetterTransformer (SDPA) enabled — faster inference")
        except Exception:
            # Not available in this transformers version — that's fine
            pass

        self._model_type = "blip2"
        _status("BLIP-2 loaded successfully")

    def _load_florence(self, _status: Callable):
        from transformers import AutoModelForCausalLM, AutoProcessor
        _ensure_hf_model_cached(self.model_name, on_status=_status)
        _status("Loading Florence-2 processor…")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True)
        _status("Loading Florence-2 model…")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self._model_type = "florence"
        _status("Florence-2 loaded successfully")

    # ── Caption generation ────────────────────────────────────────────

    def caption(self, image: Image.Image, context_tags: List[str] = None,
                verbose: bool = False) -> str:
        if self.model is None:
            self.load()
        if self.model == "fallback":
            return "image"
        try:
            if self._model_type == "blip2":
                return self._caption_blip2(image, context_tags, verbose)
            elif self._model_type == "florence":
                return self._caption_florence(image)
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return ""

    def _caption_blip2(self, image: Image.Image,
                        context_tags: List[str] = None,
                        verbose: bool = False) -> str:
        img = image.convert("RGB")

        if verbose and context_tags:
            tag_str = ", ".join(context_tags[:30])
            prompt = (
                "Give an extremely detailed description of this artwork. "
                "Describe every visible body part, genital, orifice, and "
                "non-human anatomy feature such as tails, paws, wings, horns, "
                "fur patterns, scales, or claws. Describe every action being "
                "performed, body positions, who is doing what to whom, "
                "expressions, and fluids. Be as explicit and thorough as "
                "possible. The image contains: " + tag_str + "."
            )
            max_tokens = 350
        elif context_tags:
            tag_str = ", ".join(context_tags[:20])
            prompt = (
                "Describe this image in extensive detail. Cover every "
                "character, their body features, anatomy, poses, what they "
                "are doing, their expressions, and the setting. "
                "The image contains: " + tag_str + "."
            )
            max_tokens = 300
        else:
            prompt = (
                "Describe this image in extensive detail, including all "
                "characters, their body features, anatomy, poses, actions, "
                "expressions, and the setting."
            )
            max_tokens = 300

        inputs = self.processor(img, text=prompt, return_tensors="pt").to(
            self.device, torch.float16
        )

        # ── inference_mode is faster than no_grad (skips autograd overhead) ──
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=3,                # 3 beams (was 5) — big speed win
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )

        text = self.processor.decode(output[0], skip_special_tokens=True).strip()
        text = _clean_blip2_output(text, prompt)
        return text

    def _caption_florence(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(
            self.device, torch.float16
        )
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=300,
                num_beams=3,
            )
        text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        text = text.replace("<MORE_DETAILED_CAPTION>", "").strip()
        return text

    def unload(self):
        from core.gpu_utils import flush_gpu_memory
        if self.model is not None and self.model != "fallback":
            try:
                if hasattr(self.model, "to"):
                    self.model.to("cpu")
            except Exception:
                pass
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._model_type = None
        flush_gpu_memory()


# ── Post-processing helpers ──────────────────────────────────────────────

def _clean_blip2_output(text: str, prompt: str = "") -> str:
    for prefix in ["Question:", "Answer:"]:
        idx = text.rfind(prefix)
        if idx != -1:
            text = text[idx + len(prefix):].strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for s in sentences:
        n = s.strip().lower()
        if n and n not in seen:
            seen.add(n)
            unique.append(s.strip())
    text = " ".join(unique)
    if text and text[-1] not in '.!?':
        lp = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if lp > 0:
            text = text[:lp + 1]
    return text.strip()


def _is_degenerate_caption(text: str) -> bool:
    """Check if caption is garbage (too short, mostly repeated words)."""
    if not text or len(text.strip()) < 10:
        return True
    words = text.lower().split()
    if len(words) < 3:
        return True
    unique = set(words)
    if len(unique) / len(words) < 0.3:  # >70% repeated words
        return True
    return False


# ── Tag-aware anatomy / action descriptor ────────────────────────────────

_ANATOMY_EXPANSIONS: Dict[str, str] = {
    "penis":           "an exposed penis",
    "erection":        "an erect penis",
    "large penis":     "a large, prominent penis",
    "small penis":     "a small penis",
    "foreskin":        "an uncircumcised penis with visible foreskin",
    "testicles":       "visible testicles hanging below",
    "balls":           "exposed testicles",
    "sheath":          "a genital sheath typical of non-human anatomy",
    "knot":            "a canine-style knotted penis",
    "barbed penis":    "a barbed feline-style penis",
    "hemipenis":       "a reptilian hemipenis",
    "cloaca":          "a visible cloaca",
    "genital slit":    "a visible genital slit",
    "pussy":           "exposed female genitalia",
    "vagina":          "visible vaginal opening",
    "vulva":           "visible vulva",
    "clitoris":        "an exposed clitoris",
    "anus":            "a visible anus",
    "puffy anus":      "a puffy, prominent anus",
    "gaping":          "a gaping orifice stretched wide open",
    "cervix":          "visible cervix through a deeply penetrated opening",
    "urethra":         "a visible urethra",
    "breasts":         "visible breasts",
    "large breasts":   "large, heavy breasts",
    "small breasts":   "small, modest breasts",
    "flat chest":      "a flat chest",
    "nipples":         "visible, prominent nipples",
    "areolae":         "visible areolae around the nipples",
    "ass":             "a prominently displayed rear / buttocks",
    "big ass":         "a large, prominent rear",
    "spread ass":      "buttocks spread apart to expose what is between them",
    "muscular":        "a heavily muscular physique",
    "chubby":          "a chubby, soft body",
    "plump":           "a plump, full body",
    "abs":             "a toned abdomen with visible abs",
    "tail":            "a long tail extending from the lower back",
    "cat tail":        "a slender, flexible cat tail",
    "fox tail":        "a bushy fox tail",
    "wolf tail":       "a thick wolf tail",
    "dragon tail":     "a powerful scaled dragon tail",
    "horse tail":      "a long, flowing horse tail",
    "prehensile tail": "a prehensile tail capable of grasping",
    "multiple tails":  "multiple tails",
    "wings":           "a pair of wings",
    "bat wings":       "leathery bat-like wings",
    "feathered wings": "feathered bird-like wings",
    "horns":           "prominent horns on the head",
    "antlers":         "branching antlers on the head",
    "animal ears":     "animal-style ears on top of the head",
    "cat ears":        "pointed cat ears",
    "dog ears":        "floppy dog ears",
    "fox ears":        "large fox ears",
    "wolf ears":       "pointed wolf ears",
    "rabbit ears":     "long rabbit ears",
    "paws":            "paw-like hands or feet instead of human ones",
    "pawpads":         "soft paw pads on the palms / soles",
    "claws":           "sharp claws on fingers or toes",
    "talons":          "sharp talons on the feet",
    "hooves":          "hooves instead of feet",
    "fur":             "a body covered in fur",
    "body fur":        "visible body fur covering the torso and limbs",
    "white fur":       "white-colored fur across the body",
    "black fur":       "dark black fur across the body",
    "scales":          "a body covered in reptilian scales",
    "feathers":        "a body adorned with feathers",
    "beak":            "a beak instead of a mouth",
    "snout":           "a protruding animal snout",
    "muzzle":          "a pronounced muzzle",
    "fangs":           "visible sharp fangs",
    "forked tongue":   "a forked tongue extending from the mouth",
    "tentacles":       "writhing tentacles",
    "extra arms":      "more than two arms",
    "extra limbs":     "extra limbs beyond the normal count",
    "taur":            "a taur body — humanoid upper half on a quadruped lower body",
    "cum":             "visible cum / semen",
    "cum on face":     "cum splattered across the face",
    "cum on body":     "cum dripping across the body",
    "cum in mouth":    "cum visibly inside the open mouth",
    "cum inside":      "cum visible inside a penetrated orifice",
    "cum on hair":     "strands of cum in the hair",
    "cum pool":        "a pool of cum beneath the body",
    "excessive cum":   "an excessive, overflowing amount of cum",
    "drooling":        "drool or saliva dripping from the mouth",
    "saliva":          "visible strands of saliva",
    "sweat":           "glistening beads of sweat on the skin",
    "tears":           "tears streaming down the face",
    "wet":             "a body glistening with moisture",
    "lactation":       "milk leaking from the nipples",
    "urination":       "a visible stream of urine",
    "squirting":       "fluid squirting from the genitals",
}

_ACTION_EXPANSIONS: Dict[str, str] = {
    "sex":             "actively engaged in sexual intercourse",
    "vaginal":         "engaged in vaginal penetration",
    "anal":            "engaged in anal penetration",
    "oral":            "performing or receiving oral sex",
    "fellatio":        "performing fellatio on a penis",
    "cunnilingus":     "performing cunnilingus on a vulva",
    "rimming":         "performing anilingus / rimming",
    "deepthroat":      "deep-throating a penis all the way to the base",
    "penetration":     "being penetrated",
    "double penetration": "being penetrated in two orifices simultaneously",
    "triple penetration": "being penetrated in three orifices simultaneously",
    "fisting":         "being fisted — a full hand inserted into an orifice",
    "fingering":       "being fingered — digits inserted into an orifice",
    "handjob":         "giving a handjob — manually stroking a penis",
    "footjob":         "giving a footjob — stimulating a penis with feet",
    "titfuck":         "a penis being thrust between breasts",
    "grinding":        "grinding genitals against another body part",
    "humping":         "humping — thrusting hips rhythmically",
    "riding":          "riding on top, controlling the motion",
    "cowgirl":         "riding in cowgirl position on top facing forward",
    "reverse cowgirl": "riding in reverse cowgirl, facing away",
    "missionary":      "in missionary position — lying beneath with legs spread",
    "doggystyle":      "in doggystyle — on all fours being taken from behind",
    "69":              "in a 69 position — both partners simultaneously performing oral",
    "spitroast":       "being spitroasted — penetrated from both ends",
    "gangbang":        "in a gangbang — multiple partners engaging simultaneously",
    "threesome":       "in a threesome — three participants engaged together",
    "masturbation":    "masturbating — stimulating their own genitals",
    "ejaculation":     "in the act of ejaculating",
    "orgasm":          "visibly experiencing orgasm",
    "cumshot":         "receiving a cumshot — ejaculation onto the body",
    "facial":          "receiving a facial — ejaculation onto the face",
    "creampie":        "receiving a creampie — internal ejaculation visible",
    "knotting":        "being knotted — a canine knot locked inside",
    "inflation":       "visible abdominal inflation from internal volume",
    "egg laying":      "in the process of laying or being filled with eggs",
    "oviposition":     "oviposition — eggs being deposited into or out of the body",
    "tentacle sex":    "being penetrated or restrained by tentacles",
    "bondage":         "restrained with ropes, cuffs, or bindings",
    "shibari":         "bound in decorative shibari rope patterns",
    "gagged":          "gagged — something filling or covering the mouth",
    "all fours":       "positioned on all fours",
    "spread legs":     "legs spread wide apart",
    "on back":         "lying on their back",
    "on stomach":      "lying face-down on their stomach",
    "on side":         "lying on their side",
    "sitting":         "sitting down",
    "squatting":       "squatting with knees bent",
    "standing":        "standing upright",
    "kneeling":        "kneeling on the ground",
    "bending over":    "bending over, rear presented",
    "arching back":    "arching their back",
    "legs up":         "legs raised up in the air",
    "legs over head":  "legs folded back over their head",
    "suspended":       "suspended in the air",
    "pinned down":     "pinned down against a surface",
    "straddling":      "straddling another character",
    "embracing":       "embracing another character closely",
    "kissing":         "kissing another character",
    "licking":         "licking — running tongue across a body part",
    "biting":          "biting gently on skin or body part",
    "sucking":         "sucking on a body part",
    "presenting":      "presenting — posing to display genitals or rear",
    "teasing":         "teasing — playfully showing off body",
    "looking back":    "looking back over the shoulder",
    "eye contact":     "making direct eye contact with the viewer",
    "ahegao":          "making an ahegao expression — eyes rolled back, tongue out",
    "blush":           "blushing — cheeks flushed with arousal or embarrassment",
    "open mouth":      "mouth hanging open",
    "tongue out":      "tongue extended outward",
    "one eye closed":  "one eye closed in a wink or from pleasure",
    "crying":          "crying — tears visible",
    "panting":         "panting with heavy breath",
}


def _build_verbose_description(tags: List[str], natural_desc: str,
                                content_type: str) -> str:
    tag_lower = {t.lower() for t in tags}
    anatomy_parts: List[str] = []
    action_parts: List[str] = []
    for tk, phrase in _ANATOMY_EXPANSIONS.items():
        if tk in tag_lower:
            anatomy_parts.append(phrase)
    for tk, phrase in _ACTION_EXPANSIONS.items():
        if tk in tag_lower:
            action_parts.append(phrase)

    char_count = ""
    for t in tags:
        tl = t.lower()
        if tl in ("solo", "1boy", "1girl"):
            char_count = "a single character"; break
        elif tl in ("2boys", "2girls", "duo"):
            char_count = "two characters"; break
        elif tl in ("3boys", "3girls", "trio"):
            char_count = "three characters"; break
        elif tl in ("group", "crowd", "multiple characters"):
            char_count = "a group of characters"; break

    sections: List[str] = []
    if char_count:
        opener = f"This image depicts {char_count}"
        if content_type and content_type not in ("general",):
            opener += f" in a {content_type} composition"
        opener += "."
        sections.append(opener)
    if natural_desc and natural_desc.lower() != "image":
        sections.append(natural_desc)
    if anatomy_parts:
        sections.append("Anatomical details: " + "; ".join(anatomy_parts) + ".")
    if action_parts:
        sections.append("Actions and positioning: " + "; ".join(action_parts) + ".")
    return " ".join(sections) if sections else (natural_desc or "")


# ── Content Analyzer ──────────────────────────────────────────────────────

class ContentAnalyzer:
    @staticmethod
    def detect_content_type(tags: List[str], description: str = "") -> str:
        tag_set = set(t.lower() for t in tags)
        desc_lower = description.lower()
        if tag_set & {"comic", "panel", "speech bubble", "manga",
                      "4koma", "comic panel", "sequential", "page"} \
                or "comic" in desc_lower:
            return "comic"
        if tag_set & {"multiple characters", "2girls", "2boys", "group",
                      "3girls", "3boys", "crowd", "duo", "trio"} \
                or "multiple" in desc_lower:
            return "multi_character"
        if tag_set & {"solo", "1girl", "1boy", "solo focus", "portrait"}:
            return "solo"
        if tag_set & {"scenery", "landscape", "background",
                      "no humans", "nature", "cityscape"}:
            return "scene"
        return "general"


# ═════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═════════════════════════════════════════════════════════════════════════

class AutoCaptionPipeline:
    """
    High-throughput captioning pipeline.

    Performance design:
    - WD Tagger: batched ONNX (BATCH_SIZE images per GPU call)
    - BLIP-2: 3-beam search, inference_mode, SDPA when possible
    - CPU thread pool pre-loads / resizes images while GPU captioning runs
    - Videos: bulk-extracted to PNGs first, then batch-captioned
    """

    # How many images to WD-tag in one ONNX call
    WD_BATCH = 4
    # How many images the CPU thread pool pre-loads ahead of the GPU
    PREFETCH = 8

    def __init__(self, config=None, on_status: Callable = None):
        from core.config import ConfigManager, CaptioningConfig
        if config is None:
            config = ConfigManager().config.captioning
        self.config = config
        self._on_status = on_status
        self.wd_tagger = None
        self.natural_captioner = None
        self.analyzer = ContentAnalyzer()
        self._loaded = False
        self._models_loaded = False

    def _ensure_models_loaded(self):
        """Lazily load captioning models on first use (not at __init__ time)."""
        if self._models_loaded:
            return
        config = self.config
        self.wd_tagger = WDTagger(
            model_name=config.wd_model, threshold=config.wd_threshold)
        if config.method == "florence2":
            nl_model = config.florence_model
        else:
            nl_model = config.blip2_model
        self.natural_captioner = NaturalCaptioner(model_name=nl_model)
        self.load_models()
        self._models_loaded = True

    def _status(self, msg: str):
        logger.info(msg)
        if self._on_status:
            self._on_status(msg)

    def check_dependencies(self) -> Tuple[bool, str]:
        return _check_captioning_deps(self.config.method)

    def load_models(self):
        if self._loaded:
            return
        method = self.config.method
        ok, msg = self.check_dependencies()
        if not ok:
            self._status(msg)
        if method in ("wd_tagger", "combined"):
            self._status("Preparing WD Tagger…")
            self.wd_tagger.load(on_status=self._on_status)
        if method in ("blip2", "florence2", "combined"):
            label = "Florence-2" if method == "florence2" else "BLIP-2"
            self._status(f"Preparing {label}…")
            self.natural_captioner.load(on_status=self._on_status)
        self._loaded = True
        self._status("Captioning models ready — starting dataset…")

    # ── Single-image captioning ───────────────────────────────────────

    def caption_image(self, image_path: str,
                       pil_image: Image.Image = None) -> CaptionResult:
        self._ensure_models_loaded()
        result = CaptionResult(file_path=image_path)

        if pil_image is not None:
            img = pil_image.convert("RGB")
        else:
            ext = Path(image_path).suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                img = _extract_representative_frame(image_path)
                if img is None:
                    result.combined_caption = "video"
                    return result
                img = img.convert("RGB")
            else:
                img = Image.open(image_path).convert("RGB")

        method = self.config.method
        verbose = getattr(self.config, "verbose_description", True)

        if method in ("wd_tagger", "combined"):
            tags, confs = self.wd_tagger.predict(img)
            result.tags = tags[:self.config.max_tags]
            result.tag_confidences = {
                t: confs[t] for t in result.tags if t in confs}

        if method in ("blip2", "florence2", "combined"):
            nl_text = self.natural_captioner.caption(
                img, result.tags or None, verbose=verbose)
            if _is_degenerate_caption(nl_text):
                logger.warning(
                    f"Degenerate NL caption for {image_path}, "
                    f"falling back to tags only")
                nl_text = ""
            result.natural_description = nl_text

        all_tags = result.tags or []
        desc = result.natural_description or ""
        result.content_type = self.analyzer.detect_content_type(all_tags, desc)

        if verbose and all_tags:
            result.verbose_description = _build_verbose_description(
                all_tags, result.natural_description, result.content_type)

        result.combined_caption = self._build_caption(result)
        return result

    # ── High-throughput dataset captioning ─────────────────────────────

    def caption_dataset(self, items, progress_callback=None,
                         caption_dir: str = "") -> List[CaptionResult]:
        """
        Caption an entire dataset at maximum speed.

        Images are pre-loaded on CPU threads while the GPU processes the
        previous batch.  WD Tagger runs batched.  BLIP-2/Florence
        captions are generated per-image (batch text generation is
        unreliable with variable prompts).

        Videos are extracted to frames first, then each frame is
        captioned individually.
        """
        self._ensure_models_loaded()
        results: List[CaptionResult] = []

        if caption_dir:
            _cap_dir = Path(caption_dir)
            _cap_dir.mkdir(parents=True, exist_ok=True)
        else:
            _cap_dir = None

        frame_interval = getattr(self.config, "video_frame_interval", 5) or 5
        max_vid_frames = getattr(self.config, "video_max_frames", 60) or 60
        method = self.config.method
        verbose = getattr(self.config, "verbose_description", True)
        pipeline_mode = getattr(self.config, "pipeline_mode", "image")

        # ── Resume support: load progress from previous run ──
        _progress_file = _cap_dir / "caption_progress.json" if _cap_dir else None
        _completed_files: Set[str] = set()
        if _progress_file and _progress_file.is_file() and not self.config.overwrite_existing:
            try:
                _completed_files = set(
                    json.loads(_progress_file.read_text(encoding="utf-8"))
                    .get("completed", []))
                if _completed_files:
                    self._status(
                        f"Resuming: {len(_completed_files)} files already "
                        f"captioned from previous run")
            except Exception as e:
                logger.warning(f"Could not load caption progress: {e}")

        # ── Phase 1: Separate videos from images ──
        # Each task is: (item, path, pil_or_None, video_group_id, video_source, frame_idx)
        #
        # Pipeline mode controls how videos are handled:
        #   "image"  →  videos are split into individual frames that
        #               become standalone training images
        #   "video"  →  videos stay as-is; a representative frame is
        #               used for captioning and the caption describes
        #               the whole video clip
        image_tasks = []
        total_caption_count = 0
        video_groups = {}   # group_id -> {"source": path, "frame_count": int}

        self._status("Scanning dataset…")

        for item in items:
            if not item.is_valid:
                continue

            ext = Path(item.original_path).suffix.lower()

            if ext in VIDEO_EXTENSIONS:
                if pipeline_mode == "image":
                    # ── Image model: split video into frames ──
                    self._status(f"Extracting frames from {Path(item.original_path).name}…")
                    if _cap_dir:
                        vid_frames_dir = _cap_dir / "video_frames"
                    else:
                        from core.config import CACHE_DIR
                        vid_frames_dir = CACHE_DIR / "video_frames"

                    from core.video_utils import extract_frames as _extract_shared
                    saved = _extract_shared(
                        video_path=item.original_path,
                        out_dir=str(vid_frames_dir),
                        every_n_frames=frame_interval,
                        max_frames=max_vid_frames,
                        on_progress=None,
                    )

                    # Build group ID for this video
                    vid_stem = Path(item.original_path).stem
                    vid_hash = hashlib.md5(
                        item.original_path.encode()).hexdigest()[:8]
                    group_id = f"vid_{vid_stem}_{vid_hash}"
                    video_groups[group_id] = {
                        "source": item.original_path,
                        "frame_count": len(saved),
                    }

                    for fi, fp in enumerate(saved):
                        image_tasks.append(
                            (item, fp, None, group_id,
                             item.original_path, fi))
                    total_caption_count += len(saved)

                else:
                    # ── Video model: keep video as-is, caption via
                    # representative frame ──
                    rep_frame = _extract_representative_frame(
                        item.original_path)
                    if rep_frame is None:
                        logger.warning(
                            f"Cannot read video: {item.original_path}")
                        continue

                    # Skip if already captioned (check item + caption_dir + resume)
                    if not self.config.overwrite_existing:
                        existing_caption = item.caption_text
                        if not existing_caption and _cap_dir:
                            cap_file = _cap_dir / f"{Path(item.original_path).stem}.txt"
                            if cap_file.is_file():
                                existing_caption = cap_file.read_text(
                                    encoding="utf-8").strip()
                        if not existing_caption and item.original_path in _completed_files:
                            existing_caption = "(completed)"
                        if existing_caption:
                            results.append(CaptionResult(
                                file_path=item.original_path,
                                combined_caption=existing_caption))
                            total_caption_count += 1
                            continue

                    # Use original video path as the "image" path so
                    # the caption file sits next to the video.
                    image_tasks.append((
                        item, item.original_path, rep_frame,
                        "", "", -1))
                    total_caption_count += 1

            else:
                # Regular image — no video group
                path = (item.converted_path
                        if item.converted_path else item.original_path)
                p = Path(path)
                if not p.exists():
                    continue
                if p.is_dir():
                    path = item.original_path
                    if not Path(path).is_file():
                        continue

                # Skip if already captioned (check item + caption_dir + resume)
                if not self.config.overwrite_existing:
                    existing_caption = item.caption_text
                    if not existing_caption and _cap_dir:
                        cap_file = _cap_dir / f"{Path(path).stem}.txt"
                        if cap_file.is_file():
                            existing_caption = cap_file.read_text(
                                encoding="utf-8").strip()
                    if not existing_caption and path in _completed_files:
                        existing_caption = "(completed)"
                    if existing_caption:
                        results.append(CaptionResult(
                            file_path=path,
                            combined_caption=existing_caption))
                        total_caption_count += 1
                        if progress_callback:
                            progress_callback(
                                len(results),
                                total_caption_count + len(image_tasks),
                                item.original_path)
                        continue

                image_tasks.append((item, path, None, "", "", -1))
                total_caption_count += 1

        total_work = len(results) + len(image_tasks)

        # Extraction summary (#10): log breakdown before captioning
        n_video_frames = sum(
            1 for t in image_tasks if t[3])  # has group_id
        n_standalone = len(image_tasks) - n_video_frames
        n_already = len(results)
        n_groups = len(video_groups)
        summary_parts = []
        if n_video_frames:
            summary_parts.append(
                f"{n_video_frames} video frames from {n_groups} video(s)")
        if n_standalone:
            summary_parts.append(f"{n_standalone} standalone image(s)")
        if n_already:
            summary_parts.append(f"{n_already} already captioned")
        summary = ", ".join(summary_parts)
        self._status(
            f"Extraction complete. Captioning {len(image_tasks)} items "
            f"({summary}). Total: {total_work}.")
        for gid, ginfo in video_groups.items():
            self._status(
                f"  Video group '{gid}': "
                f"{ginfo['frame_count']} frames from "
                f"{Path(ginfo['source']).name}")

        if not image_tasks:
            return results

        # ── Phase 2–4: Chunked processing ──
        # MEMORY FIX: Instead of loading ALL images into RAM at once
        # (which can exhaust 32 GB with thousands of video frames),
        # we process in small chunks.  Each chunk:
        #   1) loads a handful of images from disk
        #   2) runs WD tagging on the chunk
        #   3) runs NL captioning on the chunk
        #   4) writes caption files
        #   5) releases all PIL images before the next chunk
        #
        # Chunk size is kept small (16) so peak RAM from images is
        # ~16 × ~12 MB ≈ 200 MB instead of thousands × 12 MB.

        CHUNK_SIZE = 16  # images loaded into RAM at a time
        use_wd = method in ("wd_tagger", "combined")
        use_nl = method in ("blip2", "florence2", "combined")
        wd_batch_size = self.WD_BATCH
        t0 = time.time()
        processed = 0

        for chunk_start in range(0, len(image_tasks), CHUNK_SIZE):
            chunk = image_tasks[chunk_start:chunk_start + CHUNK_SIZE]

            # ── Load chunk images ──
            chunk_loaded = []
            for _item, fpath, _pil, _gid, _vsrc, _fidx in chunk:
                if _pil is not None:
                    pil = _pil.convert("RGB") if _pil.mode != "RGB" else _pil
                else:
                    try:
                        pil = Image.open(fpath).convert("RGB")
                    except Exception as e:
                        logger.error(f"Cannot load {fpath}: {e}")
                        pil = None
                chunk_loaded.append((_item, fpath, pil, _gid, _vsrc, _fidx))

            # ── WD Tagger on this chunk ──
            if use_wd:
                pil_images = [t[2] for t in chunk_loaded]
                chunk_tags = []
                for bi in range(0, len(pil_images), wd_batch_size):
                    batch_imgs = pil_images[bi:bi + wd_batch_size]
                    safe_batch = [
                        im if im is not None else Image.new("RGB", (64, 64))
                        for im in batch_imgs
                    ]
                    batch_results = self.wd_tagger.predict_batch(safe_batch)
                    chunk_tags.extend(batch_results)
                    del safe_batch, batch_results
                del pil_images
            else:
                chunk_tags = [None] * len(chunk_loaded)

            # ── NL captioning + write captions for this chunk ──
            for i, (item, fpath, pil_img, gid, vsrc, fidx) in enumerate(
                    chunk_loaded):
                try:
                    result = CaptionResult(
                        file_path=fpath,
                        video_group_id=gid,
                        video_source=vsrc,
                        frame_index=fidx,
                    )

                    # Tags from batch phase
                    if chunk_tags[i] is not None:
                        tags, confs = chunk_tags[i]
                        result.tags = tags[:self.config.max_tags]
                        result.tag_confidences = {
                            t: confs[t] for t in result.tags if t in confs}

                    # Natural language
                    if use_nl and pil_img is not None:
                        nl_text = self.natural_captioner.caption(
                            pil_img, result.tags or None,
                            verbose=verbose)
                        if _is_degenerate_caption(nl_text):
                            logger.warning(
                                f"Degenerate NL caption for {fpath}, "
                                f"falling back to tags only")
                            nl_text = ""
                        result.natural_description = nl_text

                    # Content type
                    result.content_type = self.analyzer.detect_content_type(
                        result.tags or [], result.natural_description or "")

                    # Verbose expansion
                    if verbose and result.tags:
                        result.verbose_description = (
                            _build_verbose_description(
                                result.tags, result.natural_description,
                                result.content_type))

                    result.combined_caption = self._build_caption(result)
                    results.append(result)

                    # Write caption file
                    if _cap_dir is not None:
                        stem = Path(fpath).stem
                        cpath = _cap_dir / f"{stem}.txt"
                    elif (item.converted_path
                          and Path(item.converted_path).is_file()):
                        cpath = Path(item.converted_path).with_suffix(".txt")
                    else:
                        from core.config import CACHE_DIR
                        fb = CACHE_DIR / "captions_tmp"
                        fb.mkdir(parents=True, exist_ok=True)
                        cpath = fb / f"{Path(fpath).stem}.txt"

                    cpath.write_text(
                        result.combined_caption, encoding="utf-8")
                    item.caption_text = result.combined_caption
                    item.caption_path = str(cpath)
                    _completed_files.add(fpath)

                except Exception as e:
                    logger.error(f"Failed to caption {fpath}: {e}")

                if progress_callback:
                    progress_callback(
                        len(results), total_work, fpath)

            # Save resume progress after each chunk
            if _progress_file:
                try:
                    _progress_file.write_text(json.dumps(
                        {"completed": list(_completed_files)},
                        ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

            processed += len(chunk_loaded)

            # ── Release chunk memory ──
            # Explicitly close PIL images and drop references so the
            # GC can reclaim RAM before loading the next chunk.
            for _item, _fpath, pil, _gid, _vsrc, _fidx in chunk_loaded:
                if pil is not None:
                    try:
                        pil.close()
                    except Exception:
                        pass
            del chunk_loaded, chunk_tags, chunk

            # Speed report every few chunks
            if processed % (CHUNK_SIZE * 2) == 0 or processed == len(image_tasks):
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                self._status(
                    f"Captioned {processed}/{len(image_tasks)} "
                    f"({rate:.1f} img/s)")

        elapsed = time.time() - t0
        rate = len(image_tasks) / elapsed if elapsed > 0 else 0
        self._status(
            f"Done — captioned {len(image_tasks)} items in {elapsed:.0f}s "
            f"({rate:.1f} img/s)")

        # ── Phase 5: Write video group manifest ──
        # A JSON file that tells the trainer which frame images belong
        # to the same source video, so they can be trained as a group
        # rather than as unrelated images.
        if video_groups:
            manifest = {}
            for gid, meta in video_groups.items():
                frame_results = [
                    r for r in results if r.video_group_id == gid]
                manifest[gid] = {
                    "source_video": meta["source"],
                    "frame_count": meta["frame_count"],
                    "frame_interval": frame_interval,
                    "frames": [
                        {
                            "image": r.file_path,
                            "caption": r.combined_caption,
                            "frame_index": r.frame_index,
                        }
                        for r in sorted(frame_results,
                                        key=lambda r: r.frame_index)
                    ],
                }

            if not _cap_dir:
                from core.config import CACHE_DIR as _cache
                _fallback = Path(_cache) / "captions_tmp"
            else:
                _fallback = None
            manifest_dir = _cap_dir if _cap_dir else _fallback
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "video_groups.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8")
            self._status(
                f"Wrote video group manifest: "
                f"{len(video_groups)} video(s), "
                f"{sum(m['frame_count'] for m in video_groups.values())} "
                f"total frames")

        return results

    # ── Caption assembly ──────────────────────────────────────────────

    def _build_caption(self, result: CaptionResult) -> str:
        parts = []
        if self.config.prepend_trigger_word and self.config.trigger_word:
            parts.append(self.config.trigger_word)

        # Video-frame prefix — ties all frames from the same video together
        # so the training model learns they share a continuous scene.
        if result.video_group_id:
            src_name = Path(result.video_source).stem if result.video_source else result.video_group_id
            parts.append(
                f"[video:{result.video_group_id} "
                f"frame {result.frame_index} "
                f"source:{src_name}]")

        fmt = self.config.caption_format
        if fmt == "tags_only":
            if result.tags:
                parts.append(", ".join(result.tags))
        elif fmt == "natural_only":
            desc = result.verbose_description or result.natural_description
            if desc:
                parts.append(desc)
        elif fmt == "tags_and_natural":
            desc = result.verbose_description or result.natural_description
            if desc:
                parts.append(desc)
            if result.tags:
                parts.append(", ".join(result.tags))
        return ", ".join(parts) if parts else "image"

    # ── Cleanup ───────────────────────────────────────────────────────

    def unload_models(self):
        if self.wd_tagger is not None:
            self.wd_tagger.unload()
        if self.natural_captioner is not None:
            self.natural_captioner.unload()
        self._loaded = False
        self._models_loaded = False
        from core.gpu_utils import flush_gpu_memory
        flush_gpu_memory()

    def generate_dataset_report(self, results: List[CaptionResult]) -> Dict:
        report = {
            "total_captioned": len(results),
            "content_types": {},
            "common_tags": {},
            "avg_tags_per_image": 0,
            "coverage": {"has_tags": 0, "has_description": 0, "has_both": 0}
        }
        all_tags = []
        for r in results:
            ct = r.content_type or "unknown"
            report["content_types"][ct] = report["content_types"].get(ct, 0) + 1
            if r.tags:
                all_tags.extend(r.tags)
                report["coverage"]["has_tags"] += 1
            if r.natural_description:
                report["coverage"]["has_description"] += 1
            if r.tags and r.natural_description:
                report["coverage"]["has_both"] += 1
        from collections import Counter
        report["common_tags"] = dict(Counter(all_tags).most_common(30))
        report["avg_tags_per_image"] = len(all_tags) / max(1, len(results))
        return report
