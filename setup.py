#!/usr/bin/env python3
"""
AI Art Studio - Automated Setup Script
Detects your hardware and installs everything needed.
Run: python setup.py
"""
import subprocess
import sys
import os
import platform
import tempfile
import shutil
from pathlib import Path


def run(cmd, check=True):
    """Run a shell command. Uses shell=True for convenience."""
    print(f"  > {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=False)


def pip_install(python, *packages, extra_args="", check=True):
    """Install packages using python -m pip (avoids Windows pip.exe self-lock).
    All version specifiers are quoted to prevent shell redirect interpretation."""
    pkg_list = " ".join(f'"{p}"' if any(c in p for c in ">=<!=") else p for p in packages)
    return run(f'"{python}" -m pip install {extra_args} {pkg_list}', check=check)


def detect_cuda():
    """Detect CUDA version."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                parts = line.split("release")[-1].strip().split(",")[0].strip()
                return parts
    except Exception:
        pass

    # Check env
    cuda_path = os.environ.get("CUDA_PATH", "")
    if "12.4" in cuda_path:
        return "12.4"
    elif "12" in cuda_path:
        return "12.x"
    elif "11" in cuda_path:
        return "11.x"

    return None


def sanitize_requirements(req_path: Path, work_dir: Path) -> Path:
    """Copy a requirements.txt, stripping any '-e .' / editable-install lines
    and file:// references that would try to build the current project."""
    clean_path = work_dir / "requirements_clean.txt"
    kept = []
    with open(req_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            # Skip blank, comments, editable installs pointing at current dir,
            # and bare '.' entries that resolve to the project root.
            if not stripped or stripped.startswith("#"):
                kept.append(line)
                continue
            low = stripped.lower()
            if low in (".", "-e .", "-e."):
                print(f"    [skip] {stripped}  (would install project root)")
                continue
            if low.startswith("-e") and ("file:///" in low or low.endswith(".")):
                print(f"    [skip] {stripped}  (editable-install of local path)")
                continue
            kept.append(line)
    with open(clean_path, "w", encoding="utf-8") as f:
        f.writelines(kept)
    return clean_path


def main():
    print("=" * 60)
    print("  AI Art Studio - Setup")
    print("=" * 60)

    # Check Python version
    py_ver = sys.version_info
    print(f"\nPython: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver.minor < 10:
        print("ERROR: Python 3.10+ required!")
        sys.exit(1)

    # ── Python 3.13 compatibility warning ────────────────────────────────
    # PyTorch wheels for Python 3.13 are only available for CUDA 11.8 and
    # 12.4 (nightly/recent) as of early 2026.  Some extensions (xformers,
    # bitsandbytes, onnxruntime-gpu) may not have 3.13 wheels yet.
    if py_ver.minor >= 13:
        print(
            "\n" + "!" * 60 + "\n"
            "  WARNING: Python 3.13 detected.\n"
            "  Some packages (xformers, bitsandbytes, onnxruntime-gpu)\n"
            "  may not have pre-built wheels for 3.13 yet.\n"
            "  If any optional installs fail below, the core app will\n"
            "  still work — you'll just miss those optimisations.\n"
            "  For maximum compatibility use Python 3.10 or 3.11.\n"
            + "!" * 60 + "\n"
        )

    # Detect CUDA
    cuda_ver = detect_cuda()
    print(f"CUDA: {cuda_ver or 'Not found'}")

    # Determine PyTorch install URL
    if cuda_ver and cuda_ver.startswith("12"):
        torch_url = "https://download.pytorch.org/whl/cu124"
        print("Installing PyTorch with CUDA 12.4 support...")
    elif cuda_ver and cuda_ver.startswith("11"):
        torch_url = "https://download.pytorch.org/whl/cu118"
        print("Installing PyTorch with CUDA 11.8 support...")
    else:
        torch_url = "https://download.pytorch.org/whl/cu124"
        print("CUDA not detected, installing CUDA 12.4 PyTorch (you have CUDA_PATH set)...")

    # Create venv
    venv_path = Path(__file__).parent / "venv"
    if not venv_path.exists():
        print("\nCreating virtual environment...")
        run(f'"{sys.executable}" -m venv "{venv_path}"')

    # Determine python path inside venv
    if platform.system() == "Windows":
        python = str(venv_path / "Scripts" / "python.exe")
    else:
        python = str(venv_path / "bin" / "python")

    # Ensure pip is available in the venv (some Python installs omit it)
    run(f'"{python}" -m ensurepip --upgrade', check=False)

    # Upgrade pip using python -m pip (avoids Windows self-overwrite lock)
    print("\nUpgrading pip...")
    run(f'"{python}" -m pip install --upgrade pip', check=False)

    # ── 1. Install PyTorch with CUDA FIRST ────────────────────────────────
    # Install torch from the CUDA wheel index BEFORE anything else so that
    # later packages see it and don't pull a CPU-only wheel from default PyPI.
    print("\nInstalling PyTorch with CUDA support (this may take a while)...")
    print(f"  Using wheel index: {torch_url}")
    run(f'"{python}" -m pip install torch torchvision torchaudio'
        f' --index-url {torch_url}')

    # ── 2. Install everything else ────────────────────────────────────────
    # Core dependencies
    print("\nInstalling core dependencies...")
    pip_install(python, "diffusers>=0.31.0", "transformers>=4.40.0", "accelerate>=0.30.0")
    pip_install(python, "safetensors", "peft>=0.10.0", "huggingface-hub>=0.23.0")

    # Image processing
    print("\nInstalling image processing...")
    pip_install(python, "Pillow>=10.0", "opencv-python>=4.8.0", "imageio", "imageio-ffmpeg")
    pip_install(python, "pillow-heif", check=False)  # Optional HEIC support

    # GUI
    print("\nInstalling GUI framework...")
    pip_install(python, "PyQt6>=6.6.0")

    # Performance
    print("\nInstalling performance optimizations...")
    pip_install(python, "xformers", check=False)  # May fail on some systems
    pip_install(python, "bitsandbytes>=0.43.0", check=False)

    # Captioning
    print("\nInstalling captioning dependencies...")
    pip_install(python, "onnxruntime-gpu>=1.17.0", "pandas")

    # Utilities
    pip_install(python, "numpy", "tqdm", "psutil", "GPUtil")

    # ── 3. kohya-ss sd-scripts ────────────────────────────────────────────
    # Their requirements.txt may contain "-e ." which would try to build
    # OUR setup.py as an editable package and recursively crash pip.
    # We sanitize the file to strip those lines before installing.
    print("\nInstalling kohya-ss training scripts...")
    sd_scripts_dir = Path(__file__).parent / "sd-scripts"
    if not sd_scripts_dir.exists():
        run(f'git clone https://github.com/kohya-ss/sd-scripts.git "{sd_scripts_dir}"', check=False)
    if sd_scripts_dir.exists():
        req_file = sd_scripts_dir / "requirements.txt"
        if req_file.exists():
            print("  Sanitizing requirements.txt (removing editable-install lines)...")
            clean_req = sanitize_requirements(req_file, sd_scripts_dir)
            run(f'"{python}" -m pip install -r "{clean_req}"', check=False)
        else:
            print("  (no requirements.txt found in sd-scripts, skipping)")

    # ── 4. Re-pin PyTorch CUDA build ──────────────────────────────────────
    # Some packages above may have pulled a CPU-only torch wheel as a
    # dependency.  Force-reinstall from the CUDA index to guarantee the
    # GPU build is what ends up in the venv.
    print("\nRe-pinning PyTorch with CUDA support...")
    run(f'"{python}" -m pip install torch torchvision torchaudio'
        f' --index-url {torch_url} --force-reinstall --no-deps')

    # ── 5. Verify CUDA is actually available ──────────────────────────────
    print("\nVerifying CUDA support...")
    verify_result = subprocess.run(
        [python, "-c",
         "import torch; print(f'PyTorch {torch.__version__}'); "
         "print(f'CUDA available: {torch.cuda.is_available()}'); "
         "print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else "
         "print('WARNING: CUDA not available - torch may be CPU-only!')"],
        capture_output=True, text=True
    )
    print(verify_result.stdout)
    if verify_result.stderr:
        print(verify_result.stderr)

    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print(f"\nTo run the application:")
    if platform.system() == "Windows":
        print(f'  .\\venv\\Scripts\\python.exe main.py')
    else:
        print(f'  ./venv/bin/python main.py')
    print(f"\nOr activate the venv first:")
    if platform.system() == "Windows":
        print(f'  .\\venv\\Scripts\\activate')
    else:
        print(f'  source venv/bin/activate')
    print(f'  python main.py')


if __name__ == "__main__":
    main()
