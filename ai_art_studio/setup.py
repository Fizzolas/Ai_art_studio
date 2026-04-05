#!/usr/bin/env python3
"""
AI Art Studio — Auto Installer
Run: python setup.py
"""
import sys, os, subprocess, platform, re
from pathlib import Path

INSTALL_DIR = Path(__file__).parent.resolve()
VENV_DIR = INSTALL_DIR / "venv"
LOG_FILE = INSTALL_DIR / "setup_log.txt"

BASE_PACKAGES = [
    "PyQt6>=6.6.0",
    "Pillow>=10.0.0",
    "pillow-heif",
    "pillow-avif-plugin",
    "safetensors>=0.4.0",
    "diffusers>=0.27.0",
    "accelerate>=0.28.0",
    "peft>=0.10.0",
    "huggingface-hub>=0.22.0",
    "psutil>=5.9.0",
    "requests>=2.31.0",
    "tqdm>=4.66.0",
    "numpy>=1.26.0",
]

OPTIONAL_PACKAGES = {
    1: {
        "name": "BLIP-2 captioning",
        "packages": ["transformers>=4.38.0", "accelerate>=0.28.0"],
    },
    2: {
        "name": "WD Tagger captioning (ONNX)",
        "packages": ["onnxruntime-gpu>=1.17.0", "timm>=0.9.0"],
    },
    3: {
        "name": "Florence-2 captioning",
        "packages": ["einops", "transformers>=4.40.0"],
    },
    4: {
        "name": "Video support",
        "packages": ["opencv-python>=4.9.0"],
    },
    5: {
        "name": "Image deduplication",
        "packages": ["imagehash>=4.3.0"],
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────

def log(msg: str):
    """Print to console and append to log file."""
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, logging output."""
    log(f"  > {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(result.stdout + "\n")
    if result.stderr.strip():
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(result.stderr + "\n")
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def pip_install(python: str, packages: list, extra_args: str = "", check: bool = True) -> bool:
    """Install packages via pip. Returns True on success."""
    pkg_str = " ".join(f'"{p}"' if any(c in p for c in ">=<!=") else p for p in packages)
    cmd = f'"{python}" -m pip install {extra_args} {pkg_str}'
    try:
        run_cmd(cmd, check=check)
        for p in packages:
            name = re.split(r"[>=<!\[]", p)[0]
            log(f"[OK] {name} installed")
        return True
    except Exception as e:
        for p in packages:
            name = re.split(r"[>=<!\[]", p)[0]
            log(f"[ERROR] Failed to install {name}: {e}")
        return False


# ── Detection ────────────────────────────────────────────────────────────

def detect_python_version():
    """Check Python version. Require 3.10+, warn on 3.13+."""
    ver = sys.version_info
    log(f"[SETUP] Python {ver.major}.{ver.minor}.{ver.micro}")
    if ver.minor < 10:
        log("[ERROR] Python 3.10+ required!")
        sys.exit(1)
    if ver.minor >= 13:
        log(
            "\n" + "!" * 60 + "\n"
            "  WARNING: Python 3.13 detected.\n"
            "  Some packages (xformers, bitsandbytes, onnxruntime-gpu)\n"
            "  may not have pre-built wheels for 3.13 yet.\n"
            "  The core app will still work.\n"
            "  For maximum compatibility use Python 3.10 or 3.11.\n"
            + "!" * 60
        )
    return ver


def detect_cuda() -> str:
    """Detect CUDA version from nvidia-smi output. Returns '12.x', '11.x', or ''."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
            if match:
                major = int(match.group(1))
                if major >= 12:
                    return "12.x"
                elif major >= 11:
                    return "11.x"
    except Exception:
        pass

    # Fallback: check nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                match = re.search(r"release\s+(\d+)", line)
                if match:
                    major = int(match.group(1))
                    if major >= 12:
                        return "12.x"
                    elif major >= 11:
                        return "11.x"
    except Exception:
        pass

    # Fallback: env var
    cuda_path = os.environ.get("CUDA_PATH", "")
    if "12" in cuda_path:
        return "12.x"
    elif "11" in cuda_path:
        return "11.x"

    return ""


# ── Core install steps ───────────────────────────────────────────────────

def create_venv(python_exe: str) -> str:
    """Create or reuse a virtual environment. Returns path to venv python."""
    if not VENV_DIR.exists():
        log("[SETUP] Creating virtual environment...")
        run_cmd(f'"{python_exe}" -m venv "{VENV_DIR}"')
        log("[OK] Virtual environment created")
    else:
        log("[SETUP] Reusing existing virtual environment")

    if platform.system() == "Windows":
        venv_python = str(VENV_DIR / "Scripts" / "python.exe")
    else:
        venv_python = str(VENV_DIR / "bin" / "python")

    # Ensure pip is available and up-to-date
    run_cmd(f'"{venv_python}" -m ensurepip --upgrade', check=False)
    run_cmd(f'"{venv_python}" -m pip install --upgrade pip', check=False)
    return venv_python


def install_torch(python: str, cuda_ver: str) -> bool:
    """Install PyTorch with appropriate CUDA support."""
    if cuda_ver == "12.x":
        torch_url = "https://download.pytorch.org/whl/cu121"
        log("[SETUP] Installing PyTorch with CUDA 12.x support...")
    elif cuda_ver == "11.x":
        torch_url = "https://download.pytorch.org/whl/cu118"
        log("[SETUP] Installing PyTorch with CUDA 11.x support...")
    else:
        torch_url = ""
        log("[SETUP] No GPU detected. Installing CPU-only PyTorch...")

    cmd = f'"{python}" -m pip install "torch==2.3.1" "torchvision" "torchaudio"'
    if torch_url:
        cmd += f" --index-url {torch_url}"

    try:
        run_cmd(cmd)
        log("[OK] PyTorch installed")
        return True
    except Exception as e:
        log(f"[ERROR] Failed to install PyTorch: {e}")
        return False


def verify_torch(python: str) -> bool:
    """Verify GPU is accessible after torch install."""
    log("[SETUP] Verifying PyTorch GPU access...")
    try:
        result = run_cmd(
            f'"{python}" -c "import torch; print(torch.cuda.is_available())"',
            check=False,
        )
        available = "True" in result.stdout
        if available:
            log("[OK] CUDA is available in PyTorch")
        else:
            log("[WARN] CUDA not available — GPU acceleration will be disabled")
        return available
    except Exception as e:
        log(f"[WARN] Could not verify torch CUDA: {e}")
        return False


def install_xformers(python: str) -> bool:
    """Install xformers for memory-efficient attention."""
    log("[SETUP] Installing xformers...")
    return pip_install(python, ["xformers"], check=False)


def install_base_packages(python: str) -> bool:
    """Install all required base packages."""
    log("[SETUP] Installing base packages...")
    success = True
    # Install in small batches for better error isolation
    for pkg in BASE_PACKAGES:
        if not pip_install(python, [pkg], check=False):
            success = False
    return success


def show_menu() -> list:
    """Show selective install menu and return chosen package lists."""
    print("\n" + "=" * 60)
    print("  Optional components:")
    print("=" * 60)
    print("  [1] BLIP-2 captioning (requires ~2GB VRAM for load) — adds: transformers, accelerate")
    print("  [2] WD Tagger captioning (ONNX) — adds: onnxruntime-gpu, timm")
    print("  [3] Florence-2 captioning — adds: transformers[torch], einops")
    print("  [4] Video support (frame extraction) — adds: opencv-python")
    print("  [5] Image deduplication — adds: imagehash")
    print("  [6] All of the above")
    print("  [7] Skip optional components")
    print()

    try:
        choice = input("  Enter numbers separated by commas (e.g. 1,2,4): ").strip()
    except (EOFError, KeyboardInterrupt):
        return []

    if not choice:
        return []

    packages = []
    try:
        nums = [int(x.strip()) for x in choice.split(",")]
    except ValueError:
        log("[WARN] Invalid input, skipping optional components")
        return []

    if 7 in nums:
        return []
    if 6 in nums:
        nums = [1, 2, 3, 4, 5]

    for n in nums:
        if n in OPTIONAL_PACKAGES:
            packages.extend(OPTIONAL_PACKAGES[n]["packages"])

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in packages:
        name = re.split(r"[>=<!\[]", p)[0]
        if name not in seen:
            seen.add(name)
            deduped.append(p)

    return deduped


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # Clear/init log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("AI Art Studio — Setup Log\n" + "=" * 60 + "\n")

    print("=" * 60)
    print("  AI Art Studio — Auto Installer")
    print("=" * 60)

    # Step 1: Check Python
    detect_python_version()

    # Step 2: Detect CUDA
    cuda_ver = detect_cuda()
    log(f"[SETUP] CUDA detected: {cuda_ver or 'None (CPU mode)'}")

    # Step 3: Create/reuse venv
    python = create_venv(sys.executable)

    # Step 4: Install torch first
    log("\n[SETUP] Installing PyTorch (step 1/4)...")
    install_torch(python, cuda_ver)

    # Step 5: Install xformers
    log("\n[SETUP] Installing xformers (step 2/4)...")
    install_xformers(python)

    # Step 6: Install diffusers + base packages
    log("\n[SETUP] Installing base packages (step 3/4)...")
    install_base_packages(python)

    # Step 7: Verify torch GPU
    verify_torch(python)

    # Step 8: Optional components
    log("\n[SETUP] Optional components (step 4/4)...")
    optional = show_menu()
    if optional:
        log(f"[SETUP] Installing {len(optional)} optional package(s)...")
        for pkg in optional:
            name = re.split(r"[>=<!\[]", pkg)[0]
            log(f"[SETUP] Installing {name}...")
            pip_install(python, [pkg], check=False)
    else:
        log("[SETUP] Skipping optional components")

    # Done
    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    log(f"\n[OK] Setup finished. Log saved to: {LOG_FILE}")
    print("\nRun run.bat to start the application")
    if platform.system() != "Windows":
        print("  Or: source venv/bin/activate && python main.py")


if __name__ == "__main__":
    main()
