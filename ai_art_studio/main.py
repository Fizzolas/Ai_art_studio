#!/usr/bin/env python3
"""
AI Art Studio - Local Training & Generation
Entry point for the application.
"""
import sys
import os
import platform
import traceback
from pathlib import Path

# ── CUDA memory-fragmentation prevention ─────────────────────────────
# Must be set BEFORE any torch import.  run.bat also sets this, but
# if the user launches  python main.py  directly, we still need it.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.6",
)
# Tell HF hub to use a consistent cache location
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
# Suppress HF hub symlinks warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Suppress known deprecation warnings from dependencies
import warnings
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*use_fast.*is unset.*", category=FutureWarning)

# Ensure the project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Centralized logging
from core.logger import get_logger, get_log_file
from core.config import LOGS_DIR

logger = get_logger("main")


def _handle_exception(exc_type, exc_value, exc_tb):
    """Log uncaught exceptions to file before crashing."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"UNCAUGHT EXCEPTION:\n{tb_str}")

    # Also show a user-facing error dialog if QApplication exists
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Unexpected Error")
            msg.setText("The application encountered an unexpected error and must close.")
            msg.setDetailedText(tb_str)
            msg.setInformativeText(f"A full error report has been saved to:\n{get_log_file()}")
            msg.exec()
    except Exception:
        pass

    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _handle_exception


def _log_startup_info():
    logger.info("=" * 60)
    logger.info("AI Art Studio starting up")
    try:
        version_file = Path(__file__).parent / "VERSION"
        logger.info(f"Version: {version_file.read_text().strip()}")
    except Exception:
        logger.info("Version: unknown")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("GPU: Not available (CPU mode)")
    except ImportError:
        logger.warning("PyTorch not installed")
    logger.info("=" * 60)


def enable_dpi_awareness():
    """
    Configure Qt environment variables for crisp high-DPI rendering.
    Qt 6 already calls SetProcessDpiAwarenessContext() internally,
    so we do NOT call it ourselves — doing so first would cause
    Qt's own call to fail with "Access is denied".
    """
    # Tell Qt to honour the OS scale factor and derive font sizes from it
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    # Use passthrough scaling so fractional scale factors (125 %, 150 %)
    # don't produce blurry text
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")


def check_dependencies():
    """Check that critical dependencies are installed."""
    missing = []

    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available! GPU acceleration will be disabled.")
            logger.warning("Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    except ImportError:
        missing.append("torch (with CUDA)")

    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        missing.append("PyQt6")

    try:
        import diffusers
    except ImportError:
        missing.append("diffusers")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        print("Or run: python setup.py")
        return False

    return True


def main():
    _log_startup_info()

    # DPI awareness must be set BEFORE QApplication is created
    enable_dpi_awareness()

    if not check_dependencies():
        sys.exit(1)

    # onnxruntime CUDA provider check (#20)
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            logger.warning("onnxruntime-gpu: CUDA provider not available. WD Tagger will use CPU.")
    except ImportError:
        logger.warning("onnxruntime not installed. WD Tagger captioning will not be available.")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from PyQt6.QtCore import Qt
    from gui.main_window import MainWindow
    from gui.theme import DARK_THEME

    # Enable high-DPI pixmap support (Qt6 default but explicit is safe)
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass  # Older PyQt6 builds may not expose this

    app = QApplication(sys.argv)
    app.setApplicationName("AI Art Studio")
    app.setOrganizationName("LocalAI")

    # Suppress harmless Qt font warnings caused by stylesheet px sizes.
    # When a stylesheet sets font-size in px, QFont.pointSize() returns -1
    # and Qt logs "QFont::setPointSize: Point size <= 0 (-1)".  Cosmetic only.
    from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
    _original_handler = None

    def _qt_msg_filter(msg_type, context, message):
        if "QFont::setPointSize" in message:
            return  # swallow
        # Forward everything else to the default handler
        if _original_handler:
            _original_handler(msg_type, context, message)
        else:
            # Fallback: print to stderr
            import sys as _sys
            print(message, file=_sys.stderr)

    _original_handler = qInstallMessageHandler(_qt_msg_filter)

    # Scale the base font size to the primary screen's logical DPI.
    # 96 DPI is the "100 %" Windows baseline; on a 144-DPI display the
    # font will be 1.5x larger automatically.
    screen = app.primaryScreen()
    if screen:
        logical_dpi = screen.logicalDotsPerInch()
        scale = logical_dpi / 96.0
    else:
        scale = 1.0

    base_pt = 10
    scaled_pt = max(8, int(base_pt * scale))

    # Prefer Segoe UI (Windows), fall back to Inter / system sans-serif
    font = QFont("Segoe UI", scaled_pt)
    if not font.exactMatch():
        font = QFont("Inter", scaled_pt)
    app.setFont(font)

    # Apply theme
    app.setStyleSheet(DARK_THEME)

    # First-run wizard
    from core.config import ConfigManager
    cfg = ConfigManager()
    if not cfg.config_file_exists():
        from gui.setup_wizard import SetupWizard
        wizard = SetupWizard(cfg)
        wizard.exec()

    # Create and show main window
    window = MainWindow()
    window.show()

    logger.info("AI Art Studio started  (DPI scale=%.2f, font=%dpt)", scale, scaled_pt)

    sys.exit(app.exec())


if __name__ == "__main__":
    # Global crash handler — logs the full traceback and shows a dialog
    # so the app never silently vanishes on the user.
    try:
        main()
    except Exception as exc:
        # Log to file even if the GUI never started
        crash_log = LOGS_DIR / "crash.log"
        with open(crash_log, "a") as f:
            f.write(f"\n{'='*72}\n")
            f.write(f"CRASH  {__import__('datetime').datetime.now().isoformat()}\n")
            traceback.print_exc(file=f)

        # Try to show a message box (Qt may or may not be available)
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(
                None, "AI Art Studio — Fatal Error",
                f"The application encountered a fatal error:\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                f"Details have been saved to:\n{crash_log}",
            )
        except Exception:
            pass

        # Print to console as well
        print(f"\nFATAL ERROR: {exc}")
        print(f"Crash log saved to: {crash_log}")
        traceback.print_exc()
        sys.exit(1)
