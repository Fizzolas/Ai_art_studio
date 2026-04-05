"""
Main application window with tabbed interface.
Fully responsive, scrollable, polished for any screen size.
Tabs: Dataset | Training | Generation | Settings

Key improvements over v1:
- Every panel is scrollable — no content cutoff on small screens
- Collapsible sections so power users can hide what they don't need
- Drag-and-drop file import on the Dataset tab
- Image gallery grid instead of a single preview
- Keyboard shortcuts (Ctrl+G = generate, Ctrl+T = start training, etc.)
- Proper minimum sizes, stretch factors, and size policies
- Status bar with VRAM gauge, GPU temp, and generation seed
- All widgets auto-save on change via config manager
"""
import sys
import os
import time
from datetime import timedelta
from pathlib import Path
from functools import partial
from io import BytesIO

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QSplitter,
    QTextEdit, QProgressBar, QFileDialog, QListWidget, QListWidgetItem,
    QGridLayout, QFrame, QMessageBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QLineEdit, QCheckBox, QSizePolicy, QApplication,
    QPlainTextEdit, QStackedWidget, QToolButton, QStatusBar,
    QMenu, QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QInputDialog,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QMimeData,
    QPropertyAnimation, QAbstractAnimation,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QFont, QAction, QKeySequence,
    QShortcut, QDragEnterEvent, QDropEvent, QColor,
)

from gui.theme import DARK_THEME
from gui.widgets import (
    LabeledSlider, LabeledCombo, LabeledCheck, PathSelector,
    StatusCard, Separator, CollapsibleSection, make_scroll_panel,
)
from core.config import ConfigManager
from core.logger import get_logger

logger = get_logger(__name__)


# ── Worker Threads ──────────────────────────────────────────────────────────

class DatasetWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, dataset_manager, action="scan", **kwargs):
        super().__init__()
        self.dm = dataset_manager
        self.action = action
        self.kwargs = kwargs

    def run(self):
        try:
            if self.action == "scan":
                result = self.dm.scan_directory(
                    progress_callback=lambda c, t, f: self.progress.emit(c, t, f)
                )
                self.finished.emit(result)
            elif self.action == "convert":
                result = self.dm.validate_and_convert(
                    progress_callback=lambda c, t, f: self.progress.emit(c, t, f),
                    **self.kwargs
                )
                self.finished.emit(result)
            elif self.action == "prepare":
                result = self.dm.prepare_training_dir(**self.kwargs)
                self.finished.emit(result)
        except Exception as e:
            import traceback
            from core.logger import get_logger
            get_logger("worker").error(
                f"{self.__class__.__name__} failed: {e}\n{traceback.format_exc()}"
            )
            self.error.emit(str(e))


class CaptionWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)           # status / log messages from the pipeline

    def __init__(self, pipeline, items, caption_dir: str = ""):
        super().__init__()
        self.pipeline = pipeline
        self.items = items
        self.caption_dir = caption_dir

        # Wire the pipeline's on_status callback so every status message
        # (dependency checks, model downloads, load progress) reaches the GUI.
        self.pipeline._on_status = lambda msg: self.log.emit(msg)

    def run(self):
        try:
            results = self.pipeline.caption_dataset(
                self.items,
                progress_callback=lambda c, t, f: self.progress.emit(c, t, f),
                caption_dir=self.caption_dir,
            )
            self.finished.emit(results)
        except Exception as e:
            import traceback
            from core.logger import get_logger
            get_logger("worker").error(
                f"{self.__class__.__name__} failed: {e}\n{traceback.format_exc()}"
            )
            self.error.emit(str(e))
        finally:
            # Flush GPU memory when captioning thread exits
            try:
                from core.gpu_utils import flush_gpu_memory
                flush_gpu_memory()
            except Exception:
                pass


class TrainingWorker(QThread):
    progress = pyqtSignal(int, int, float, float)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self):
        try:
            self.job.on_progress = lambda s, t, l, lr: self.progress.emit(s, t, l, lr)
            self.job.on_log = lambda m: self.log.emit(m)
            self.job.on_complete = lambda p: self.finished.emit(p)
            self.job.on_error = lambda e: self.error.emit(e)

            from core.config import ConfigManager
            hw = ConfigManager().config.hardware
            self.job.start(hw)

            while self.job.is_running:
                self.msleep(500)

        except Exception as e:
            import traceback
            from core.logger import get_logger
            get_logger("worker").error(
                f"{self.__class__.__name__} failed: {e}\n{traceback.format_exc()}"
            )
            self.error.emit(str(e))
        finally:
            # Flush GPU memory when the training thread exits
            from core.gpu_utils import flush_gpu_memory
            flush_gpu_memory()


class GenerationWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, generator, gen_type="image", **kwargs):
        super().__init__()
        self.generator = generator
        self.gen_type = gen_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.gen_type == "image":
                self.kwargs["callback"] = lambda s, t: self.progress.emit(s, t)
                images = self.generator.generate(**self.kwargs)
                self.finished.emit(images)
            elif self.gen_type == "video":
                self.kwargs["callback"] = lambda s, t: self.progress.emit(s, t)
                frames = self.generator.generate(**self.kwargs)
                self.finished.emit([frames])
        except Exception as e:
            import traceback
            from core.logger import get_logger
            get_logger("worker").error(
                f"{self.__class__.__name__} failed: {e}\n{traceback.format_exc()}"
            )
            self.error.emit(str(e))
        finally:
            # Prevent QThread + GPU memory leak: flush CUDA cache after
            # every generation pass regardless of success/failure.
            from core.gpu_utils import flush_gpu_memory
            flush_gpu_memory()


class ModelLoadWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, generator, model_path, model_type):
        super().__init__()
        self.generator = generator
        self.model_path = model_path
        self.model_type = model_type

    def run(self):
        try:
            self.generator.load_model(
                self.model_path, self.model_type,
                on_progress=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit(self.model_path)
        except Exception as e:
            import traceback
            from core.logger import get_logger
            get_logger("worker").error(
                f"{self.__class__.__name__} failed: {e}\n{traceback.format_exc()}"
            )
            self.error.emit(str(e))
        finally:
            from core.gpu_utils import flush_gpu_memory
            flush_gpu_memory()


class ModelDownloadWorker(QThread):
    """Download a HuggingFace model in the background."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)  # model_key
    error = pyqtSignal(str, str)  # model_key, error message

    def __init__(self, model_key: str, repo_id: str):
        super().__init__()
        self.model_key = model_key
        self.repo_id = repo_id

    def run(self):
        try:
            from huggingface_hub import snapshot_download
            import os
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
            self.progress.emit(f"Downloading {self.repo_id}...")
            snapshot_download(repo_id=self.repo_id, repo_type="model")
            self.finished.emit(self.model_key)
        except Exception as e:
            import traceback
            from core.logger import get_logger
            get_logger("worker").error(
                f"{self.__class__.__name__} failed: {e}\n{traceback.format_exc()}"
            )
            self.error.emit(self.model_key, str(e))


# ── LossGraphWidget ────────────────────────────────────────────────────────

class LossGraphWidget(QWidget):
    """Simple real-time loss graph."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []
        self._max_points = 500
        self.setStyleSheet("background-color: #010409; border: 1px solid #21262d; border-radius: 8px;")

    def add_point(self, loss: float):
        self._data.append(loss)
        if len(self._data) > self._max_points:
            self._data = self._data[-self._max_points:]
        self.update()

    def clear(self):
        self._data.clear()
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QPen, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 10

        if len(self._data) < 2:
            painter.setPen(QColor("#484f58"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loss data will appear here")
            painter.end()
            return

        data = self._data
        min_v = min(data)
        max_v = max(data)
        if max_v == min_v:
            max_v = min_v + 0.01

        # Draw grid lines
        painter.setPen(QPen(QColor("#21262d"), 1))
        for i in range(5):
            y = margin + (h - 2 * margin) * i / 4
            painter.drawLine(int(margin), int(y), int(w - margin), int(y))

        # Draw loss curve
        pen = QPen(QColor("#e94560"), 2)
        painter.setPen(pen)

        n = len(data)
        x_step = (w - 2 * margin) / max(1, n - 1)

        for i in range(1, n):
            x0 = margin + (i - 1) * x_step
            x1 = margin + i * x_step
            y0 = margin + (h - 2 * margin) * (1 - (data[i-1] - min_v) / (max_v - min_v))
            y1 = margin + (h - 2 * margin) * (1 - (data[i] - min_v) / (max_v - min_v))
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        # Labels
        painter.setPen(QColor("#8b949e"))
        painter.drawText(margin, margin + 10, f"{max_v:.4f}")
        painter.drawText(margin, h - margin, f"{min_v:.4f}")
        painter.drawText(w - margin - 60, h - margin, f"Steps: {n}")

        painter.end()


# ── Drag-and-Drop Zone ──────────────────────────────────────────────────────

class DropZone(QLabel):
    """A label that accepts drag-and-drop files."""
    filesDropped = pyqtSignal(list)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(100)
        self.setWordWrap(True)
        self._default_style = """
            QLabel {
                border: 2px dashed #30363d;
                border-radius: 12px;
                color: #484f58;
                font-size: 13px;
                padding: 20px;
                background-color: #0d1117;
            }
        """
        self._hover_style = """
            QLabel {
                border: 2px dashed #e94560;
                border-radius: 12px;
                color: #e94560;
                font-size: 13px;
                padding: 20px;
                background-color: #1a0a10;
            }
        """
        self.setStyleSheet(self._default_style)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._hover_style)

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._default_style)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._default_style)
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                files.append(path)
        if files:
            self.filesDropped.emit(files)


# ── Toast Notification ──────────────────────────────────────────────────────

class ToastNotification(QLabel):
    """Temporary overlay notification that fades out."""
    def __init__(self, message: str, level: str = "info", parent=None):
        super().__init__(message, parent)
        colors = {
            "info": ("#1f6feb", "#cae8ff"),
            "success": ("#238636", "#aff5b4"),
            "warning": ("#9e6a03", "#ffd8a8"),
            "error": ("#da3633", "#ffc1c0"),
        }
        bg, fg = colors.get(level, colors["info"])
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 8px; "
            f"padding: 10px 16px; font-size: 13px; font-weight: 500;")
        self.setWordWrap(True)
        self.setMaximumWidth(320)
        self.adjustSize()
        QTimer.singleShot(3500, self._fade_out)

    def _fade_out(self):
        try:
            anim = QPropertyAnimation(self, b"windowOpacity")
            anim.setDuration(400)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.finished.connect(self.deleteLater)
            anim.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
            self._anim = anim
        except Exception:
            self.deleteLater()


# ── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.cfg = ConfigManager()
        self.dataset_manager = None
        self.caption_pipeline = None
        self.image_generator = None
        self.video_generator = None
        self.training_job = None
        self._current_images = []
        self._history_images = []
        self._workers = []  # prevent GC of workers
        self._undo_stack = []  # List of (widget_name, old_value) tuples
        self._redo_stack = []
        self._max_undo = 50
        self._param_widgets = {}
        self._prompt_queue_list = []
        self._prompt_queue_index = 0
        self._review_items = []
        self._review_index = 0

        self.setWindowTitle("AI Art Studio")
        self.setMinimumSize(900, 600)
        self.resize(1300, 850)

        # Restore geometry
        if self.cfg.config.window_geometry:
            try:
                from PyQt6.QtCore import QByteArray
                geo = QByteArray.fromBase64(
                    self.cfg.config.window_geometry.encode()
                )
                self.restoreGeometry(geo)
            except Exception:
                pass

        self._build_ui()
        self._connect_autosave()
        self._setup_shortcuts()
        self._start_vram_monitor()
        self._register_param_widgets()

    # ═══════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ═══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.tabs.addTab(self._build_dataset_tab(), "📂  Dataset")
        self.tabs.addTab(self._build_training_tab(), "🎯  Training")
        self.tabs.addTab(self._build_generation_tab(), "🎨  Generate")
        self.tabs.addTab(self._build_img2img_tab(), "🖌  img2img")
        self.tabs.addTab(self._build_settings_tab(), "⚙️  Settings")
        self.tabs.addTab(self._build_gallery_tab(), "🖼  Gallery")
        self.tabs.addTab(self._build_models_tab(), "📦  Models")

        main_layout.addWidget(self.tabs, 1)

        # Status bar
        self._build_status_bar()

        # Menu bar
        self._build_menu()

    def _build_status_bar(self):
        """Build a 4-zone status bar: [Activity] [Progress] [VRAM] [GPU Temp]"""
        status = self.statusBar()
        status.setStyleSheet(
            "QStatusBar { background: #010409; border-top: 1px solid #21262d; }"
            "QStatusBar::item { border: none; }")
        self.status_activity = QLabel("Ready")
        self.status_activity.setStyleSheet("color: #8b949e; padding: 0 12px;")
        status.addWidget(self.status_activity, 1)
        self.status_progress = QProgressBar()
        self.status_progress.setMaximumWidth(150)
        self.status_progress.setMaximumHeight(14)
        self.status_progress.setTextVisible(False)
        self.status_progress.setVisible(False)
        self.status_progress.setStyleSheet(
            "QProgressBar { background: #21262d; border-radius: 7px; }"
            "QProgressBar::chunk { background: #1f6feb; border-radius: 7px; }")
        status.addPermanentWidget(self.status_progress)
        self.status_vram = QLabel("VRAM: —")
        self.status_vram.setStyleSheet(
            "color: #8b949e; padding: 0 12px; border-left: 1px solid #21262d;")
        status.addPermanentWidget(self.status_vram)
        self.status_temp = QLabel("Temp: —")
        self.status_temp.setStyleSheet(
            "color: #8b949e; padding: 0 12px; border-left: 1px solid #21262d;")
        status.addPermanentWidget(self.status_temp)
        self.vram_label = self.status_vram

    def _set_status(self, msg: str, progress: int = -1):
        """Update activity label. progress: 0-100 to show bar, -1 to hide."""
        self.status_activity.setText(msg)
        if progress == -1:
            self.status_progress.setVisible(False)
        elif progress == 0:
            self.status_progress.setVisible(True)
            self.status_progress.setRange(0, 0)
        else:
            self.status_progress.setVisible(True)
            self.status_progress.setRange(0, 100)
            self.status_progress.setValue(progress)

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        open_dataset = QAction("Open Dataset Folder...", self)
        open_dataset.setShortcut(QKeySequence("Ctrl+O"))
        open_dataset.triggered.connect(self._open_dataset_folder)
        file_menu.addAction(open_dataset)

        file_menu.addSeparator()

        quit_action = QAction("Exit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tools_menu = menu.addMenu("Tools")
        clear_cache = QAction("Clear VRAM Cache", self)
        clear_cache.triggered.connect(self._clear_vram)
        tools_menu.addAction(clear_cache)

        help_menu = menu.addMenu("Help")
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

    def _show_shortcuts(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.setMinimumWidth(400)
        layout = QVBoxLayout(dlg)

        shortcuts = [
            ("Ctrl+Z", "Undo parameter change"),
            ("Ctrl+Y", "Redo parameter change"),
            ("Ctrl+G", "Start generation"),
            ("Ctrl+T", "Start training"),
            ("Ctrl+S", "Save current image"),
            ("Ctrl+,", "Go to Settings"),
            ("F5", "Refresh gallery"),
            ("Escape", "Cancel current operation"),
            ("Space", "Next item in caption review"),
        ]

        table = QTableWidget(len(shortcuts), 2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for i, (key, action) in enumerate(shortcuts):
            table.setItem(i, 0, QTableWidgetItem(key))
            table.setItem(i, 1, QTableWidgetItem(action))

        layout.addWidget(table)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec()

    def _setup_shortcuts(self):
        # Ctrl+G = generate
        gen_shortcut = QShortcut(QKeySequence("Ctrl+G"), self)
        gen_shortcut.activated.connect(self._generate)

        # Ctrl+T = start training
        train_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        train_shortcut.activated.connect(self._start_training)

        # Ctrl+Z / Ctrl+Y = undo/redo parameter changes
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo_param)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo_param)

    # ═══════════════════════════════════════════════════════════════════════
    #  DATASET TAB
    # ═══════════════════════════════════════════════════════════════════════

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel (controls) ──
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setContentsMargins(12, 12, 8, 12)
        left_layout.setSpacing(10)

        # Pipeline mode selector — determines how videos are handled
        mode_section = CollapsibleSection("Pipeline Mode")
        self.pipeline_mode = LabeledCombo("Training Target",
            ["image", "video"],
            self.cfg.config.pipeline_mode,
            tooltip=(
                "Image: videos are split into frames for image model "
                "training (SD, SDXL, Flux).\n"
                "Video: videos are kept as-is for video model training "
                "(Wan2.1, AnimateDiff, etc.)."))
        mode_section.addWidget(self.pipeline_mode)

        self.pipeline_mode_info = QLabel("")
        self.pipeline_mode_info.setObjectName("muted")
        self.pipeline_mode_info.setWordWrap(True)
        mode_section.addWidget(self.pipeline_mode_info)
        self._update_pipeline_mode_info(
            self.cfg.config.pipeline_mode)
        left_layout.addWidget(mode_section)

        # Drop zone
        self.drop_zone = DropZone("Drop images/videos here\nor use controls below")
        self.drop_zone.filesDropped.connect(self._on_files_dropped)
        left_layout.addWidget(self.drop_zone)

        # Source
        src_section = CollapsibleSection("Data Source")
        self.dataset_path = PathSelector("Dataset Folder", mode="dir")
        self.dataset_path.pathChanged.connect(self._on_dataset_path_changed)
        src_section.addWidget(self.dataset_path)

        btn_row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan Directory")
        self.scan_btn.setObjectName("primary")
        self.scan_btn.clicked.connect(self._scan_dataset)
        btn_row.addWidget(self.scan_btn)
        self.add_files_btn = QPushButton("Add Files...")
        self.add_files_btn.clicked.connect(self._add_individual_files)
        btn_row.addWidget(self.add_files_btn)
        src_section.addLayout(btn_row)
        dedup_btn = QPushButton("Find Duplicates")
        dedup_btn.clicked.connect(self._find_duplicates)
        src_section.addWidget(dedup_btn)
        left_layout.addWidget(src_section)

        # Stats
        stats_section = CollapsibleSection("Statistics")
        stats_grid = QGridLayout()
        stats_grid.setSpacing(6)
        self.stat_total = StatusCard("Total Files")
        self.stat_images = StatusCard("Images")
        self.stat_videos = StatusCard("Videos")
        self.stat_captioned = StatusCard("Captioned")
        self.stat_size = StatusCard("Total Size")
        self.stat_avgres = StatusCard("Avg Resolution")
        stats_grid.addWidget(self.stat_total, 0, 0)
        stats_grid.addWidget(self.stat_images, 0, 1)
        stats_grid.addWidget(self.stat_videos, 0, 2)
        stats_grid.addWidget(self.stat_captioned, 1, 0)
        stats_grid.addWidget(self.stat_size, 1, 1)
        stats_grid.addWidget(self.stat_avgres, 1, 2)
        stats_section.addLayout(stats_grid)
        left_layout.addWidget(stats_section)

        # Dataset summary
        self.dataset_summary = QLabel("No dataset loaded")
        self.dataset_summary.setStyleSheet(
            "color: #8b949e; font-size: 12px; padding: 6px 0;")
        left_layout.addWidget(self.dataset_summary)

        # Processing
        proc_section = CollapsibleSection("Processing", start_collapsed=True)
        self.max_resolution = LabeledSlider("Max Resolution", 256, 2048, 1024, 64,
                                            tooltip="Images larger than this will be downscaled")
        proc_section.addWidget(self.max_resolution)
        self.convert_btn = QPushButton("Convert & Validate All")
        self.convert_btn.clicked.connect(self._convert_dataset)
        proc_section.addWidget(self.convert_btn)
        left_layout.addWidget(proc_section)

        # Captioning
        cap_section = CollapsibleSection("Auto-Captioning")
        self.caption_method = LabeledCombo("Method",
            ["combined", "wd_tagger", "blip2", "florence2"],
            self.cfg.config.captioning.method,
            tooltip="combined = WD tags + natural language (best quality)")
        cap_section.addWidget(self.caption_method)

        self.caption_threshold = LabeledSlider("Tag Threshold", 0.1, 0.9, 0.35, 0.05, 2,
                                               tooltip="Lower = more tags, higher = stricter")
        cap_section.addWidget(self.caption_threshold)

        self.trigger_word = QLineEdit()
        self.trigger_word.setPlaceholderText("Trigger word (e.g., sks, ohwx)")
        self.trigger_word.setText(self.cfg.config.captioning.trigger_word)
        cap_section.addWidget(self.trigger_word)

        self.caption_format = LabeledCombo("Format",
            ["tags_and_natural", "tags_only", "natural_only"],
            self.cfg.config.captioning.caption_format)
        cap_section.addWidget(self.caption_format)

        self.overwrite_captions = LabeledCheck("Overwrite existing captions")
        cap_section.addWidget(self.overwrite_captions)

        self.verbose_description = LabeledCheck(
            "Verbose descriptions",
            description="Tag-aware prose with anatomy & action detail")
        self.verbose_description.setChecked(self.cfg.config.captioning.verbose_description)
        cap_section.addWidget(self.verbose_description)

        self.keep_caption_models = LabeledCheck("Keep captioning models in memory", False)
        self.keep_caption_models.setChecked(self.cfg.config.captioning.keep_models_in_memory)
        self.keep_caption_models.toggled.connect(
            lambda v: self.cfg.update_and_save("captioning", "keep_models_in_memory", v))
        cap_section.addWidget(self.keep_caption_models)

        # Video frame extraction controls (only relevant in image mode)
        vid_frame_layout = QHBoxLayout()
        vid_frame_layout.setSpacing(6)
        vid_frame_lbl = QLabel("Video: every")
        vid_frame_lbl.setObjectName("muted")
        self.video_frame_interval = QSpinBox()
        self.video_frame_interval.setRange(1, 60)
        self.video_frame_interval.setValue(self.cfg.config.captioning.video_frame_interval)
        self.video_frame_interval.setSuffix(" frames")
        self.video_frame_interval.setToolTip("Extract one frame every N video frames (image mode only)")
        vid_frame_layout.addWidget(vid_frame_lbl)
        vid_frame_layout.addWidget(self.video_frame_interval)
        vid_frame_layout.addStretch()
        vid_frame_w = QWidget()
        vid_frame_w.setLayout(vid_frame_layout)
        cap_section.addWidget(vid_frame_w)

        vid_max_layout = QHBoxLayout()
        vid_max_layout.setSpacing(6)
        vid_max_lbl = QLabel("Max frames")
        vid_max_lbl.setObjectName("muted")
        self.video_max_frames = QSpinBox()
        self.video_max_frames.setRange(1, 500)
        self.video_max_frames.setValue(self.cfg.config.captioning.video_max_frames)
        self.video_max_frames.setToolTip("Maximum frames to extract per video")
        vid_max_layout.addWidget(vid_max_lbl)
        vid_max_layout.addWidget(self.video_max_frames)
        vid_max_layout.addStretch()
        vid_max_w = QWidget()
        vid_max_w.setLayout(vid_max_layout)
        cap_section.addWidget(vid_max_w)

        # Keep references to toggle visibility when pipeline mode changes
        self._vid_frame_controls = [vid_frame_w, vid_max_w]
        if self.cfg.config.pipeline_mode == "video":
            vid_frame_w.setVisible(False)
            vid_max_w.setVisible(False)

        self.caption_btn = QPushButton("Auto-Caption Dataset")
        self.caption_btn.setObjectName("primary")
        self.caption_btn.clicked.connect(self._run_captioning)
        cap_section.addWidget(self.caption_btn)
        left_layout.addWidget(cap_section)

        # Progress
        self.dataset_progress = QProgressBar()
        self.dataset_progress.setVisible(False)
        left_layout.addWidget(self.dataset_progress)

        self.dataset_status = QLabel("")
        self.dataset_status.setObjectName("muted")
        left_layout.addWidget(self.dataset_status)

        # Log panel
        log_section = CollapsibleSection("Log", start_collapsed=True)
        self.dataset_log = QPlainTextEdit()
        self.dataset_log.setReadOnly(True)
        self.dataset_log.setMaximumBlockCount(2000)
        self.dataset_log.setMaximumHeight(150)
        self.dataset_log.setStyleSheet(
            "font-family: 'Cascadia Code', 'Consolas', monospace;"
            "font-size: 11px; background-color: #010409; color: #8b949e;"
            "border: 1px solid #21262d; border-radius: 8px; padding: 6px;")
        log_section.addWidget(self.dataset_log)
        left_layout.addWidget(log_section)

        left_layout.addStretch()

        left_scroll = make_scroll_panel(left_inner)
        left_scroll.setMinimumWidth(320)

        # ── Right panel (file list + preview) ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 12, 12, 12)
        right_layout.setSpacing(8)

        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(False)
        self.file_list.currentRowChanged.connect(self._preview_file)
        self.file_list.setAcceptDrops(True)
        self.file_list.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.file_list.installEventFilter(self)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._dataset_context_menu)
        right_layout.addWidget(self.file_list, 1)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(6)

        self.preview_image = QLabel("Select a file to preview")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumHeight(180)
        self.preview_image.setStyleSheet("""
            background-color: #161b22; border-radius: 8px;
            color: #484f58; font-size: 13px; padding: 20px;
        """)
        preview_layout.addWidget(self.preview_image)

        self.preview_caption = QTextEdit()
        self.preview_caption.setMaximumHeight(80)
        self.preview_caption.setPlaceholderText("Caption will appear here...")
        preview_layout.addWidget(self.preview_caption)

        save_caption_btn = QPushButton("Save Caption")
        save_caption_btn.clicked.connect(self._save_caption)
        preview_layout.addWidget(save_caption_btn)

        review_btn = QPushButton("Review Captions")
        review_btn.clicked.connect(self._start_caption_review)
        preview_layout.addWidget(review_btn)

        review_nav = QHBoxLayout()
        prev_review_btn = QPushButton("< Prev")
        prev_review_btn.clicked.connect(self._prev_review_item)
        next_review_btn = QPushButton("Next >")
        next_review_btn.clicked.connect(self._next_review_item)
        review_nav.addWidget(prev_review_btn)
        review_nav.addWidget(next_review_btn)
        preview_layout.addLayout(review_nav)

        right_layout.addWidget(preview_group)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([420, 580])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        return tab

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAINING TAB
    # ═══════════════════════════════════════════════════════════════════════

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Config (scrollable) ──
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setContentsMargins(12, 12, 8, 12)
        left_layout.setSpacing(10)

        # Presets
        preset_section = CollapsibleSection("Quick Presets")
        self.preset_combo = LabeledCombo("Preset",
            ["(Custom)", "SD 1.5 LoRA (8GB Safe)", "SDXL LoRA (8GB Tight)",
             "FLUX LoRA (8GB Split)", "SDXL Character LoRA (Recommended)"],
            "(Custom)")
        preset_section.addWidget(self.preset_combo)
        apply_preset_btn = QPushButton("Apply Preset")
        apply_preset_btn.clicked.connect(self._apply_training_preset)
        preset_section.addWidget(apply_preset_btn)
        # Use-case quick presets
        usecase_row = QHBoxLayout()
        _USECASE_PRESETS = {
            "Quick Test": {"max_train_steps": 200, "learning_rate": 1e-4, "batch_size": 1},
            "Character": {"max_train_steps": 1500, "learning_rate": 1e-4, "batch_size": 2},
            "Style": {"max_train_steps": 2000, "learning_rate": 5e-5, "batch_size": 1},
            "Concept": {"max_train_steps": 1000, "learning_rate": 1e-4, "batch_size": 2},
        }
        for uc_name, uc_vals in _USECASE_PRESETS.items():
            uc_btn = QPushButton(uc_name)
            uc_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
            uc_btn.clicked.connect(lambda _, v=uc_vals, n=uc_name: self._apply_usecase_preset(v, n))
            usecase_row.addWidget(uc_btn)
        uc_widget = QWidget()
        uc_widget.setLayout(usecase_row)
        preset_section.addWidget(uc_widget)
        left_layout.addWidget(preset_section)

        # Model selection
        model_section = CollapsibleSection("Base Model")
        self.model_type = LabeledCombo("Model Type",
            ["sd15", "sdxl", "flux", "wan21"],
            self.cfg.config.training.model_type,
            tooltip=(
                "sd15 = low VRAM, sdxl = best quality, flux = latest arch\n"
                "wan21 = Wan2.1 video model (set Pipeline Mode to Video)"))
        model_section.addWidget(self.model_type)

        self.base_model_path = PathSelector("Model Path", mode="file",
            filter="Model files (*.safetensors *.ckpt *.bin);;All files (*)")
        self.base_model_path.setPath(self.cfg.config.training.base_model)
        model_section.addWidget(self.base_model_path)

        self.training_type = LabeledCombo("Training Type",
            ["lora", "dreambooth_lora", "textual_inversion"],
            self.cfg.config.training.training_type)
        model_section.addWidget(self.training_type)
        left_layout.addWidget(model_section)

        # LoRA
        lora_section = CollapsibleSection("LoRA Configuration")
        self.lora_rank = LabeledSlider("Network Rank", 1, 128, 32, 1,
            tooltip="Higher = more capacity, more VRAM. 32-64 is good.")
        lora_section.addWidget(self.lora_rank)
        self.lora_alpha = LabeledSlider("Network Alpha", 1, 128, 16, 1,
            tooltip="Usually half the rank. Controls learning strength.")
        lora_section.addWidget(self.lora_alpha)
        left_layout.addWidget(lora_section)

        # Hyperparameters
        hyper_section = CollapsibleSection("Hyperparameters")
        self.learning_rate = LabeledSlider("Learning Rate", 0.00001, 0.01, 0.0003, 0.00001, 5,
            tooltip="Start low. 1e-4 to 3e-4 works for most LoRA training.")
        hyper_section.addWidget(self.learning_rate)
        self.train_steps = LabeledSlider("Max Steps", 100, 10000, 2000, 100,
            tooltip="Total optimization steps. 1500-3000 for most datasets.")
        hyper_section.addWidget(self.train_steps)
        self.train_epochs = LabeledSlider("Epochs", 1, 100, 10, 1)
        hyper_section.addWidget(self.train_epochs)
        self.batch_size = LabeledSlider("Batch Size", 1, 8, 1, 1,
            tooltip="Keep at 1 for 8GB VRAM. Higher needs more memory.")
        hyper_section.addWidget(self.batch_size)
        self.train_resolution = LabeledSlider("Resolution", 256, 1024, 512, 64,
            tooltip="Training image resolution. 512 for 8GB VRAM.")
        hyper_section.addWidget(self.train_resolution)
        self.num_repeats = LabeledSlider("Dataset Repeats", 1, 50, 10, 1,
            tooltip="How many times each image is seen per epoch.")
        hyper_section.addWidget(self.num_repeats)
        left_layout.addWidget(hyper_section)

        # Optimizer
        opt_section = CollapsibleSection("Optimizer & Scheduler", start_collapsed=True)
        self.optimizer = LabeledCombo("Optimizer",
            ["Adafactor", "AdamW8bit", "Prodigy", "Lion8bit", "AdamW"],
            self.cfg.config.training.optimizer,
            tooltip="Adafactor = memory efficient. Prodigy = auto LR.")
        opt_section.addWidget(self.optimizer)
        self.lr_scheduler = LabeledCombo("LR Scheduler",
            ["constant", "cosine", "cosine_with_restarts", "polynomial"],
            self.cfg.config.training.lr_scheduler)
        opt_section.addWidget(self.lr_scheduler)
        left_layout.addWidget(opt_section)

        # Regularization
        reg_section = CollapsibleSection("Regularization & Augmentation", start_collapsed=True)
        self.noise_offset = LabeledSlider("Noise Offset", 0, 0.2, 0.05, 0.01, 2,
            tooltip="Helps with very dark/bright images. 0.05-0.1 recommended.")
        reg_section.addWidget(self.noise_offset)
        self.flip_aug = LabeledCheck("Horizontal Flip",
            description="Don't use for asymmetric characters")
        reg_section.addWidget(self.flip_aug)
        self.color_aug = LabeledCheck("Color Augmentation")
        reg_section.addWidget(self.color_aug)
        left_layout.addWidget(reg_section)

        # Prompts
        prompt_section = CollapsibleSection("Prompts & Output")
        self.instance_prompt = QLineEdit()
        self.instance_prompt.setPlaceholderText("Instance prompt (e.g., sks character)")
        self.instance_prompt.setText(self.cfg.config.training.instance_prompt)
        prompt_section.addWidget(QLabel("Instance Prompt:"))
        prompt_section.addWidget(self.instance_prompt)

        self.class_prompt = QLineEdit()
        self.class_prompt.setPlaceholderText("Class prompt (e.g., character, creature)")
        self.class_prompt.setText(self.cfg.config.training.class_prompt)
        prompt_section.addWidget(QLabel("Class Prompt:"))
        prompt_section.addWidget(self.class_prompt)

        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Output model name")
        self.output_name.setText(self.cfg.config.training.output_name)
        prompt_section.addWidget(QLabel("Output Name:"))
        prompt_section.addWidget(self.output_name)

        config_btn_row = QHBoxLayout()
        export_btn = QPushButton("Export Config")
        export_btn.clicked.connect(self._export_training_config)
        config_btn_row.addWidget(export_btn)
        import_btn = QPushButton("Import Config")
        import_btn.clicked.connect(self._import_training_config)
        config_btn_row.addWidget(import_btn)
        prompt_section.addLayout(config_btn_row)

        left_layout.addWidget(prompt_section)

        # Resume from checkpoint
        resume_section = CollapsibleSection("Resume Training", start_collapsed=True)
        self.resume_check = LabeledCheck("Resume from checkpoint", False)
        self.resume_path = PathSelector("Checkpoint Path", mode="directory")
        resume_section.addWidget(self.resume_check)
        resume_section.addWidget(self.resume_path)
        left_layout.addWidget(resume_section)

        left_layout.addStretch()

        left_scroll = make_scroll_panel(left_inner)
        left_scroll.setMinimumWidth(320)

        # ── Right: Controls + Log ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 12, 12, 12)
        right_layout.setSpacing(8)

        # Controls
        ctrl_group = QGroupBox("Training Control")
        ctrl_layout = QVBoxLayout(ctrl_group)

        btn_row = QHBoxLayout()
        self.start_train_btn = QPushButton("Start Training")
        self.start_train_btn.setObjectName("primary")
        self.start_train_btn.setMinimumHeight(38)
        self.start_train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self.start_train_btn)

        self.stop_train_btn = QPushButton("Cancel")
        self.stop_train_btn.setObjectName("danger")
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.setMinimumHeight(38)
        self.stop_train_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self.stop_train_btn)
        ctrl_layout.addLayout(btn_row)

        self.train_progress = QProgressBar()
        ctrl_layout.addWidget(self.train_progress)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(6)
        self.train_step_label = StatusCard("Step", "0 / 0")
        self.train_loss_label = StatusCard("Loss", "—")
        self.train_lr_label = StatusCard("LR", "—")
        self.train_eta_label = StatusCard("ETA", "—")
        self.train_vram_label = StatusCard("VRAM", "—")
        for card in [self.train_step_label, self.train_loss_label,
                     self.train_lr_label, self.train_eta_label, self.train_vram_label]:
            stats_row.addWidget(card)
        ctrl_layout.addLayout(stats_row)
        right_layout.addWidget(ctrl_group)

        # Loss graph
        loss_group = QGroupBox("Loss History")
        loss_layout = QVBoxLayout(loss_group)
        self.loss_graph = LossGraphWidget()
        self.loss_graph.setMinimumHeight(120)
        self.loss_graph.setMaximumHeight(200)
        loss_layout.addWidget(self.loss_graph)
        right_layout.addWidget(loss_group)

        # Log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(5000)
        self.train_log.setStyleSheet(
            "font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;"
            "font-size: 11px; background-color: #010409; color: #8b949e;"
            "border: 1px solid #21262d; border-radius: 8px; padding: 8px;"
        )
        log_layout.addWidget(self.train_log)
        right_layout.addWidget(log_group, 1)

        # ── Sample Generation During Training ──
        sample_section = CollapsibleSection("Sample Generation", start_collapsed=True)
        self.sample_enabled = LabeledCheck("Enable sample generation during training", False)
        self.sample_prompt = QLineEdit()
        self.sample_prompt.setPlaceholderText("Enter a test prompt for mid-training samples...")
        self.sample_every_n = LabeledSlider("Generate sample every N steps", 100, 5000, 500, decimals=0)
        sample_section.addWidget(self.sample_enabled)
        sample_section.addWidget(self.sample_prompt)
        sample_section.addWidget(self.sample_every_n)
        self.sample_preview = QLabel("Samples will appear here during training")
        self.sample_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_preview.setMinimumHeight(200)
        self.sample_preview.setStyleSheet(
            "background-color: #010409; border: 1px solid #21262d; border-radius: 8px;")
        sample_section.addWidget(self.sample_preview)
        right_layout.addWidget(sample_section)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        return tab

    # ═══════════════════════════════════════════════════════════════════════
    #  GENERATION TAB
    # ═══════════════════════════════════════════════════════════════════════

    def _build_generation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Controls (scrollable) ──
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setContentsMargins(12, 12, 8, 12)
        left_layout.setSpacing(10)

        # Mode
        mode_section = CollapsibleSection("Generation Mode")
        self.gen_mode = LabeledCombo("Mode", ["Image", "Video"], "Image")
        self.gen_mode.currentTextChanged.connect(self._on_gen_mode_changed)
        mode_section.addWidget(self.gen_mode)
        left_layout.addWidget(mode_section)

        # Model
        load_section = CollapsibleSection("Model")
        self.gen_model_type = LabeledCombo("Type",
            ["sdxl", "sd15", "flux", "wan21", "animatediff"],
            self.cfg.config.training.model_type)
        load_section.addWidget(self.gen_model_type)

        self.gen_model_path = PathSelector("Base Model", mode="file",
            filter="Model files (*.safetensors *.ckpt *.bin);;All files (*)")
        load_section.addWidget(self.gen_model_path)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setObjectName("success")
        self.load_model_btn.clicked.connect(self._load_gen_model)
        load_section.addWidget(self.load_model_btn)

        left_layout.addWidget(load_section)

        # Multi-LoRA stacking
        lora_section = CollapsibleSection("LoRA Models")
        self.lora_list = QListWidget()
        self.lora_list.setMaximumHeight(120)
        add_lora_btn = QPushButton("Add LoRA")
        add_lora_btn.clicked.connect(self._add_lora)
        remove_lora_btn = QPushButton("Remove Selected")
        remove_lora_btn.clicked.connect(self._remove_lora)
        self.lora_weight_slider = LabeledSlider("Weight", 0.0, 2.0, 1.0, 0.05, 2)

        lora_btn_row = QHBoxLayout()
        lora_btn_row.addWidget(add_lora_btn)
        lora_btn_row.addWidget(remove_lora_btn)
        lora_btn_widget = QWidget()
        lora_btn_widget.setLayout(lora_btn_row)

        lora_section.addWidget(self.lora_list)
        lora_section.addWidget(lora_btn_widget)
        lora_section.addWidget(self.lora_weight_slider)
        left_layout.addWidget(lora_section)

        # Textual Inversion Embeddings
        embed_section = CollapsibleSection("Textual Inversion Embeddings", start_collapsed=True)
        self.embed_list = QListWidget()
        self.embed_list.setMaximumHeight(100)
        add_embed_btn = QPushButton("Add Embedding")
        add_embed_btn.clicked.connect(self._add_embedding)
        remove_embed_btn = QPushButton("Remove Selected")
        remove_embed_btn.clicked.connect(self._remove_embedding)
        embed_btn_row = QHBoxLayout()
        embed_btn_row.addWidget(add_embed_btn)
        embed_btn_row.addWidget(remove_embed_btn)
        embed_btn_widget = QWidget()
        embed_btn_widget.setLayout(embed_btn_row)
        embed_section.addWidget(self.embed_list)
        embed_section.addWidget(embed_btn_widget)
        left_layout.addWidget(embed_section)

        # Prompt
        prompt_section = CollapsibleSection("Prompt")

        # Style preset bar
        preset_row = QHBoxLayout()
        self.style_preset_combo = QComboBox()
        self.style_preset_combo.setPlaceholderText("Style preset...")
        self._reload_style_presets()
        preset_row.addWidget(self.style_preset_combo, 1)
        apply_preset_btn = QPushButton("Apply")
        apply_preset_btn.setFixedWidth(60)
        apply_preset_btn.clicked.connect(self._apply_style_preset)
        preset_row.addWidget(apply_preset_btn)
        save_preset_btn = QPushButton("Save")
        save_preset_btn.setFixedWidth(50)
        save_preset_btn.clicked.connect(self._save_style_preset)
        preset_row.addWidget(save_preset_btn)
        delete_preset_btn = QPushButton("Del")
        delete_preset_btn.setFixedWidth(40)
        delete_preset_btn.clicked.connect(self._delete_style_preset)
        preset_row.addWidget(delete_preset_btn)
        preset_widget = QWidget()
        preset_widget.setLayout(preset_row)
        prompt_section.addWidget(preset_widget)

        prompt_section.addWidget(QLabel("Positive:"))
        self.gen_prompt = QTextEdit()
        self.gen_prompt.setMaximumHeight(90)
        self.gen_prompt.setMinimumHeight(60)
        self.gen_prompt.setPlaceholderText("Describe what you want to generate...")
        prompt_section.addWidget(self.gen_prompt)

        prompt_section.addWidget(QLabel("Negative:"))
        self.gen_negative = QTextEdit()
        self.gen_negative.setMaximumHeight(55)
        self.gen_negative.setPlainText(self.cfg.config.generation.img_negative_prompt)
        prompt_section.addWidget(self.gen_negative)
        left_layout.addWidget(prompt_section)

        # Image parameters
        self.img_params_section = CollapsibleSection("Image Parameters")
        self.img_width = LabeledSlider("Width", 256, 2048, 768, 64)
        self.img_params_section.addWidget(self.img_width)
        self.img_height = LabeledSlider("Height", 256, 2048, 768, 64)
        self.img_params_section.addWidget(self.img_height)
        # Quick resolution presets
        res_presets = [("512\u00b2", 512, 512), ("768\u00b2", 768, 768),
                       ("1024\u00b2", 1024, 1024), ("1024\u00d7576", 1024, 576),
                       ("576\u00d71024", 576, 1024), ("1216\u00d7832", 1216, 832)]
        res_btn_row = QHBoxLayout()
        for label, w, h in res_presets:
            btn = QPushButton(label)
            btn.setMaximumWidth(72)
            btn.setStyleSheet("font-size: 11px; padding: 4px 6px;")
            btn.clicked.connect(lambda _, w=w, h=h: (
                self.img_width.setValue(w), self.img_height.setValue(h)))
            res_btn_row.addWidget(btn)
        res_btn_widget = QWidget()
        res_btn_widget.setLayout(res_btn_row)
        self.img_params_section.addWidget(res_btn_widget)
        self.img_steps = LabeledSlider("Steps", 1, 150, 30, 1,
            tooltip="More steps = better quality but slower. 20-40 is typical.")
        self.img_params_section.addWidget(self.img_steps)
        self.img_cfg = LabeledSlider("CFG Scale", 1, 30, 7.5, 0.5, 1,
            tooltip="How closely to follow the prompt. 5-12 is typical.")
        self.img_params_section.addWidget(self.img_cfg)
        self.img_seed = LabeledSlider("Seed (-1 = random)", -1, 999999999, -1, 1)
        self.img_params_section.addWidget(self.img_seed)
        self.seed_lock = LabeledCheck("Lock Seed", False)
        self.seed_lock.toggled.connect(lambda locked: self.img_seed.setEnabled(not locked))
        self.img_params_section.addWidget(self.seed_lock)
        self.img_sampler = LabeledCombo("Sampler",
            ["euler_a", "euler", "dpm++_2m", "dpm++_2m_karras", "dpm++_sde",
             "dpm++_sde_karras", "ddim", "uni_pc", "heun", "lms", "pndm"],
            "euler_a", tooltip="euler_a = fast and good. dpm++_2m_karras = sharp details.")
        self.img_params_section.addWidget(self.img_sampler)
        self.img_clip_skip = LabeledSlider("CLIP Skip", 1, 4, 2, 1,
            tooltip="2 for anime/art models. 1 for realistic models.")
        self.img_params_section.addWidget(self.img_clip_skip)
        self.img_batch = LabeledSlider("Batch Size", 1, 4, 1, 1,
            tooltip="Generate multiple images at once. Uses more VRAM.")
        self.img_params_section.addWidget(self.img_batch)
        left_layout.addWidget(self.img_params_section)

        # Advanced Image Settings (Part B)
        adv_section = CollapsibleSection("Advanced Image Settings", start_collapsed=True)
        self.img_eta = LabeledSlider("DDIM Eta", 0.0, 1.0,
            self.cfg.config.generation.img_eta, 0.01, 2)
        self.img_tiling = LabeledCheck("Seamless Tiling",
            default=self.cfg.config.generation.img_tiling)
        self.img_karras = LabeledCheck("Karras Sigmas",
            default=self.cfg.config.generation.img_karras_sigmas)
        self.img_rescale_cfg = LabeledSlider("CFG Rescale", 0.0, 1.0,
            self.cfg.config.generation.img_rescale_cfg, 0.01, 2)
        self.img_aes_score = LabeledSlider("Aesthetic Score (SDXL)", 1.0, 10.0,
            self.cfg.config.generation.img_aesthetic_score, 0.5, 1)
        self.img_neg_aes = LabeledSlider("Negative Aesthetic Score", 1.0, 10.0,
            self.cfg.config.generation.img_negative_aesthetic_score, 0.5, 1)
        self.img_denoise_start = LabeledSlider("Denoise Start", 0.0, 1.0,
            self.cfg.config.generation.img_denoising_start, 0.01, 2)
        self.img_denoise_end = LabeledSlider("Denoise End", 0.0, 1.0,
            self.cfg.config.generation.img_denoising_end, 0.01, 2)
        adv_section.addWidget(self.img_eta)
        adv_section.addWidget(self.img_tiling)
        adv_section.addWidget(self.img_karras)
        adv_section.addWidget(self.img_rescale_cfg)
        adv_section.addWidget(self.img_aes_score)
        adv_section.addWidget(self.img_neg_aes)
        adv_section.addWidget(self.img_denoise_start)
        adv_section.addWidget(self.img_denoise_end)
        # Auto-save advanced image settings
        self.img_eta.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_eta", v))
        self.img_tiling.toggled.connect(
            lambda v: self.cfg.update_and_save("generation", "img_tiling", v))
        self.img_karras.toggled.connect(
            lambda v: self.cfg.update_and_save("generation", "img_karras_sigmas", v))
        self.img_rescale_cfg.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_rescale_cfg", v))
        self.img_aes_score.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_aesthetic_score", v))
        self.img_neg_aes.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_negative_aesthetic_score", v))
        self.img_denoise_start.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_denoising_start", v))
        self.img_denoise_end.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_denoising_end", v))
        left_layout.addWidget(adv_section)

        # Hi-res fix
        hires_section = CollapsibleSection("Hi-Res Fix", start_collapsed=True)
        self.hires_check = LabeledCheck("Enable Hi-Res Fix",
            description="Upscale after initial generation")
        hires_section.addWidget(self.hires_check)
        self.hires_scale = LabeledSlider("Upscale Factor", 1.0, 3.0, 1.5, 0.1, 1)
        hires_section.addWidget(self.hires_scale)
        self.hires_steps = LabeledSlider("Hires Steps", 1, 50, 15, 1)
        hires_section.addWidget(self.hires_steps)
        self.hires_denoise = LabeledSlider("Hires Denoising", 0, 1.0, 0.55, 0.05, 2)
        hires_section.addWidget(self.hires_denoise)
        left_layout.addWidget(hires_section)

        # ControlNet
        controlnet_section = CollapsibleSection("ControlNet", start_collapsed=True)
        self.controlnet_enabled = LabeledCheck("Enable ControlNet", False)
        self.controlnet_model = LabeledCombo("ControlNet Model", [
            "lllyasviel/control_v11p_sd15_canny",
            "lllyasviel/control_v11f1p_sd15_depth",
            "lllyasviel/control_v11p_sd15_openpose",
            "diffusers/controlnet-canny-sdxl-1.0",
            "diffusers/controlnet-depth-sdxl-1.0",
        ], self.cfg.config.generation.controlnet_model_id)
        self.controlnet_preprocessor = LabeledCombo("Preprocessor", [
            "canny", "depth", "openpose", "none"
        ], self.cfg.config.generation.controlnet_preprocessor)
        self.controlnet_input = PathSelector("Control Image", mode="file",
            filter="Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)")
        self.controlnet_preview_btn = QPushButton("Preview Conditioning")
        self.controlnet_preview_btn.clicked.connect(self._preview_controlnet)
        self.controlnet_preview_label = QLabel()
        self.controlnet_preview_label.setFixedSize(160, 160)
        self.controlnet_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.controlnet_preview_label.setStyleSheet(
            "background-color: #0a1628; border: 1px solid #21262d; border-radius: 6px;")
        self.controlnet_strength = LabeledSlider("Control Strength", 0.0, 2.0,
            self.cfg.config.generation.controlnet_strength, 0.05, 2)
        self.controlnet_guidance_start = LabeledSlider("Guidance Start", 0.0, 1.0,
            self.cfg.config.generation.controlnet_guidance_start, 0.05, 2)
        self.controlnet_guidance_end = LabeledSlider("Guidance End", 0.0, 1.0,
            self.cfg.config.generation.controlnet_guidance_end, 0.05, 2)

        controlnet_section.addWidget(self.controlnet_enabled)
        controlnet_section.addWidget(self.controlnet_model)
        controlnet_section.addWidget(self.controlnet_preprocessor)
        controlnet_section.addWidget(self.controlnet_input)
        cn_preview_row = QHBoxLayout()
        cn_preview_row.addWidget(self.controlnet_preview_btn)
        cn_preview_row.addStretch()
        cn_preview_widget = QWidget()
        cn_preview_widget.setLayout(cn_preview_row)
        controlnet_section.addWidget(cn_preview_widget)
        controlnet_section.addWidget(self.controlnet_preview_label)
        controlnet_section.addWidget(self.controlnet_strength)
        controlnet_section.addWidget(self.controlnet_guidance_start)
        controlnet_section.addWidget(self.controlnet_guidance_end)

        # Auto-save ControlNet settings
        self.controlnet_enabled.toggled.connect(
            lambda v: self.cfg.update_and_save("generation", "controlnet_enabled", v))
        self.controlnet_model.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "controlnet_model_id", v))
        self.controlnet_preprocessor.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "controlnet_preprocessor", v))
        self.controlnet_input.pathChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "controlnet_input_image", v))
        self.controlnet_strength.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "controlnet_strength", v))
        self.controlnet_guidance_start.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "controlnet_guidance_start", v))
        self.controlnet_guidance_end.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "controlnet_guidance_end", v))

        left_layout.addWidget(controlnet_section)

        # IP-Adapter Style Reference
        ip_section = CollapsibleSection("Style Reference (IP-Adapter)", start_collapsed=True)
        self.ip_adapter_enabled = LabeledCheck("Enable IP-Adapter",
            default=self.cfg.config.generation.ip_adapter_enabled)
        self.ip_adapter_image = PathSelector("Reference Image", mode="file",
            filter="Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)")
        self.ip_adapter_preview = QLabel()
        self.ip_adapter_preview.setFixedSize(100, 100)
        self.ip_adapter_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ip_adapter_preview.setStyleSheet(
            "background-color: #0a1628; border: 1px solid #21262d; border-radius: 6px;")
        self.ip_adapter_image.pathChanged.connect(self._update_ip_adapter_preview)
        self.ip_adapter_scale = LabeledSlider("IP-Adapter Scale", 0.0, 1.0,
            self.cfg.config.generation.ip_adapter_scale, 0.05, 2)
        ip_info = QLabel("Style reference steers visual style without full img2img.\n"
                         "Uses IP-Adapter from h94/IP-Adapter.")
        ip_info.setStyleSheet("font-size: 11px; color: #556a8b;")
        ip_info.setWordWrap(True)

        ip_section.addWidget(self.ip_adapter_enabled)
        ip_section.addWidget(self.ip_adapter_image)
        ip_section.addWidget(self.ip_adapter_preview)
        ip_section.addWidget(self.ip_adapter_scale)
        ip_section.addWidget(ip_info)

        # Auto-save IP-Adapter settings
        self.ip_adapter_enabled.toggled.connect(
            lambda v: self.cfg.update_and_save("generation", "ip_adapter_enabled", v))
        self.ip_adapter_image.pathChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "ip_adapter_image", v))
        self.ip_adapter_scale.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "ip_adapter_scale", v))

        left_layout.addWidget(ip_section)

        # Image Upscaling
        upscale_section = CollapsibleSection("Image Upscaling", start_collapsed=True)
        self.upscale_input = PathSelector("Input Image", mode="file",
            filter="Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)")
        self.upscale_factor = LabeledCombo("Scale Factor",
            ["2x (AI)", "4x (AI)", "2x (Lanczos)", "4x (Lanczos)"], "4x (AI)")
        upscale_btn = QPushButton("Upscale Image")
        upscale_btn.clicked.connect(self._upscale_image)
        upscale_section.addWidget(self.upscale_input)
        upscale_section.addWidget(self.upscale_factor)
        upscale_section.addWidget(upscale_btn)
        # Show availability note
        try:
            from generation.upscaler import is_realesrgan_available
            if not is_realesrgan_available():
                esrgan_note = QLabel("Install realesrgan for AI upscaling (pip install realesrgan)")
                esrgan_note.setStyleSheet("font-size: 11px; color: #f0883e;")
                esrgan_note.setWordWrap(True)
                upscale_section.addWidget(esrgan_note)
        except Exception:
            pass
        left_layout.addWidget(upscale_section)

        # Video parameters
        self.vid_params_section = CollapsibleSection("Video Parameters")
        self.vid_width = LabeledSlider("Width", 256, 1280, 512, 64)
        self.vid_params_section.addWidget(self.vid_width)
        self.vid_height = LabeledSlider("Height", 256, 720, 512, 64)
        self.vid_params_section.addWidget(self.vid_height)
        self.vid_frames = LabeledSlider("Frames", 9, 201, 49, 4)
        self.vid_params_section.addWidget(self.vid_frames)
        self.vid_fps = LabeledSlider("FPS", 8, 30, 16, 1)
        self.vid_params_section.addWidget(self.vid_fps)
        self.vid_steps = LabeledSlider("Steps", 1, 100, 30, 1)
        self.vid_params_section.addWidget(self.vid_steps)
        self.vid_cfg = LabeledSlider("CFG Scale", 1, 20, 6, 0.5, 1)
        self.vid_params_section.addWidget(self.vid_cfg)
        self.vid_seed = LabeledSlider("Seed (-1 = random)", -1, 999999999, -1, 1)
        self.vid_params_section.addWidget(self.vid_seed)

        # Video resolution presets
        vid_res_presets = [
            ("480p", 854, 480), ("480p\u2191", 480, 854),
            ("720p", 1280, 720), ("720p\u2191", 720, 1280),
            ("512\u00b2", 512, 512), ("480\u00b2", 480, 480),
        ]
        vid_res_row = QHBoxLayout()
        for label, w, h in vid_res_presets:
            btn = QPushButton(label)
            btn.setMaximumWidth(68)
            btn.setStyleSheet("font-size: 11px; padding: 4px 5px;")
            btn.clicked.connect(lambda _, w=w, h=h: (
                self.vid_width.setValue(w), self.vid_height.setValue(h)))
            vid_res_row.addWidget(btn)
        vid_res_widget = QWidget()
        vid_res_widget.setLayout(vid_res_row)
        self.vid_params_section.addWidget(vid_res_widget)

        # Video negative prompt
        self.vid_negative = QPlainTextEdit()
        self.vid_negative.setPlaceholderText("Negative prompt for video...")
        self.vid_negative.setMaximumHeight(60)
        self.vid_params_section.addWidget(self.vid_negative)

        # Video output format
        self.vid_format_combo = LabeledCombo("Output Format", ["mp4", "gif", "webm"], "mp4")
        self.vid_format_combo.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "vid_format", v))
        self.vid_params_section.addWidget(self.vid_format_combo)

        # Advanced Video Settings
        adv_vid_section = CollapsibleSection("Advanced Video Settings", start_collapsed=True)
        self.vid_flow_shift = LabeledSlider("Flow Shift (WAN)", 0.0, 10.0, 3.0, 0.1, 1)
        self.vid_flow_shift.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "vid_flow_shift", v))
        adv_vid_section.addWidget(self.vid_flow_shift)
        self.vid_decode_chunks = LabeledSlider("Decode Chunk Size", 1, 32, 8, 1, 0)
        self.vid_decode_chunks.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "vid_decode_chunk_size", int(v)))
        adv_vid_section.addWidget(self.vid_decode_chunks)
        self.vid_overlap = LabeledSlider("Clip Overlap Frames", 0, 16, 4, 1, 0)
        self.vid_overlap.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "vid_overlap_frames", int(v)))
        adv_vid_section.addWidget(self.vid_overlap)
        self.vid_clip_count = LabeledSlider("Clip Count (stitching)", 1, 8, 1, 1, 0)
        self.vid_clip_count.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "vid_clip_count", int(v)))
        adv_vid_section.addWidget(self.vid_clip_count)
        self.vid_tiling_check = LabeledCheck("Spatial Tiling", False)
        self.vid_tiling_check.toggled.connect(
            lambda v: self.cfg.update_and_save("generation", "vid_tiling", v))
        adv_vid_section.addWidget(self.vid_tiling_check)
        self.vid_sample_method = LabeledCombo(
            "Sampling Method", ["euler", "euler_a", "dpm++_2m", "ddim"], "euler")
        self.vid_sample_method.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "vid_sample_method", v))
        adv_vid_section.addWidget(self.vid_sample_method)

        stitch_info = QLabel(
            "Clip Count > 1: generates multiple clips sequentially,\n"
            "each seeded from the last frame of the previous clip,\n"
            "then stitches them into one long video."
        )
        stitch_info.setObjectName("muted")
        stitch_info.setWordWrap(True)
        adv_vid_section.addWidget(stitch_info)
        self.vid_params_section.addWidget(adv_vid_section)

        self.vid_params_section.setVisible(False)
        left_layout.addWidget(self.vid_params_section)

        # ── Prompt Queue ──
        # ── Generation Queue ──
        queue_section = CollapsibleSection("Generation Queue", start_collapsed=True)
        queue_btn_row = QHBoxLayout()
        add_queue_btn = QPushButton("Add to Queue")
        add_queue_btn.clicked.connect(self._add_to_gen_queue)
        queue_btn_row.addWidget(add_queue_btn)
        clear_queue_btn = QPushButton("Clear")
        clear_queue_btn.clicked.connect(self._clear_gen_queue)
        queue_btn_row.addWidget(clear_queue_btn)
        self.pause_queue_btn = QPushButton("Pause")
        self.pause_queue_btn.clicked.connect(self._toggle_queue_pause)
        queue_btn_row.addWidget(self.pause_queue_btn)
        queue_btn_widget = QWidget()
        queue_btn_widget.setLayout(queue_btn_row)
        queue_section.addWidget(queue_btn_widget)
        self.gen_queue_list = QListWidget()
        self.gen_queue_list.setMaximumHeight(150)
        self.gen_queue_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.gen_queue_list.customContextMenuRequested.connect(self._queue_context_menu)
        queue_section.addWidget(self.gen_queue_list)
        self.queue_progress_label = QLabel("")
        queue_section.addWidget(self.queue_progress_label)
        left_layout.addWidget(queue_section)

        # Prompt Queue (legacy batch-by-line)
        pq_section = CollapsibleSection("Prompt Queue (Batch)", start_collapsed=True)
        self.prompt_queue = QPlainTextEdit()
        self.prompt_queue.setPlaceholderText("Enter one prompt per line for batch generation...")
        self.prompt_queue.setMaximumHeight(150)
        pq_section.addWidget(self.prompt_queue)
        run_queue_btn = QPushButton("Run Queue")
        run_queue_btn.clicked.connect(self._run_prompt_queue)
        pq_section.addWidget(run_queue_btn)
        left_layout.addWidget(pq_section)

        # ── Audio Generation (Experimental) ──
        audio_section = CollapsibleSection("Audio Generation (Experimental)", start_collapsed=True)
        self.audio_status_label = QLabel("No audio model loaded")
        self.audio_status_label.setObjectName("muted")
        audio_section.addWidget(self.audio_status_label)
        self.audio_model_combo = LabeledCombo("Audio Model", [
            "facebook/musicgen-small",
            "facebook/musicgen-medium",
            "CVSSP/audioldm2",
            "CVSSP/audioldm2-music",
            "stabilityai/stable-audio-open-1.0",
        ], "facebook/musicgen-small")
        audio_section.addWidget(self.audio_model_combo)
        load_audio_btn = QPushButton("Load Audio Model")
        load_audio_btn.clicked.connect(self._load_audio_model)
        audio_section.addWidget(load_audio_btn)
        self.audio_prompt = QPlainTextEdit()
        self.audio_prompt.setPlaceholderText("Describe the audio/music to generate...")
        self.audio_prompt.setMaximumHeight(70)
        audio_section.addWidget(self.audio_prompt)
        self.audio_duration = LabeledSlider("Duration (seconds)", 1.0, 60.0, 10.0, 0.5, 1)
        self.audio_cfg = LabeledSlider("Guidance Scale", 1.0, 10.0, 3.5, 0.5, 1)
        audio_section.addWidget(self.audio_duration)
        audio_section.addWidget(self.audio_cfg)
        self.audio_sync_check = LabeledCheck("Sync to generated video", False)
        audio_section.addWidget(self.audio_sync_check)
        self.audio_format = LabeledCombo("Output Format", ["wav", "mp3", "flac"], "wav")
        audio_section.addWidget(self.audio_format)
        gen_audio_btn = QPushButton("Generate Audio")
        gen_audio_btn.clicked.connect(self._generate_audio)
        audio_section.addWidget(gen_audio_btn)
        scaffold_note = QLabel(
            "Audio generation is a work-in-progress feature.\n"
            "MusicGen and AudioLDM2 backends are supported when transformers/diffusers are installed."
        )
        scaffold_note.setObjectName("muted")
        scaffold_note.setWordWrap(True)
        audio_section.addWidget(scaffold_note)
        left_layout.addWidget(audio_section)

        # Generate button (always visible at bottom)
        left_layout.addStretch()

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setObjectName("primary")
        self.generate_btn.setMinimumHeight(44)
        gen_font = self.generate_btn.font()
        gen_font.setPointSize(13)
        gen_font.setBold(True)
        self.generate_btn.setFont(gen_font)
        self.generate_btn.clicked.connect(self._generate)
        left_layout.addWidget(self.generate_btn)

        self.gen_progress = QProgressBar()
        self.gen_progress.setVisible(False)
        left_layout.addWidget(self.gen_progress)

        left_scroll = make_scroll_panel(left_inner)
        left_scroll.setMinimumWidth(320)

        # ── Right: Output display ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 12, 12, 12)
        right_layout.setSpacing(8)

        # Output image
        self.gen_output_image = QLabel("Generated images will appear here")
        self.gen_output_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gen_output_image.setMinimumSize(300, 300)
        self.gen_output_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gen_output_image.setStyleSheet("""
            background-color: #161b22; border-radius: 12px;
            color: #484f58; font-size: 14px; border: 1px solid #21262d;
        """)
        self.gen_output_image.setScaledContents(False)
        right_layout.addWidget(self.gen_output_image, 1)

        # Controls
        out_ctrl = QHBoxLayout()
        out_ctrl.setSpacing(8)
        self.save_output_btn = QPushButton("Save Image")
        self.save_output_btn.clicked.connect(self._save_generated)
        self.save_output_btn.setEnabled(False)
        out_ctrl.addWidget(self.save_output_btn)

        self.gen_save_format = QComboBox()
        self.gen_save_format.addItems(["png", "jpg", "webp"])
        self.gen_save_format.setFixedWidth(80)
        out_ctrl.addWidget(self.gen_save_format)

        out_ctrl.addStretch()
        right_layout.addLayout(out_ctrl)

        # Info
        self.gen_info = QLabel("")
        self.gen_info.setObjectName("muted")
        self.gen_info.setWordWrap(True)
        right_layout.addWidget(self.gen_info)

        # History thumbnails
        history_group = QGroupBox("History")
        history_layout = QHBoxLayout(history_group)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(100)
        self.history_list.setIconSize(QSize(64, 64))
        self.history_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.history_list.setGridSize(QSize(80, 90))
        self.history_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.history_list.setWordWrap(True)
        self.history_list.setFlow(QListWidget.Flow.LeftToRight)
        self.history_list.setWrapping(False)
        self.history_list.setSpacing(4)
        self.history_list.currentRowChanged.connect(self._show_history_item)
        history_layout.addWidget(self.history_list)
        right_layout.addWidget(history_group)

        # ── A/B Comparison ──
        compare_section = CollapsibleSection("A/B Comparison", start_collapsed=True)
        compare_layout_h = QHBoxLayout()
        self.compare_a = QLabel("Image A")
        self.compare_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compare_a.setMinimumSize(256, 256)
        self.compare_a.setStyleSheet(
            "background-color: #010409; border: 1px solid #21262d; border-radius: 8px;")
        self.compare_b = QLabel("Image B")
        self.compare_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compare_b.setMinimumSize(256, 256)
        self.compare_b.setStyleSheet(
            "background-color: #010409; border: 1px solid #21262d; border-radius: 8px;")
        compare_layout_h.addWidget(self.compare_a)
        compare_layout_h.addWidget(self.compare_b)
        compare_widget = QWidget()
        compare_widget.setLayout(compare_layout_h)
        compare_section.addWidget(compare_widget)
        set_a_btn = QPushButton("Set Current as A")
        set_a_btn.clicked.connect(lambda: self._set_compare("a"))
        set_b_btn = QPushButton("Set Current as B")
        set_b_btn.clicked.connect(lambda: self._set_compare("b"))
        btn_row = QHBoxLayout()
        btn_row.addWidget(set_a_btn)
        btn_row.addWidget(set_b_btn)
        btn_widget = QWidget()
        btn_widget.setLayout(btn_row)
        compare_section.addWidget(btn_widget)
        right_layout.addWidget(compare_section)

        # ── Image Metadata ──
        meta_section = CollapsibleSection("Image Metadata", start_collapsed=True)
        self.meta_display = QPlainTextEdit()
        self.meta_display.setReadOnly(True)
        self.meta_display.setMaximumHeight(120)
        self.meta_display.setStyleSheet(
            "font-family: 'Cascadia Code', 'Consolas', monospace;"
            "font-size: 11px; background-color: #010409; color: #8b949e;"
            "border: 1px solid #21262d; border-radius: 8px; padding: 6px;")
        meta_section.addWidget(self.meta_display)
        right_layout.addWidget(meta_section)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        return tab

    # ═══════════════════════════════════════════════════════════════════════
    #  IMG2IMG TAB
    # ═══════════════════════════════════════════════════════════════════════

    def _build_img2img_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Controls ──
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setContentsMargins(12, 12, 8, 12)
        left_layout.setSpacing(10)

        # Input image
        input_section = CollapsibleSection("Input Image")
        self.i2i_input = PathSelector("Image", mode="file",
            filter="Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)")
        self.i2i_preview = QLabel("Drag & drop or browse")
        self.i2i_preview.setFixedSize(256, 256)
        self.i2i_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.i2i_preview.setStyleSheet(
            "background-color: #0a1628; border: 1px solid #21262d; border-radius: 8px; color: #484f58;")
        self.i2i_input.pathChanged.connect(self._update_i2i_preview)
        input_section.addWidget(self.i2i_input)
        input_section.addWidget(self.i2i_preview)
        left_layout.addWidget(input_section)

        # Prompt
        prompt_section = CollapsibleSection("Prompt")
        prompt_section.addWidget(QLabel("Positive:"))
        self.i2i_prompt = QTextEdit()
        self.i2i_prompt.setMaximumHeight(90)
        self.i2i_prompt.setMinimumHeight(60)
        self.i2i_prompt.setPlaceholderText("Describe the desired output...")
        prompt_section.addWidget(self.i2i_prompt)
        prompt_section.addWidget(QLabel("Negative:"))
        self.i2i_negative = QTextEdit()
        self.i2i_negative.setMaximumHeight(55)
        self.i2i_negative.setPlainText(self.cfg.config.generation.img_negative_prompt)
        prompt_section.addWidget(self.i2i_negative)
        left_layout.addWidget(prompt_section)

        # Parameters
        params_section = CollapsibleSection("Parameters")
        self.i2i_denoising = LabeledSlider("Denoising Strength", 0.0, 1.0, 0.75, 0.01, 2,
            tooltip="0.0 = no change, 1.0 = completely new image")
        params_section.addWidget(self.i2i_denoising)
        self.i2i_steps = LabeledSlider("Steps", 1, 150, 30, 1)
        params_section.addWidget(self.i2i_steps)
        self.i2i_cfg = LabeledSlider("CFG Scale", 1, 30, 7.5, 0.5, 1)
        params_section.addWidget(self.i2i_cfg)
        self.i2i_sampler = LabeledCombo("Sampler",
            ["euler_a", "euler", "dpm++_2m", "dpm++_2m_karras", "dpm++_sde",
             "dpm++_sde_karras", "ddim", "uni_pc", "heun", "lms", "pndm"],
            "euler_a")
        params_section.addWidget(self.i2i_sampler)
        self.i2i_seed = LabeledSlider("Seed (-1 = random)", -1, 999999999, -1, 1)
        params_section.addWidget(self.i2i_seed)
        self.i2i_width = LabeledSlider("Width", 256, 2048, 768, 64)
        params_section.addWidget(self.i2i_width)
        self.i2i_height = LabeledSlider("Height", 256, 2048, 768, 64)
        params_section.addWidget(self.i2i_height)
        left_layout.addWidget(params_section)

        # Generate button
        left_layout.addStretch()
        self.i2i_generate_btn = QPushButton("Generate img2img")
        self.i2i_generate_btn.setObjectName("primary")
        self.i2i_generate_btn.setMinimumHeight(44)
        btn_font = self.i2i_generate_btn.font()
        btn_font.setPointSize(13)
        btn_font.setBold(True)
        self.i2i_generate_btn.setFont(btn_font)
        self.i2i_generate_btn.clicked.connect(self._generate_img2img)
        left_layout.addWidget(self.i2i_generate_btn)
        self.i2i_progress = QProgressBar()
        self.i2i_progress.setVisible(False)
        left_layout.addWidget(self.i2i_progress)

        left_scroll = make_scroll_panel(left_inner)
        left_scroll.setMinimumWidth(320)

        # ── Right: Output ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 12, 12, 12)
        right_layout.setSpacing(8)

        self.i2i_output = QLabel("img2img results will appear here")
        self.i2i_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.i2i_output.setMinimumSize(300, 300)
        self.i2i_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.i2i_output.setStyleSheet(
            "background-color: #161b22; border-radius: 12px;"
            "color: #484f58; font-size: 14px; border: 1px solid #21262d;")
        right_layout.addWidget(self.i2i_output, 1)

        out_ctrl = QHBoxLayout()
        self.i2i_save_btn = QPushButton("Save Image")
        self.i2i_save_btn.clicked.connect(self._save_i2i)
        self.i2i_save_btn.setEnabled(False)
        out_ctrl.addWidget(self.i2i_save_btn)
        out_ctrl.addStretch()
        right_layout.addLayout(out_ctrl)

        self.i2i_info = QLabel("")
        self.i2i_info.setObjectName("muted")
        self.i2i_info.setWordWrap(True)
        right_layout.addWidget(self.i2i_info)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        return tab

    def _update_i2i_preview(self, path):
        if not path or not Path(path).exists():
            self.i2i_preview.setText("Drag & drop or browse")
            return
        try:
            pixmap = QPixmap(path).scaled(
                256, 256, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.i2i_preview.setPixmap(pixmap)
        except Exception:
            self.i2i_preview.setText("Preview failed")

    def _generate_img2img(self):
        if not self.image_generator or not self.image_generator.pipe:
            QMessageBox.warning(self, "Error", "Load a model first (Generate tab).")
            return
        prompt = self.i2i_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Error", "Enter a prompt.")
            return
        input_path = self.i2i_input.path()
        if not input_path or not Path(input_path).exists():
            QMessageBox.warning(self, "Error", "Select an input image.")
            return

        self.i2i_generate_btn.setEnabled(False)
        self.i2i_progress.setVisible(True)
        self.i2i_progress.setValue(0)
        self.i2i_info.setText("Generating...")

        kwargs = {
            "prompt": prompt,
            "negative_prompt": self.i2i_negative.toPlainText().strip(),
            "width": int(self.i2i_width.value()),
            "height": int(self.i2i_height.value()),
            "steps": int(self.i2i_steps.value()),
            "cfg_scale": self.i2i_cfg.value(),
            "seed": int(self.i2i_seed.value()),
            "sampler": self.i2i_sampler.currentText(),
            "init_image": input_path,
            "strength": self.i2i_denoising.value(),
        }

        worker = GenerationWorker(self.image_generator, "image", **kwargs)
        worker.progress.connect(lambda s, t: (
            self.i2i_progress.setMaximum(t),
            self.i2i_progress.setValue(s),
        ))
        worker.finished.connect(self._on_i2i_complete)
        worker.error.connect(self._on_i2i_error)
        self._track_worker(worker)
        worker.start()

    def _on_i2i_complete(self, images):
        self.i2i_generate_btn.setEnabled(True)
        self.i2i_progress.setVisible(False)
        self._i2i_images = images
        if images:
            img = images[0]
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            qimg = QImage()
            qimg.loadFromData(buffer.read())
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.i2i_output.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.i2i_output.setPixmap(pixmap)
            self.i2i_save_btn.setEnabled(True)
            self.i2i_info.setText(f"Generated {len(images)} image(s)  |  {img.width}x{img.height}")

    def _on_i2i_error(self, error):
        self.i2i_generate_btn.setEnabled(True)
        self.i2i_progress.setVisible(False)
        self.i2i_info.setText(f"Error: {error}")
        QMessageBox.critical(self, "img2img Error", error)

    def _save_i2i(self):
        if not hasattr(self, '_i2i_images') or not self._i2i_images:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
            "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)")
        if path:
            self._i2i_images[0].save(path, quality=95)
            self.i2i_info.setText(f"Saved: {path}")

    # ═══════════════════════════════════════════════════════════════════════
    #  MODEL BROWSER TAB (Task 11)
    # ═══════════════════════════════════════════════════════════════════════

    MODEL_REGISTRY = [
        {"name": "Stable Diffusion 1.5", "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
         "type": "Image — SD1.5", "size": "~4.3 GB",
         "description": "Classic SD 1.5. Fast, well-supported, huge ecosystem of LoRAs.",
         "tags": ["sd15", "image"]},
        {"name": "SDXL Base 1.0", "repo": "stabilityai/stable-diffusion-xl-base-1.0",
         "type": "Image — SDXL", "size": "~6.9 GB",
         "description": "High-resolution 1024px generation. State of the art for photorealism.",
         "tags": ["sdxl", "image"]},
        {"name": "SDXL VAE fp16-fix", "repo": "madebyollin/sdxl-vae-fp16-fix",
         "type": "VAE — SDXL", "size": "~160 MB",
         "description": "Fixed SDXL VAE that doesn't NaN in fp16. Required for SDXL quality.",
         "tags": ["sdxl", "vae"]},
        {"name": "FLUX.1-dev", "repo": "black-forest-labs/FLUX.1-dev",
         "type": "Image — Flux", "size": "~32 GB",
         "description": "State of the art diffusion transformer. Requires 16GB+ VRAM.",
         "tags": ["flux", "image"]},
        {"name": "FLUX.1-schnell", "repo": "black-forest-labs/FLUX.1-schnell",
         "type": "Image — Flux (fast)", "size": "~32 GB",
         "description": "4-step distilled FLUX model. Apache 2.0 license.",
         "tags": ["flux", "image"]},
        {"name": "Wan 2.1 T2V 1.3B", "repo": "Wan-AI/Wan2.1-T2V-1.3B",
         "type": "Video", "size": "~11 GB",
         "description": "Text-to-video model. Works on 8GB VRAM with balanced offloading.",
         "tags": ["video"]},
        {"name": "ControlNet Canny SD1.5", "repo": "lllyasviel/control_v11p_sd15_canny",
         "type": "ControlNet — SD1.5", "size": "~1.5 GB",
         "description": "Canny edge ControlNet for SD1.5.",
         "tags": ["controlnet", "sd15"]},
        {"name": "ControlNet Depth SD1.5", "repo": "lllyasviel/control_v11f1p_sd15_depth",
         "type": "ControlNet — SD1.5", "size": "~1.5 GB",
         "description": "Depth map ControlNet for SD1.5.",
         "tags": ["controlnet", "sd15"]},
        {"name": "ControlNet Canny SDXL", "repo": "diffusers/controlnet-canny-sdxl-1.0",
         "type": "ControlNet — SDXL", "size": "~2.5 GB",
         "description": "Canny edge ControlNet for SDXL.",
         "tags": ["controlnet", "sdxl"]},
        {"name": "IP-Adapter SD1.5", "repo": "h94/IP-Adapter",
         "type": "IP-Adapter — SD1.5", "size": "~360 MB",
         "description": "Style reference adapter. Steers generation toward reference image style.",
         "tags": ["ip-adapter", "sd15"]},
        {"name": "BLIP-2 (2.7B)", "repo": "Salesforce/blip2-opt-2.7b",
         "type": "Captioning", "size": "~5.5 GB",
         "description": "Powerful image captioning for dataset preparation.",
         "tags": ["captioning"]},
        {"name": "WD SwinV2 Tagger v3", "repo": "SmilingWolf/wd-swinv2-tagger-v3",
         "type": "Captioning", "size": "~350 MB",
         "description": "Fast tag-based captioner. Essential for anime/art datasets.",
         "tags": ["captioning"]},
    ]

    def _build_models_tab(self) -> QWidget:
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(8)

        # Filter bar
        filter_row = QHBoxLayout()
        self.model_search = QLineEdit()
        self.model_search.setPlaceholderText("Search models...")
        self.model_search.textChanged.connect(self._filter_model_cards)
        filter_row.addWidget(self.model_search, 1)
        self._model_tag_filter = "all"
        for tag_label in ["All", "Image", "Video", "ControlNet", "Captioning"]:
            btn = QPushButton(tag_label)
            btn.setCheckable(True)
            btn.setChecked(tag_label == "All")
            btn.setMaximumWidth(100)
            btn.setStyleSheet("QPushButton:checked { background: #1f6feb; color: white; }")
            btn.clicked.connect(lambda _, t=tag_label: self._set_model_tag_filter(t))
            filter_row.addWidget(btn)
        outer_layout.addLayout(filter_row)

        # Scrollable card area
        self._model_card_widgets = []
        scroll_inner = QWidget()
        self._model_cards_layout = QVBoxLayout(scroll_inner)
        self._model_cards_layout.setSpacing(8)

        for entry in self.MODEL_REGISTRY:
            card = self._create_model_card(entry)
            self._model_cards_layout.addWidget(card)
            self._model_card_widgets.append((entry, card))

        self._model_cards_layout.addStretch()
        scroll = make_scroll_panel(scroll_inner)
        outer_layout.addWidget(scroll, 1)
        return tab

    def _create_model_card(self, entry: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background: #161b22; border: 1px solid #21262d;"
            "border-radius: 8px; padding: 10px; }")
        layout = QHBoxLayout(card)
        layout.setSpacing(12)

        # Info side
        info_layout = QVBoxLayout()
        name_lbl = QLabel(f"<b>{entry['name']}</b>")
        name_lbl.setStyleSheet("font-size: 14px; color: #c9d1d9; background: transparent;")
        info_layout.addWidget(name_lbl)

        type_lbl = QLabel(f"{entry['type']}  •  {entry['size']}")
        type_lbl.setStyleSheet("font-size: 11px; color: #8b949e; background: transparent;")
        info_layout.addWidget(type_lbl)

        desc_lbl = QLabel(entry['description'])
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size: 12px; color: #8b949e; background: transparent;")
        info_layout.addWidget(desc_lbl)

        status_lbl = QLabel()
        status_lbl.setObjectName("model_status")
        status_lbl.setStyleSheet("font-size: 11px; background: transparent;")
        info_layout.addWidget(status_lbl)
        card._status_label = status_lbl
        card._repo = entry['repo']
        card._tags = entry.get('tags', [])
        card._name = entry['name']
        layout.addLayout(info_layout, 1)

        # Button side
        btn_layout = QVBoxLayout()
        dl_btn = QPushButton("Download")
        dl_btn.setFixedWidth(100)
        dl_btn.clicked.connect(lambda _, r=entry['repo'], k=entry['name'], b=dl_btn:
                               self._download_model_card(r, k, b))
        btn_layout.addWidget(dl_btn)
        card._dl_btn = dl_btn
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Check status
        self._check_model_status(card)
        return card

    def _check_model_status(self, card):
        """Check if a model is downloaded in HF cache."""
        try:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            slug = "models--" + card._repo.replace("/", "--")
            if (cache_dir / slug).exists():
                card._status_label.setText("Downloaded ✓")
                card._status_label.setStyleSheet(
                    "color: #3fb950; font-size: 11px; background: transparent;")
                card._dl_btn.setEnabled(False)
                card._dl_btn.setText("Downloaded")
            else:
                card._status_label.setText("Not downloaded")
                card._status_label.setStyleSheet(
                    "color: #8b949e; font-size: 11px; background: transparent;")
        except Exception:
            pass

    def _download_model_card(self, repo: str, key: str, btn):
        btn.setEnabled(False)
        btn.setText("Downloading...")
        worker = ModelDownloadWorker(key, repo)
        worker.progress.connect(lambda msg: btn.setText(msg[:18] + "..."))
        worker.finished.connect(lambda k: (
            btn.setText("Downloaded"),
            self._refresh_model_cards()))
        worker.error.connect(lambda k, e: (
            btn.setText("Download"),
            btn.setEnabled(True),
            QMessageBox.warning(self, "Download Error", e)))
        self._track_worker(worker)
        worker.start()

    def _refresh_model_cards(self):
        for entry, card in self._model_card_widgets:
            self._check_model_status(card)

    def _filter_model_cards(self, text=""):
        search = text.lower() if text else self.model_search.text().lower()
        tag_filter = self._model_tag_filter.lower()
        for entry, card in self._model_card_widgets:
            name_match = search in entry['name'].lower() or search in entry.get('description', '').lower()
            tag_match = tag_filter == "all" or tag_filter in [t.lower() for t in entry.get('tags', [])]
            card.setVisible(name_match and tag_match)

    def _set_model_tag_filter(self, tag):
        self._model_tag_filter = tag
        self._filter_model_cards()

    # ═══════════════════════════════════════════════════════════════════════
    #  SETTINGS TAB
    # ═══════════════════════════════════════════════════════════════════════

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        hw = self.cfg.config.hardware

        # Hardware info
        hw_section = CollapsibleSection("Hardware Profile")
        hw_grid = QGridLayout()
        hw_grid.setSpacing(6)
        hw_grid.addWidget(StatusCard("GPU", hw.gpu_name), 0, 0, 1, 2)
        hw_grid.addWidget(StatusCard("VRAM", f"{hw.vram_total_mb} MB"), 0, 2)
        hw_grid.addWidget(StatusCard("RAM", f"{hw.ram_total_gb} GB"), 1, 0)
        hw_grid.addWidget(StatusCard("CPU Cores", str(hw.cpu_cores)), 1, 1)
        hw_grid.addWidget(StatusCard("CUDA", hw.cuda_version), 1, 2)
        hw_section.addLayout(hw_grid)
        refresh_hw_btn = QPushButton("Refresh Hardware Info")
        refresh_hw_btn.clicked.connect(self._refresh_hardware)
        hw_section.addWidget(refresh_hw_btn)
        layout.addWidget(hw_section)

        # Offloading
        off_section = CollapsibleSection("VRAM Offloading")
        self.offload_mode = LabeledCombo("Offload Mode",
            ["none", "balanced", "aggressive", "cpu_only"], hw.offload_mode,
            tooltip="balanced = recommended for 8GB. aggressive = slower but fits more.")
        self.offload_mode.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("hardware", "offload_mode", v))
        self.offload_mode.currentTextChanged.connect(lambda _: self._update_vram_budget())
        off_section.addWidget(self.offload_mode)

        offload_desc = QLabel(
            "none — Everything on GPU (needs lots of VRAM)\n"
            "balanced — Smart CPU offload (recommended for 8GB)\n"
            "aggressive — Sequential offload (slower, less VRAM)\n"
            "cpu_only — Maximum offload, minimum VRAM usage"
        )
        offload_desc.setObjectName("muted")
        offload_desc.setWordWrap(True)
        off_section.addWidget(offload_desc)

        off_section.addWidget(Separator())

        self.transformer_offload = LabeledSlider("Transformer Offload %",
            0, 100, hw.transformer_offload_pct, 10)
        self.transformer_offload.valueChanged.connect(
            lambda v: self.cfg.update_and_save("hardware", "transformer_offload_pct", int(v)))
        off_section.addWidget(self.transformer_offload)

        self.te_offload = LabeledSlider("Text Encoder Offload %",
            0, 100, hw.text_encoder_offload_pct, 10)
        self.te_offload.valueChanged.connect(
            lambda v: self.cfg.update_and_save("hardware", "text_encoder_offload_pct", int(v)))
        off_section.addWidget(self.te_offload)

        self.max_vram = LabeledSlider("Max VRAM Usage (MB)",
            4000, 8000, hw.max_vram_usage_mb, 500)
        self.max_vram.valueChanged.connect(
            lambda v: self.cfg.update_and_save("hardware", "max_vram_usage_mb", int(v)))
        off_section.addWidget(self.max_vram)

        off_section.addWidget(Separator())

        # Toggle options
        checks = [
            ("VAE Offload", "vae_offload", hw.vae_offload),
            ("Attention Slicing", "attention_slicing", hw.attention_slicing),
            ("VAE Slicing", "vae_slicing", hw.vae_slicing),
            ("VAE Tiling", "vae_tiling", hw.vae_tiling),
            ("xformers", "xformers", hw.xformers),
            ("FP8 Base Model", "use_fp8", hw.use_fp8),
            ("BF16 Precision", "use_bf16", hw.use_bf16),
            ("Gradient Checkpointing", "gradient_checkpointing", hw.gradient_checkpointing),
        ]
        for label, key, default in checks:
            cb = LabeledCheck(label, default=default)
            cb.toggled.connect(
                lambda v, k=key: self.cfg.update_and_save("hardware", k, v))
            off_section.addWidget(cb)

        layout.addWidget(off_section)

        # ── VRAM Budget Estimate ──
        vram_budget_section = CollapsibleSection("VRAM Budget Estimate")
        self.vram_budget_label = QLabel("Select a model and offload mode to see estimate")
        self.vram_budget_label.setWordWrap(True)
        self.vram_budget_label.setStyleSheet("color: #8b949e; padding: 8px;")
        vram_budget_section.addWidget(self.vram_budget_label)
        layout.addWidget(vram_budget_section)

        # Paths
        paths_section = CollapsibleSection("Default Paths")
        from core.config import MODELS_DIR, OUTPUTS_DIR, DATASETS_DIR, CACHE_DIR
        for name, path in [("Models", MODELS_DIR), ("Outputs", OUTPUTS_DIR),
                           ("Datasets", DATASETS_DIR), ("Cache", CACHE_DIR)]:
            row = QHBoxLayout()
            lbl = QLabel(f"{name}:")
            lbl.setStyleSheet("font-weight: bold; color: #8b949e;")
            lbl.setFixedWidth(80)
            row.addWidget(lbl)
            path_lbl = QLabel(str(path))
            path_lbl.setObjectName("muted")
            path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(path_lbl, 1)
            paths_section.addLayout(row)
        layout.addWidget(paths_section)

        # ── Theme toggle ────────────────────────────────────
        theme_section = CollapsibleSection("Appearance")
        self.theme_toggle = LabeledCombo("Theme", ["dark", "light"], self.cfg.config.theme)
        self.theme_toggle.currentTextChanged.connect(self._toggle_theme)
        theme_section.addWidget(self.theme_toggle)
        layout.addWidget(theme_section)

        # ── Model Manager ──
        layout.addWidget(self._build_model_manager_section())

        # ── Log Viewer ──
        log_section = CollapsibleSection("Log Viewer", start_collapsed=True)

        log_toolbar = QHBoxLayout()
        self.log_level_combo = LabeledCombo("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], "INFO")
        self.log_level_combo.currentTextChanged.connect(
            lambda v: __import__('core.logger', fromlist=['set_log_level']).set_log_level(v))
        refresh_log_btn = QPushButton("Refresh")
        refresh_log_btn.clicked.connect(self._refresh_log_viewer)
        open_log_btn = QPushButton("Open Log Folder")
        open_log_btn.clicked.connect(self._open_log_folder)
        clear_log_btn = QPushButton("Clear Display")
        clear_log_btn.clicked.connect(lambda: self.log_viewer.clear())
        log_toolbar.addWidget(self.log_level_combo)
        log_toolbar.addWidget(refresh_log_btn)
        log_toolbar.addWidget(open_log_btn)
        log_toolbar.addWidget(clear_log_btn)
        log_toolbar_widget = QWidget()
        log_toolbar_widget.setLayout(log_toolbar)

        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(300)
        self.log_viewer.setStyleSheet(
            "font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;"
            "font-size: 11px; background-color: #010409; color: #8b949e;"
            "border: 1px solid #21262d; border-radius: 8px; padding: 6px;")

        log_section.addWidget(log_toolbar_widget)
        log_section.addWidget(self.log_viewer)
        layout.addWidget(log_section)

        # Auto-refresh log viewer every 5 seconds
        self._log_refresh_timer = QTimer(self)
        self._log_refresh_timer.timeout.connect(self._refresh_log_viewer)
        self._log_refresh_timer.start(5000)

        layout.addStretch()

        scroll = make_scroll_panel(inner)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

        return tab

    # ═══════════════════════════════════════════════════════════════════════
    #  MODEL MANAGER (Settings tab)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_model_manager_section(self) -> CollapsibleSection:
        """Build the Model Manager collapsible section for Settings tab."""
        from core.model_downloader import MANAGED_MODELS

        section = CollapsibleSection("Model Manager", start_collapsed=True)
        self._model_cards = {}

        for key, info in MANAGED_MODELS.items():
            card = QFrame()
            card.setStyleSheet(
                "QFrame { border: 1px solid #21262d; border-radius: 6px; "
                "padding: 8px; margin: 2px 0px; }"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(4)

            top_row = QHBoxLayout()
            label = QLabel(f"<b>{info['label']}</b>  ({info['repo']})")
            label.setWordWrap(True)
            top_row.addWidget(label, 1)

            download_btn = QPushButton("Download")
            download_btn.setFixedWidth(90)
            download_btn.clicked.connect(lambda _, k=key: self._download_model(k))
            top_row.addWidget(download_btn)

            remove_btn = QPushButton("Remove")
            remove_btn.setFixedWidth(80)
            remove_btn.setStyleSheet("color: #e94560;")
            remove_btn.clicked.connect(lambda _, k=key: self._remove_model(k))
            top_row.addWidget(remove_btn)

            card_layout.addLayout(top_row)

            status_row = QHBoxLayout()
            status_label = QLabel("Status: Checking...")
            status_label.setObjectName("muted")
            status_row.addWidget(status_label)
            size_label = QLabel(f"Size: ~{info['size_gb']} GB")
            size_label.setObjectName("muted")
            status_row.addWidget(size_label)
            status_row.addStretch()
            card_layout.addLayout(status_row)

            progress = QProgressBar()
            progress.setVisible(False)
            progress.setRange(0, 0)
            progress.setMaximumHeight(16)
            card_layout.addWidget(progress)

            section.addWidget(card)
            self._model_cards[key] = {
                "download_btn": download_btn,
                "remove_btn": remove_btn,
                "status_label": status_label,
                "progress": progress,
            }

        QTimer.singleShot(500, self._refresh_model_status)
        return section

    def _refresh_model_status(self):
        """Check download status for all managed models."""
        from core.model_downloader import MANAGED_MODELS, check_model_cached
        for key, info in MANAGED_MODELS.items():
            if key not in self._model_cards:
                continue
            card = self._model_cards[key]
            try:
                cached = check_model_cached(info["repo"])
            except Exception:
                cached = False
            if cached:
                card["status_label"].setText("Status: Downloaded \u2713")
                card["status_label"].setStyleSheet("color: #3fb950;")
                card["download_btn"].setEnabled(False)
                card["download_btn"].setText("Downloaded")
                card["remove_btn"].setVisible(True)
            else:
                card["status_label"].setText("Status: Not downloaded")
                card["status_label"].setStyleSheet("color: #8b949e;")
                card["download_btn"].setEnabled(True)
                card["download_btn"].setText("Download")
                card["remove_btn"].setVisible(False)

    def _download_model(self, model_key: str):
        """Start downloading a managed model."""
        from core.model_downloader import MANAGED_MODELS
        info = MANAGED_MODELS.get(model_key)
        if not info:
            return
        card = self._model_cards[model_key]
        card["download_btn"].setEnabled(False)
        card["download_btn"].setText("Downloading...")
        card["progress"].setVisible(True)
        card["status_label"].setText("Status: Downloading...")
        card["status_label"].setStyleSheet("color: #1f6feb;")
        worker = ModelDownloadWorker(model_key, info["repo"])
        worker.progress.connect(lambda msg: card["status_label"].setText(f"Status: {msg}"))
        worker.finished.connect(self._on_model_download_complete)
        worker.error.connect(self._on_model_download_error)
        self._workers.append(worker)
        worker.start()

    def _on_model_download_complete(self, model_key: str):
        """Handle completed model download."""
        if model_key in self._model_cards:
            card = self._model_cards[model_key]
            card["progress"].setVisible(False)
            card["status_label"].setText("Status: Downloaded \u2713")
            card["status_label"].setStyleSheet("color: #3fb950;")
            card["download_btn"].setEnabled(False)
            card["download_btn"].setText("Downloaded")
            card["remove_btn"].setVisible(True)
        self.statusBar().showMessage(f"Model '{model_key}' downloaded successfully", 5000)

    def _on_model_download_error(self, model_key: str, error_msg: str):
        """Handle failed model download."""
        if model_key in self._model_cards:
            card = self._model_cards[model_key]
            card["progress"].setVisible(False)
            card["status_label"].setText("Status: Download failed")
            card["status_label"].setStyleSheet("color: #e94560;")
            card["download_btn"].setEnabled(True)
            card["download_btn"].setText("Retry")
        QMessageBox.warning(self, "Download Error",
                            f"Failed to download model '{model_key}':\n{error_msg}")

    def _remove_model(self, model_key: str):
        """Remove a downloaded model after confirmation."""
        from core.model_downloader import MANAGED_MODELS, remove_model_cache
        info = MANAGED_MODELS.get(model_key)
        if not info:
            return
        reply = QMessageBox.question(
            self, "Remove Model",
            f"Remove {info['label']} (~{info['size_gb']} GB) from cache?\n\n"
            f"This will delete the downloaded model files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if remove_model_cache(info["repo"]):
            self.statusBar().showMessage(f"Model '{info['label']}' removed", 5000)
        else:
            QMessageBox.warning(self, "Error", f"Failed to remove model '{info['label']}'")
        self._refresh_model_status()

    # ═══════════════════════════════════════════════════════════════════════
    #  GALLERY TAB
    # ═══════════════════════════════════════════════════════════════════════

    def _build_gallery_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Toolbar
        toolbar = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_gallery)
        self.gallery_filter = QLineEdit()
        self.gallery_filter.setPlaceholderText("Filter by filename...")
        self.gallery_filter.textChanged.connect(self._filter_gallery)
        self.gallery_sort = LabeledCombo("Sort", ["Newest First", "Oldest First", "Name A-Z"], "Newest First")
        self.gallery_sort.currentTextChanged.connect(lambda _: self._refresh_gallery())
        toolbar.addWidget(refresh_btn)
        toolbar.addWidget(self.gallery_filter)
        toolbar.addWidget(self.gallery_sort)
        layout.addLayout(toolbar)

        # Gallery grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.gallery_container = QWidget()
        self.gallery_grid = QGridLayout(self.gallery_container)
        self.gallery_grid.setSpacing(8)
        scroll.setWidget(self.gallery_container)
        layout.addWidget(scroll)

        return tab

    def _refresh_gallery(self):
        output_dir = self.cfg.config.generation.output_dir
        if not output_dir or not Path(output_dir).exists():
            return

        # Clear existing
        while self.gallery_grid.count():
            item = self.gallery_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Collect images
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        files = [f for f in Path(output_dir).iterdir() if f.suffix.lower() in exts]

        sort = self.gallery_sort.currentText()
        if sort == "Newest First":
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        elif sort == "Oldest First":
            files.sort(key=lambda f: f.stat().st_mtime)
        else:
            files.sort(key=lambda f: f.name.lower())

        filter_text = self.gallery_filter.text().lower()
        if filter_text:
            files = [f for f in files if filter_text in f.name.lower()]

        cols = 4
        for i, filepath in enumerate(files[:100]):
            try:
                pixmap = QPixmap(str(filepath))
                if pixmap.isNull():
                    continue
                thumb = pixmap.scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                label = QLabel()
                label.setPixmap(thumb)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setToolTip(filepath.name)
                label.setStyleSheet("border: 1px solid #21262d; border-radius: 6px; padding: 4px;")
                self.gallery_grid.addWidget(label, i // cols, i % cols)
            except Exception:
                continue

    def _filter_gallery(self, text: str):
        self._refresh_gallery()

    # ═══════════════════════════════════════════════════════════════════════
    #  ACTIONS — Dataset
    # ═══════════════════════════════════════════════════════════════════════

    def _update_pipeline_mode_info(self, mode: str):
        """Update the info label and toggle video frame controls."""
        if mode == "image":
            self.pipeline_mode_info.setText(
                "Image model pipeline: videos will be split into "
                "individual frames for captioning and training.")
            # Show frame extraction controls
            if hasattr(self, "_vid_frame_controls"):
                for w in self._vid_frame_controls:
                    w.setVisible(True)
        else:
            self.pipeline_mode_info.setText(
                "Video model pipeline: videos are kept as video "
                "clips. Captions describe each video as a whole. "
                "Use with Wan2.1 or similar video architectures.")
            # Hide frame extraction controls (not relevant for video mode)
            if hasattr(self, "_vid_frame_controls"):
                for w in self._vid_frame_controls:
                    w.setVisible(False)

    def _on_pipeline_mode_changed(self, mode: str):
        mode = mode.strip().lower()
        self.cfg.update_and_save("app", "pipeline_mode", mode)
        self._update_pipeline_mode_info(mode)

    def _on_dataset_path_changed(self, path):
        self.cfg.config.last_dataset_dir = path
        self.cfg.save()

    def _open_dataset_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self.dataset_path.setPath(path)

    def eventFilter(self, obj, event):
        """Forward drag-and-drop on file_list to the dataset drop handler."""
        if obj is self.file_list:
            if event.type() == event.Type.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif event.type() == event.Type.Drop:
                files = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
                if files:
                    self._on_files_dropped(files)
                return True
        return super().eventFilter(obj, event)

    def _on_files_dropped(self, files):
        """Handle drag-and-drop files."""
        from core.dataset import DatasetManager
        if not self.dataset_manager:
            # Use parent dir of first file
            parent = os.path.dirname(files[0])
            self.dataset_manager = DatasetManager(parent)
        self.dataset_manager.scan_files(files)
        self._on_scan_complete(self.dataset_manager.items)
        self.dataset_status.setText(f"Dropped {len(files)} files")

    def _scan_dataset(self):
        """Scan a directory for images/videos and populate the dataset list.

        Flow:
        1. Validate the selected directory path
        2. Create a DatasetManager for the directory
        3. Create DatasetWorker to scan files in a background thread
        4. Connect progress/finished/error signals to update UI
        5. Populate self.dataset_list with discovered items on completion
        6. Update status bar with item count and media types found
        """
        path = self.dataset_path.path()
        if not path or not os.path.isdir(path):
            QMessageBox.warning(self, "Error", "Select a valid dataset directory.")
            return

        from core.dataset import DatasetManager
        self.dataset_manager = DatasetManager(path)

        self.dataset_status.setText("Scanning...")
        self.scan_btn.setEnabled(False)

        worker = DatasetWorker(self.dataset_manager, "scan")
        worker.finished.connect(self._on_scan_complete)
        worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        worker.finished.connect(lambda: self.scan_btn.setEnabled(True))
        self._track_worker(worker)
        worker.start()

    def _on_scan_complete(self, items):
        self.file_list.clear()
        for item in items:
            name = os.path.basename(item.original_path)
            icon = "IMG" if item.media_type == "image" else ("VID" if item.media_type == "video" else "ANI")
            cap = "  [captioned]" if item.caption_text else ""
            list_item = QListWidgetItem(f"[{icon}]  {name}{cap}")
            # Color-code by state
            if not item.is_valid:
                list_item.setForeground(QColor("#f85149"))  # red
            elif item.converted_path and item.caption_text:
                list_item.setForeground(QColor("#3fb950"))  # green = fully ready
            elif item.caption_text:
                list_item.setForeground(QColor("#d29922"))  # yellow = captioned only
            elif item.converted_path:
                list_item.setForeground(QColor("#1f6feb"))  # blue = processed only
            else:
                list_item.setForeground(QColor("#8b949e"))  # grey = raw
            self.file_list.addItem(list_item)

        s = self.dataset_manager.stats
        self.stat_total.setValue(str(s.total_files))
        self.stat_images.setValue(str(s.valid_images + s.animated_images))
        self.stat_videos.setValue(str(s.valid_videos))
        self.stat_captioned.setValue(str(s.captioned_files))
        self.stat_size.setValue(f"{s.total_size_mb:.1f} MB")
        self.stat_avgres.setValue(f"{s.avg_resolution[0]}x{s.avg_resolution[1]}")
        self.dataset_status.setText(f"Found {s.total_files} files")
        # Update summary
        n_images = sum(1 for i in items if i.media_type == "image")
        n_videos = sum(1 for i in items if i.media_type == "video")
        n_anim = sum(1 for i in items if i.media_type == "animated")
        self.dataset_summary.setText(
            f"{len(items)} items \u2014 {n_images} images, {n_videos} videos, {n_anim} animated")

    def _add_individual_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "",
            "All supported (*.png *.jpg *.jpeg *.gif *.webp *.bmp *.tiff *.mp4 *.avi *.mov *.mkv);;"
            "Images (*.png *.jpg *.jpeg *.gif *.webp *.bmp *.tiff *.heic *.avif);;"
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;"
            "All files (*)"
        )
        if files:
            from core.dataset import DatasetManager
            if not self.dataset_manager:
                self.dataset_manager = DatasetManager(os.path.dirname(files[0]))
            self.dataset_manager.scan_files(files)
            self._on_scan_complete(self.dataset_manager.items)

    def _dataset_context_menu(self, pos):
        """Right-click context menu on dataset list."""
        item = self.file_list.itemAt(pos)
        if not item:
            return
        row = self.file_list.row(item)
        menu = QMenu(self)
        menu.addAction("Open in Explorer", lambda: self._open_item_in_explorer(row))
        menu.addAction("View Caption", lambda: self._view_item_caption(row))
        menu.addAction("Copy Path", lambda: QApplication.clipboard().setText(
            self._item_path_for_row(row)))
        menu.exec(self.file_list.mapToGlobal(pos))

    def _item_path_for_row(self, row):
        if self.dataset_manager and 0 <= row < len(self.dataset_manager.items):
            return self.dataset_manager.items[row].original_path
        return ""

    def _open_item_in_explorer(self, row):
        path = self._item_path_for_row(row)
        if path:
            import subprocess, sys as _sys
            folder = os.path.dirname(path)
            if _sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", path])
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["xdg-open", folder])

    def _view_item_caption(self, row):
        if self.dataset_manager and 0 <= row < len(self.dataset_manager.items):
            item = self.dataset_manager.items[row]
            cap = item.caption_text or "(no caption)"
            self.preview_caption.setPlainText(cap)

    def _find_duplicates(self):
        if not self.dataset_manager or not self.dataset_manager.items:
            QMessageBox.information(self, "No Data", "Scan a dataset first.")
            return
        self.dataset_status.setText("Scanning for duplicates...")
        QApplication.processEvents()
        dupes = self.dataset_manager.find_duplicates()
        if not dupes:
            self.dataset_status.setText("No duplicates found.")
            self.dataset_log.appendPlainText("Duplicate scan: no duplicates found.")
            return
        msg = f"Found {len(dupes)} potential duplicate pair(s):\n\n"
        for a, b, dist in dupes[:20]:
            msg += f"  Distance {dist}: {os.path.basename(a)} <-> {os.path.basename(b)}\n"
        if len(dupes) > 20:
            msg += f"\n... and {len(dupes) - 20} more"
        self.dataset_log.appendPlainText(msg)
        self.dataset_status.setText(f"Found {len(dupes)} duplicate pair(s). See log for details.")

    def _start_caption_review(self):
        if not self.dataset_manager or not self.dataset_manager.items:
            return
        self._review_items = [it for it in self.dataset_manager.items if it.caption_text]
        if not self._review_items:
            QMessageBox.information(self, "No Captions", "No captions to review. Run captioning first.")
            return
        self._review_index = 0
        self._show_review_item()

    def _show_review_item(self):
        if self._review_index >= len(self._review_items):
            self.dataset_status.setText("Caption review complete!")
            return
        item = self._review_items[self._review_index]
        # Show image in preview
        try:
            path = item.converted_path if item.converted_path and os.path.isfile(item.converted_path) else item.original_path
            if os.path.isfile(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        self.preview_image.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                    self.preview_image.setPixmap(scaled)
        except Exception:
            pass
        self.preview_caption.setPlainText(item.caption_text or "")
        self.dataset_status.setText(
            f"Review: {self._review_index + 1}/{len(self._review_items)} — "
            f"{os.path.basename(item.original_path)}")

    def _next_review_item(self):
        self._review_index += 1
        self._show_review_item()

    def _prev_review_item(self):
        if self._review_index > 0:
            self._review_index -= 1
            self._show_review_item()

    def _convert_dataset(self):
        """Convert dataset images to training-ready format (PNG/RGB).

        Flow:
        1. Verify dataset is loaded with items
        2. Determine target format and resolution from config
        3. Create DatasetWorker for batch conversion in background thread
        4. Convert HEIC/AVIF/BMP/TIFF to PNG, resize if needed
        5. Extract video frames if pipeline_mode is 'image'
        6. Update item.converted_path for each processed file
        7. Log conversion stats (success/fail/skip counts)
        """
        if not self.dataset_manager or not self.dataset_manager.items:
            QMessageBox.warning(self, "Error", "No dataset loaded. Scan a directory first.")
            return

        from core.gpu_utils import warn_if_low_disk
        disk_warn = warn_if_low_disk(str(Path.home()))
        if disk_warn:
            reply = QMessageBox.warning(self, "Low Disk Space",
                f"{disk_warn}\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        self.dataset_progress.setVisible(True)
        self.dataset_progress.setValue(0)
        self.convert_btn.setEnabled(False)

        max_res = int(self.max_resolution.value())

        worker = DatasetWorker(self.dataset_manager, "convert",
                               target_format="png", max_resolution=max_res)
        worker.progress.connect(lambda c, t, f: (
            self.dataset_progress.setMaximum(t),
            self.dataset_progress.setValue(c),
            self.dataset_status.setText(f"Converting: {os.path.basename(f)}"),
        ))
        worker.finished.connect(lambda _: (
            self.dataset_progress.setVisible(False),
            self.convert_btn.setEnabled(True),
            self.dataset_status.setText("Conversion complete!"),
            self._on_scan_complete(self.dataset_manager.items),
        ))
        worker.error.connect(lambda e: (
            QMessageBox.critical(self, "Error", e),
            self.convert_btn.setEnabled(True),
        ))
        self._track_worker(worker)
        worker.start()

    def _run_captioning(self):
        """Run auto-captioning on the current dataset.

        Flow:
        1. Gather captioning config (method, format, overwrite flag)
        2. Check disk space and warn if low
        3. Create CaptionWorker with the pipeline and dataset items
        4. Connect progress/log/finished/error signals
        5. Start the worker thread (non-blocking)
        6. Track start time for ETA calculation
        """
        if not self.dataset_manager or not self.dataset_manager.items:
            QMessageBox.warning(self, "Error", "No dataset loaded.")
            return

        from core.gpu_utils import warn_if_low_disk
        disk_warn = warn_if_low_disk(str(Path.home()))
        if disk_warn:
            reply = QMessageBox.warning(self, "Low Disk Space",
                f"{disk_warn}\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        from captioning.auto_caption import AutoCaptionPipeline
        from core.config import CaptioningConfig

        cap_cfg = self.cfg.config.captioning
        cap_cfg.method = self.caption_method.currentText()
        cap_cfg.wd_threshold = self.caption_threshold.value()
        cap_cfg.trigger_word = self.trigger_word.text()
        cap_cfg.caption_format = self.caption_format.currentText()
        cap_cfg.overwrite_existing = self.overwrite_captions.isChecked()
        cap_cfg.verbose_description = self.verbose_description.isChecked()
        cap_cfg.video_frame_interval = self.video_frame_interval.value()
        cap_cfg.pipeline_mode = self.pipeline_mode.currentText()
        cap_cfg.video_max_frames = self.video_max_frames.value()
        self.cfg.save()

        self.caption_pipeline = AutoCaptionPipeline(cap_cfg)

        items = self.dataset_manager.get_all_valid_items()
        if not items:
            QMessageBox.warning(self, "Error", "No valid items to caption.")
            return

        logger.info("Starting captioning: %d items", len(items))
        self.dataset_progress.setVisible(True)
        self.dataset_progress.setValue(0)
        self.caption_btn.setEnabled(False)
        self.dataset_status.setText("Loading captioning models...")

        # Captions go into the work_dir so originals are never touched
        cap_dir = str(self.dataset_manager.work_dir / "captions")
        worker = CaptionWorker(self.caption_pipeline, items, caption_dir=cap_dir)

        # Relay pipeline status messages (dependency checks, downloads,
        # model loading progress) to the dataset status label and log panel.
        worker.log.connect(lambda msg: self.dataset_status.setText(msg))
        worker.log.connect(lambda msg: self.dataset_log.appendPlainText(msg))

        import time as _time
        _cap_start = _time.time()

        def _cap_progress(c, t, f):
            self.dataset_progress.setMaximum(t)
            self.dataset_progress.setValue(c)
            elapsed = _time.time() - _cap_start
            if c > 0:
                eta = (elapsed / c) * (t - c)
                eta_str = f"  ~{int(eta)}s remaining" if eta < 3600 else f"  ~{eta/3600:.1f}h remaining"
            else:
                eta_str = ""
            self.dataset_status.setText(f"Captioning: {os.path.basename(f)}{eta_str}")

        worker.progress.connect(_cap_progress)
        worker.finished.connect(lambda results: (
            self.dataset_progress.setVisible(False),
            self.caption_btn.setEnabled(True),
            self.dataset_status.setText(f"Captioned {len(results)} files"),
            self._on_scan_complete(self.dataset_manager.items),
            self.caption_pipeline.unload_models() if self.caption_pipeline and not self.cfg.config.captioning.keep_models_in_memory else None,
            self._notify_completion("Captioning Complete"),
        ))
        worker.error.connect(lambda e: (
            QMessageBox.critical(self, "Error", f"Captioning error: {e}"),
            self.caption_btn.setEnabled(True),
            self.dataset_progress.setVisible(False),
        ))
        self._track_worker(worker)
        worker.start()

    def _save_caption(self):
        row = self.file_list.currentRow()
        if not self.dataset_manager or row < 0 or row >= len(self.dataset_manager.items):
            return
        item = self.dataset_manager.items[row]
        text = self.preview_caption.toPlainText().strip()
        # Save to work_dir captions
        cap_dir = self.dataset_manager.work_dir / "captions"
        cap_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(item.original_path).stem
        cap_path = cap_dir / f"{stem}.txt"
        cap_path.write_text(text, encoding="utf-8")
        item.caption_text = text
        item.caption_path = str(cap_path)
        self.dataset_status.setText(f"Caption saved: {stem}.txt")
        # Update file list display
        self._on_scan_complete(self.dataset_manager.items)
        self.file_list.setCurrentRow(row)

    def _preview_file(self, row):
        if not self.dataset_manager or row < 0 or row >= len(self.dataset_manager.items):
            return

        item = self.dataset_manager.items[row]

        try:
            path = item.converted_path if item.converted_path and os.path.isfile(item.converted_path) else item.original_path
            if os.path.isfile(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull() and (pixmap.width() > 800 or pixmap.height() > 800):
                    pixmap = pixmap.scaled(800, 800,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        self.preview_image.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.preview_image.setPixmap(scaled)
        except Exception:
            self.preview_image.setText("Cannot preview this file type")

        self.preview_caption.setPlainText(item.caption_text or "(No caption)")

        # Show video frame info if applicable
        if getattr(item, 'media_type', '') == 'video':
            self._preview_video_frames(item)

    def _preview_video_frames(self, item):
        """Show extracted frames as a filmstrip for video items."""
        if not self.dataset_manager:
            return
        work_dir = self.dataset_manager.work_dir
        if not work_dir:
            return

        stem = Path(item.original_path).stem
        converted_dir = work_dir / "converted"
        if not converted_dir.exists():
            return

        frames = sorted(converted_dir.glob(f"{stem}_frame_*"))
        if not frames:
            return

        info_parts = [f"Video: {os.path.basename(item.original_path)}", f"Extracted frames: {len(frames)}"]

        caption_dir = work_dir / "captions"
        for frame_path in frames[:10]:
            cap_file = caption_dir / f"{frame_path.stem}.txt"
            caption = cap_file.read_text(encoding="utf-8").strip() if cap_file.exists() else "(no caption)"
            info_parts.append(f"  {frame_path.name}: {caption[:80]}")

        if len(frames) > 10:
            info_parts.append(f"  ... and {len(frames) - 10} more frames")

        self.dataset_log.appendPlainText("\n".join(info_parts))

    # ═══════════════════════════════════════════════════════════════════════
    #  ACTIONS — Training
    # ═══════════════════════════════════════════════════════════════════════

    def _check_vram_params(self):
        """Warn if current parameters might exceed VRAM."""
        vram = self.cfg.config.hardware.vram_total_mb
        model = self.model_type.currentText()
        res = int(self.train_resolution.value())
        batch = int(self.batch_size.value())

        base = {"sd15": 2000, "sdxl": 4000, "flux": 6000}.get(model, 4000)
        per_pixel = res * res / (512 * 512)
        estimated = base * per_pixel * batch

        if estimated > vram:
            suggested_batch = max(1, int(batch * vram / estimated))
            suggested_res = int(res * (vram / estimated) ** 0.5)
            suggested_res = max(256, (suggested_res // 64) * 64)

            reply = QMessageBox.warning(self, "VRAM Warning",
                f"Current settings may use ~{estimated:.0f} MB VRAM (available: {vram} MB).\n\n"
                f"Suggested adjustments:\n"
                f"  Batch size: {batch} -> {suggested_batch}\n"
                f"  Resolution: {res} -> {suggested_res}\n\n"
                f"Auto-adjust?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self.batch_size.setValue(suggested_batch)
                self.train_resolution.setValue(suggested_res)

    def _start_training(self):
        """Launch LoRA/DreamBooth training via kohya-ss sd-scripts.

        Flow:
        1. Validate dataset is loaded and has captioned items
        2. Run VRAM budget check and offer auto-adjustment if over budget
        3. Collect all training hyperparams from GUI widgets
        4. Build TrainingJob with collected config
        5. Optionally set resume checkpoint path
        6. Create TrainingWorker and connect progress/log/finished signals
        7. Start training in background thread with live log streaming
        8. Enable stop button and disable start button
        """
        if not self.dataset_manager:
            QMessageBox.warning(self, "Error", "No dataset loaded. Go to the Dataset tab first.")
            return

        self._check_vram_params()

        from core.gpu_utils import warn_if_low_disk
        disk_warn = warn_if_low_disk(str(Path.home()))
        if disk_warn:
            reply = QMessageBox.warning(self, "Low Disk Space",
                f"{disk_warn}\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        tc = self.cfg.config.training
        tc.model_type = self.model_type.currentText()
        logger.info("Starting training: %s", tc.model_type)

        # Resolve base model — auto-fill default if empty
        user_path = self.base_model_path.path()
        if user_path:
            tc.base_model = user_path
        else:
            from core.model_downloader import DEFAULT_MODELS
            tc.base_model = DEFAULT_MODELS.get(tc.model_type, tc.base_model)
        tc.training_type = self.training_type.currentText()
        tc.lora_rank = int(self.lora_rank.value())
        tc.lora_alpha = int(self.lora_alpha.value())
        tc.learning_rate = self.learning_rate.value()
        tc.max_train_steps = int(self.train_steps.value())
        tc.epochs = int(self.train_epochs.value())
        tc.batch_size = int(self.batch_size.value())
        tc.resolution = int(self.train_resolution.value())
        tc.optimizer = self.optimizer.currentText()
        tc.lr_scheduler = self.lr_scheduler.currentText()
        tc.noise_offset = self.noise_offset.value()
        tc.flip_aug = self.flip_aug.isChecked()
        tc.color_aug = self.color_aug.isChecked()
        tc.instance_prompt = self.instance_prompt.text()
        tc.class_prompt = self.class_prompt.text()
        tc.output_name = self.output_name.text() or "my_lora"
        self.cfg.save()

        self.dataset_status.setText("Preparing training directory...")
        train_dir = self.dataset_manager.prepare_training_dir(
            num_repeats=int(self.num_repeats.value()),
            instance_prompt=tc.instance_prompt or "sks",
            class_prompt=tc.class_prompt or "character",
        )
        tc.dataset_dir = train_dir

        from training.trainer import TrainingJob
        self.training_job = TrainingJob(tc)

        self.train_log.clear()
        self.loss_graph.clear()
        self.train_progress.setValue(0)
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self._set_vram_poll_rate(True)

        worker = TrainingWorker(self.training_job)
        worker.progress.connect(self._on_train_progress)
        worker.log.connect(self._on_train_log)
        worker.finished.connect(self._on_train_complete)
        worker.error.connect(self._on_train_error)
        self._track_worker(worker)
        worker.start()

    def _on_train_progress(self, step, total, loss, lr):
        self.train_progress.setMaximum(total)
        self.train_progress.setValue(step)
        self.train_step_label.setValue(f"{step} / {total}")
        self.train_loss_label.setValue(f"{loss:.4f}")
        self.train_lr_label.setValue(f"{lr:.2e}")
        self.loss_graph.add_point(loss)
        # ETA: prefer training_job.get_eta(), fallback to time-based calc
        if self.training_job:
            self.train_eta_label.setValue(self.training_job.get_eta())
        elif step > 0:
            if not hasattr(self, '_train_start_time'):
                self._train_start_time = time.time()
            elapsed = time.time() - self._train_start_time
            secs_per_step = elapsed / step
            remaining = (total - step) * secs_per_step
            self.train_eta_label.setValue(f"ETA: {timedelta(seconds=int(remaining))}")
        if (self.sample_enabled.isChecked() and step > 0 and
                step % int(self.sample_every_n.value()) == 0):
            self._generate_training_sample(step)

    def _on_train_log(self, message):
        self.train_log.appendPlainText(message)

    def _on_train_complete(self, output_path):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self._set_vram_poll_rate(False)
        self.train_log.appendPlainText(
            f"\n{'=' * 50}\nTraining Complete!\nModel: {output_path}\n{'=' * 50}")
        self._notify_completion(f"Training Complete — saved to {os.path.basename(output_path)}")

        if output_path not in self.cfg.config.recent_models:
            self.cfg.config.recent_models.insert(0, output_path)
            self.cfg.config.recent_models = self.cfg.config.recent_models[:10]
            self.cfg.save()

    def _on_train_error(self, error):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self._set_vram_poll_rate(False)
        self.train_log.appendPlainText(f"\nERROR: {error}")
        logger.error("Training error: %s", error)
        QMessageBox.critical(self, "Training Error", error)

    def _stop_training(self):
        if self.training_job:
            self.training_job.cancel()
            self.train_log.appendPlainText("Cancelling training...")

    def _apply_training_preset(self):
        from configs.presets import PRESETS
        preset_map = {
            "SD 1.5 LoRA (8GB Safe)": "sd15_lora_8gb",
            "SDXL LoRA (8GB Tight)": "sdxl_lora_8gb",
            "FLUX LoRA (8GB Split)": "flux_lora_8gb",
            "SDXL Character LoRA (Recommended)": "sdxl_character_lora",
        }
        name = self.preset_combo.currentText()
        key = preset_map.get(name)
        if not key:
            return
        preset = PRESETS.get(key, {})
        config = preset.get("config", {})

        # Apply to GUI widgets
        field_map = {
            "model_type": self.model_type,
            "training_type": self.training_type,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }
        slider_map = {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "max_train_steps": self.train_steps,
            "batch_size": self.batch_size,
            "resolution": self.train_resolution,
            "noise_offset": self.noise_offset,
        }
        check_map = {
            "flip_aug": self.flip_aug,
            "color_aug": self.color_aug,
        }
        for k, widget in field_map.items():
            if k in config:
                widget.setCurrentText(str(config[k]))
        for k, widget in slider_map.items():
            if k in config:
                widget.setValue(config[k])
        for k, widget in check_map.items():
            if k in config:
                widget.setChecked(config[k])

        self._set_status(f"Applied preset: {name}")

    def _export_training_config(self):
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Export Training Config", "",
            "JSON (*.json);;TOML (*.toml)")
        if path:
            tc = self.cfg.config.training
            from dataclasses import asdict
            data = asdict(tc)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self._set_status(f"Config exported: {path}")

    def _import_training_config(self):
        import json
        path, _ = QFileDialog.getOpenFileName(self, "Import Training Config", "",
            "JSON (*.json);;All files (*)")
        if path:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(self.cfg.config.training, k):
                        setattr(self.cfg.config.training, k, v)
                self.cfg.save()
                self._set_status(f"Config imported: {path}")
                self._notify("Config imported successfully", "success")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", str(e))

    # ═══════════════════════════════════════════════════════════════════════
    #  ACTIONS — Generation
    # ═══════════════════════════════════════════════════════════════════════

    def _on_gen_mode_changed(self, mode):
        is_image = mode == "Image"
        self.img_params_section.setVisible(is_image)
        self.vid_params_section.setVisible(not is_image)

    def _load_gen_model(self):
        """Load a Stable Diffusion / FLUX model for image generation.

        Flow:
        1. Read model path and type from GUI widgets
        2. Create ImageGenerator if not already initialized
        3. Spawn ModelLoadWorker to load in background thread
        4. Auto-download from HuggingFace if model not found locally
        5. Apply VRAM optimizations (offloading, attention slicing, xformers)
        6. Update status bar and enable generate button on success
        """
        model_path = self.gen_model_path.path()
        model_type = self.gen_model_type.currentText()
        logger.info("Loading model: %s (%s)", model_path, model_type)

        # Auto-resolve: leave empty to use the default for this model type
        if not model_path:
            from core.model_downloader import DEFAULT_MODELS
            model_path = DEFAULT_MODELS.get(model_type, DEFAULT_MODELS["sdxl"])

        # Validate model path
        if model_path and not Path(model_path).exists():
            from core.model_downloader import is_hf_repo_id
            if not is_hf_repo_id(model_path):
                QMessageBox.warning(self, "Invalid Model Path",
                    f"'{model_path}' is not a valid local path or Hugging Face repo ID.\n\n"
                    "Examples:\n"
                    "  Local: D:\\models\\my_model.safetensors\n"
                    "  HF: stabilityai/stable-diffusion-xl-base-1.0")
                return

        self.gen_info.setText("Loading model...")
        self.load_model_btn.setEnabled(False)

        if model_type in ("wan21", "animatediff"):
            from generation.video_gen import VideoGenerator
            self.video_generator = VideoGenerator(self.cfg.config.hardware)
            generator = self.video_generator
        else:
            from generation.image_gen import ImageGenerator
            self.image_generator = ImageGenerator(self.cfg.config.hardware)
            self.image_generator._oom_callback = lambda mode: self._notify(
                f"Low VRAM — auto-switched to {mode} offloading", level="warning")
            generator = self.image_generator

        worker = ModelLoadWorker(generator, model_path, model_type)
        worker.progress.connect(lambda msg: self.gen_info.setText(msg))
        worker.finished.connect(lambda p: (
            self.gen_info.setText(f"Model loaded: {p}"),
            self.load_model_btn.setEnabled(True),
        ))
        worker.error.connect(lambda e: (
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}"),
            self.gen_info.setText(f"Error: {e}"),
            self.load_model_btn.setEnabled(True),
        ))
        self._track_worker(worker)
        worker.start()

    def _add_lora(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select LoRA", "",
            "SafeTensors (*.safetensors);;All files (*)")
        if path:
            weight = self.lora_weight_slider.value()
            item = QListWidgetItem(f"{os.path.basename(path)} (w={weight:.2f})")
            item.setData(Qt.ItemDataRole.UserRole, {"path": path, "weight": weight})
            self.lora_list.addItem(item)

    def _remove_lora(self):
        row = self.lora_list.currentRow()
        if row >= 0:
            self.lora_list.takeItem(row)

    def _add_embedding(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Embedding", "",
            "SafeTensors (*.safetensors);;Bin files (*.bin);;All files (*)")
        if path:
            name = Path(path).stem
            item = QListWidgetItem(f"{name}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.embed_list.addItem(item)

    def _remove_embedding(self):
        row = self.embed_list.currentRow()
        if row >= 0:
            self.embed_list.takeItem(row)

    def _reload_style_presets(self):
        from configs.prompt_presets import load_presets
        self._style_presets = load_presets()
        self.style_preset_combo.clear()
        self.style_preset_combo.addItems(list(self._style_presets.keys()))

    def _apply_style_preset(self):
        name = self.style_preset_combo.currentText()
        suffix = self._style_presets.get(name, "")
        current = self.gen_prompt.toPlainText().rstrip()
        if suffix and suffix not in current:
            self.gen_prompt.setPlainText(current + (", " if current else "") + suffix)

    def _save_style_preset(self):
        from configs.prompt_presets import save_presets
        name, ok = QInputDialog.getText(self, "Save Style Preset", "Preset name:")
        if ok and name:
            self._style_presets[name] = self.gen_prompt.toPlainText().strip()
            save_presets(self._style_presets)
            self._reload_style_presets()
            self._notify(f"Preset '{name}' saved", "success")

    def _delete_style_preset(self):
        from configs.prompt_presets import save_presets
        name = self.style_preset_combo.currentText()
        if name and name in self._style_presets:
            del self._style_presets[name]
            save_presets(self._style_presets)
            self._reload_style_presets()

    def _preview_controlnet(self):
        """Run the ControlNet preprocessor on the input image and show preview."""
        input_path = self.controlnet_input.path()
        if not input_path or not Path(input_path).exists():
            QMessageBox.warning(self, "No Input", "Select a control image first.")
            return
        try:
            from generation.image_gen import ImageGenerator
            gen = ImageGenerator()
            preprocessor = self.controlnet_preprocessor.currentText()
            result = gen._preprocess_controlnet_image(
                input_path, preprocessor, 512, 512)
            buffer = BytesIO()
            result.save(buffer, format="PNG")
            buffer.seek(0)
            qimg = QImage()
            qimg.loadFromData(buffer.read())
            pixmap = QPixmap.fromImage(qimg).scaled(
                160, 160, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.controlnet_preview_label.setPixmap(pixmap)
        except Exception as e:
            self.controlnet_preview_label.setText(f"Error: {e}")

    def _update_ip_adapter_preview(self, path):
        """Show thumbnail of IP-Adapter reference image."""
        if not path or not Path(path).exists():
            self.ip_adapter_preview.clear()
            return
        try:
            pixmap = QPixmap(path).scaled(
                100, 100, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.ip_adapter_preview.setPixmap(pixmap)
        except Exception:
            self.ip_adapter_preview.setText("Preview failed")

    def _upscale_image(self):
        input_path = self.upscale_input.path()
        if not input_path or not Path(input_path).exists():
            QMessageBox.warning(self, "No Input", "Select an input image first.")
            return

        self._set_status("Upscaling image...")
        QApplication.processEvents()

        try:
            from PIL import Image
            img = Image.open(input_path)
            choice = self.upscale_factor.currentText()
            factor = 2 if "2x" in choice else 4
            use_ai = "AI" in choice

            if use_ai:
                from generation.upscaler import upscale_image
                upscaled = upscale_image(img, scale=factor)
            else:
                new_size = (img.width * factor, img.height * factor)
                upscaled = img.resize(new_size, Image.LANCZOS)

            save_path, _ = QFileDialog.getSaveFileName(self, "Save Upscaled Image", "",
                "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)")
            if save_path:
                upscaled.save(save_path)
                self._set_status(f"Upscaled image saved: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Upscale Error", str(e))

    def _add_to_gen_queue(self):
        """Add current prompt + params as a job to the generation queue."""
        from core.generation_queue import GenerationJob, GenerationQueue
        if not hasattr(self, '_gen_queue'):
            self._gen_queue = GenerationQueue()
        prompt = self.gen_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Error", "Enter a prompt first.")
            return
        job = GenerationJob(prompt=prompt, params={
            "negative_prompt": self.gen_negative.toPlainText().strip(),
            "width": int(self.img_width.value()),
            "height": int(self.img_height.value()),
            "steps": int(self.img_steps.value()),
            "cfg_scale": self.img_cfg.value(),
            "seed": int(self.img_seed.value()),
            "sampler": self.img_sampler.currentText(),
        })
        self._gen_queue.add_job(job)
        status_icons = {"pending": "⏸", "running": "▶", "done": "✓", "failed": "✗", "cancelled": "⊘"}
        self.gen_queue_list.addItem(
            f"{status_icons.get(job.status.value, '●')} {job.job_id} — {prompt[:40]}")
        self.queue_progress_label.setText(
            f"{self._gen_queue.get_pending_count()} job(s) queued")

    def _clear_gen_queue(self):
        if hasattr(self, '_gen_queue'):
            self._gen_queue.clear()
        self.gen_queue_list.clear()
        self.queue_progress_label.setText("")

    def _toggle_queue_pause(self):
        if not hasattr(self, '_gen_queue'):
            return
        if self._gen_queue.is_paused:
            self._gen_queue.resume()
            self.pause_queue_btn.setText("Pause")
        else:
            self._gen_queue.pause()
            self.pause_queue_btn.setText("Resume")

    def _queue_context_menu(self, pos):
        menu = QMenu(self)
        cancel_action = menu.addAction("Cancel this job")
        action = menu.exec(self.gen_queue_list.mapToGlobal(pos))
        if action == cancel_action and hasattr(self, '_gen_queue'):
            row = self.gen_queue_list.currentRow()
            jobs = self._gen_queue.get_jobs()
            if 0 <= row < len(jobs):
                self._gen_queue.cancel_job(jobs[row].job_id)
                item = self.gen_queue_list.item(row)
                if item:
                    item.setText(f"⊘ {jobs[row].job_id} — {jobs[row].prompt[:40]}")

    def _generate(self):
        """Run image or video generation with current parameters.

        Flow:
        1. Read generation mode (txt2img / img2img / video)
        2. Validate prompt is not empty
        3. Collect all generation params (resolution, steps, CFG, seed, sampler)
        4. Gather multi-LoRA list with per-LoRA weights
        5. Create GenerationWorker with collected kwargs
        6. Connect step-progress callback for progress bar updates
        7. Start worker thread; disable generate button until complete
        8. Handle prompt queue advancement if queue mode is active
        """
        mode = self.gen_mode.currentText()
        logger.info("Generating: mode=%s", mode)
        prompt = self.gen_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Error", "Enter a prompt.")
            return
        if mode == "Image":
            self._generate_image(prompt)
        else:
            self._generate_video(prompt)

    def _generate_image(self, prompt):
        if not self.image_generator or not self.image_generator.pipe:
            QMessageBox.warning(self, "Error", "Load a model first.")
            return

        self.generate_btn.setEnabled(False)
        self.gen_progress.setVisible(True)
        self.gen_progress.setValue(0)
        self.gen_info.setText("Generating...")
        self._set_vram_poll_rate(True)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": self.gen_negative.toPlainText().strip(),
            "width": int(self.img_width.value()),
            "height": int(self.img_height.value()),
            "steps": int(self.img_steps.value()),
            "cfg_scale": self.img_cfg.value(),
            "seed": int(self.img_seed.value()),
            "batch_size": int(self.img_batch.value()),
            "sampler": self.img_sampler.currentText(),
            "clip_skip": int(self.img_clip_skip.value()),
            "hires_fix": self.hires_check.isChecked(),
            "hires_scale": self.hires_scale.value(),
            "hires_steps": int(self.hires_steps.value()),
            "hires_denoising": self.hires_denoise.value(),
        }

        # Advanced image params
        if hasattr(self, 'img_eta'):
            kwargs["eta"] = self.img_eta.value()
        if hasattr(self, 'img_tiling') and self.img_tiling.isChecked():
            kwargs["tiling"] = True
        if hasattr(self, 'img_karras') and self.img_karras.isChecked():
            kwargs["karras_sigmas"] = True
        if hasattr(self, 'img_rescale_cfg') and self.img_rescale_cfg.value() > 0:
            kwargs["guidance_rescale"] = self.img_rescale_cfg.value()
        if hasattr(self, 'img_aes_score'):
            kwargs["aesthetic_score"] = self.img_aes_score.value()
        if hasattr(self, 'img_neg_aes'):
            kwargs["negative_aesthetic_score"] = self.img_neg_aes.value()
        if hasattr(self, 'img_denoise_start') and self.img_denoise_start.value() > 0:
            kwargs["denoising_start"] = self.img_denoise_start.value()
        if hasattr(self, 'img_denoise_end') and self.img_denoise_end.value() < 1.0:
            kwargs["denoising_end"] = self.img_denoise_end.value()

        if hasattr(self, 'img2img_enabled') and self.img2img_enabled.isChecked():
            kwargs["init_image"] = self.img2img_input.path()
            kwargs["strength"] = self.img2img_strength.value()

        # ControlNet params
        if hasattr(self, 'controlnet_enabled') and self.controlnet_enabled.isChecked():
            kwargs["controlnet_enabled"] = True
            kwargs["controlnet_model_id"] = self.controlnet_model.currentText()
            kwargs["controlnet_preprocessor"] = self.controlnet_preprocessor.currentText()
            kwargs["controlnet_input_image"] = self.controlnet_input.path()
            kwargs["controlnet_strength"] = self.controlnet_strength.value()
            if hasattr(self, 'controlnet_guidance_start'):
                kwargs["controlnet_guidance_start"] = self.controlnet_guidance_start.value()
            if hasattr(self, 'controlnet_guidance_end'):
                kwargs["controlnet_guidance_end"] = self.controlnet_guidance_end.value()

        # IP-Adapter params
        if hasattr(self, 'ip_adapter_enabled') and self.ip_adapter_enabled.isChecked():
            kwargs["ip_adapter_enabled"] = True
            kwargs["ip_adapter_image"] = self.ip_adapter_image.path()
            kwargs["ip_adapter_scale"] = self.ip_adapter_scale.value()

        # Gather multi-LoRA data
        loras = []
        for i in range(self.lora_list.count()):
            data = self.lora_list.item(i).data(Qt.ItemDataRole.UserRole)
            loras.append(data)
        if loras:
            kwargs["loras"] = loras

        worker = GenerationWorker(self.image_generator, "image", **kwargs)
        worker.progress.connect(lambda s, t: (
            self.gen_progress.setMaximum(t),
            self.gen_progress.setValue(s),
        ))
        worker.finished.connect(self._on_gen_complete)
        worker.error.connect(self._on_gen_error)
        self._track_worker(worker)
        worker.start()

    def _generate_video(self, prompt):
        if not self.video_generator or not self.video_generator.pipe:
            QMessageBox.warning(self, "Error", "Load a video model first.")
            return

        # Route to long-video generation if clip_count > 1
        clip_count = int(self.vid_clip_count.value()) if hasattr(self, 'vid_clip_count') else 1
        if clip_count > 1:
            self._generate_long_video(prompt)
            return

        self.generate_btn.setEnabled(False)
        self.gen_progress.setVisible(True)
        self._set_vram_poll_rate(True)

        neg = self.vid_negative.toPlainText().strip() if hasattr(self, 'vid_negative') else ""
        kwargs = {
            "prompt": prompt,
            "negative_prompt": neg or self.gen_negative.toPlainText().strip(),
            "width": int(self.vid_width.value()),
            "height": int(self.vid_height.value()),
            "num_frames": int(self.vid_frames.value()),
            "fps": int(self.vid_fps.value()),
            "steps": int(self.vid_steps.value()),
            "cfg_scale": self.vid_cfg.value(),
            "seed": int(self.vid_seed.value()),
        }

        worker = GenerationWorker(self.video_generator, "video", **kwargs)
        worker.progress.connect(lambda s, t: (
            self.gen_progress.setMaximum(t),
            self.gen_progress.setValue(s),
        ))
        worker.finished.connect(self._on_video_gen_complete)
        worker.error.connect(self._on_gen_error)
        self._track_worker(worker)
        worker.start()

    def _generate_long_video(self, prompt):
        """Generate a long video by stitching multiple clips together."""
        self.generate_btn.setEnabled(False)
        self.gen_progress.setVisible(True)
        self.gen_progress.setRange(0, 0)
        self._set_vram_poll_rate(True)
        self.gen_info.setText("Generating long video (multi-clip stitching)...")

        neg = self.vid_negative.toPlainText().strip() if hasattr(self, 'vid_negative') else ""
        kwargs = {
            "prompt": prompt,
            "negative_prompt": neg or self.gen_negative.toPlainText().strip(),
            "width": int(self.vid_width.value()),
            "height": int(self.vid_height.value()),
            "frames_per_clip": int(self.vid_frames.value()),
            "clip_count": int(self.vid_clip_count.value()),
            "overlap_frames": int(self.vid_overlap.value()) if hasattr(self, 'vid_overlap') else 4,
            "fps": int(self.vid_fps.value()),
            "steps": int(self.vid_steps.value()),
            "cfg_scale": self.vid_cfg.value(),
            "seed": int(self.vid_seed.value()),
            "flow_shift": self.vid_flow_shift.value() if hasattr(self, 'vid_flow_shift') else 3.0,
            "output_dir": self.cfg.config.generation.output_dir,
        }

        class LongVideoWorker(QThread):
            finished = pyqtSignal(str)
            error = pyqtSignal(str)
            progress = pyqtSignal(int, int, str)

            def __init__(self, generator, **kw):
                super().__init__()
                self.generator = generator
                self.kw = kw

            def run(self):
                try:
                    self.kw["callback"] = lambda done, total, msg: self.progress.emit(done, total, msg)
                    path = self.generator.generate_long_video(**self.kw)
                    self.finished.emit(path)
                except Exception as e:
                    import traceback
                    from core.logger import get_logger
                    get_logger("worker").error(f"LongVideoWorker failed: {e}\n{traceback.format_exc()}")
                    self.error.emit(str(e))
                finally:
                    from core.gpu_utils import flush_gpu_memory
                    flush_gpu_memory()

        worker = LongVideoWorker(self.video_generator, **kwargs)
        worker.progress.connect(lambda done, total, msg: (
            self.gen_progress.setRange(0, max(total, 1)),
            self.gen_progress.setValue(done),
            self.gen_info.setText(msg),
        ))
        worker.finished.connect(self._on_long_video_complete)
        worker.error.connect(self._on_gen_error)
        self._track_worker(worker)
        worker.start()

    def _on_long_video_complete(self, filepath):
        self.generate_btn.setEnabled(True)
        self.gen_progress.setVisible(False)
        self._set_vram_poll_rate(False)
        self._last_video_path = filepath
        self.gen_info.setText(f"Long video saved: {filepath}")
        self.gen_output_image.setText(f"Long video saved to:\n{filepath}")
        self._notify("Long video generation complete!", "success")

    # ── Audio Generation ──────────────────────────────────────────────────

    def _load_audio_model(self):
        model_id = self.audio_model_combo.currentText()
        self.audio_status_label.setText(f"Loading {model_id}...")
        try:
            from generation.audio_gen import AudioGenerator
        except ImportError:
            self.audio_status_label.setText("audio_gen module not found")
            return
        if not hasattr(self, 'audio_generator'):
            self.audio_generator = AudioGenerator()
        try:
            self.audio_generator.load_model(
                model_id, on_progress=lambda m: self.audio_status_label.setText(m))
            self.audio_status_label.setText(f"Loaded: {model_id}")
            self._notify(f"Audio model loaded: {model_id}", "success")
        except Exception as e:
            self.audio_status_label.setText(f"Failed: {e}")
            self._notify(f"Audio model load failed: {e}", "error")

    def _generate_audio(self):
        if not hasattr(self, 'audio_generator') or self.audio_generator.pipe is None:
            self._notify("Load an audio model first", "warning")
            return
        prompt = self.audio_prompt.toPlainText().strip()
        if not prompt:
            self._notify("Enter an audio prompt first", "warning")
            return
        duration = self.audio_duration.value()
        guidance = self.audio_cfg.value()
        fmt = self.audio_format.currentText()
        output_dir = self.cfg.config.generation.output_dir
        self._set_status("Generating audio...", 0)
        try:
            audio, sr = self.audio_generator.generate(
                prompt, duration_seconds=duration, guidance_scale=guidance)
            path = self.audio_generator.save_audio(audio, sr, output_dir, format=fmt)
            self._set_status(f"Audio saved: {path}", -1)
            self._notify("Audio generation complete!", "success")

            # If sync to video is checked and we have a last video path, embed
            if (self.audio_sync_check.isChecked()
                    and hasattr(self, '_last_video_path')
                    and self._last_video_path):
                base, ext = os.path.splitext(self._last_video_path)
                out_path = f"{base}_with_audio.mp4"
                self.audio_generator.embed_audio_in_video(
                    audio, sr, self._last_video_path, out_path)
                self._notify(f"Audio embedded in video: {os.path.basename(out_path)}", "success")
        except Exception as e:
            self._set_status("Audio generation failed", -1)
            self._notify(f"Audio failed: {e}", "error")
            logger.error(f"Audio generation error: {e}", exc_info=True)

    def _on_gen_complete(self, images):
        """Handle completed image generation results.

        Flow:
        1. Re-enable the generate button and hide progress bar
        2. Store images in _current_images for display
        3. Show first image in the preview panel
        4. Auto-save images to output directory if enabled
        5. Append to _history_images for undo/redo support
        6. Update gallery tab if visible
        7. Advance prompt queue if in queue mode
        """
        self.generate_btn.setEnabled(True)
        self.gen_progress.setVisible(False)
        self._current_images = images

        if images:
            img = images[0]
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            qimg = QImage()
            qimg.loadFromData(buffer.read())
            pixmap = QPixmap.fromImage(qimg)

            scaled = pixmap.scaled(
                self.gen_output_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.gen_output_image.setPixmap(scaled)
            self.save_output_btn.setEnabled(True)
            self.gen_info.setText(f"Generated {len(images)} image(s)  |  {img.width}x{img.height}")

            # Store generation metadata for history display
            if not hasattr(self, '_generation_metadata'):
                self._generation_metadata = []
            meta = {"prompt": self.gen_prompt.toPlainText(),
                    "negative_prompt": self.gen_negative.toPlainText(),
                    "steps": int(self.img_steps.value()),
                    "cfg_scale": self.img_cfg.value(),
                    "width": int(self.img_width.value()),
                    "height": int(self.img_height.value()),
                    "seed": str(int(self.img_seed.value())),
                    "sampler": self.img_sampler.currentText()}
            for _ in images:
                self._generation_metadata.append(meta.copy())

            # Auto-save (async to avoid blocking GUI)
            if self.cfg.config.generation.auto_save:
                import threading
                threading.Thread(target=self.image_generator.save_images,
                    args=(images, self.cfg.config.generation.output_dir),
                    kwargs={"format": self.gen_save_format.currentText(),
                            "metadata": {"prompt": self.gen_prompt.toPlainText()}},
                    daemon=True).start()

            # History — cap at MAX_HISTORY
            MAX_HISTORY = 50
            if len(self._history_images) >= MAX_HISTORY:
                self._history_images = self._history_images[-MAX_HISTORY + len(images):]

            self._history_images.extend(images)

            for img in images:
                item = QListWidgetItem()
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                qi = QImage()
                qi.loadFromData(buf.read())
                icon = QIcon(QPixmap.fromImage(qi).scaled(
                    72, 72, Qt.AspectRatioMode.KeepAspectRatio))
                item.setIcon(icon)
                self.history_list.addItem(item)

        self._set_vram_poll_rate(False)
        self._notify_completion("Generation Complete")

        # Prompt queue chaining
        if hasattr(self, '_prompt_queue_list') and self._prompt_queue_index < len(self._prompt_queue_list):
            self._prompt_queue_index += 1
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, self._run_next_queue_prompt)

    def _on_video_gen_complete(self, results):
        self.generate_btn.setEnabled(True)
        self.gen_progress.setVisible(False)
        self._set_vram_poll_rate(False)

        if results and results[0]:
            frames = results[0]
            filepath = self.video_generator.save_video(
                frames, self.cfg.config.generation.output_dir,
                fps=int(self.vid_fps.value()),
            )
            self._last_video_path = filepath
            self.gen_info.setText(f"Video saved: {filepath}")

            # Show first frame as preview
            try:
                if hasattr(frames[0], 'save'):  # PIL Image
                    buffer = BytesIO()
                    frames[0].save(buffer, format="PNG")
                    buffer.seek(0)
                    qimg = QImage()
                    qimg.loadFromData(buffer.read())
                    pixmap = QPixmap.fromImage(qimg)
                    scaled = pixmap.scaled(self.gen_output_image.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                    self.gen_output_image.setPixmap(scaled)
            except Exception:
                self.gen_output_image.setText(f"Video saved to:\n{filepath}")

    def _on_gen_error(self, error):
        self.generate_btn.setEnabled(True)
        self.gen_progress.setVisible(False)
        self._set_vram_poll_rate(False)
        self.gen_info.setText(f"Error: {error}")
        logger.error("Generation error: %s", error)
        QMessageBox.critical(self, "Generation Error", error)

    def _save_generated(self):
        if not self._current_images:
            return
        fmt = self.gen_save_format.currentText()
        filters = {"png": "PNG (*.png)", "jpg": "JPEG (*.jpg)", "webp": "WebP (*.webp)"}
        default_filter = filters.get(fmt, "PNG (*.png)")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)",
            default_filter)
        if path:
            self._current_images[0].save(path, quality=95)
            self.gen_info.setText(f"Saved: {path}")

    def _show_history_item(self, row):
        if row < 0 or row >= len(self._history_images):
            return
        img = self._history_images[row]
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        qimg = QImage()
        qimg.loadFromData(buffer.read())
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.gen_output_image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.gen_output_image.setPixmap(scaled)
        self._current_images = [img]
        self.gen_info.setText(f"History [{row + 1}/{len(self._history_images)}]  |  {img.width}x{img.height}")
        self.save_output_btn.setEnabled(True)

        # Display metadata
        if hasattr(self, 'meta_display'):
            meta = getattr(img, 'info', {}) if hasattr(img, 'info') else {}
            if not meta and hasattr(self, '_generation_metadata'):
                idx = row if row < len(self._generation_metadata) else -1
                meta = self._generation_metadata[idx] if idx >= 0 else {}
            meta_text = "\n".join(f"{k}: {v}" for k, v in meta.items()) if meta else "No metadata available"
            self.meta_display.setPlainText(meta_text)

    # ═══════════════════════════════════════════════════════════════════════
    #  UTILITY
    # ═══════════════════════════════════════════════════════════════════════

    def _show_error(self, message: str):
        """Show an error message in a dialog and log it."""
        logger.error(message)
        QMessageBox.critical(self, "Error", message)

    def _refresh_log_viewer(self):
        from core.logger import get_log_file
        log_file = get_log_file()
        if log_file.exists():
            text = log_file.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            self.log_viewer.setPlainText("\n".join(lines[-200:]))
            scrollbar = self.log_viewer.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _open_log_folder(self):
        from core.logger import get_log_dir
        import subprocess as _sp
        import sys as _sys
        log_dir = str(get_log_dir())
        if _sys.platform == "win32":
            _sp.Popen(["explorer", log_dir])
        elif _sys.platform == "darwin":
            _sp.Popen(["open", log_dir])
        else:
            _sp.Popen(["xdg-open", log_dir])

    def _toggle_theme(self, theme: str):
        self.cfg.update_and_save("app", "theme", theme)
        from gui.theme import DARK_THEME, LIGHT_THEME
        stylesheet = LIGHT_THEME if theme == "light" else DARK_THEME
        QApplication.instance().setStyleSheet(stylesheet)

    def _set_compare(self, slot: str):
        if not self._current_images:
            return
        img = self._current_images[0]
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        qimg = QImage()
        qimg.loadFromData(buffer.read())
        pixmap = QPixmap.fromImage(qimg)
        target = self.compare_a if slot == "a" else self.compare_b
        scaled = pixmap.scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        target.setPixmap(scaled)

    def _run_prompt_queue(self):
        text = self.prompt_queue.toPlainText().strip()
        if not text:
            return
        prompts = [line.strip() for line in text.split("\n") if line.strip()]
        self._prompt_queue_list = prompts
        self._prompt_queue_index = 0
        self._run_next_queue_prompt()

    def _run_next_queue_prompt(self):
        if self._prompt_queue_index >= len(self._prompt_queue_list):
            self.queue_progress_label.setText(
                f"Queue complete: {len(self._prompt_queue_list)} prompts processed")
            self._notify_completion("Prompt Queue Complete")
            return
        prompt = self._prompt_queue_list[self._prompt_queue_index]
        self.queue_progress_label.setText(
            f"Queue: {self._prompt_queue_index + 1}/{len(self._prompt_queue_list)}")
        self.gen_prompt.setPlainText(prompt)
        self._generate()

    def _generate_training_sample(self, step: int):
        """Generate a sample image using the current training checkpoint."""
        prompt = self.sample_prompt.text().strip()
        if not prompt:
            return
        try:
            self.sample_preview.setText(f"Generating sample at step {step}...")
            QApplication.processEvents()
        except Exception:
            pass

    def _update_vram_budget(self):
        mode = self.cfg.config.hardware.offload_mode
        model_type = self.model_type.currentText() if hasattr(self, 'model_type') else "sdxl"
        base_estimates = {
            "sd15": {"none": 4000, "balanced": 2500, "aggressive": 1500, "cpu_only": 500},
            "sdxl": {"none": 7000, "balanced": 4500, "aggressive": 2500, "cpu_only": 800},
            "flux": {"none": 12000, "balanced": 6000, "aggressive": 3500, "cpu_only": 1000},
        }
        estimates = base_estimates.get(model_type, base_estimates["sdxl"])
        est_mb = estimates.get(mode, 4500)
        total_vram = self.cfg.config.hardware.vram_total_mb
        color = "#3fb950" if est_mb < total_vram * 0.85 else "#d29922" if est_mb < total_vram else "#f85149"
        self.vram_budget_label.setText(
            f"<span style='color:{color}; font-size:14px;'>~{est_mb} MB</span> estimated for "
            f"<b>{model_type}</b> with <b>{mode}</b> offloading<br>"
            f"<span style='color:#8b949e;'>Available: {total_vram} MB</span>")

    def _register_param_widgets(self):
        """Map widget names to references for undo/redo."""
        self._param_widgets = {
            "img_width": self.img_width, "img_height": self.img_height,
            "img_steps": self.img_steps, "img_cfg": self.img_cfg,
            "img_clip_skip": self.img_clip_skip,
            "img_sampler": self.img_sampler,
            "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "train_epochs": self.train_epochs,
            "batch_size": self.batch_size,
            "train_resolution": self.train_resolution,
        }

    def _record_param_change(self, name: str, old_value, new_value):
        self._undo_stack.append((name, old_value))
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo_param(self):
        if not self._undo_stack:
            return
        name, old_value = self._undo_stack.pop()
        widget = self._param_widgets.get(name)
        if widget:
            current = self._get_widget_value(widget)
            self._redo_stack.append((name, current))
            self._set_widget_value(widget, old_value)

    def _redo_param(self):
        if not self._redo_stack:
            return
        name, value = self._redo_stack.pop()
        widget = self._param_widgets.get(name)
        if widget:
            current = self._get_widget_value(widget)
            self._undo_stack.append((name, current))
            self._set_widget_value(widget, value)

    def _get_widget_value(self, widget):
        if isinstance(widget, LabeledSlider):
            return widget.value()
        elif isinstance(widget, LabeledCombo):
            return widget.currentText()
        elif isinstance(widget, LabeledCheck):
            return widget.isChecked()
        return None

    def _set_widget_value(self, widget, value):
        if isinstance(widget, LabeledSlider):
            widget.setValue(value)
        elif isinstance(widget, LabeledCombo):
            widget.setCurrentText(str(value))
        elif isinstance(widget, LabeledCheck):
            widget.setChecked(bool(value))

    def _apply_usecase_preset(self, vals: dict, name: str):
        """Apply a use-case training preset (Quick Test, Character, etc.)."""
        if "max_train_steps" in vals:
            self.train_steps.setValue(vals["max_train_steps"])
        if "learning_rate" in vals:
            self.learning_rate.setValue(vals["learning_rate"])
        if "batch_size" in vals:
            self.batch_size.setValue(vals["batch_size"])
        self._set_status(f"Applied preset: {name}")

    def _notify(self, message: str, level: str = "info"):
        """Show a temporary toast notification."""
        toast = ToastNotification(message, level, parent=self)
        x = self.width() - toast.width() - 20
        y = self.height() - toast.height() - 50
        toast.move(x, y)
        toast.show()
        toast.raise_()

    def _notify_completion(self, title: str):
        """Flash taskbar, beep, and show toast on task completion."""
        try:
            QApplication.instance().alert(self, 0)
            QApplication.beep()
        except Exception:
            pass
        self._notify(title, "success")

    def _refresh_hardware(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.cfg.config.hardware.gpu_name = torch.cuda.get_device_name(0)
                self.cfg.config.hardware.vram_total_mb = int(
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
                self.cfg.config.hardware.cuda_version = torch.version.cuda or "N/A"
            import psutil
            self.cfg.config.hardware.ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
            self.cfg.config.hardware.cpu_cores = psutil.cpu_count(logical=True)
            self.cfg.save()
            self._set_status("Hardware info refreshed")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not refresh hardware: {e}")

    def _track_worker(self, worker):
        """Add worker to list and auto-remove when finished."""
        self._workers.append(worker)
        worker.finished.connect(lambda *a: self._cleanup_worker(worker))

    def _cleanup_worker(self, worker):
        try:
            self._workers.remove(worker)
        except ValueError:
            pass

    def _connect_autosave(self):
        # Generation auto-save
        self.img_width.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_width", int(v)))
        self.img_height.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_height", int(v)))
        self.img_steps.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_steps", int(v)))
        self.img_cfg.valueChanged.connect(
            lambda v: self.cfg.update_and_save("generation", "img_cfg_scale", v))

        # Captioning auto-save
        self.verbose_description.toggled.connect(
            lambda v: self.cfg.update_and_save("captioning", "verbose_description", v))
        self.video_frame_interval.valueChanged.connect(
            lambda v: self.cfg.update_and_save("captioning", "video_frame_interval", v))
        self.video_max_frames.valueChanged.connect(
            lambda v: self.cfg.update_and_save("captioning", "video_max_frames", v))

        # Training auto-save
        self.model_type.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("training", "model_type", v))
        self.training_type.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("training", "training_type", v))
        self.lora_rank.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "lora_rank", int(v)))
        self.lora_alpha.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "lora_alpha", int(v)))
        self.learning_rate.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "learning_rate", v))
        self.train_steps.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "max_train_steps", int(v)))
        self.train_epochs.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "epochs", int(v)))
        self.batch_size.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "batch_size", int(v)))
        self.train_resolution.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "resolution", int(v)))
        self.num_repeats.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "num_repeats", int(v)))
        self.optimizer.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("training", "optimizer", v))
        self.lr_scheduler.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("training", "lr_scheduler", v))
        self.noise_offset.valueChanged.connect(
            lambda v: self.cfg.update_and_save("training", "noise_offset", v))
        self.flip_aug.toggled.connect(
            lambda v: self.cfg.update_and_save("training", "flip_aug", v))
        self.color_aug.toggled.connect(
            lambda v: self.cfg.update_and_save("training", "color_aug", v))
        self.caption_method.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("captioning", "method", v))
        self.caption_format.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("captioning", "caption_format", v))

        # Pipeline mode
        self.pipeline_mode.currentTextChanged.connect(
            self._on_pipeline_mode_changed)

    def _start_vram_monitor(self):
        self.vram_timer = QTimer()
        self.vram_timer.timeout.connect(self._update_vram)
        self.vram_timer.start(5000)

    def _set_vram_poll_rate(self, fast: bool):
        self.vram_timer.setInterval(1000 if fast else 5000)

    def _update_vram(self):
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                pct = (alloc / total * 100) if total > 0 else 0
                self.status_vram.setText(f"VRAM: {alloc:.0f}/{total:.0f} MB ({pct:.0f}%)")

                # GPU temp
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        temp = result.stdout.strip()
                        self.status_temp.setText(f"Temp: {temp}°C")
                except Exception:
                    pass

                if self.training_job and self.training_job.is_running:
                    stats = self.training_job.vram_monitor.get_stats()
                    self.train_vram_label.setValue(f"{stats['current_mb']:.0f} MB")
        except Exception:
            pass

    def _clear_vram(self):
        import gc
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            gc.collect()
        self._set_status("VRAM cache cleared")

    def closeEvent(self, event):
        geo = self.saveGeometry().toBase64().data().decode()
        self.cfg.config.window_geometry = geo
        self.cfg.save()

        if self.training_job and self.training_job.is_running:
            reply = QMessageBox.question(
                self, "Training in Progress",
                "Training is still running. Cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.training_job.cancel()

        event.accept()
