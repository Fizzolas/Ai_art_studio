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
import logging
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
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QMimeData
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QFont, QAction, QKeySequence,
    QShortcut, QDragEnterEvent, QDropEvent,
)

from gui.theme import DARK_THEME
from gui.widgets import (
    LabeledSlider, LabeledCombo, LabeledCheck, PathSelector,
    StatusCard, Separator, CollapsibleSection, make_scroll_panel,
)
from core.config import ConfigManager

logger = logging.getLogger(__name__)


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
                result = self.dm.scan_directory()
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
            self.error.emit(str(e))
        finally:
            # Prevent QThread + GPU memory leak: flush CUDA cache after
            # every generation pass regardless of success/failure.
            from core.gpu_utils import flush_gpu_memory
            flush_gpu_memory()


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
        self._workers = []  # prevent GC of workers

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

        self.tabs.addTab(self._build_dataset_tab(), "  Dataset  ")
        self.tabs.addTab(self._build_training_tab(), "  Training  ")
        self.tabs.addTab(self._build_generation_tab(), "  Generation  ")
        self.tabs.addTab(self._build_settings_tab(), "  Settings  ")

        main_layout.addWidget(self.tabs, 1)

        # Status bar
        self._build_status_bar()

        # Menu bar
        self._build_menu()

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.showMessage("Ready")

        self.vram_label = QLabel("VRAM: —")
        self.vram_label.setObjectName("stat")
        sb.addPermanentWidget(self.vram_label)

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

    def _setup_shortcuts(self):
        # Ctrl+G = generate
        gen_shortcut = QShortcut(QKeySequence("Ctrl+G"), self)
        gen_shortcut.activated.connect(self._generate)

        # Ctrl+T = start training
        train_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        train_shortcut.activated.connect(self._start_training)

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
        left_layout.addWidget(prompt_section)

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

        # LoRA sub-section
        load_section.addWidget(Separator())
        lora_label = QLabel("LoRA")
        lora_label.setObjectName("subheader")
        load_section.addWidget(lora_label)

        self.gen_lora_path = PathSelector("LoRA Path", mode="file",
            filter="LoRA files (*.safetensors *.pt);;All files (*)")
        load_section.addWidget(self.gen_lora_path)
        self.lora_weight = LabeledSlider("LoRA Weight", 0, 2.0, 0.8, 0.05, 2,
            tooltip="0.6-1.0 is typical. Higher = stronger effect.")
        load_section.addWidget(self.lora_weight)

        lora_btn_row = QHBoxLayout()
        self.load_lora_btn = QPushButton("Load LoRA")
        self.load_lora_btn.clicked.connect(self._load_lora)
        lora_btn_row.addWidget(self.load_lora_btn)
        self.unload_lora_btn = QPushButton("Unload LoRA")
        self.unload_lora_btn.clicked.connect(self._unload_lora)
        lora_btn_row.addWidget(self.unload_lora_btn)
        load_section.addLayout(lora_btn_row)
        left_layout.addWidget(load_section)

        # Prompt
        prompt_section = CollapsibleSection("Prompt")
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
        self.img_steps = LabeledSlider("Steps", 1, 150, 30, 1,
            tooltip="More steps = better quality but slower. 20-40 is typical.")
        self.img_params_section.addWidget(self.img_steps)
        self.img_cfg = LabeledSlider("CFG Scale", 1, 30, 7.5, 0.5, 1,
            tooltip="How closely to follow the prompt. 5-12 is typical.")
        self.img_params_section.addWidget(self.img_cfg)
        self.img_seed = LabeledSlider("Seed (-1 = random)", -1, 999999999, -1, 1)
        self.img_params_section.addWidget(self.img_seed)
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
        self.vid_params_section.setVisible(False)
        left_layout.addWidget(self.vid_params_section)

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
        self.history_list.setMaximumHeight(90)
        self.history_list.setFlow(QListWidget.Flow.LeftToRight)
        self.history_list.setWrapping(False)
        self.history_list.setIconSize(QSize(72, 72))
        self.history_list.setSpacing(4)
        self.history_list.currentRowChanged.connect(self._show_history_item)
        history_layout.addWidget(self.history_list)
        right_layout.addWidget(history_group)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        return tab

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
        layout.addWidget(hw_section)

        # Offloading
        off_section = CollapsibleSection("VRAM Offloading")
        self.offload_mode = LabeledCombo("Offload Mode",
            ["none", "balanced", "aggressive", "cpu_only"], hw.offload_mode,
            tooltip="balanced = recommended for 8GB. aggressive = slower but fits more.")
        self.offload_mode.currentTextChanged.connect(
            lambda v: self.cfg.update_and_save("hardware", "offload_mode", v))
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

        layout.addStretch()

        scroll = make_scroll_panel(inner)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

        return tab

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
        self.cfg.update_and_save("app", "pipeline_mode", mode)
        self._update_pipeline_mode_info(mode)

    def _on_dataset_path_changed(self, path):
        self.cfg.config.last_dataset_dir = path
        self.cfg.save()

    def _open_dataset_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self.dataset_path.setPath(path)

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
        self._workers.append(worker)
        worker.start()

    def _on_scan_complete(self, items):
        self.file_list.clear()
        for item in items:
            name = os.path.basename(item.original_path)
            icon = "IMG" if item.media_type == "image" else ("VID" if item.media_type == "video" else "ANI")
            cap = "  [captioned]" if item.caption_text else ""
            self.file_list.addItem(f"[{icon}]  {name}{cap}")

        s = self.dataset_manager.stats
        self.stat_total.setValue(str(s.total_files))
        self.stat_images.setValue(str(s.valid_images + s.animated_images))
        self.stat_videos.setValue(str(s.valid_videos))
        self.stat_captioned.setValue(str(s.captioned_files))
        self.stat_size.setValue(f"{s.total_size_mb:.1f} MB")
        self.stat_avgres.setValue(f"{s.avg_resolution[0]}x{s.avg_resolution[1]}")
        self.dataset_status.setText(f"Found {s.total_files} files")

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

    def _convert_dataset(self):
        if not self.dataset_manager or not self.dataset_manager.items:
            QMessageBox.warning(self, "Error", "No dataset loaded. Scan a directory first.")
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
        self._workers.append(worker)
        worker.start()

    def _run_captioning(self):
        if not self.dataset_manager or not self.dataset_manager.items:
            QMessageBox.warning(self, "Error", "No dataset loaded.")
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

        self.dataset_progress.setVisible(True)
        self.dataset_progress.setValue(0)
        self.caption_btn.setEnabled(False)
        self.dataset_status.setText("Loading captioning models...")

        # Captions go into the work_dir so originals are never touched
        cap_dir = str(self.dataset_manager.work_dir / "captions")
        worker = CaptionWorker(self.caption_pipeline, items, caption_dir=cap_dir)

        # Relay pipeline status messages (dependency checks, downloads,
        # model loading progress) to the dataset status label.
        worker.log.connect(lambda msg: self.dataset_status.setText(msg))

        worker.progress.connect(lambda c, t, f: (
            self.dataset_progress.setMaximum(t),
            self.dataset_progress.setValue(c),
            self.dataset_status.setText(f"Captioning: {os.path.basename(f)}"),
        ))
        worker.finished.connect(lambda results: (
            self.dataset_progress.setVisible(False),
            self.caption_btn.setEnabled(True),
            self.dataset_status.setText(f"Captioned {len(results)} files"),
            self._on_scan_complete(self.dataset_manager.items),
            self.caption_pipeline.unload_models() if self.caption_pipeline else None,
        ))
        worker.error.connect(lambda e: (
            QMessageBox.critical(self, "Error", f"Captioning error: {e}"),
            self.caption_btn.setEnabled(True),
            self.dataset_progress.setVisible(False),
        ))
        self._workers.append(worker)
        worker.start()

    def _preview_file(self, row):
        if not self.dataset_manager or row < 0 or row >= len(self.dataset_manager.items):
            return

        item = self.dataset_manager.items[row]

        try:
            path = item.converted_path if item.converted_path and os.path.isfile(item.converted_path) else item.original_path
            if os.path.isfile(path):
                pixmap = QPixmap(path)
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

    # ═══════════════════════════════════════════════════════════════════════
    #  ACTIONS — Training
    # ═══════════════════════════════════════════════════════════════════════

    def _start_training(self):
        if not self.dataset_manager:
            QMessageBox.warning(self, "Error", "No dataset loaded. Go to the Dataset tab first.")
            return

        tc = self.cfg.config.training
        tc.model_type = self.model_type.currentText()

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
        self.train_progress.setValue(0)
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)

        worker = TrainingWorker(self.training_job)
        worker.progress.connect(self._on_train_progress)
        worker.log.connect(self._on_train_log)
        worker.finished.connect(self._on_train_complete)
        worker.error.connect(self._on_train_error)
        self._workers.append(worker)
        worker.start()

    def _on_train_progress(self, step, total, loss, lr):
        self.train_progress.setMaximum(total)
        self.train_progress.setValue(step)
        self.train_step_label.setValue(f"{step} / {total}")
        self.train_loss_label.setValue(f"{loss:.4f}")
        self.train_lr_label.setValue(f"{lr:.2e}")
        if self.training_job:
            self.train_eta_label.setValue(self.training_job.get_eta())

    def _on_train_log(self, message):
        self.train_log.appendPlainText(message)

    def _on_train_complete(self, output_path):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.train_log.appendPlainText(
            f"\n{'=' * 50}\nTraining Complete!\nModel: {output_path}\n{'=' * 50}")
        QMessageBox.information(self, "Training Complete", f"LoRA saved to:\n{output_path}")

        if output_path not in self.cfg.config.recent_models:
            self.cfg.config.recent_models.insert(0, output_path)
            self.cfg.config.recent_models = self.cfg.config.recent_models[:10]
            self.cfg.save()

    def _on_train_error(self, error):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.train_log.appendPlainText(f"\nERROR: {error}")
        QMessageBox.critical(self, "Training Error", error)

    def _stop_training(self):
        if self.training_job:
            self.training_job.cancel()
            self.train_log.appendPlainText("Cancelling training...")

    # ═══════════════════════════════════════════════════════════════════════
    #  ACTIONS — Generation
    # ═══════════════════════════════════════════════════════════════════════

    def _on_gen_mode_changed(self, mode):
        is_image = mode == "Image"
        self.img_params_section.setVisible(is_image)
        self.vid_params_section.setVisible(not is_image)

    def _load_gen_model(self):
        model_path = self.gen_model_path.path()
        model_type = self.gen_model_type.currentText()

        # Auto-resolve: leave empty to use the default for this model type
        if not model_path:
            from core.model_downloader import DEFAULT_MODELS
            model_path = DEFAULT_MODELS.get(model_type, DEFAULT_MODELS["sdxl"])

        self.gen_info.setText("Checking model availability...")
        self.load_model_btn.setEnabled(False)
        QApplication.processEvents()

        def _status(msg):
            # Update the info label from whatever thread calls us
            self.gen_info.setText(msg)
            QApplication.processEvents()

        try:
            if model_type in ("wan21", "animatediff"):
                from generation.video_gen import VideoGenerator
                self.video_generator = VideoGenerator(self.cfg.config.hardware)
                self.video_generator.load_model(model_path, model_type,
                                                on_progress=_status)
            else:
                from generation.image_gen import ImageGenerator
                self.image_generator = ImageGenerator(self.cfg.config.hardware)
                self.image_generator.load_model(model_path, model_type,
                                                on_progress=_status)

            self.gen_info.setText(f"Model loaded: {model_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            self.gen_info.setText(f"Error: {e}")
        finally:
            self.load_model_btn.setEnabled(True)

    def _load_lora(self):
        if not self.image_generator:
            QMessageBox.warning(self, "Error", "Load a base model first.")
            return
        lora_path = self.gen_lora_path.path()
        if not lora_path:
            QMessageBox.warning(self, "Error", "Select a LoRA file.")
            return
        try:
            weight = self.lora_weight.value()
            self.image_generator.load_lora(lora_path, weight)
            self.gen_info.setText(f"LoRA loaded: {os.path.basename(lora_path)} (weight={weight})")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load LoRA:\n{e}")

    def _unload_lora(self):
        if self.image_generator:
            self.image_generator.unload_lora()
            self.gen_info.setText("LoRA unloaded")

    def _generate(self):
        mode = self.gen_mode.currentText()
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

        worker = GenerationWorker(self.image_generator, "image", **kwargs)
        worker.progress.connect(lambda s, t: (
            self.gen_progress.setMaximum(t),
            self.gen_progress.setValue(s),
        ))
        worker.finished.connect(self._on_gen_complete)
        worker.error.connect(self._on_gen_error)
        self._workers.append(worker)
        worker.start()

    def _generate_video(self, prompt):
        if not self.video_generator or not self.video_generator.pipe:
            QMessageBox.warning(self, "Error", "Load a video model first.")
            return

        self.generate_btn.setEnabled(False)
        self.gen_progress.setVisible(True)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": self.gen_negative.toPlainText().strip(),
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
        self._workers.append(worker)
        worker.start()

    def _on_gen_complete(self, images):
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

            # Auto-save
            if self.cfg.config.generation.auto_save:
                self.image_generator.save_images(
                    images, self.cfg.config.generation.output_dir,
                    format=self.gen_save_format.currentText(),
                    metadata={"prompt": self.gen_prompt.toPlainText()},
                )

            # History
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

    def _on_video_gen_complete(self, results):
        self.generate_btn.setEnabled(True)
        self.gen_progress.setVisible(False)

        if results and results[0]:
            frames = results[0]
            filepath = self.video_generator.save_video(
                frames, self.cfg.config.generation.output_dir,
                fps=int(self.vid_fps.value()),
            )
            self.gen_info.setText(f"Video saved: {filepath}")
            self.gen_output_image.setText(f"Video saved to:\n{filepath}")

    def _on_gen_error(self, error):
        self.generate_btn.setEnabled(True)
        self.gen_progress.setVisible(False)
        self.gen_info.setText(f"Error: {error}")
        QMessageBox.critical(self, "Generation Error", error)

    def _save_generated(self):
        if not self._current_images:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)")
        if path:
            self._current_images[0].save(path, quality=95)
            self.gen_info.setText(f"Saved: {path}")

    def _show_history_item(self, row):
        pass  # History is icon-based; selecting shows via icon

    # ═══════════════════════════════════════════════════════════════════════
    #  UTILITY
    # ═══════════════════════════════════════════════════════════════════════

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

        # Pipeline mode
        self.pipeline_mode.currentTextChanged.connect(
            self._on_pipeline_mode_changed)

    def _start_vram_monitor(self):
        self.vram_timer = QTimer()
        self.vram_timer.timeout.connect(self._update_vram)
        self.vram_timer.start(2000)

    def _update_vram(self):
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                pct = (alloc / total * 100) if total > 0 else 0
                self.vram_label.setText(f"VRAM: {alloc:.0f} / {total:.0f} MB ({pct:.0f}%)")

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
        self.statusBar().showMessage("VRAM cache cleared", 3000)

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
