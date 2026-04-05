"""
First-run setup wizard — 3-page QWizard.
Shown on first launch when no config file exists.
"""
from pathlib import Path

from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QLabel, QCheckBox,
    QHBoxLayout, QWidget, QPushButton, QComboBox, QFrame, QGroupBox,
)
from PyQt6.QtCore import Qt

from gui.widgets import PathSelector
from core.model_downloader import MANAGED_MODELS
from core.gpu_utils import detect_hardware_profile
from core.config import ConfigManager


class WelcomePage(QWizardPage):
    """Page 1 — Welcome with hardware detection."""

    OFFLOAD_MODES = ["none", "balanced", "aggressive", "cpu_only"]
    OFFLOAD_LABELS = {
        "none": "None (full VRAM)",
        "balanced": "Balanced",
        "aggressive": "Aggressive",
        "cpu_only": "CPU Only",
    }

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.setTitle("Welcome to AI Art Studio")
        layout = QVBoxLayout(self)

        desc = QLabel(
            "AI Art Studio is a local-first tool for training LoRA models, "
            "generating images and videos, and managing datasets — all from "
            "a single desktop application.\n\n"
            "This wizard will help you configure initial settings.\n\n"
            "Click Next to get started."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 13px; padding: 16px 0;")
        layout.addWidget(desc)

        # --- Hardware profile card ---
        hw_profile = detect_hardware_profile()

        hw_group = QGroupBox("Detected Hardware")
        hw_layout = QVBoxLayout(hw_group)

        gpu_name = hw_profile.get("gpu_name", "None")
        vram_mb = hw_profile.get("vram_total_mb", 0)
        vram_display = f"{vram_mb} MB" if vram_mb > 0 else "N/A"
        recommended = hw_profile.get("recommended_offload", "cpu_only")

        gpu_label = QLabel(f"<b>GPU:</b> {gpu_name}")
        vram_label = QLabel(f"<b>VRAM:</b> {vram_display}")
        rec_label = QLabel(
            f"<b>Recommended offload mode:</b> "
            f"{self.OFFLOAD_LABELS.get(recommended, recommended)}"
        )
        for lbl in (gpu_label, vram_label, rec_label):
            lbl.setStyleSheet("font-size: 12px; padding: 2px 0;")
            hw_layout.addWidget(lbl)

        layout.addWidget(hw_group)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        self._btn_recommended = QPushButton("Use Recommended")
        self._btn_manual = QPushButton("Choose Manually")
        btn_row.addWidget(self._btn_recommended)
        btn_row.addWidget(self._btn_manual)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # --- Manual combo (hidden by default) ---
        self._manual_widget = QWidget()
        manual_layout = QHBoxLayout(self._manual_widget)
        manual_layout.setContentsMargins(0, 4, 0, 0)
        manual_layout.addWidget(QLabel("Offload mode:"))
        self._offload_combo = QComboBox()
        for mode in self.OFFLOAD_MODES:
            self._offload_combo.addItem(self.OFFLOAD_LABELS[mode], mode)
        # Pre-select the recommended mode in the combo
        rec_index = self.OFFLOAD_MODES.index(recommended) if recommended in self.OFFLOAD_MODES else 1
        self._offload_combo.setCurrentIndex(rec_index)
        manual_layout.addWidget(self._offload_combo)
        manual_layout.addStretch()
        self._manual_widget.setVisible(False)
        layout.addWidget(self._manual_widget)

        # Store recommended value for the button handler
        self._recommended_offload = recommended

        # Connect signals
        self._btn_recommended.clicked.connect(self._apply_recommended)
        self._btn_manual.clicked.connect(self._show_manual)
        self._offload_combo.currentIndexChanged.connect(self._on_combo_changed)

        layout.addStretch()

    # --- Slots ---

    def _apply_recommended(self):
        """Set offload mode to the recommended value and save."""
        if self.cfg is not None:
            self.cfg.config.hardware.offload_mode = self._recommended_offload
            self.cfg.save()
        self._manual_widget.setVisible(False)

    def _show_manual(self):
        """Reveal the manual offload-mode combo box."""
        self._manual_widget.setVisible(True)

    def _on_combo_changed(self, index):
        """Apply the manually-chosen offload mode and save."""
        mode = self._offload_combo.currentData()
        if mode and self.cfg is not None:
            self.cfg.config.hardware.offload_mode = mode
            self.cfg.save()


class PathsPage(QWizardPage):
    """Page 2 — Output and cache directory selection."""

    def __init__(self):
        super().__init__()
        self.setTitle("Configure Paths")
        self.setSubTitle("Choose where to store outputs and model files.")
        layout = QVBoxLayout(self)

        self.output_path = PathSelector(
            "Output Directory", mode="directory",
            default=str(Path.home() / "ai_art_studio_output"),
        )
        layout.addWidget(self.output_path)

        self.cache_path = PathSelector(
            "Models Cache Directory", mode="directory",
            default=str(Path.home() / ".cache" / "huggingface"),
        )
        layout.addWidget(self.cache_path)

        note = QLabel(
            "Output directory is where generated images and videos are saved.\n"
            "Models cache is where HuggingFace model weights are stored."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        layout.addStretch()


class ModelSelectionPage(QWizardPage):
    """Page 3 — Select models to pre-download."""

    def __init__(self):
        super().__init__()
        self.setTitle("Download Models")
        self.setSubTitle("Select models to download now (you can also do this later from Settings).")
        layout = QVBoxLayout(self)

        self._checkboxes = {}
        for key, info in MANAGED_MODELS.items():
            cb = QCheckBox(f"{info['label']}  (~{info['size_gb']} GB)  [{info['type']}]")
            cb.setChecked(False)
            layout.addWidget(cb)
            self._checkboxes[key] = cb

        note = QLabel(
            "\nUnchecked models can be downloaded later from "
            "Settings → Model Manager."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        layout.addStretch()

    def selected_models(self) -> list:
        """Return list of model keys the user wants to download."""
        return [key for key, cb in self._checkboxes.items() if cb.isChecked()]


class SetupWizard(QWizard):
    """First-run wizard with 3 pages: Welcome, Paths, Model Selection."""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("AI Art Studio — Setup")
        self.setMinimumSize(600, 450)

        self._welcome = WelcomePage(cfg=cfg)
        self._paths = PathsPage()
        self._models = ModelSelectionPage()

        self.addPage(self._welcome)
        self.addPage(self._paths)
        self.addPage(self._models)

        self.finished.connect(self._on_finish)

    def _on_finish(self, result):
        """Apply wizard settings to config and start downloads if selected."""
        if result != QWizard.DialogCode.Accepted:
            return

        # Apply paths
        output_dir = self._paths.output_path.path()
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.cfg.update_and_save("generation", "output_dir", output_dir)

        cache_dir = self._paths.cache_path.path()
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            import os
            os.environ["HF_HOME"] = cache_dir

        # Save config to create the file (marks first-run as done)
        self.cfg.save()

        # Start model downloads if any selected
        selected = self._models.selected_models()
        if selected:
            self._start_downloads(selected)

    def _start_downloads(self, model_keys: list):
        """Spawn download workers for selected models."""
        from core.model_downloader import MANAGED_MODELS
        from PyQt6.QtCore import QThread, pyqtSignal

        class _DownloadWorker(QThread):
            status = pyqtSignal(str)
            done = pyqtSignal()

            def __init__(self, repos):
                super().__init__()
                self.repos = repos

            def run(self):
                for repo in self.repos:
                    try:
                        self.status.emit(f"Downloading {repo}...")
                        from huggingface_hub import snapshot_download
                        snapshot_download(repo_id=repo, repo_type="model")
                        self.status.emit(f"Downloaded {repo}")
                    except Exception as e:
                        self.status.emit(f"Failed {repo}: {e}")
                self.done.emit()

        repos = [MANAGED_MODELS[k]["repo"] for k in model_keys if k in MANAGED_MODELS]
        if repos:
            self._dl_worker = _DownloadWorker(repos)
            self._dl_worker.start()
