"""
First-run setup wizard — 3-page QWizard.
Shown on first launch when no config file exists.
"""
from pathlib import Path

from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QLabel, QCheckBox,
    QHBoxLayout, QWidget,
)
from PyQt6.QtCore import Qt

from gui.widgets import PathSelector
from core.model_downloader import MANAGED_MODELS


class WelcomePage(QWizardPage):
    """Page 1 — Welcome."""

    def __init__(self):
        super().__init__()
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
        layout.addStretch()


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

        self._welcome = WelcomePage()
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
