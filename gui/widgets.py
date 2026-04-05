"""
Reusable custom widgets for the GUI.
Responsive, polished, with tooltips, reset buttons, collapsible sections,
and proper minimum/maximum sizing for all screen sizes.
"""
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QGroupBox, QFrame, QPushButton, QFileDialog, QLineEdit,
    QSizePolicy, QToolButton, QScrollArea, QGridLayout,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, QSize,
)
from PyQt6.QtGui import QFont, QColor, QIcon


# ── Collapsible Section ─────────────────────────────────────────────────

class CollapsibleSection(QWidget):
    """A group box that collapses/expands with a toggle arrow."""

    def __init__(self, title: str, parent=None, start_collapsed: bool = False):
        super().__init__(parent)
        self._collapsed = start_collapsed

        # Toggle button styled as header
        self.toggle_btn = QToolButton()
        self.toggle_btn.setStyleSheet("""
            QToolButton {
                border: none;
                color: #e94560;
                font-weight: bold;
                font-size: 13px;
                padding: 6px 8px;
                text-align: left;
            }
            QToolButton:hover { color: #ff5a7a; }
        """)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(not start_collapsed)
        self.toggle_btn.clicked.connect(self._toggle)
        self._update_arrow()

        # Header line
        self.header_line = QFrame()
        self.header_line.setFrameShape(QFrame.Shape.HLine)
        self.header_line.setStyleSheet("background-color: #0f3460; max-height: 1px;")

        # Content container
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(6)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(self.toggle_btn)
        header_row.addWidget(self.header_line, 1)
        main_layout.addLayout(header_row)
        main_layout.addWidget(self.content_widget)

        if start_collapsed:
            self.content_widget.setVisible(False)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self.content_widget.setVisible(not self._collapsed)
        self._update_arrow()

    def _update_arrow(self):
        if self._collapsed:
            self.toggle_btn.setText("▶  " + self.toggle_btn.text().lstrip("▶▼ "))
        else:
            self.toggle_btn.setText("▼  " + self.toggle_btn.text().lstrip("▶▼ "))

    def setCollapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self.content_widget.setVisible(not collapsed)
        self._update_arrow()


# ── Labeled Slider (improved) ────────────────────────────────────────────

class LabeledSlider(QWidget):
    """Slider with label, numeric display, optional tooltip and reset button."""
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, step: float = 1.0, decimals: int = 0,
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self.step = step
        self._default = default
        self._scale = 10 ** decimals

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        # Top row: label + value
        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        self.label = QLabel(label)
        self.label.setStyleSheet("font-size: 12px; color: #8892b0;")
        top_row.addWidget(self.label)

        top_row.addStretch()

        if decimals > 0:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(decimals)
            self.spinbox.setMinimum(min_val)
            self.spinbox.setMaximum(max_val)
            self.spinbox.setSingleStep(step)
            self.spinbox.setValue(default)
        else:
            self.spinbox = QSpinBox()
            self.spinbox.setMinimum(int(min_val))
            self.spinbox.setMaximum(int(max_val))
            self.spinbox.setSingleStep(max(1, int(step)))
            self.spinbox.setValue(int(default))

        self.spinbox.setFixedWidth(90)
        self.spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        top_row.addWidget(self.spinbox)

        # Reset button
        self.reset_btn = QToolButton()
        self.reset_btn.setText("↺")
        self.reset_btn.setToolTip(f"Reset to {default}")
        self.reset_btn.setFixedSize(22, 22)
        self.reset_btn.setStyleSheet("""
            QToolButton {
                border: none; color: #555; font-size: 14px;
                background: transparent; border-radius: 4px;
            }
            QToolButton:hover { color: #e94560; background: #1a1a2e; }
        """)
        self.reset_btn.clicked.connect(self._reset)
        top_row.addWidget(self.reset_btn)

        layout.addLayout(top_row)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self._scale))
        self.slider.setMaximum(int(max_val * self._scale))
        self.slider.setValue(int(default * self._scale))
        self.slider.setSingleStep(max(1, int(step * self._scale)))
        self.slider.setMinimumHeight(20)
        layout.addWidget(self.slider)

        # Tooltip
        if tooltip:
            self.setToolTip(tooltip)

        # Sync
        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def _slider_changed(self, value):
        real_value = value / self._scale
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(real_value if self.decimals > 0 else int(real_value))
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(real_value)

    def _spinbox_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int(float(value) * self._scale))
        self.slider.blockSignals(False)
        self.valueChanged.emit(float(value))

    def _reset(self):
        self.spinbox.setValue(self._default if self.decimals > 0 else int(self._default))

    def value(self):
        if self.decimals > 0:
            return self.spinbox.value()
        return int(self.spinbox.value())

    def setValue(self, v):
        self.spinbox.setValue(v)


# ── Labeled Combo (improved) ─────────────────────────────────────────────

class LabeledCombo(QWidget):
    """Combo box with label, now stacked vertically for narrower panels."""
    currentTextChanged = pyqtSignal(str)

    def __init__(self, label: str, items: list, default: str = "",
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        self.label = QLabel(label)
        self.label.setStyleSheet("font-size: 12px; color: #8892b0;")
        layout.addWidget(self.label)

        self.combo = QComboBox()
        self.combo.addItems(items)
        if default and default in items:
            self.combo.setCurrentText(default)
        self.combo.setMinimumHeight(32)
        layout.addWidget(self.combo)

        if tooltip:
            self.setToolTip(tooltip)

        self.combo.currentTextChanged.connect(self.currentTextChanged.emit)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def currentText(self):
        return self.combo.currentText()

    def setCurrentText(self, text):
        self.combo.setCurrentText(text)


# ── Labeled Check (improved) ────────────────────────────────────────────

class LabeledCheck(QWidget):
    """Checkbox with optional description text."""
    toggled = pyqtSignal(bool)

    def __init__(self, label: str, description: str = "", default: bool = False,
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.checkbox = QCheckBox(label)
        self.checkbox.setChecked(default)
        layout.addWidget(self.checkbox)

        if description:
            desc = QLabel(description)
            desc.setStyleSheet("font-size: 11px; color: #556a8b;")
            layout.addWidget(desc)

        layout.addStretch()
        self.checkbox.toggled.connect(self.toggled.emit)

        if tooltip:
            self.setToolTip(tooltip)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def isChecked(self):
        return self.checkbox.isChecked()

    def setChecked(self, v):
        self.checkbox.setChecked(v)


# ── Path Selector (improved) ────────────────────────────────────────────

class PathSelector(QWidget):
    """File/folder path selector, now stacked for narrow layouts."""
    pathChanged = pyqtSignal(str)

    def __init__(self, label: str, mode: str = "dir", filter: str = "",
                 parent=None):
        super().__init__(parent)
        self.mode = mode
        self.filter = filter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        self.label = QLabel(label)
        self.label.setStyleSheet("font-size: 12px; color: #8892b0;")
        layout.addWidget(self.label)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select path...")
        self.path_edit.setMinimumHeight(32)
        row.addWidget(self.path_edit, 1)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.setMinimumHeight(32)
        self.browse_btn.clicked.connect(self._browse)
        row.addWidget(self.browse_btn)
        layout.addLayout(row)

        self.path_edit.textChanged.connect(self.pathChanged.emit)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def _browse(self):
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        elif self.mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.filter)
        elif self.mode == "files":
            paths, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", self.filter)
            path = ";".join(paths) if paths else ""
        else:
            path = ""
        if path:
            self.path_edit.setText(path)

    def path(self):
        return self.path_edit.text()

    def setPath(self, p):
        self.path_edit.setText(p)


# ── Status Card (improved) ──────────────────────────────────────────────

class StatusCard(QFrame):
    """Compact status display card with subtle glow."""

    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setObjectName("StatusCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(1)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 10px; color: #556a8b; font-weight: bold; background: transparent;")
        layout.addWidget(self.title_label)

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 15px; color: #64ffda; font-weight: bold; background: transparent;")
        layout.addWidget(self.value_label)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(48)

    def setValue(self, v: str):
        self.value_label.setText(v)


# ── Separator ────────────────────────────────────────────────────────────

class Separator(QFrame):
    """Horizontal line separator."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: #0f3460;")


# ── Image Thumbnail Grid ────────────────────────────────────────────────

class ThumbnailLabel(QLabel):
    """Clickable image thumbnail."""
    clicked = pyqtSignal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self._index = index
        self.setFixedSize(80, 80)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid transparent;
                border-radius: 6px;
                background-color: #0a1628;
            }
            QLabel:hover {
                border-color: #533483;
            }
        """)
        self.setScaledContents(True)

    def mousePressEvent(self, event):
        self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def setSelected(self, selected: bool):
        if selected:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #e94560;
                    border-radius: 6px;
                    background-color: #0a1628;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid transparent;
                    border-radius: 6px;
                    background-color: #0a1628;
                }
                QLabel:hover { border-color: #533483; }
            """)


# ── Scroll Panel helper ─────────────────────────────────────────────────

def make_scroll_panel(widget: QWidget) -> QScrollArea:
    """Wrap a widget in a scroll area with proper theming."""
    scroll = QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setStyleSheet("""
        QScrollArea {
            border: none;
            background: transparent;
        }
        QScrollArea > QWidget > QWidget {
            background: transparent;
        }
    """)
    return scroll
