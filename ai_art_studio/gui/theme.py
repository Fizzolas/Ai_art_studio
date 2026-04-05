"""
Dark theme stylesheet for the application.
Professional creative-tool aesthetic. DPI-aware, smooth,
with proper contrast ratios and hover/focus feedback.
"""

DARK_THEME = """
/* ─────────────────── Base ─────────────────── */
QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
}

QMainWindow {
    background-color: #0d1117;
}

/* ─────────────────── Menu Bar ─────────────────── */
QMenuBar {
    background-color: #161b22;
    color: #c9d1d9;
    border-bottom: 1px solid #21262d;
    padding: 2px 4px;
    spacing: 2px;
}
QMenuBar::item {
    padding: 5px 12px;
    border-radius: 6px;
}
QMenuBar::item:selected {
    background-color: #21262d;
}
QMenu {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item {
    padding: 8px 28px 8px 16px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: #1f6feb33;
    color: #58a6ff;
}
QMenu::separator {
    height: 1px;
    background: #21262d;
    margin: 4px 8px;
}

/* ─────────────────── Tab Widget ─────────────────── */
QTabWidget::pane {
    border: none;
    background-color: #0d1117;
}
QTabBar {
    qproperty-drawBase: 0;
}
QTabBar::tab {
    background-color: transparent;
    color: #8b949e;
    border: none;
    border-bottom: 3px solid transparent;
    padding: 10px 20px;
    margin: 0 1px;
    min-width: 100px;
    font-weight: 500;
    font-size: 13px;
}
QTabBar::tab:selected {
    color: #ffffff;
    border-bottom: 3px solid #e94560;
}
QTabBar::tab:hover:!selected {
    color: #c9d1d9;
    border-bottom: 3px solid #30363d;
}

/* ─────────────────── Group Box ─────────────────── */
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 16px;
    padding: 20px 12px 12px 12px;
    font-weight: 600;
    color: #e94560;
    background-color: #0d1117;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 8px;
    color: #e94560;
    background-color: #0d1117;
    border-radius: 4px;
}

/* ─────────────────── Buttons ─────────────────── */
QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 600;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}
QPushButton:pressed {
    background-color: #161b22;
}
QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}
QPushButton#primary {
    background-color: #e94560;
    color: #ffffff;
    border: none;
}
QPushButton#primary:hover {
    background-color: #ff5a7a;
}
QPushButton#primary:pressed {
    background-color: #c73050;
}
QPushButton#primary:disabled {
    background-color: #3d1525;
    color: #8b4050;
}
QPushButton#danger {
    background-color: #da3633;
    color: #ffffff;
    border: none;
}
QPushButton#danger:hover {
    background-color: #f85149;
}
QPushButton#success {
    background-color: #238636;
    color: #ffffff;
    border: none;
}
QPushButton#success:hover {
    background-color: #2ea043;
}

/* ─────────────────── Input Fields ─────────────────── */
QLineEdit, QPlainTextEdit, QTextEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #c9d1d9;
    selection-background-color: #1f6feb55;
    selection-color: #f0f6fc;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid #58a6ff;
    outline: none;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover {
    border-color: #484f58;
}
QLineEdit[readOnly="true"], QPlainTextEdit[readOnly="true"], QTextEdit[readOnly="true"] {
    background-color: #161b22;
    color: #8b949e;
}

/* ─────────────────── Spin Boxes ─────────────────── */
QSpinBox, QDoubleSpinBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 8px;
    color: #c9d1d9;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #58a6ff;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #484f58;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #21262d;
    border: none;
    border-top-right-radius: 5px;
    width: 20px;
    subcontrol-position: top right;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #21262d;
    border: none;
    border-bottom-right-radius: 5px;
    width: 20px;
    subcontrol-position: bottom right;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #30363d;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 8px; height: 6px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 8px; height: 6px;
}

/* ─────────────────── Combo Box ─────────────────── */
QComboBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 12px;
    color: #c9d1d9;
    min-height: 26px;
}
QComboBox:hover {
    border-color: #484f58;
}
QComboBox:focus, QComboBox:on {
    border-color: #58a6ff;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
    subcontrol-position: center right;
    padding-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    selection-background-color: #1f6feb33;
    selection-color: #58a6ff;
    color: #c9d1d9;
    padding: 4px;
    outline: none;
}
QComboBox QAbstractItemView::item {
    padding: 6px 12px;
    border-radius: 4px;
}

/* ─────────────────── Sliders ─────────────────── */
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #21262d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #e94560;
    border: 2px solid #0d1117;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #ff5a7a;
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 10px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560, stop:1 #8b5cf6);
    border-radius: 2px;
}

/* ─────────────────── Progress Bar ─────────────────── */
QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #21262d;
    text-align: center;
    color: #f0f6fc;
    font-weight: 600;
    font-size: 11px;
    min-height: 18px;
    max-height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560, stop:1 #8b5cf6);
    border-radius: 4px;
}

/* ─────────────────── Scroll Bars ─────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 40px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #484f58;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #30363d;
    min-width: 40px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #484f58;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}

/* ─────────────────── Check Box ─────────────────── */
QCheckBox {
    spacing: 8px;
    color: #c9d1d9;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #30363d;
    background: #0d1117;
}
QCheckBox::indicator:checked {
    background: #e94560;
    border-color: #e94560;
}
QCheckBox::indicator:hover {
    border-color: #58a6ff;
}
QCheckBox::indicator:checked:hover {
    background: #ff5a7a;
    border-color: #ff5a7a;
}

/* ─────────────────── Labels ─────────────────── */
QLabel {
    color: #c9d1d9;
    background: transparent;
}
QLabel#header {
    font-size: 18px;
    font-weight: bold;
    color: #f0f6fc;
}
QLabel#subheader {
    font-size: 13px;
    color: #8b949e;
}
QLabel#stat {
    font-size: 12px;
    color: #64ffda;
    font-weight: bold;
}
QLabel#muted {
    color: #484f58;
    font-size: 11px;
}

/* ─────────────────── List Widget ─────────────────── */
QListWidget {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    color: #c9d1d9;
    outline: none;
    padding: 4px;
}
QListWidget::item {
    padding: 6px 10px;
    border-radius: 6px;
    margin: 1px 2px;
}
QListWidget::item:selected {
    background-color: #1f6feb33;
    color: #58a6ff;
}
QListWidget::item:hover:!selected {
    background-color: #161b22;
}
QListWidget::item:alternate {
    background-color: transparent;
}

/* ─────────────────── Splitter ─────────────────── */
QSplitter::handle {
    background-color: #21262d;
    width: 3px;
    margin: 2px;
    border-radius: 1px;
}
QSplitter::handle:hover {
    background-color: #e94560;
}

/* ─────────────────── Scroll Area ─────────────────── */
QScrollArea {
    border: none;
    background: transparent;
}

/* ─────────────────── Status Bar ─────────────────── */
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #21262d;
    font-size: 12px;
    padding: 2px 8px;
}
QStatusBar QLabel {
    color: #8b949e;
    padding: 0 8px;
}

/* ─────────────────── Tool Tip ─────────────────── */
QToolTip {
    background-color: #1c2128;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #c9d1d9;
    padding: 6px 10px;
    font-size: 12px;
}

/* ─────────────────── StatusCard Widget ─────────────────── */
#StatusCard {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
}

/* ─────────────────── Focus Outlines (accessibility) ─────────────────── */
*:focus {
    outline: none;
}
"""

LIGHT_THEME = """
/* ─────────────────── Base ─────────────────── */
QWidget {
    background-color: #ffffff;
    color: #24292f;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
}

QMainWindow {
    background-color: #f6f8fa;
}

/* ─────────────────── Menu Bar ─────────────────── */
QMenuBar {
    background-color: #f6f8fa;
    color: #24292f;
    border-bottom: 1px solid #d0d7de;
    padding: 2px 4px;
    spacing: 2px;
}
QMenuBar::item {
    padding: 5px 12px;
    border-radius: 6px;
}
QMenuBar::item:selected {
    background-color: #eaeef2;
}
QMenu {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item {
    padding: 8px 28px 8px 16px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: #ddf4ff;
    color: #1f6feb;
}
QMenu::separator {
    height: 1px;
    background: #d0d7de;
    margin: 4px 8px;
}

/* ─────────────────── Tab Widget ─────────────────── */
QTabWidget::pane {
    border: none;
    background-color: #ffffff;
}
QTabBar {
    qproperty-drawBase: 0;
}
QTabBar::tab {
    background-color: transparent;
    color: #656d76;
    border: none;
    border-bottom: 3px solid transparent;
    padding: 10px 20px;
    margin: 0 1px;
    min-width: 100px;
    font-weight: 500;
    font-size: 13px;
}
QTabBar::tab:selected {
    color: #24292f;
    border-bottom: 3px solid #e94560;
}
QTabBar::tab:hover:!selected {
    color: #24292f;
    border-bottom: 2px solid #d0d7de;
}

/* ─────────────────── Group Box ─────────────────── */
QGroupBox {
    border: 1px solid #d0d7de;
    border-radius: 8px;
    margin-top: 16px;
    padding: 20px 12px 12px 12px;
    font-weight: 600;
    color: #e94560;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 8px;
    color: #e94560;
    background-color: #ffffff;
    border-radius: 4px;
}

/* ─────────────────── Buttons ─────────────────── */
QPushButton {
    background-color: #f6f8fa;
    color: #24292f;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 600;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #eaeef2;
    border-color: #afb8c1;
}
QPushButton:pressed {
    background-color: #d0d7de;
}
QPushButton:disabled {
    background-color: #f6f8fa;
    color: #8c959f;
    border-color: #d0d7de;
}
QPushButton#primary {
    background-color: #e94560;
    color: #ffffff;
    border: none;
}
QPushButton#primary:hover {
    background-color: #ff5a7a;
}
QPushButton#primary:pressed {
    background-color: #c73050;
}
QPushButton#primary:disabled {
    background-color: #ffb3c1;
    color: #ffffff;
}
QPushButton#danger {
    background-color: #da3633;
    color: #ffffff;
    border: none;
}
QPushButton#danger:hover {
    background-color: #f85149;
}
QPushButton#success {
    background-color: #238636;
    color: #ffffff;
    border: none;
}
QPushButton#success:hover {
    background-color: #2ea043;
}

/* ─────────────────── Input Fields ─────────────────── */
QLineEdit, QPlainTextEdit, QTextEdit {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 6px 10px;
    color: #24292f;
    selection-background-color: #ddf4ff;
    selection-color: #24292f;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid #1f6feb;
    outline: none;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover {
    border-color: #afb8c1;
}
QLineEdit[readOnly="true"], QPlainTextEdit[readOnly="true"], QTextEdit[readOnly="true"] {
    background-color: #f6f8fa;
    color: #656d76;
}

/* ─────────────────── Spin Boxes ─────────────────── */
QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 4px 8px;
    color: #24292f;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #1f6feb;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #afb8c1;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #f6f8fa;
    border: none;
    border-top-right-radius: 5px;
    width: 20px;
    subcontrol-position: top right;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #f6f8fa;
    border: none;
    border-bottom-right-radius: 5px;
    width: 20px;
    subcontrol-position: bottom right;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #eaeef2;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 8px; height: 6px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 8px; height: 6px;
}

/* ─────────────────── Combo Box ─────────────────── */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 6px 12px;
    color: #24292f;
    min-height: 26px;
}
QComboBox:hover {
    border-color: #afb8c1;
}
QComboBox:focus, QComboBox:on {
    border-color: #1f6feb;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
    subcontrol-position: center right;
    padding-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    selection-background-color: #ddf4ff;
    selection-color: #1f6feb;
    color: #24292f;
    padding: 4px;
    outline: none;
}
QComboBox QAbstractItemView::item {
    padding: 6px 12px;
    border-radius: 4px;
}

/* ─────────────────── Sliders ─────────────────── */
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #d0d7de;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #e94560;
    border: 2px solid #ffffff;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #ff5a7a;
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 10px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560, stop:1 #8b5cf6);
    border-radius: 2px;
}

/* ─────────────────── Progress Bar ─────────────────── */
QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #eaeef2;
    text-align: center;
    color: #24292f;
    font-weight: 600;
    font-size: 11px;
    min-height: 18px;
    max-height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560, stop:1 #8b5cf6);
    border-radius: 4px;
}

/* ─────────────────── Scroll Bars ─────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #d0d7de;
    min-height: 40px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #afb8c1;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #d0d7de;
    min-width: 40px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #afb8c1;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}

/* ─────────────────── Check Box ─────────────────── */
QCheckBox {
    spacing: 8px;
    color: #24292f;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #d0d7de;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #e94560;
    border-color: #e94560;
}
QCheckBox::indicator:hover {
    border-color: #1f6feb;
}
QCheckBox::indicator:checked:hover {
    background: #ff5a7a;
    border-color: #ff5a7a;
}

/* ─────────────────── Labels ─────────────────── */
QLabel {
    color: #24292f;
    background: transparent;
}
QLabel#header {
    font-size: 18px;
    font-weight: bold;
    color: #24292f;
}
QLabel#subheader {
    font-size: 13px;
    color: #656d76;
}
QLabel#stat {
    font-size: 12px;
    color: #0969da;
    font-weight: bold;
}
QLabel#muted {
    color: #8c959f;
    font-size: 11px;
}

/* ─────────────────── List Widget ─────────────────── */
QListWidget {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    color: #24292f;
    outline: none;
    padding: 4px;
}
QListWidget::item {
    padding: 6px 10px;
    border-radius: 6px;
    margin: 1px 2px;
}
QListWidget::item:selected {
    background-color: #ddf4ff;
    color: #1f6feb;
}
QListWidget::item:hover:!selected {
    background-color: #f6f8fa;
}

/* ─────────────────── Splitter ─────────────────── */
QSplitter::handle {
    background-color: #d0d7de;
    width: 3px;
    margin: 2px;
    border-radius: 1px;
}
QSplitter::handle:hover {
    background-color: #e94560;
}

/* ─────────────────── Scroll Area ─────────────────── */
QScrollArea {
    border: none;
    background: transparent;
}

/* ─────────────────── Status Bar ─────────────────── */
QStatusBar {
    background-color: #f6f8fa;
    color: #656d76;
    border-top: 1px solid #d0d7de;
    font-size: 12px;
    padding: 2px 8px;
}
QStatusBar QLabel {
    color: #656d76;
    padding: 0 8px;
}

/* ─────────────────── Tool Tip ─────────────────── */
QToolTip {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    color: #24292f;
    padding: 6px 10px;
    font-size: 12px;
}

/* ─────────────────── StatusCard Widget ─────────────────── */
#StatusCard {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 8px;
}

/* ─────────────────── Focus Outlines (accessibility) ─────────────────── */
*:focus {
    outline: none;
}
"""
