#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capture Widget.

GUI component for screen capture interface.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Import capture system components
try:
    from ...capture.window_capture import WindowCaptureSystem, WindowInfo
    from ...utils.config import CaptureConfig

    CAPTURE_AVAILABLE = True
except ImportError:
    CAPTURE_AVAILABLE = False
    WindowCaptureSystem = None  # type: ignore
    WindowInfo = None  # type: ignore
    CaptureConfig = None  # type: ignore


class CaptureWidget(QWidget):
    """Screen capture interface."""

    # Signals for thread-safe UI updates
    preview_updated = Signal(QPixmap)
    status_updated = Signal(str)
    stats_updated = Signal(dict)

    def __init__(self, main_window: Any) -> None:
        """
        Initialize the CaptureWidget.

        Args:
            main_window: The main window instance.
        """
        super().__init__()
        self.main_window = main_window

        # Capture system
        self.capture_system: Optional[WindowCaptureSystem] = None
        self.current_session: Optional[Any] = None

        # UI components
        self.window_combo: QComboBox
        self.fps_spinbox: QSpinBox
        self.resolution_combo: QComboBox
        self.preview_label: QLabel
        self.status_label: QLabel
        self.stats_label: QLabel
        self.start_btn: QPushButton
        self.stop_btn: QPushButton
        self.pause_btn: QPushButton

        # Preview timer
        self.preview_timer: Optional[QTimer] = None

        # Available windows
        self.available_windows: List[WindowInfo] = []

        self.init_ui()
        self.init_capture_system()

        # Connect signals
        self.status_updated.connect(self.status_label.setText)
        self.preview_updated.connect(self.preview_label.setPixmap)

    def init_ui(self) -> None:
        """Initialize the user interface for the capture widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        self.create_header(layout)

        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left panel - Controls
        left_panel = self.create_controls_panel()
        content_layout.addWidget(left_panel, 1)

        # Right panel - Preview and status
        right_panel = self.create_preview_panel()
        content_layout.addWidget(right_panel, 2)

        layout.addLayout(content_layout)

    def create_header(self, layout: QVBoxLayout) -> None:
        """Create the capture header."""
        header_widget = QWidget()
        header_widget.setObjectName("dashboard-header")
        header_widget.setFixedHeight(60)

        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title_label = QLabel("Screen Capture")
        title_label.setObjectName("welcome-title")
        header_layout.addWidget(title_label)

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("time-label")
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)

        layout.addWidget(header_widget)

    def create_controls_panel(self) -> QWidget:
        """Create the controls panel for capture settings and actions."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Window selection
        window_group = QGroupBox("Window Selection")
        window_group.setObjectName("dashboard-group")
        window_layout = QVBoxLayout(window_group)

        # Window combo box
        self.window_combo = QComboBox()
        self.window_combo.addItem("Full Screen", None)
        self.window_combo.currentIndexChanged.connect(self.on_window_changed)
        window_layout.addWidget(self.window_combo)

        # Refresh button
        refresh_btn = QPushButton("Refresh Windows")
        refresh_btn.setObjectName("secondary-btn")
        refresh_btn.clicked.connect(self.refresh_windows)
        window_layout.addWidget(refresh_btn)

        layout.addWidget(window_group)

        # Capture settings
        settings_group = QGroupBox("Capture Settings")
        settings_group.setObjectName("dashboard-group")
        settings_layout = QFormLayout(settings_group)

        # FPS setting
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 30)
        self.fps_spinbox.setValue(5)
        self.fps_spinbox.setSuffix(" FPS")
        settings_layout.addRow("Frame Rate:", self.fps_spinbox)

        # Resolution setting
        self.resolution_combo = QComboBox()
        resolutions = [
            ("640x640", (640, 640)),
            ("800x600", (800, 600)),
            ("1024x768", (1024, 768)),
            ("1280x720", (1280, 720)),
            ("1920x1080", (1920, 1080)),
        ]
        for name, res in resolutions:
            self.resolution_combo.addItem(name, res)
        settings_layout.addRow("Resolution:", self.resolution_combo)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setText("data/captured")
        output_layout.addWidget(self.output_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("secondary-btn")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(browse_btn)

        settings_layout.addRow("Output Directory:", output_layout)

        layout.addWidget(settings_group)

        # Recording controls
        controls_group = QGroupBox("Recording Controls")
        controls_group.setObjectName("dashboard-group")
        controls_layout = QVBoxLayout(controls_group)

        # Control buttons
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Capture")
        self.start_btn.setObjectName("quick-action-btn")
        self.start_btn.clicked.connect(self.start_capture)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Capture")
        self.stop_btn.setObjectName("secondary-btn")
        self.stop_btn.clicked.connect(self.stop_capture)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("secondary-btn")
        self.pause_btn.clicked.connect(self.pause_capture)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)

        controls_layout.addLayout(btn_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        layout.addWidget(controls_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_group.setObjectName("dashboard-group")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_label = QLabel("No capture session active")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

        layout.addStretch()
        return panel

    def create_preview_panel(self) -> QWidget:
        """Create the preview panel for live capture preview and save options."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Preview group
        preview_group = QGroupBox("Live Preview")
        preview_group.setObjectName("dashboard-group")
        preview_layout = QVBoxLayout(preview_group)

        # Preview label
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "border: 2px dashed #404040; border-radius: 8px;"
        )
        self.preview_label.setText("No preview available")
        preview_layout.addWidget(self.preview_label)

        # Preview controls
        preview_controls = QHBoxLayout()

        preview_btn = QPushButton("Enable Preview")
        preview_btn.setObjectName("secondary-btn")
        preview_btn.clicked.connect(self.toggle_preview)
        preview_controls.addWidget(preview_btn)

        preview_controls.addStretch()
        preview_layout.addLayout(preview_controls)

        layout.addWidget(preview_group)

        # Save options
        save_group = QGroupBox("Save Options")
        save_group.setObjectName("dashboard-group")
        save_layout = QVBoxLayout(save_group)

        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG"])
        format_layout.addWidget(self.format_combo)

        format_layout.addStretch()
        save_layout.addLayout(format_layout)

        # Quality slider (for JPEG)
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))

        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(50, 100)
        self.quality_slider.setValue(95)
        self.quality_slider.setEnabled(False)
        quality_layout.addWidget(self.quality_slider)

        self.quality_label = QLabel("95%")
        self.quality_label.setEnabled(False)
        quality_layout.addWidget(self.quality_label)

        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_label.setText(f"{v}%")
        )
        self.format_combo.currentTextChanged.connect(self.on_format_changed)

        save_layout.addLayout(quality_layout)

        layout.addWidget(save_group)

        layout.addStretch()
        return panel

    def init_capture_system(self) -> None:
        """Initialize the capture system and refresh available windows."""
        if not CAPTURE_AVAILABLE:
            self.status_updated.emit("Capture system not available")
            return

        try:
            # Create capture config
            config = CaptureConfig()
            self.capture_system = WindowCaptureSystem(config)

            # Refresh windows
            self.refresh_windows()

            self.status_updated.emit("Capture system ready")

        except Exception as e:
            self.status_updated.emit(f"Failed to initialize capture system: {e}")

    def refresh_windows(self) -> None:
        """Refresh the list of available windows for capture."""
        if not self.capture_system:
            return

        try:
            self.available_windows = self.capture_system.get_available_windows()

            # Update combo box
            self.window_combo.clear()
            self.window_combo.addItem("Full Screen", None)

            for window in self.available_windows:
                self.window_combo.addItem(window.title, window)

            self.status_updated.emit(f"Found {len(self.available_windows)} windows")

        except Exception as e:
            self.status_updated.emit(f"Error refreshing windows: {e}")

    def on_window_changed(self, index: int) -> None:
        """Handle window selection change in the combo box."""
        if index == 0:  # Full screen
            self.status_updated.emit("Selected: Full Screen")
        else:
            window = self.window_combo.currentData()
            if window:
                self.status_updated.emit(f"Selected: {window.title}")

    def on_format_changed(self, format_name: str) -> None:
        """Handle format selection change for image saving."""
        is_jpeg = format_name == "JPEG"
        self.quality_slider.setEnabled(is_jpeg)
        self.quality_label.setEnabled(is_jpeg)

    def browse_output_dir(self) -> None:
        """Open a dialog to browse and select the output directory."""
        current_dir = self.output_edit.text()
        if not current_dir:
            current_dir = str(Path.cwd())

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", current_dir
        )

        if dir_path:
            self.output_edit.setText(dir_path)

    def start_capture(self) -> None:
        """Start the capture session with the selected settings."""
        if not self.capture_system:
            QMessageBox.warning(self, "Error", "Capture system not available")
            return

        try:
            # Get settings
            fps = self.fps_spinbox.value()
            resolution = self.resolution_combo.currentData()
            output_dir = self.output_edit.text()

            # Validate output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create session ID
            session_id = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get selected window
            window = self.window_combo.currentData()
            window_title = window.title if window else None

            # Start session
            self.current_session = self.capture_system.start_session(
                session_id=session_id,
                output_dir=output_path,
                window_title=window_title,
                fps=fps,
                resolution=resolution,
            )

            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.progress_bar.setVisible(True)

            # Start preview timer
            self.start_preview_timer()

            self.status_updated.emit(f"Capture started: {session_id}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start capture: {e}")

    def stop_capture(self) -> None:
        """Stop the current capture session and update the UI."""
        if not self.current_session:
            return

        try:
            # Stop session
            if self.capture_system is not None:
                self.capture_system.stop_session(self.current_session.session_id)
            self.current_session = None

            # Update UI
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.progress_bar.setVisible(False)

            # Stop preview timer
            self.stop_preview_timer()

            self.status_updated.emit("Capture stopped")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop capture: {e}")

    def pause_capture(self) -> None:
        """Pause or resume the current capture session."""
        if not self.current_session:
            return

        try:
            if self.current_session.is_paused:
                self.current_session.resume_capture()
                self.pause_btn.setText("Pause")
                self.status_updated.emit("Capture resumed")
            else:
                self.current_session.pause_capture()
                self.pause_btn.setText("Resume")
                self.status_updated.emit("Capture paused")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to pause/resume capture: {e}")

    def toggle_preview(self) -> None:
        """Toggle the preview timer for live image preview."""
        if self.preview_timer and self.preview_timer.isActive():
            self.stop_preview_timer()
        else:
            self.start_preview_timer()

    def start_preview_timer(self) -> None:
        """Start the preview timer for updating the live preview."""
        if not self.preview_timer:
            self.preview_timer = QTimer()
            self.preview_timer.timeout.connect(self.update_preview)

        self.preview_timer.start(100)  # 10 FPS for preview

    def stop_preview_timer(self) -> None:
        """Stop the preview timer."""
        if self.preview_timer:
            self.preview_timer.stop()

    def update_preview(self) -> None:
        """Update the preview image and statistics during capture."""
        if not self.current_session or not self.current_session.is_active:
            return

        try:
            # Get latest captured image (this would need to be implemented in the
            # capture system). For now, show a placeholder.
            preview_text = (
                f"Capturing...\nSession: {self.current_session.session_id}\n"
                f"FPS: {self.current_session.fps}"
            )
            self.preview_label.setText(preview_text)

            # Update statistics
            stats = self.current_session.get_stats()
            stats_text = (
                f"Images Captured: {stats['images_captured']}\n"
                f"Errors: {stats['errors']}\n"
                f"Duration: {stats['total_duration']:.1f}s\n"
                f"Average FPS: {stats['average_fps']:.1f}"
            )
            self.stats_label.setText(stats_text)

        except Exception as e:
            self.status_updated.emit(f"Preview error: {e}")

    def cleanup(self) -> None:
        """Cleanup resources and stop any active capture session or timers."""
        if self.current_session:
            self.stop_capture()

        if self.preview_timer:
            self.preview_timer.stop()
