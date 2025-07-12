#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS Main Window

The central GUI window for the Detection Model Suite (DMS).
Provides a modern, intuitive interface for managing object detection projects.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# GUI Framework - try multiple options for compatibility
try:
    from PySide6.QtWidgets import *
    from PySide6.QtCore import *
    from PySide6.QtGui import *

    GUI_FRAMEWORK = "PySide6"
except ImportError:
    try:
        from PyQt6.QtWidgets import *
        from PyQt6.QtCore import *
        from PyQt6.QtGui import *

        GUI_FRAMEWORK = "PyQt6"
    except ImportError:
        try:
            from PyQt5.QtWidgets import *
            from PyQt5.QtCore import *
            from PyQt5.QtGui import *

            GUI_FRAMEWORK = "PyQt5"
        except ImportError:
            raise ImportError(
                "No compatible GUI framework found. Please install PySide6, PyQt6, or PyQt5."
            )

from .components.dashboard import DashboardWidget
from .components.project_manager import ProjectManagerWidget
from .components.training import TrainingWidget
from .components.annotation import AnnotationWidget
from .components.capture import CaptureWidget
from .components.system_monitor import SystemMonitorWidget
from .components.settings import SettingsWidget

from .utils.styles import get_dark_style, get_light_style
from .utils.icons import IconManager

# Import utilities with error handling
try:
    STYLES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Styles not available: {e}")
    STYLES_AVAILABLE = False

    def get_dark_style():
        return ""

    def get_light_style():
        return ""


try:
    ICONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Icons not available: {e}")
    ICONS_AVAILABLE = False

    # Create a simple icon manager
    class IconManager:
        @classmethod
        def get_icon(cls, name):
            from PySide6.QtGui import QIcon

            return QIcon()


class DMSMainWindow(QMainWindow):
    """
    Main window for the DMS application.

    Features:
    - Modern, responsive design
    - Tabbed interface for different modules
    - Real-time system monitoring
    - Project management
    - Training interface
    - Annotation tools
    - Screen capture system
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # Set GUI framework as instance attribute
        self.GUI_FRAMEWORK = GUI_FRAMEWORK

        # Configuration
        self.config = config or {}
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.current_project = None

        # Initialize GPU detection attribute
        self._gpu_detected = None

        # Initialize UI
        self.init_ui()
        self.setup_menu()
        self.setup_status_bar()
        self.setup_toolbar()

        # Load configuration
        self.load_config()

        # Start monitoring
        self.start_system_monitoring()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("DMS - Detection Model Suite")
        self.setGeometry(100, 100, 1400, 900)

        # Set window icon
        if ICONS_AVAILABLE:
            self.setWindowIcon(IconManager.get_icon("app"))

        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create main layout
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Create sidebar
        self.create_sidebar()

        # Create main content area
        self.create_main_content()

        # Apply styling
        self.apply_styling()

    def create_sidebar(self):
        """Create the sidebar navigation."""
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(270)
        self.sidebar.setObjectName("sidebar")

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo/Title area
        self.create_logo_area(sidebar_layout)

        # Navigation buttons
        self.create_navigation_buttons(sidebar_layout)

        # System info area
        self.create_system_info_area(sidebar_layout)

        self.main_layout.addWidget(self.sidebar)

    def create_logo_area(self, layout):
        """Create the logo and title area."""
        logo_widget = QWidget()
        logo_widget.setFixedHeight(70)
        logo_widget.setObjectName("logo-area")

        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        # Logo icon
        logo_label = QLabel()
        if ICONS_AVAILABLE:
            logo_label.setPixmap(IconManager.get_icon("logo").pixmap(32, 32))
        else:
            logo_label.setText("🎯")  # Unicode fallback
            logo_label.setStyleSheet("font-size: 22px; color: white;")
        logo_label.setAlignment(Qt.AlignCenter)

        # Title
        title_label = QLabel("DMS")
        title_label.setObjectName("app-title")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        # Subtitle
        subtitle_label = QLabel("Detection Model Suite")
        subtitle_label.setObjectName("app-subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 10px; color: #CCCCCC;")

        logo_layout.addWidget(logo_label)
        logo_layout.addWidget(title_label)
        logo_layout.addWidget(subtitle_label)

        layout.addWidget(logo_widget)

    def create_navigation_buttons(self, layout):
        """Create navigation buttons."""
        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)

        # Navigation buttons
        self.nav_buttons = {}

        nav_items = [
            ("dashboard", "Dashboard", "dashboard", "Main overview and quick actions"),
            ("projects", "Projects", "folder", "Manage your object detection projects"),
            ("capture", "Capture", "camera", "Screen capture and data collection"),
            ("annotation", "Annotation", "edit", "Label and annotate your datasets"),
            ("training", "Training", "brain", "Train YOLO models"),
            ("monitor", "System", "monitor", "System resources and performance"),
            ("settings", "Settings", "settings", "Configure DMS settings"),
        ]

        for key, text, icon_name, tooltip in nav_items:
            btn = QPushButton()
            btn.setObjectName(f"nav-{key}")
            btn.setToolTip(tooltip)
            btn.setCheckable(True)

            # Create button layout
            btn_layout = QHBoxLayout(btn)
            btn_layout.setContentsMargins(8, 6, 8, 6)

            # Icon
            icon_label = QLabel()
            if ICONS_AVAILABLE:
                icon_label.setPixmap(IconManager.get_icon(icon_name).pixmap(20, 20))
            else:
                # Unicode fallback icons
                fallback_icons = {
                    "dashboard": "📊",
                    "folder": "📁",
                    "camera": "📷",
                    "edit": "✏️",
                    "brain": "🧠",
                    "monitor": "🖥️",
                    "settings": "⚙️",
                }
                icon_label.setText(fallback_icons.get(icon_name, "📄"))
                icon_label.setStyleSheet("font-size: 16px;")
            btn_layout.addWidget(icon_label)

            # Text
            text_label = QLabel(text)
            text_label.setObjectName("nav-text")
            text_label.setStyleSheet("font-size: 13px; color: white; font-weight: 500;")
            btn_layout.addWidget(text_label)

            btn_layout.addStretch()

            # Connect signal
            btn.clicked.connect(lambda checked, k=key: self.show_page(k))

            self.nav_buttons[key] = btn
            nav_layout.addWidget(btn)

        # Select dashboard by default
        self.nav_buttons["dashboard"].setChecked(True)

        nav_layout.addStretch()
        layout.addWidget(nav_widget)

    def create_system_info_area(self, layout):
        """Create system information area."""
        info_widget = QWidget()
        info_widget.setObjectName("system-info")
        info_widget.setFixedHeight(120)

        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(15, 15, 15, 15)
        info_layout.setSpacing(8)

        # System status
        self.system_status_label = QLabel("System: Ready")
        self.system_status_label.setObjectName("system-status")

        # GPU info
        self.gpu_info_label = QLabel("GPU: Detecting...")
        self.gpu_info_label.setObjectName("system-info-text")

        # Memory info
        self.memory_info_label = QLabel("Memory: --")
        self.memory_info_label.setObjectName("system-info-text")

        # Project info
        self.project_info_label = QLabel("Project: None")
        self.project_info_label.setObjectName("system-info-text")

        info_layout.addWidget(self.system_status_label)
        info_layout.addWidget(self.gpu_info_label)
        info_layout.addWidget(self.memory_info_label)
        info_layout.addWidget(self.project_info_label)

        layout.addWidget(info_widget)

    def create_main_content(self):
        """Create the main content area."""
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("content-stack")

        # Create pages
        self.pages = {}

        # Dashboard
        self.pages["dashboard"] = DashboardWidget(self)
        self.content_stack.addWidget(self.pages["dashboard"])

        # Projects
        self.pages["projects"] = ProjectManagerWidget(self)
        self.content_stack.addWidget(self.pages["projects"])

        # Capture
        self.pages["capture"] = CaptureWidget(self)
        self.content_stack.addWidget(self.pages["capture"])

        # Annotation
        self.pages["annotation"] = AnnotationWidget(self)
        self.content_stack.addWidget(self.pages["annotation"])

        # Training
        self.pages["training"] = TrainingWidget(self)
        self.content_stack.addWidget(self.pages["training"])

        # System Monitor
        self.pages["monitor"] = SystemMonitorWidget(self)
        self.content_stack.addWidget(self.pages["monitor"])

        # Settings
        self.pages["settings"] = SettingsWidget(self)
        self.content_stack.addWidget(self.pages["settings"])

        self.main_layout.addWidget(self.content_stack)

    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("&Open Project", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        capture_action = QAction("&Screen Capture", self)
        capture_action.triggered.connect(lambda: self.show_page("capture"))
        tools_menu.addAction(capture_action)

        annotation_action = QAction("&Annotation", self)
        annotation_action.triggered.connect(lambda: self.show_page("annotation"))
        tools_menu.addAction(annotation_action)

        training_action = QAction("&Training", self)
        training_action.triggered.connect(lambda: self.show_page("training"))
        tools_menu.addAction(training_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = self.statusBar()

        # Status message
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def setup_toolbar(self):
        """Setup the toolbar."""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)

        # New project
        if ICONS_AVAILABLE:
            new_project_action = QAction(
                IconManager.get_icon("new"), "New Project", self
            )
        else:
            new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self.new_project)
        toolbar.addAction(new_project_action)

        # Open project
        if ICONS_AVAILABLE:
            open_project_action = QAction(
                IconManager.get_icon("open"), "Open Project", self
            )
        else:
            open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self.open_project)
        toolbar.addAction(open_project_action)

        toolbar.addSeparator()

        # Quick actions
        if ICONS_AVAILABLE:
            capture_action = QAction(IconManager.get_icon("camera"), "Capture", self)
            annotation_action = QAction(IconManager.get_icon("edit"), "Annotate", self)
            training_action = QAction(IconManager.get_icon("brain"), "Train", self)
        else:
            capture_action = QAction("Capture", self)
            annotation_action = QAction("Annotate", self)
            training_action = QAction("Train", self)

        capture_action.triggered.connect(lambda: self.show_page("capture"))
        toolbar.addAction(capture_action)

        annotation_action.triggered.connect(lambda: self.show_page("annotation"))
        toolbar.addAction(annotation_action)

        training_action.triggered.connect(lambda: self.show_page("training"))
        toolbar.addAction(training_action)

    def show_page(self, page_name: str):
        """Show a specific page."""
        if page_name in self.pages:
            # Update navigation buttons
            for key, btn in self.nav_buttons.items():
                btn.setChecked(key == page_name)

            # Show the page
            self.content_stack.setCurrentWidget(self.pages[page_name])

            # Update status
            self.status_label.setText(f"Showing {page_name.title()}")

    def new_project(self):
        """Create a new project."""
        # This will be implemented in the project manager
        self.show_page("projects")
        self.pages["projects"].new_project()

    def open_project(self):
        """Open an existing project."""
        # This will be implemented in the project manager
        self.show_page("projects")
        self.pages["projects"].open_project()

    def load_config(self):
        """Load configuration."""
        # Load from config file
        config_path = self.project_root / "config" / "studio_config.yaml"
        if config_path.exists():
            try:
                import yaml

                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config: {e}")

    def start_system_monitoring(self):
        """Start system monitoring."""
        # Create timer for system updates
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self.update_system_info)
        self.system_timer.start(2000)  # Update every 2 seconds

    def update_system_info(self):
        """Update system information display."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_gb = memory.used / (1024**3)
            self.memory_info_label.setText(
                f"Memory: {memory_percent:.1f}% ({memory_gb:.1f}GB)"
            )
            if not hasattr(self, "_gpu_detected"):
                self._gpu_detected = self.detect_gpu()
                if self._gpu_detected:
                    if getattr(self, "_gpu_type", None) == "cuda":
                        self.gpu_info_label.setText("GPU: Available (CUDA)")
                    elif getattr(self, "_gpu_type", None) == "amd":
                        self.gpu_info_label.setText("GPU: Available (AMD)")
                    else:
                        self.gpu_info_label.setText("GPU: Available")
                else:
                    self.gpu_info_label.setText("GPU: CPU Only")
        except ImportError:
            self.memory_info_label.setText("Memory: psutil not available")

    def detect_gpu(self):
        """Detect GPU availability (NVIDIA CUDA or AMD)."""
        try:
            import torch

            if torch.cuda.is_available():
                self._gpu_type = "cuda"
                return True
        except ImportError:
            pass
        # Check for AMD GPU using WMI (Windows only)
        try:
            import wmi

            computer = wmi.WMI()
            for gpu in computer.Win32_VideoController():
                if "amd" in gpu.Name.lower() or "radeon" in gpu.Name.lower():
                    self._gpu_type = "amd"
                    return True
        except ImportError:
            pass
        except Exception as e:
            print(f"GPU detection error: {e}")
        self._gpu_type = None
        return False

    def apply_styling(self):
        """Apply styling to the application."""
        if not STYLES_AVAILABLE:
            print("Warning: Styles not available, using default styling")
            return

        # Get theme from config
        theme = self.config.get("annotation", {}).get("ui", {}).get("theme", "dark")

        if theme == "dark":
            self.setStyleSheet(get_dark_style())
        else:
            self.setStyleSheet(get_light_style())

    def closeEvent(self, event):
        """Handle application close event."""
        # Save any unsaved work
        # Stop monitoring
        if hasattr(self, "system_timer"):
            self.system_timer.stop()

        # Close all pages
        for page in self.pages.values():
            if hasattr(page, "cleanup"):
                page.cleanup()

        event.accept()


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("DMS")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("DMS Team")

    # Create and show main window
    window = DMSMainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
