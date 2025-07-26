#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS Main Window.

The central GUI window for the Detection Model Suite (DMS).
Provides a modern, intuitive interface for managing object detection projects.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import psutil
import torch
import wmi
import yaml

# GUI Framework detection and imports
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .components.annotation import AnnotationWidget
from .components.capture import CaptureWidget
from .components.dashboard import DashboardWidget
from .components.project_manager import ProjectManagerWidget
from .components.settings import SettingsWidget
from .components.system_monitor import SystemMonitorWidget
from .components.training import TrainingWidget
from .utils.icons import IconManager
from .utils.styles import get_dark_style, get_light_style

GUI_FRAMEWORK = "PySide6"


def get_style_functions() -> Tuple[Callable[[], str], Callable[[], str]]:
    """Get style functions with fallback.

    Returns:
        Tuple of (get_dark_style, get_light_style) functions.
    """
    try:
        return get_dark_style, get_light_style
    except ImportError as e:
        print(f"Warning: Styles not available: {e}")

        def fallback_dark_style() -> str:
            """Fallback dark style function."""
            return ""

        def fallback_light_style() -> str:
            """Fallback light style function."""
            return ""

        return fallback_dark_style, fallback_light_style


def get_icon_manager() -> Any:
    """Get icon manager with fallback.

    Returns:
        IconManager class or fallback.
    """
    try:
        return IconManager
    except ImportError as e:
        print(f"Warning: Icons not available: {e}")

        class FallbackIconManager:
            """Fallback icon manager."""

            @classmethod
            def get_icon(cls, name: str) -> QIcon:
                """Get fallback icon."""
                return QIcon()

        return FallbackIconManager


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

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the main window.

        Args:
            config: Optional configuration dictionary.
        """
        super().__init__()

        # Set GUI framework as instance attribute
        self.GUI_FRAMEWORK = GUI_FRAMEWORK

        # Configuration
        self.config = config or {}
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.current_project: Optional[str] = None

        # Initialize GPU detection attribute
        self._gpu_detected: Optional[bool] = None
        self._gpu_type: Optional[str] = None
        self._gpu_name: Optional[str] = None

        # Get style and icon functions
        self.get_dark_style, self.get_light_style = get_style_functions()
        self.IconManager = get_icon_manager()

        # Initialize UI
        self.init_ui()
        self.setup_menu()
        self.setup_status_bar()
        self.setup_toolbar()

        # Load configuration
        self.load_config()

        # Start monitoring
        self.start_system_monitoring()
        
        # Initialize GPU detection immediately
        self._gpu_detected = self.detect_gpu()
        self.update_gpu_label()
        
        # Update dashboard GPU status if it exists
        if hasattr(self, 'pages') and 'dashboard' in self.pages:
            dashboard = self.pages['dashboard']
            if hasattr(dashboard, 'update_gpu_status'):
                dashboard.update_gpu_status()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("DMS - Detection Model Suite")
        self.setGeometry(100, 100, 1400, 900)

        # Set window icon
        self.setWindowIcon(self.IconManager.get_icon("app"))

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

    def create_sidebar(self) -> None:
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

    def create_logo_area(self, layout: QVBoxLayout) -> None:
        """Create the logo and title area.

        Args:
            layout: The layout to add the logo area to.
        """
        logo_widget = QWidget()
        logo_widget.setFixedHeight(70)
        logo_widget.setObjectName("logo-area")

        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        # Logo icon
        logo_label = QLabel()
        logo_label.setPixmap(self.IconManager.get_icon("logo").pixmap(32, 32))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Title
        title_label = QLabel("DMS")
        title_label.setObjectName("app-title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        # Subtitle
        subtitle_label = QLabel("Detection Model Suite")
        subtitle_label.setObjectName("app-subtitle")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 10px; color: #CCCCCC;")

        logo_layout.addWidget(logo_label)
        logo_layout.addWidget(title_label)
        logo_layout.addWidget(subtitle_label)

        layout.addWidget(logo_widget)

    def create_navigation_buttons(self, layout: QVBoxLayout) -> None:
        """Create navigation buttons.

        Args:
            layout: The layout to add navigation buttons to.
        """
        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)

        # Navigation buttons
        self.nav_buttons: Dict[str, QPushButton] = {}

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
            icon_label.setPixmap(self.IconManager.get_icon(icon_name).pixmap(20, 20))
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

    def create_system_info_area(self, layout: QVBoxLayout) -> None:
        """Create system information area.

        Args:
            layout: The layout to add system info area to.
        """
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

    def create_main_content(self) -> None:
        """Create the main content area."""
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("content-stack")

        # Create pages
        self.pages: Dict[str, QWidget] = {}

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

    def setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        if menubar is None:
            return  # type: ignore[unreachable]

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
        menubar.addMenu("&Help")
        # TODO: Add help menu items

    def setup_status_bar(self) -> None:
        """Set up the status bar."""
        self.status_bar = self.statusBar()
        if self.status_bar is None:
            return  # type: ignore[unreachable]

        # Status message
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def setup_toolbar(self) -> None:
        """Set up the toolbar."""
        toolbar = self.addToolBar("Main Toolbar")
        if toolbar is None:
            return  # type: ignore[unreachable]

        toolbar.setMovable(False)

        # New project
        new_project_action = QAction(
            self.IconManager.get_icon("new"), "New Project", self
        )
        new_project_action.triggered.connect(self.new_project)
        toolbar.addAction(new_project_action)

        # Open project
        open_project_action = QAction(
            self.IconManager.get_icon("open"), "Open Project", self
        )
        open_project_action.triggered.connect(self.open_project)
        toolbar.addAction(open_project_action)

        toolbar.addSeparator()

        # Quick actions
        capture_action = QAction(self.IconManager.get_icon("camera"), "Capture", self)
        annotation_action = QAction(self.IconManager.get_icon("edit"), "Annotate", self)
        training_action = QAction(self.IconManager.get_icon("brain"), "Train", self)

        capture_action.triggered.connect(lambda: self.show_page("capture"))
        toolbar.addAction(capture_action)

        annotation_action.triggered.connect(lambda: self.show_page("annotation"))
        toolbar.addAction(annotation_action)

        training_action.triggered.connect(lambda: self.show_page("training"))
        toolbar.addAction(training_action)

    def show_page(self, page_name: str) -> None:
        """Show a specific page.

        Args:
            page_name: Name of the page to show.
        """
        if page_name in self.pages:
            # Update navigation buttons
            for key, btn in self.nav_buttons.items():
                btn.setChecked(key == page_name)

            # Show the page
            self.content_stack.setCurrentWidget(self.pages[page_name])

            # Update status
            self.status_label.setText(f"Showing {page_name.title()}")

    def new_project(self) -> None:
        """Create a new project."""
        # Show the projects page and trigger new project creation
        self.show_page("projects")
        if hasattr(self.pages["projects"], "new_project"):
            self.pages["projects"].new_project()
            # Refresh project list after creation
            if hasattr(self.pages["projects"], "refresh_project_list"):
                self.pages["projects"].refresh_project_list()

    def open_project(self) -> None:
        """Open an existing project."""
        # Show the projects page and trigger project opening
        self.show_page("projects")
        if hasattr(self.pages["projects"], "open_project"):
            self.pages["projects"].open_project()
            # Update status after opening project
            if self.current_project:
                self.status_label.setText(
                    f"Project opened: {Path(self.current_project).name}"
                )
                # Update window title to show current project
                project_name = Path(self.current_project).name
                self.setWindowTitle(f"DMS - Detection Model Suite - {project_name}")

    def get_current_project(self) -> Optional[str]:
        """Get the current project path.

        Returns:
            Current project path or None if no project is open.
        """
        return self.current_project

    def set_current_project(self, project_path: str) -> None:
        """Set the current project.

        Args:
            project_path: Path to the project directory.
        """
        self.current_project = project_path
        if project_path:
            project_name = Path(project_path).name
            self.setWindowTitle(f"DMS - Detection Model Suite - {project_name}")
            self.status_label.setText(f"Current project: {project_name}")
        else:
            self.setWindowTitle("DMS - Detection Model Suite")
            self.status_label.setText("Ready")

    def refresh_project_list(self) -> None:
        """Refresh the project list in the project manager."""
        if hasattr(self.pages["projects"], "refresh_project_list"):
            self.pages["projects"].refresh_project_list()

    def load_config(self) -> None:
        """Load configuration."""
        # Load from config file
        config_path = self.project_root / "config" / "studio_config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config: {e}")

    def start_system_monitoring(self) -> None:
        """Start the system monitoring."""
        # Create timer for system updates
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self.update_system_info)
        self.system_timer.start(2000)  # Update every 2 seconds

    def update_system_info(self) -> None:
        """Update system information display."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_gb = memory.used / (1024**3)
            self.memory_info_label.setText(
                f"Memory: {memory_percent:.1f}% ({memory_gb:.1f}GB)"
            )
            if not hasattr(self, "_gpu_detected"):
                self._gpu_detected = self.detect_gpu()
                self.update_gpu_label()
        except ImportError:
            self.memory_info_label.setText("Memory: psutil not available")

    def update_gpu_label(self) -> None:
        """Update the GPU status label."""
        if hasattr(self, 'gpu_info_label'):
            if self._gpu_detected:
                if self._gpu_type == "cuda":
                    gpu_text = f"GPU: {self._gpu_name} (CUDA)" if self._gpu_name else "GPU: Available (CUDA)"
                    self.gpu_info_label.setText(gpu_text)
                elif self._gpu_type == "directml":
                    gpu_text = f"GPU: {self._gpu_name} (DirectML)" if self._gpu_name else "GPU: Available (DirectML)"
                    self.gpu_info_label.setText(gpu_text)
                else:
                    gpu_text = f"GPU: {self._gpu_name}" if self._gpu_name else "GPU: Available"
                    self.gpu_info_label.setText(gpu_text)
            else:
                self.gpu_info_label.setText("GPU: CPU Only")

    def detect_gpu(self) -> bool:
        """Detect GPU availability using the hardware detection system.

        Returns:
            True if GPU is available, False otherwise.
        """
        try:
            from utils.hardware import get_hardware_detector
            
            detector = get_hardware_detector()
            specs = detector.detect_hardware()
            
            if specs.device_type in ["cuda", "directml"]:
                self._gpu_type = specs.device_type
                # Get the first GPU name
                if specs.gpus and len(specs.gpus) > 0:
                    self._gpu_name = specs.gpus[0].name
                return True
            else:
                self._gpu_type = None
                self._gpu_name = None
                return False
                
        except Exception as e:
            print(f"GPU detection error: {e}")
            self._gpu_type = None
            self._gpu_name = None
            return False

    def apply_styling(self) -> None:
        """Apply styling to the application."""
        # Get theme from config
        theme = self.config.get("annotation", {}).get("ui", {}).get("theme", "dark")

        if theme == "dark":
            self.setStyleSheet(self.get_dark_style())
        else:
            self.setStyleSheet(self.get_light_style())

    def closeEvent(self, event: Any) -> None:
        """Handle application close event.

        Args:
            event: The close event.
        """
        # Save any unsaved work
        # Stop monitoring
        if hasattr(self, "system_timer"):
            self.system_timer.stop()

        # Close all pages
        for page in self.pages.values():
            if hasattr(page, "cleanup"):
                page.cleanup()

        event.accept()


def main() -> None:
    """Start the main GUI application."""
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
