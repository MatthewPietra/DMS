import json
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from ..utils.icons import IconManager
        import sys
            import psutil

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Widget

The main dashboard providing an overview of the DMS system,
quick actions, and project status.
"""

class DashboardWidget(QWidget):
    """
    Dashboard widget providing overview and quick actions.

    Features:
    - Project overview
    - Quick action buttons
    - System status
    - Recent activity
    - Statistics
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.project_root = main_window.project_root

        self.init_ui()
        self.load_data()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        self.create_header(layout)

        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left column - Quick actions and stats
        left_column = QVBoxLayout()
        left_column.setSpacing(20)

        self.create_quick_actions(left_column)
        self.create_statistics(left_column)

        content_layout.addLayout(left_column, 1)

        # Right column - Recent activity and system info
        right_column = QVBoxLayout()
        right_column.setSpacing(20)

        self.create_recent_activity(right_column)
        self.create_system_overview(right_column)

        content_layout.addLayout(right_column, 1)

        layout.addLayout(content_layout)

    def create_header(self, layout):
        """Create the dashboard header."""
        header_widget = QWidget()
        header_widget.setObjectName("dashboard-header")
        header_widget.setFixedHeight(80)

        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Welcome message
        welcome_label = QLabel("Welcome to DMS")
        welcome_label.setObjectName("welcome-title")

        header_layout.addWidget(welcome_label)
        header_layout.addStretch()

        layout.addWidget(header_widget)

    def create_quick_actions(self, layout):
        """Create quick action buttons."""
        group_box = QGroupBox("Quick Actions")
        group_box.setObjectName("dashboard-group")
        group_box.setMinimumWidth(340)
        group_box.setStyleSheet(
            "QGroupBox::title { color: #2196F3; font-size: 15px; font-weight: bold; subcontrol-origin: margin; subcontrol-position: top center; }"
        )

        group_layout = QVBoxLayout(group_box)
        group_layout.setSpacing(10)

        # Action buttons
        actions = [
            (
                "New Project",
                "new",
                "Create a new object detection project",
                lambda: self.main_window.show_page("projects"),
            ),
            (
                "Screen Capture",
                "camera",
                "Start capturing screen data",
                lambda: self.main_window.show_page("capture"),
            ),
            (
                "Annotate Data",
                "edit",
                "Open annotation interface",
                lambda: self.main_window.show_page("annotation"),
            ),
            (
                "Train Model",
                "brain",
                "Start model training",
                lambda: self.main_window.show_page("training"),
            ),
            (
                "System Monitor",
                "monitor",
                "View system resources",
                lambda: self.main_window.show_page("monitor"),
            ),
        ]

        for text, icon_name, tooltip, callback in actions:
            btn = QPushButton(text)
            btn.setObjectName("quick-action-btn")
            btn.setToolTip(tooltip)
            btn.clicked.connect(callback)

            # Set icon
            btn.setIcon(IconManager.get_icon(icon_name))

            group_layout.addWidget(btn)

        layout.addWidget(group_box)

    def create_statistics(self, layout):
        """Create statistics display."""
        group_box = QGroupBox("Statistics")
        group_box.setObjectName("dashboard-group")
        group_box.setMinimumWidth(340)
        group_box.setStyleSheet(
            "QGroupBox::title { color: #2196F3; font-size: 15px; font-weight: bold; subcontrol-origin: margin; subcontrol-position: top center; }"
        )

        group_layout = QVBoxLayout(group_box)
        group_layout.setSpacing(15)

        # Stats grid
        stats_layout = QGridLayout()
        stats_layout.setSpacing(15)

        # Project count
        self.project_count_label = QLabel("0")
        self.project_count_label.setObjectName("stat-number")
        project_count_title = QLabel("Projects")
        project_count_title.setObjectName("stat-title")

        stats_layout.addWidget(self.project_count_label, 0, 0)
        stats_layout.addWidget(project_count_title, 0, 1)

        # Image count
        self.image_count_label = QLabel("0")
        self.image_count_label.setObjectName("stat-number")
        image_count_title = QLabel("Images")
        image_count_title.setObjectName("stat-title")

        stats_layout.addWidget(self.image_count_label, 0, 2)
        stats_layout.addWidget(image_count_title, 0, 3)

        # Model count
        self.model_count_label = QLabel("0")
        self.model_count_label.setObjectName("stat-number")
        model_count_title = QLabel("Models")
        model_count_title.setObjectName("stat-title")

        stats_layout.addWidget(self.model_count_label, 1, 0)
        stats_layout.addWidget(model_count_title, 1, 1)

        # Annotation count
        self.annotation_count_label = QLabel("0")
        self.annotation_count_label.setObjectName("stat-number")
        annotation_count_title = QLabel("Annotations")
        annotation_count_title.setObjectName("stat-title")

        stats_layout.addWidget(self.annotation_count_label, 1, 2)
        stats_layout.addWidget(annotation_count_title, 1, 3)

        group_layout.addLayout(stats_layout)

        # Refresh button
        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.setObjectName("secondary-btn")
        refresh_btn.clicked.connect(self.load_data)
        group_layout.addWidget(refresh_btn)

        layout.addWidget(group_box)

    def create_recent_activity(self, layout):
        """Create recent activity display."""
        group_box = QGroupBox("Recent Activity")
        group_box.setObjectName("dashboard-group")
        group_box.setMinimumWidth(340)
        group_box.setStyleSheet(
            "QGroupBox::title { color: #2196F3; font-size: 15px; font-weight: bold; subcontrol-origin: margin; subcontrol-position: top center; }"
        )
        group_layout = QVBoxLayout(group_box)
        group_layout.setSpacing(10)
        group_layout.addSpacing(24)
        self.activity_list = QListWidget()
        self.activity_list.setObjectName("activity-list")
        self.activity_list.setAlternatingRowColors(True)
        group_layout.addWidget(self.activity_list)
        clear_btn = QPushButton("Clear History")
        clear_btn.setObjectName("secondary-btn")
        clear_btn.clicked.connect(self.clear_activity_history)
        group_layout.addWidget(clear_btn)
        layout.addWidget(group_box)

    def create_system_overview(self, layout):
        """Create system overview display."""
        group_box = QGroupBox("System Overview")
        group_box.setObjectName("dashboard-group")
        group_box.setMinimumWidth(340)
        group_box.setStyleSheet(
            "QGroupBox::title { color: #2196F3; font-size: 15px; font-weight: bold; subcontrol-origin: margin; subcontrol-position: top center; }"
        )

        group_layout = QVBoxLayout(group_box)
        group_layout.setSpacing(15)

        # System info
        info_layout = QFormLayout()
        info_layout.setSpacing(10)

        # Python version
        python_version = "{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        info_layout.addRow("Python:", QLabel(python_version))

        # GUI framework
        info_layout.addRow("GUI Framework:", QLabel(self.main_window.GUI_FRAMEWORK))

        # Project root
        project_root_label = QLabel(str(self.project_root))
        project_root_label.setWordWrap(True)
        info_layout.addRow("Project Root:", project_root_label)

        # GPU status
        self.gpu_status_label = QLabel("Detecting...")
        info_layout.addRow("GPU Status:", self.gpu_status_label)

        group_layout.addLayout(info_layout)

        # System health indicators
        health_layout = QHBoxLayout()

        # CPU indicator
        self.cpu_indicator = QProgressBar()
        self.cpu_indicator.setMaximum(100)
        self.cpu_indicator.setFormat("CPU: %p%")
        health_layout.addWidget(self.cpu_indicator)

        # Memory indicator
        self.memory_indicator = QProgressBar()
        self.memory_indicator.setMaximum(100)
        self.memory_indicator.setFormat("RAM: %p%")
        health_layout.addWidget(self.memory_indicator)

        group_layout.addLayout(health_layout)

        layout.addWidget(group_box)

    def load_data(self):
        """Load dashboard data."""
        self.load_statistics()
        self.load_recent_activity()
        self.update_system_info()

    def load_statistics(self):
        """Load and display statistics."""
        try:
            # Count projects
            projects_dir = self.project_root / "data" / "projects"
            project_count = 0
            if projects_dir.exists():
                project_count = len([d for d in projects_dir.iterdir() if d.is_dir()])

            # Count images (simplified)
            image_count = 0
            if projects_dir.exists():
                for project_dir in projects_dir.iterdir():
                    if project_dir.is_dir():
                        images_dir = project_dir / "images"
                        if images_dir.exists():
                            image_count += len(list(images_dir.glob("*.jpg")))
                            image_count += len(list(images_dir.glob("*.png")))

            # Count models
            models_dir = self.project_root / "data" / "models"
            model_count = 0
            if models_dir.exists():
                model_count = len(list(models_dir.glob("*.pt")))

            # Count annotations (simplified)
            annotation_count = 0
            if projects_dir.exists():
                for project_dir in projects_dir.iterdir():
                    if project_dir.is_dir():
                        annotations_dir = project_dir / "annotations"
                        if annotations_dir.exists():
                            annotation_count += len(
                                list(annotations_dir.glob("*.json"))
                            )
                            annotation_count += len(list(annotations_dir.glob("*.txt")))

            # Update labels
            self.project_count_label.setText(str(project_count))
            self.image_count_label.setText(str(image_count))
            self.model_count_label.setText(str(model_count))
            self.annotation_count_label.setText(str(annotation_count))

        except Exception as e:
            print("Error loading statistics: {e}")

    def load_recent_activity(self):
        """Load recent activity."""
        self.activity_list.clear()

        try:
            # Load from activity log
            activity_file = self.project_root / "logs" / "activity.log"
            if activity_file.exists():
                with open(activity_file, "r") as f:
                    lines = f.readlines()
                    # Show last 10 activities
                    for line in lines[-10:]:
                        if line.strip():
                            item = QListWidgetItem(line.strip())
                            self.activity_list.addItem(item)
            else:
                # Add some default activities
                activities = [
                    "DMS started successfully",
                    "System initialized",
                    "Ready for new projects",
                ]
                for activity in activities:
                    item = QListWidgetItem(activity)
                    self.activity_list.addItem(item)

        except Exception as e:
            print("Error loading activity: {e}")

    def update_system_info(self):
        """Update system information."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_indicator.setValue(int(cpu_percent))

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_indicator.setValue(int(memory_percent))

            # GPU status
            if self.main_window._gpu_detected:
                self.gpu_status_label.setText("Available (CUDA)")
            else:
                self.gpu_status_label.setText("CPU Only")

        except ImportError:
            self.gpu_status_label.setText("psutil not available")

    def clear_activity_history(self):
        """Clear activity history."""
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear the activity history?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.activity_list.clear()

            # Clear activity log file
            activity_file = self.project_root / "logs" / "activity.log"
            if activity_file.exists():
                activity_file.unlink()

    def add_activity(self, activity: str):
        """Add a new activity to the history."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        activity_text = "[{timestamp}] {activity}"

        # Add to list
        item = QListWidgetItem(activity_text)
        self.activity_list.insertItem(0, item)  # Insert at top

        # Keep only last 50 items
        while self.activity_list.count() > 50:
            self.activity_list.takeItem(self.activity_list.count() - 1)

        # Save to file
        try:
            activity_file = self.project_root / "logs" / "activity.log"
            activity_file.parent.mkdir(exist_ok=True)

            with open(activity_file, "a") as f:
                f.write(activity_text + "\n")

        except Exception as e:
            print("Error saving activity: {e}")

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "time_timer"):
            self.time_timer.stop()
