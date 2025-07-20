#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings Widget.

GUI component for application settings interface.
"""

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...utils.config import ConfigManager


class SettingsWidget(QWidget):
    """Application settings interface."""

    def __init__(self, main_window: Any) -> None:
        """
        Initialize the settings widget.

        Args:
            main_window: The main window instance.
        """
        super().__init__()
        self.main_window = main_window
        self.config_manager = ConfigManager()

        # Initialize UI components
        self.tab_widget: QTabWidget
        self.general_tab: QWidget
        self.hardware_tab: QWidget
        self.training_tab: QWidget
        self.annotation_tab: QWidget
        self.capture_tab: QWidget
        self.theme_tab: QWidget

        # Settings controls
        self.debug_mode_checkbox: QCheckBox
        self.log_level_combo: QComboBox
        self.auto_save_interval_spin: QSpinBox
        self.max_projects_spin: QSpinBox

        # Hardware settings
        self.auto_detect_gpu_checkbox: QCheckBox
        self.preferred_device_combo: QComboBox
        self.gpu_memory_slider: QSlider
        self.cpu_threads_spin: QSpinBox

        # Training settings
        self.default_epochs_spin: QSpinBox
        self.default_batch_size_spin: QSpinBox
        self.default_image_size_spin: QSpinBox
        self.min_map50_spin: QSpinBox
        self.min_precision_spin: QSpinBox
        self.min_recall_spin: QSpinBox

        # Annotation settings
        self.annotation_theme_combo: QComboBox
        self.font_size_spin: QSpinBox
        self.zoom_sensitivity_spin: QSpinBox
        self.validation_checkbox: QCheckBox
        self.min_box_size_spin: QSpinBox

        # Capture settings
        self.default_fps_spin: QSpinBox
        self.default_resolution_combo: QComboBox
        self.image_format_combo: QComboBox
        self.jpeg_quality_spin: QSpinBox
        self.preview_checkbox: QCheckBox

        # Theme settings
        self.theme_combo: QComboBox
        self.apply_theme_btn: QPushButton

        # Save/Reset buttons
        self.save_btn: QPushButton
        self.reset_btn: QPushButton

        self.init_ui()
        self.load_settings()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        self.create_header(layout)

        # Create scrollable content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("settings-tabs")

        # Create tabs
        self.create_general_tab()
        self.create_hardware_tab()
        self.create_training_tab()
        self.create_annotation_tab()
        self.create_capture_tab()
        self.create_theme_tab()

        scroll_area.setWidget(self.tab_widget)
        layout.addWidget(scroll_area)

        # Create action buttons
        self.create_action_buttons(layout)

    def create_header(self, layout: QVBoxLayout) -> None:
        """
        Create the settings header.

        Args:
            layout: The parent layout to add the header to.
        """
        header_widget = QWidget()
        header_widget.setObjectName("settings-header")
        header_widget.setFixedHeight(80)

        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title_label = QLabel("Settings")
        title_label.setObjectName("settings-title")

        # Subtitle
        subtitle_label = QLabel("Configure application preferences and behavior")
        subtitle_label.setObjectName("settings-subtitle")

        title_layout = QVBoxLayout()
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        layout.addWidget(header_widget)

    def create_general_tab(self) -> None:
        """Create the general settings tab."""
        self.general_tab = QWidget()
        layout = QVBoxLayout(self.general_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Studio Settings
        studio_group = QGroupBox("Studio Settings")
        studio_group.setObjectName("settings-group")
        studio_layout = QFormLayout(studio_group)

        # Debug mode
        self.debug_mode_checkbox = QCheckBox("Enable debug mode")
        self.debug_mode_checkbox.setToolTip(
            "Enable detailed logging and debug features"
        )
        studio_layout.addRow("Debug Mode:", self.debug_mode_checkbox)

        # Log level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setToolTip("Set the logging level for the application")
        studio_layout.addRow("Log Level:", self.log_level_combo)

        # Auto save interval
        self.auto_save_interval_spin = QSpinBox()
        self.auto_save_interval_spin.setRange(30, 3600)
        self.auto_save_interval_spin.setSuffix(" seconds")
        self.auto_save_interval_spin.setToolTip("Interval for automatic project saving")
        studio_layout.addRow("Auto Save Interval:", self.auto_save_interval_spin)

        # Max concurrent projects
        self.max_projects_spin = QSpinBox()
        self.max_projects_spin.setRange(1, 20)
        self.max_projects_spin.setToolTip("Maximum number of concurrent projects")
        studio_layout.addRow("Max Concurrent Projects:", self.max_projects_spin)

        layout.addWidget(studio_group)
        layout.addStretch()

        self.tab_widget.addTab(self.general_tab, "General")

    def create_hardware_tab(self) -> None:
        """Create the hardware settings tab."""
        self.hardware_tab = QWidget()
        layout = QVBoxLayout(self.hardware_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Hardware Detection
        detection_group = QGroupBox("Hardware Detection")
        detection_group.setObjectName("settings-group")
        detection_layout = QFormLayout(detection_group)

        # Auto detect GPU
        self.auto_detect_gpu_checkbox = QCheckBox("Auto-detect GPU")
        self.auto_detect_gpu_checkbox.setToolTip(
            "Automatically detect and configure GPU settings"
        )
        detection_layout.addRow("Auto Detect GPU:", self.auto_detect_gpu_checkbox)

        # Preferred device
        self.preferred_device_combo = QComboBox()
        self.preferred_device_combo.addItems(["auto", "cuda", "directml", "cpu"])
        self.preferred_device_combo.setToolTip(
            "Preferred computing device for training"
        )
        detection_layout.addRow("Preferred Device:", self.preferred_device_combo)

        layout.addWidget(detection_group)

        # Performance Settings
        performance_group = QGroupBox("Performance Settings")
        performance_group.setObjectName("settings-group")
        performance_layout = QFormLayout(performance_group)

        # GPU memory fraction
        self.gpu_memory_slider = QSlider(Qt.Orientation.Horizontal)
        self.gpu_memory_slider.setRange(10, 100)
        self.gpu_memory_slider.setToolTip("Fraction of GPU memory to use (10-100%)")
        performance_layout.addRow("GPU Memory Fraction (%):", self.gpu_memory_slider)

        # CPU threads
        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setRange(-1, 32)
        self.cpu_threads_spin.setSpecialValueText("Auto")
        self.cpu_threads_spin.setToolTip("Number of CPU threads (-1 for auto-detect)")
        performance_layout.addRow("CPU Threads:", self.cpu_threads_spin)

        layout.addWidget(performance_group)
        layout.addStretch()

        self.tab_widget.addTab(self.hardware_tab, "Hardware")

    def create_training_tab(self) -> None:
        """Create the training settings tab."""
        self.training_tab = QWidget()
        layout = QVBoxLayout(self.training_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Default Training Parameters
        defaults_group = QGroupBox("Default Training Parameters")
        defaults_group.setObjectName("settings-group")
        defaults_layout = QFormLayout(defaults_group)

        # Default epochs
        self.default_epochs_spin = QSpinBox()
        self.default_epochs_spin.setRange(10, 1000)
        self.default_epochs_spin.setToolTip("Default number of training epochs")
        defaults_layout.addRow("Default Epochs:", self.default_epochs_spin)

        # Default batch size
        self.default_batch_size_spin = QSpinBox()
        self.default_batch_size_spin.setRange(-1, 128)
        self.default_batch_size_spin.setSpecialValueText("Auto")
        self.default_batch_size_spin.setToolTip(
            "Default batch size (-1 for auto-calculate)"
        )
        defaults_layout.addRow("Default Batch Size:", self.default_batch_size_spin)

        # Default image size
        self.default_image_size_spin = QSpinBox()
        self.default_image_size_spin.setRange(320, 1280)
        self.default_image_size_spin.setSuffix(" px")
        self.default_image_size_spin.setToolTip("Default input image size")
        defaults_layout.addRow("Default Image Size:", self.default_image_size_spin)

        layout.addWidget(defaults_group)

        # Performance Thresholds
        thresholds_group = QGroupBox("Performance Thresholds")
        thresholds_group.setObjectName("settings-group")
        thresholds_layout = QFormLayout(thresholds_group)

        # Min mAP50
        self.min_map50_spin = QSpinBox()
        self.min_map50_spin.setRange(50, 100)
        self.min_map50_spin.setSuffix("%")
        self.min_map50_spin.setToolTip("Minimum mAP@0.5 threshold for model acceptance")
        thresholds_layout.addRow("Min mAP@0.5:", self.min_map50_spin)

        # Min precision
        self.min_precision_spin = QSpinBox()
        self.min_precision_spin.setRange(50, 100)
        self.min_precision_spin.setSuffix("%")
        self.min_precision_spin.setToolTip("Minimum precision threshold")
        thresholds_layout.addRow("Min Precision:", self.min_precision_spin)

        # Min recall
        self.min_recall_spin = QSpinBox()
        self.min_recall_spin.setRange(50, 100)
        self.min_recall_spin.setSuffix("%")
        self.min_recall_spin.setToolTip("Minimum recall threshold")
        thresholds_layout.addRow("Min Recall:", self.min_recall_spin)

        layout.addWidget(thresholds_group)
        layout.addStretch()

        self.tab_widget.addTab(self.training_tab, "Training")

    def create_annotation_tab(self) -> None:
        """Create the annotation settings tab."""
        self.annotation_tab = QWidget()
        layout = QVBoxLayout(self.annotation_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # UI Settings
        ui_group = QGroupBox("Interface Settings")
        ui_group.setObjectName("settings-group")
        ui_layout = QFormLayout(ui_group)

        # Theme
        self.annotation_theme_combo = QComboBox()
        self.annotation_theme_combo.addItems(["dark", "light"])
        self.annotation_theme_combo.setToolTip("Annotation interface theme")
        ui_layout.addRow("Theme:", self.annotation_theme_combo)

        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setSuffix(" px")
        self.font_size_spin.setToolTip("Font size for annotation interface")
        ui_layout.addRow("Font Size:", self.font_size_spin)

        # Zoom sensitivity
        self.zoom_sensitivity_spin = QSpinBox()
        self.zoom_sensitivity_spin.setRange(1, 50)
        self.zoom_sensitivity_spin.setSuffix("%")
        self.zoom_sensitivity_spin.setToolTip("Zoom sensitivity for annotation tools")
        ui_layout.addRow("Zoom Sensitivity:", self.zoom_sensitivity_spin)

        layout.addWidget(ui_group)

        # Quality Settings
        quality_group = QGroupBox("Quality Settings")
        quality_group.setObjectName("settings-group")
        quality_layout = QFormLayout(quality_group)

        # Validation
        self.validation_checkbox = QCheckBox("Enable validation")
        self.validation_checkbox.setToolTip("Enable annotation quality validation")
        quality_layout.addRow("Enable Validation:", self.validation_checkbox)

        # Min box size
        self.min_box_size_spin = QSpinBox()
        self.min_box_size_spin.setRange(1, 100)
        self.min_box_size_spin.setSuffix(" px")
        self.min_box_size_spin.setToolTip("Minimum bounding box size")
        quality_layout.addRow("Min Box Size:", self.min_box_size_spin)

        layout.addWidget(quality_group)
        layout.addStretch()

        self.tab_widget.addTab(self.annotation_tab, "Annotation")

    def create_capture_tab(self) -> None:
        """Create the capture settings tab."""
        self.capture_tab = QWidget()
        layout = QVBoxLayout(self.capture_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Capture Settings
        capture_group = QGroupBox("Capture Settings")
        capture_group.setObjectName("settings-group")
        capture_layout = QFormLayout(capture_group)

        # Default FPS
        self.default_fps_spin = QSpinBox()
        self.default_fps_spin.setRange(1, 30)
        self.default_fps_spin.setSuffix(" fps")
        self.default_fps_spin.setToolTip("Default capture frame rate")
        capture_layout.addRow("Default FPS:", self.default_fps_spin)

        # Default resolution
        self.default_resolution_combo = QComboBox()
        self.default_resolution_combo.addItems(["320x320", "640x640", "1280x1280"])
        self.default_resolution_combo.setToolTip("Default capture resolution")
        capture_layout.addRow("Default Resolution:", self.default_resolution_combo)

        # Image format
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["PNG", "JPEG", "BMP"])
        self.image_format_combo.setToolTip("Default image format for captures")
        capture_layout.addRow("Image Format:", self.image_format_combo)

        # JPEG quality
        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(50, 100)
        self.jpeg_quality_spin.setSuffix("%")
        self.jpeg_quality_spin.setToolTip("JPEG compression quality")
        capture_layout.addRow("JPEG Quality:", self.jpeg_quality_spin)

        layout.addWidget(capture_group)

        # Preview Settings
        preview_group = QGroupBox("Preview Settings")
        preview_group.setObjectName("settings-group")
        preview_layout = QFormLayout(preview_group)

        # Preview enabled
        self.preview_checkbox = QCheckBox("Enable preview")
        self.preview_checkbox.setToolTip("Enable real-time capture preview")
        preview_layout.addRow("Enable Preview:", self.preview_checkbox)

        layout.addWidget(preview_group)
        layout.addStretch()

        self.tab_widget.addTab(self.capture_tab, "Capture")

    def create_theme_tab(self) -> None:
        """Create the theme settings tab."""
        self.theme_tab = QWidget()
        layout = QVBoxLayout(self.theme_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Theme Selection
        theme_group = QGroupBox("Theme Selection")
        theme_group.setObjectName("settings-group")
        theme_layout = QFormLayout(theme_group)

        # Theme combo
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setToolTip("Application theme")
        theme_layout.addRow("Theme:", self.theme_combo)

        # Apply theme button
        self.apply_theme_btn = QPushButton("Apply Theme")
        self.apply_theme_btn.setObjectName("primary-btn")
        self.apply_theme_btn.clicked.connect(self.apply_theme)
        theme_layout.addRow("", self.apply_theme_btn)

        layout.addWidget(theme_group)
        layout.addStretch()

        self.tab_widget.addTab(self.theme_tab, "Theme")

    def create_action_buttons(self, layout: QVBoxLayout) -> None:
        """
        Create action buttons for saving and resetting settings.

        Args:
            layout: The parent layout to add the buttons to.
        """
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Reset button
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setObjectName("secondary-btn")
        self.reset_btn.clicked.connect(self.reset_settings)
        button_layout.addWidget(self.reset_btn)

        # Save button
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setObjectName("primary-btn")
        self.save_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

    def load_settings(self) -> None:
        """Load current settings into the UI controls."""
        try:
            # General settings
            studio_config = self.config_manager.get_studio_config()
            self.debug_mode_checkbox.setChecked(studio_config.debug_mode)
            self.log_level_combo.setCurrentText(studio_config.log_level)
            self.auto_save_interval_spin.setValue(studio_config.auto_save_interval)
            self.max_projects_spin.setValue(studio_config.max_concurrent_projects)

            # Hardware settings
            hardware_config = self.config_manager.get_hardware_config()
            self.auto_detect_gpu_checkbox.setChecked(hardware_config.auto_detect_gpu)
            self.preferred_device_combo.setCurrentText(hardware_config.preferred_device)
            self.gpu_memory_slider.setValue(
                int(hardware_config.gpu_memory_fraction * 100)
            )
            self.cpu_threads_spin.setValue(hardware_config.cpu_threads)

            # Training settings
            training_config = self.config_manager.get_training_config()
            self.default_epochs_spin.setValue(training_config.epochs)
            self.default_batch_size_spin.setValue(training_config.batch_size)
            self.default_image_size_spin.setValue(training_config.image_size)
            self.min_map50_spin.setValue(int(training_config.min_map50 * 100))
            self.min_precision_spin.setValue(int(training_config.min_precision * 100))
            self.min_recall_spin.setValue(int(training_config.min_recall * 100))

            # Annotation settings
            annotation_config = self.config_manager.get_annotation_config()
            self.annotation_theme_combo.setCurrentText(annotation_config.theme)
            self.font_size_spin.setValue(annotation_config.font_size)
            self.zoom_sensitivity_spin.setValue(
                int(annotation_config.zoom_sensitivity * 100)
            )
            self.validation_checkbox.setChecked(annotation_config.enable_validation)
            self.min_box_size_spin.setValue(annotation_config.min_box_size)

            # Capture settings
            capture_config = self.config_manager.get_capture_config()
            self.default_fps_spin.setValue(capture_config.default_fps)
            if capture_config.default_resolution is not None:
                resolution_text = (
                    f"{capture_config.default_resolution[0]}x"
                    f"{capture_config.default_resolution[1]}"
                )
                self.default_resolution_combo.setCurrentText(resolution_text)
            self.image_format_combo.setCurrentText(capture_config.image_format)
            self.jpeg_quality_spin.setValue(capture_config.jpeg_quality)
            if capture_config.preview is not None:
                self.preview_checkbox.setChecked(
                    capture_config.preview.get("enabled", True)
                )

            # Theme settings
            self.theme_combo.setCurrentText("dark")  # Default theme

        except Exception as e:
            logging.error(f"Error loading settings: {e}")

    def save_settings(self) -> None:
        """Save current settings from UI controls."""
        try:
            # General settings
            self.config_manager.set(
                "studio.debug_mode", self.debug_mode_checkbox.isChecked()
            )
            self.config_manager.set(
                "studio.log_level", self.log_level_combo.currentText()
            )
            self.config_manager.set(
                "studio.auto_save_interval", self.auto_save_interval_spin.value()
            )
            self.config_manager.set(
                "studio.max_concurrent_projects", self.max_projects_spin.value()
            )

            # Hardware settings
            self.config_manager.set(
                "hardware.auto_detect_gpu", self.auto_detect_gpu_checkbox.isChecked()
            )
            self.config_manager.set(
                "hardware.preferred_device",
                self.preferred_device_combo.currentText(),
            )
            self.config_manager.set(
                "hardware.gpu_memory_fraction",
                self.gpu_memory_slider.value() / 100.0,
            )
            self.config_manager.set(
                "hardware.cpu_threads", self.cpu_threads_spin.value()
            )

            # Training settings
            self.config_manager.set("training.epochs", self.default_epochs_spin.value())
            self.config_manager.set(
                "training.batch_size", self.default_batch_size_spin.value()
            )
            self.config_manager.set(
                "training.image_size", self.default_image_size_spin.value()
            )
            self.config_manager.set(
                "training.min_map50", self.min_map50_spin.value() / 100.0
            )
            self.config_manager.set(
                "training.min_precision", self.min_precision_spin.value() / 100.0
            )
            self.config_manager.set(
                "training.min_recall", self.min_recall_spin.value() / 100.0
            )

            # Annotation settings
            self.config_manager.set(
                "annotation.theme",
                self.annotation_theme_combo.currentText(),
            )
            self.config_manager.set("annotation.font_size", self.font_size_spin.value())
            self.config_manager.set(
                "annotation.zoom_sensitivity",
                self.zoom_sensitivity_spin.value() / 100.0,
            )
            self.config_manager.set(
                "annotation.enable_validation", self.validation_checkbox.isChecked()
            )
            self.config_manager.set(
                "annotation.min_box_size", self.min_box_size_spin.value()
            )

            # Capture settings
            self.config_manager.set(
                "capture.default_fps", self.default_fps_spin.value()
            )
            resolution = self.default_resolution_combo.currentText().split("x")
            self.config_manager.set(
                "capture.default_resolution",
                [int(resolution[0]), int(resolution[1])],
            )
            self.config_manager.set(
                "capture.image_format", self.image_format_combo.currentText()
            )
            self.config_manager.set(
                "capture.jpeg_quality", self.jpeg_quality_spin.value()
            )
            self.config_manager.set(
                "capture.preview.enabled", self.preview_checkbox.isChecked()
            )

            # Save configuration
            self.config_manager.save_config()

            # Show success message
            self.show_success_message(
                "Settings have been saved successfully!\n\n"
                "Your changes will take effect immediately."
            )

        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            self.show_error_message(f"Error saving settings: {e}")

    def reset_settings(self) -> None:
        """Reset all settings to default values."""
        # Show confirmation dialog
        if not self.show_confirm_dialog(
            "Reset Settings",
            "Are you sure you want to reset all settings to their default values?\n\n"
            "This action cannot be undone.",
        ):
            return

        try:
            # Reset to default configuration
            self.config_manager = ConfigManager()
            self.load_settings()
            self.show_success_message(
                "Settings have been reset to their default values!"
            )

        except Exception as e:
            logging.error(f"Error resetting settings: {e}")
            self.show_error_message(f"Error resetting settings: {e}")

    def apply_theme(self) -> None:
        """Apply the selected theme to the application."""
        try:
            theme = self.theme_combo.currentText()

            # Apply styling if available
            if hasattr(self.main_window, "apply_styling"):
                self.main_window.apply_styling()

            # Show success message with theme name
            self.show_success_message(
                f"Theme '{theme.capitalize()}' has been applied successfully!\n\n"
                f"The new theme is now active throughout the application."
            )

            # Update status bar if available
            if hasattr(self.main_window, "statusBar"):
                self.main_window.statusBar().showMessage(
                    f"Theme changed to: {theme.capitalize()}", 3000
                )

        except Exception as e:
            logging.error(f"Error applying theme: {e}")
            self.show_error_message(f"Error applying theme: {e}")

    def show_success_message(self, message: str) -> None:
        """
        Show a success message to the user.

        Args:
            message: The message to display.
        """
        # Show message box
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Settings")
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

        # Also log for debugging
        logging.info(f"Settings: {message}")

        # Show status bar message if available
        if hasattr(self.main_window, "statusBar"):
            self.main_window.statusBar().showMessage(
                message, 3000
            )  # Show for 3 seconds

    def show_error_message(self, message: str) -> None:
        """
        Show an error message to the user.

        Args:
            message: The message to display.
        """
        # Show error message box
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Settings Error")
        msg_box.setText("An error occurred while processing settings.")
        msg_box.setInformativeText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

        # Also log for debugging
        logging.error(f"Settings Error: {message}")

        # Show status bar message if available
        if hasattr(self.main_window, "statusBar"):
            self.main_window.statusBar().showMessage(
                f"Error: {message}", 5000
            )  # Show for 5 seconds

    def show_confirm_dialog(self, title: str, message: str) -> bool:
        """
        Show a confirmation dialog to the user.

        Args:
            title: The dialog title.
            message: The confirmation message.

        Returns:
            True if user confirms, False otherwise.
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        result = msg_box.exec()
        return result == QMessageBox.StandardButton.Yes

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
