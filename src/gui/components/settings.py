#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings Widget

GUI component for application settings interface.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class SettingsWidget(QWidget):
    """Application settings interface."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Placeholder content
        label = QLabel("Settings - Coming Soon")
        label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(label)

        # TODO: Implement settings interface
        # - General settings
        # - Hardware settings
        # - Training settings
        # - Annotation settings
        # - Capture settings
        # - Theme selection

    def cleanup(self):
        """Cleanup resources."""
        pass
