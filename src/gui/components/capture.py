from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capture Widget

GUI component for screen capture interface.
"""


class CaptureWidget(QWidget):
    """Screen capture interface."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Placeholder content
        label = QLabel("Screen Capture - Coming Soon")
        label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(label)

        # TODO: Implement capture interface
        # - Window selection
        # - Capture settings
        # - Live preview
        # - Recording controls
        # - Save options

    def cleanup(self):
        """Cleanup resources."""
        pass
