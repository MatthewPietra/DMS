#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS GUI Styles

Modern styling for the DMS GUI application.
Provides dark and light themes with consistent design.
"""


def get_dark_style():
    """Get the dark theme stylesheet."""
    return """
    /* Main Application */
    QMainWindow {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Sidebar */
    #sidebar {
        background-color: #2d2d2d;
        border-right: 1px solid #404040;
    }

    #logo-area {
        background-color: #1e1e1e;
        border-bottom: 1px solid #404040;
    }

    #app-title {
        color: #4a9eff;
        font-size: 18px;
        font-weight: bold;
    }

    #app-subtitle {
        color: #888888;
        font-size: 12px;
    }

    /* Navigation Buttons */
    QPushButton#nav-dashboard,
    QPushButton#nav-projects,
    QPushButton#nav-capture,
    QPushButton#nav-annotation,
    QPushButton#nav-training,
    QPushButton#nav-monitor,
    QPushButton#nav-settings {
        background-color: transparent;
        border: none;
        border-radius: 8px;
        padding: 8px;
        text-align: left;
        color: #cccccc;
        font-size: 14px;
    }

    QPushButton#nav-dashboard:hover,
    QPushButton#nav-projects:hover,
    QPushButton#nav-capture:hover,
    QPushButton#nav-annotation:hover,
    QPushButton#nav-training:hover,
    QPushButton#nav-monitor:hover,
    QPushButton#nav-settings:hover {
        background-color: #404040;
        color: #ffffff;
    }

    QPushButton#nav-dashboard:checked,
    QPushButton#nav-projects:checked,
    QPushButton#nav-capture:checked,
    QPushButton#nav-annotation:checked,
    QPushButton#nav-training:checked,
    QPushButton#nav-monitor:checked,
    QPushButton#nav-settings:checked {
        background-color: #4a9eff;
        color: #ffffff;
    }

    #nav-text {
        color: inherit;
        font-size: 14px;
        font-weight: 500;
    }

    /* System Info Area */
    #system-info {
        background-color: #1e1e1e;
        border-top: 1px solid #404040;
    }

    #system-status {
        color: #4a9eff;
        font-weight: bold;
        font-size: 12px;
    }

    #system-info-text {
        color: #888888;
        font-size: 11px;
    }

    /* Content Area */
    #content-stack {
        background-color: #1e1e1e;
    }

    /* Dashboard */
    #dashboard-header {
        background-color: #2d2d2d;
        border-bottom: 1px solid #404040;
        border-radius: 8px;
    }

    #welcome-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: bold;
    }

    #time-label {
        color: #888888;
        font-size: 14px;
    }

    #dashboard-group {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 8px;
        color: #ffffff;
        font-weight: bold;
    }

    QGroupBox::title {
        color: #4a9eff;
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
    }

    /* Quick Action Buttons */
    #quick-action-btn {
        background-color: #404040;
        border: 1px solid #555555;
        border-radius: 6px;
        padding: 12px;
        color: #ffffff;
        font-size: 14px;
        font-weight: 500;
        text-align: left;
    }

    #quick-action-btn:hover {
        background-color: #4a9eff;
        border-color: #4a9eff;
    }

    #quick-action-btn:pressed {
        background-color: #3a7ecc;
    }

    /* Statistics */
    #stat-number {
        color: #4a9eff;
        font-size: 24px;
        font-weight: bold;
    }

    #stat-title {
        color: #888888;
        font-size: 14px;
    }

    /* Activity List */
    #activity-list {
        background-color: #1e1e1e;
        border: 1px solid #404040;
        border-radius: 6px;
        color: #cccccc;
        alternate-background-color: #2d2d2d;
    }

    #activity-list::item {
        padding: 8px;
        border-bottom: 1px solid #404040;
    }

    #activity-list::item:selected {
        background-color: #4a9eff;
        color: #ffffff;
    }

    /* Buttons */
    #secondary-btn {
        background-color: #404040;
        border: 1px solid #555555;
        border-radius: 6px;
        padding: 8px 16px;
        color: #ffffff;
        font-size: 12px;
    }

    #secondary-btn:hover {
        background-color: #555555;
        border-color: #666666;
    }

    /* Progress Bars */
    QProgressBar {
        border: 1px solid #404040;
        border-radius: 4px;
        text-align: center;
        background-color: #1e1e1e;
        color: #ffffff;
    }

    QProgressBar::chunk {
        background-color: #4a9eff;
        border-radius: 3px;
    }

    /* Form Layout */
    QFormLayout {
        color: #cccccc;
    }

    QLabel {
        color: #cccccc;
    }

    /* Menu Bar */
    QMenuBar {
        background-color: #2d2d2d;
        color: #ffffff;
        border-bottom: 1px solid #404040;
    }

    QMenuBar::item {
        background-color: transparent;
        padding: 8px 12px;
    }

    QMenuBar::item:selected {
        background-color: #404040;
    }

    QMenu {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        color: #ffffff;
    }

    QMenu::item {
        padding: 8px 20px;
    }

    QMenu::item:selected {
        background-color: #4a9eff;
    }

    /* Toolbar */
    QToolBar {
        background-color: #2d2d2d;
        border-bottom: 1px solid #404040;
        spacing: 5px;
        padding: 5px;
    }

    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 6px;
    }

    QToolButton:hover {
        background-color: #404040;
        border-color: #555555;
    }

    QToolButton:pressed {
        background-color: #4a9eff;
    }

    /* Status Bar */
    QStatusBar {
        background-color: #2d2d2d;
        color: #cccccc;
        border-top: 1px solid #404040;
    }

    /* Scrollbars */
    QScrollBar:vertical {
        background-color: #1e1e1e;
        width: 12px;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical {
        background-color: #404040;
        border-radius: 6px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #555555;
    }

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0px;
    }

    /* Input Fields */
    QLineEdit {
        background-color: #1e1e1e;
        border: 1px solid #404040;
        border-radius: 4px;
        padding: 8px;
        color: #ffffff;
    }

    QLineEdit:focus {
        border-color: #4a9eff;
    }

    /* Combo Boxes */
    QComboBox {
        background-color: #1e1e1e;
        border: 1px solid #404040;
        border-radius: 4px;
        padding: 8px;
        color: #ffffff;
    }

    QComboBox::drop-down {
        border: none;
        width: 20px;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #888888;
    }

    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        color: #ffffff;
        selection-background-color: #4a9eff;
    }
    """


def get_light_style():
    """Get the light theme stylesheet."""
    return """
    /* Main Application */
    QMainWindow {
        background-color: #f5f5f5;
        color: #333333;
    }

    /* Sidebar */
    #sidebar {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    #logo-area {
        background-color: #f8f9fa;
        border-bottom: 1px solid #e0e0e0;
    }

    #app-title {
        color: #007bff;
        font-size: 18px;
        font-weight: bold;
    }

    #app-subtitle {
        color: #6c757d;
        font-size: 12px;
    }

    /* Navigation Buttons */
    QPushButton#nav-dashboard,
    QPushButton#nav-projects,
    QPushButton#nav-capture,
    QPushButton#nav-annotation,
    QPushButton#nav-training,
    QPushButton#nav-monitor,
    QPushButton#nav-settings {
        background-color: transparent;
        border: none;
        border-radius: 8px;
        padding: 8px;
        text-align: left;
        color: #495057;
        font-size: 14px;
    }

    QPushButton#nav-dashboard:hover,
    QPushButton#nav-projects:hover,
    QPushButton#nav-capture:hover,
    QPushButton#nav-annotation:hover,
    QPushButton#nav-training:hover,
    QPushButton#nav-monitor:hover,
    QPushButton#nav-settings:hover {
        background-color: #e9ecef;
        color: #212529;
    }

    QPushButton#nav-dashboard:checked,
    QPushButton#nav-projects:checked,
    QPushButton#nav-capture:checked,
    QPushButton#nav-annotation:checked,
    QPushButton#nav-training:checked,
    QPushButton#nav-monitor:checked,
    QPushButton#nav-settings:checked {
        background-color: #007bff;
        color: #ffffff;
    }

    #nav-text {
        color: inherit;
        font-size: 14px;
        font-weight: 500;
    }

    /* System Info Area */
    #system-info {
        background-color: #f8f9fa;
        border-top: 1px solid #e0e0e0;
    }

    #system-status {
        color: #007bff;
        font-weight: bold;
        font-size: 12px;
    }

    #system-info-text {
        color: #6c757d;
        font-size: 11px;
    }

    /* Content Area */
    #content-stack {
        background-color: #f5f5f5;
    }

    /* Dashboard */
    #dashboard-header {
        background-color: #ffffff;
        border-bottom: 1px solid #e0e0e0;
        border-radius: 8px;
    }

    #welcome-title {
        color: #212529;
        font-size: 24px;
        font-weight: bold;
    }

    #time-label {
        color: #6c757d;
        font-size: 14px;
    }

    #dashboard-group {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        color: #212529;
        font-weight: bold;
    }

    QGroupBox::title {
        color: #007bff;
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
    }

    /* Quick Action Buttons */
    #quick-action-btn {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 12px;
        color: #495057;
        font-size: 14px;
        font-weight: 500;
        text-align: left;
    }

    #quick-action-btn:hover {
        background-color: #007bff;
        border-color: #007bff;
        color: #ffffff;
    }

    #quick-action-btn:pressed {
        background-color: #0056b3;
    }

    /* Statistics */
    #stat-number {
        color: #007bff;
        font-size: 24px;
        font-weight: bold;
    }

    #stat-title {
        color: #6c757d;
        font-size: 14px;
    }

    /* Activity List */
    #activity-list {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        color: #495057;
        alternate-background-color: #f8f9fa;
    }

    #activity-list::item {
        padding: 8px;
        border-bottom: 1px solid #e0e0e0;
    }

    #activity-list::item:selected {
        background-color: #007bff;
        color: #ffffff;
    }

    /* Buttons */
    #secondary-btn {
        background-color: #6c757d;
        border: 1px solid #6c757d;
        border-radius: 6px;
        padding: 8px 16px;
        color: #ffffff;
        font-size: 12px;
    }

    #secondary-btn:hover {
        background-color: #5a6268;
        border-color: #545b62;
    }

    /* Progress Bars */
    QProgressBar {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        text-align: center;
        background-color: #f8f9fa;
        color: #495057;
    }

    QProgressBar::chunk {
        background-color: #007bff;
        border-radius: 3px;
    }

    /* Form Layout */
    QFormLayout {
        color: #495057;
    }

    QLabel {
        color: #495057;
    }

    /* Menu Bar */
    QMenuBar {
        background-color: #ffffff;
        color: #495057;
        border-bottom: 1px solid #e0e0e0;
    }

    QMenuBar::item {
        background-color: transparent;
        padding: 8px 12px;
    }

    QMenuBar::item:selected {
        background-color: #e9ecef;
    }

    QMenu {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        color: #495057;
    }

    QMenu::item {
        padding: 8px 20px;
    }

    QMenu::item:selected {
        background-color: #007bff;
        color: #ffffff;
    }

    /* Toolbar */
    QToolBar {
        background-color: #ffffff;
        border-bottom: 1px solid #e0e0e0;
        spacing: 5px;
        padding: 5px;
    }

    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 6px;
    }

    QToolButton:hover {
        background-color: #e9ecef;
        border-color: #dee2e6;
    }

    QToolButton:pressed {
        background-color: #007bff;
        color: #ffffff;
    }

    /* Status Bar */
    QStatusBar {
        background-color: #ffffff;
        color: #495057;
        border-top: 1px solid #e0e0e0;
    }

    /* Scrollbars */
    QScrollBar:vertical {
        background-color: #f8f9fa;
        width: 12px;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical {
        background-color: #dee2e6;
        border-radius: 6px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #adb5bd;
    }

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0px;
    }

    /* Input Fields */
    QLineEdit {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 8px;
        color: #495057;
    }

    QLineEdit:focus {
        border-color: #007bff;
    }

    /* Combo Boxes */
    QComboBox {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 8px;
        color: #495057;
    }

    QComboBox::drop-down {
        border: none;
        width: 20px;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #6c757d;
    }

    QComboBox QAbstractItemView {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        color: #495057;
        selection-background-color: #007bff;
        selection-color: #ffffff;
    }
    """
