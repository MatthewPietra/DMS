"""
DMS GUI Module

Modern GUI interface for the Detection Model Suite (DMS).
Provides an intuitive, information-rich interface for managing
object detection projects, training, and annotation workflows.
"""

from .components import *
from .main_window import DMSMainWindow
from .utils import *

__all__ = [
    "DMSMainWindow",
    "DashboardWidget",
    "ProjectManagerWidget",
    "TrainingWidget",
    "AnnotationWidget",
    "CaptureWidget",
    "SystemMonitorWidget",
    "SettingsWidget",
]
