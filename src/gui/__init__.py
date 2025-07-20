"""DMS GUI Module.

Modern GUI interface for the Detection Model Suite (DMS).
Provides an intuitive, information-rich interface for managing
object detection projects, training, and annotation workflows.
"""

from .components.annotation import AnnotationWidget
from .components.capture import CaptureWidget
from .components.dashboard import DashboardWidget
from .components.project_manager import ProjectManagerWidget
from .components.settings import SettingsWidget
from .components.system_monitor import SystemMonitorWidget
from .components.training import TrainingWidget
from .main_window import DMSMainWindow

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
