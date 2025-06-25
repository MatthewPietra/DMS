"""
DMS GUI Components

Individual GUI widgets and components for the DMS application.
Each component provides specific functionality for different parts of the pipeline.
"""

from .dashboard import DashboardWidget
from .project_manager import ProjectManagerWidget
from .training import TrainingWidget
from .annotation import AnnotationWidget
from .capture import CaptureWidget
from .system_monitor import SystemMonitorWidget
from .settings import SettingsWidget

__all__ = [
    'DashboardWidget',
    'ProjectManagerWidget',
    'TrainingWidget', 
    'AnnotationWidget',
    'CaptureWidget',
    'SystemMonitorWidget',
    'SettingsWidget'
] 