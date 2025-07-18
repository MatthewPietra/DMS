"""YOLO Vision Studio - Training Module.

This module provides comprehensive YOLO model training capabilities including:
- Multi-GPU training support with automatic device detection
- Cross-platform compatibility (CUDA, DirectML, CPU)
- Advanced training configuration and optimization
- Model management and versioning
- Comprehensive metrics tracking and validation
"""

from .yolo_trainer import (
    ModelManager,
    TrainingConfig,
    TrainingResults,
    YOLOTrainer,
)

__all__ = [
    "ModelManager",
    "TrainingConfig",
    "TrainingResults",
    "YOLOTrainer",
]

__version__ = "1.0.0"
