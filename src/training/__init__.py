"""
YOLO Vision Studio - Training Module

This module provides comprehensive YOLO model training capabilities including:
- Multi-YOLO architecture support (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11)
- Cross-vendor GPU support (NVIDIA CUDA, AMD DirectML, CPU)
- Automated hyperparameter optimization
- Real-time training monitoring and evaluation
"""

from .yolo_trainer import ModelManager, TrainingConfig, TrainingResults, YOLOTrainer

__all__ = ["YOLOTrainer", "ModelManager", "TrainingConfig", "TrainingResults"]

__version__ = "1.0.0"
