"""
YOLO Vision Studio - Auto-Annotation Module

This module provides intelligent auto-annotation capabilities including:
- Confidence-based annotation acceptance (0.60/0.20 thresholds)
- ACC framework (Accuracy, Credibility, Consistency) quality assessment
- Active learning workflow for continuous improvement
- Multi-model ensemble predictions
"""

from .auto_annotator import AutoAnnotator, AutoAnnotationConfig, AutoAnnotationResult
from .acc_framework import ACCFramework, ACCScores
from .confidence_manager import ConfidenceManager

__all__ = [
    'AutoAnnotator',
    'AutoAnnotationConfig',
    'AutoAnnotationResult',
    'ACCFramework',
    'ACCScores',
    'ConfidenceManager'
]

__version__ = "1.0.0" 