"""
DMS - Annotation Module

This module provides comprehensive annotation capabilities including:
- Interactive annotation interface with bounding box, polygon, and point tools
- Collaborative annotation features with version control
- Quality assurance and inter-annotator agreement tracking
- COCO format support with YOLO conversion
"""

from .annotation_interface import (
 coco_exporter, 
 COCOExporter,
 Annotation,
 AnnotationCanvas,
 AnnotationInterface,
 ClassDialog,
 launch_annotation_interface,
)

__all__ = [
    "AnnotationInterface",
    "Annotation",
    "AnnotationCanvas",
    "ClassDialog",
    "launch_annotation_interface",
    "COCOExporter",
]

__version__ = "1.0.0"
