"""
Capture module for YOLO Vision Studio

Provides window capture, screen recording, and image acquisition
functionality with cross-platform support.
"""

from .window_capture import WindowCaptureSystem, WindowDetector, CaptureSession
from .image_processor import ImageProcessor

__all__ = ["WindowCaptureSystem", "WindowDetector", "CaptureSession", "ImageProcessor"]
