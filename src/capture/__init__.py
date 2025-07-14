from .image_processor import ImageProcessor
from .window_capture import CaptureSession, WindowCaptureSystem, WindowDetector

"""
Capture module for YOLO Vision Studio

Provides window capture, screen recording, and image acquisition
functionality with cross-platform support.
"""

__all__ = ["WindowCaptureSystem", "WindowDetector", "CaptureSession", "ImageProcessor"]
