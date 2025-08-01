"""Window Capture System.

Cross-platform window detection and capture system with real-time preview,
configurable frame rates, and automatic image processing.
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, cast

import mss
from PIL import Image

# Cross-platform window detection
try:
    import pygetwindow as gw  # type: ignore

    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False
    gw = None

# Windows-specific imports
if sys.platform == "win32":
    try:
        import win32api
        import win32con
        import win32gui
        import win32process

        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
        win32api = cast(ModuleType, None)
        win32con = cast(ModuleType, None)
        win32gui = cast(ModuleType, None)
        win32process = cast(ModuleType, None)
else:
    WIN32_AVAILABLE = False
    win32api = cast(ModuleType, None)
    win32con = cast(ModuleType, None)
    win32gui = cast(ModuleType, None)
    win32process = cast(ModuleType, None)

from ..utils.config import CaptureConfig
from ..utils.logger import get_component_logger, setup_logger
from .image_processor import ImageProcessor


@dataclass
class WindowInfo:
    """Window information container."""

    title: str
    pid: int
    handle: int
    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom)
    is_visible: bool
    is_minimized: bool


@dataclass
class CaptureStats:
    """Capture session statistics."""

    images_captured: int = 0
    total_duration: float = 0.0
    average_fps: float = 0.0
    last_capture_time: float = 0.0
    errors: int = 0


class WindowDetector:
    """Cross-platform window detection and management."""

    def __init__(self) -> None:
        """Initialize the window detector."""
        self.logger = get_component_logger("window_detector")
        self._windows_cache: Dict[str, WindowInfo] = {}
        self._cache_timeout = 5.0  # seconds
        self._last_cache_update = 0.0

    def get_available_windows(
        self, include_minimized: bool = False
    ) -> List[WindowInfo]:
        """Get list of available windows.

        Args:
            include_minimized: Whether to include minimized windows.

        Returns:
            List of available windows.
        """
        current_time = time.time()

        # Check cache validity
        if (current_time - self._last_cache_update) < self._cache_timeout:
            cached_windows = list(self._windows_cache.values())
            if not include_minimized:
                cached_windows = [w for w in cached_windows if not w.is_minimized]
            return cached_windows

        # Refresh window list
        detected_windows: List[WindowInfo] = []

        if PYGETWINDOW_AVAILABLE and gw is not None:
            detected_windows = self._get_windows_pygetwindow(include_minimized)
        elif WIN32_AVAILABLE and sys.platform == "win32" and win32gui is not None:
            detected_windows = self._get_windows_win32(include_minimized)
        else:
            self.logger.warning("No window detection method available")
            return []

        # Update cache
        self._windows_cache = {w.title: w for w in detected_windows}
        self._last_cache_update = current_time

        return detected_windows

    def _get_windows_pygetwindow(self, include_minimized: bool) -> List[WindowInfo]:
        """Get windows using pygetwindow.

        Args:
            include_minimized: Whether to include minimized windows.

        Returns:
            List of windows found using pygetwindow.
        """
        windows: List[WindowInfo] = []

        try:
            if gw is not None:
                for window in gw.getAllWindows():
                    if not window.title.strip():
                        continue

                    is_minimized = (
                        window.isMinimized if hasattr(window, "isMinimized") else False
                    )
                    is_visible = window.visible if hasattr(window, "visible") else True

                    if not include_minimized and is_minimized:
                        continue

                    windows.append(
                        WindowInfo(
                            title=window.title,
                            pid=getattr(window, "_hWnd", 0),  # Handle if available
                            handle=getattr(window, "_hWnd", 0),
                            bbox=(window.left, window.top, window.right, window.bottom),
                            is_visible=is_visible,
                            is_minimized=is_minimized,
                        )
                    )

        except Exception as e:
            self.logger.error(f"Error getting windows with pygetwindow: {e}")

        return windows

    def _get_windows_win32(self, include_minimized: bool) -> List[WindowInfo]:
        """Get windows using Win32 API.

        Args:
            include_minimized: Whether to include minimized windows.

        Returns:
            List of windows found using Win32 API.
        """
        windows: List[WindowInfo] = []

        def enum_windows_callback(hwnd: int, windows: List[WindowInfo]) -> bool:
            should_continue = True

            win32gui_available = win32gui is not None
            window_visible = win32gui_available and win32gui.IsWindowVisible(hwnd)

            if window_visible:
                title = win32gui.GetWindowText(hwnd)
                title_not_empty = bool(title.strip())

                if title_not_empty:
                    try:
                        bbox = win32gui.GetWindowRect(hwnd)
                        is_minimized = bool(win32gui.IsIconic(hwnd))

                        should_process = include_minimized or not is_minimized

                        if should_process:
                            try:
                                if win32process is not None:
                                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                                else:
                                    pid = 0
                            except Exception as e:
                                self.logger.debug(
                                    f"Could not get process ID for window {hwnd}: {e}"
                                )
                                pid = 0

                            windows.append(
                                WindowInfo(
                                    title=title,
                                    pid=pid,
                                    handle=hwnd,
                                    bbox=bbox,
                                    is_visible=True,
                                    is_minimized=is_minimized,
                                )
                            )
                    except Exception as e:
                        # Skip problematic windows - log for debugging
                        self.logger.debug(f"Skipping window due to error: {e}")

            pass
            return should_continue

        try:
            if win32gui is not None:
                win32gui.EnumWindows(enum_windows_callback, windows)
        except Exception as e:
            self.logger.error(f"Error getting windows with Win32: {e}")

        return windows

    def find_window_by_title(
        self, title: str, partial_match: bool = True
    ) -> Optional[WindowInfo]:
        """Find window by title.

        Args:
            title: Window title to search for.
            partial_match: Whether to use partial matching.

        Returns:
            Window info if found, None otherwise.
        """
        windows = self.get_available_windows(include_minimized=True)

        for window in windows:
            if partial_match:
                if title.lower() in window.title.lower():
                    return window
            else:
                if title == window.title:
                    return window

        return None

    def get_active_window(self) -> Optional[WindowInfo]:
        """Get currently active window.

        Returns:
            Active window info if found, None otherwise.
        """
        if WIN32_AVAILABLE and sys.platform == "win32" and win32gui is not None:
            try:
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                if title:
                    bbox = win32gui.GetWindowRect(hwnd)
                    return WindowInfo(
                        title=title,
                        pid=0,
                        handle=hwnd,
                        bbox=bbox,
                        is_visible=True,
                        is_minimized=bool(win32gui.IsIconic(hwnd)),
                    )
            except Exception as e:
                self.logger.error(f"Error getting active window: {e}")

        return None


class CaptureSession:
    """Individual capture session management."""

    def __init__(
        self, session_id: str, output_dir: Path, config: CaptureConfig
    ) -> None:
        """Initialize capture session.

        Args:
            session_id: Unique session identifier.
            output_dir: Directory to save captured images.
            config: Capture configuration.
        """
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = get_component_logger(f"capture_session_{session_id}")

        # Session state
        self.is_active = False
        self.is_paused = False
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Capture settings
        self.target_window: Optional[WindowInfo] = None
        self.fps = config.default_fps
        min_res = config.min_resolution or [1, 1]
        max_res = config.max_resolution or [4096, 4096]
        default_res = config.default_resolution or [640, 480]
        self.resolution: Tuple[int, int] = (
            max(min_res[0], min(default_res[0], max_res[0])),
            max(min_res[1], min(default_res[1], max_res[1])),
        )
        self.frame_interval = 1.0 / self.fps

        # Statistics
        self.stats = CaptureStats()

        # MSS instance will be created in capture thread
        self.mss_instance: Optional[Any] = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Image processor
        self.image_processor = ImageProcessor(config)

    def set_target_window(self, window: WindowInfo) -> None:
        """Set the target window for capture.

        Args:
            window: Window to capture.
        """
        self.target_window = window
        self.logger.info(f"Target window set: {window.title}")

    def set_fps(self, fps: int) -> None:
        """Set capture frame rate.

        Args:
            fps: Frames per second.
        """
        fps = max(self.config.min_fps, min(fps, self.config.max_fps))
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.logger.info(f"FPS set to: {fps}")

    def set_resolution(self, resolution: Tuple[int, int]) -> None:
        """Set capture resolution.

        Args:
            resolution: Target resolution (width, height).
        """
        min_res = tuple(self.config.min_resolution or (1, 1))
        max_res = tuple(self.config.max_resolution or (4096, 4096))

        width = max(min_res[0], min(resolution[0], max_res[0]))
        height = max(min_res[1], min(resolution[1], max_res[1]))

        self.resolution = (width, height)
        self.logger.info(f"Resolution set to: {self.resolution}")

    def start_capture(self) -> None:
        """Start the capture session."""
        if self.is_active:
            self.logger.warning("Capture session already active")
            return

        self.is_active = True
        self.stop_event.clear()

        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop, name=f"capture_{self.session_id}", daemon=True
        )
        self.capture_thread.start()

        self.logger.info(f"Capture session started: {self.session_id}")

    def stop_capture(self) -> None:
        """Stop the capture session."""
        if not self.is_active:
            return

        self.is_active = False
        self.stop_event.set()

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)

        self.logger.info(f"Capture session stopped: {self.session_id}")

    def pause_capture(self) -> None:
        """Pause the capture session."""
        self.is_paused = True
        self.logger.info("Capture session paused")

    def resume_capture(self) -> None:
        """Resume the capture session."""
        self.is_paused = False
        self.logger.info("Capture session resumed")

    def _capture_loop(self) -> None:
        """Run the main capture loop."""
        # Create MSS instance in this thread to avoid threading issues
        try:
            self.mss_instance = mss.mss()
        except Exception as e:
            self.logger.error(f"Failed to initialize MSS in capture thread: {e}")
            self.is_active = False
            return

        last_capture_time = 0.0
        session_start_time = time.time()

        while self.is_active and not self.stop_event.is_set():
            try:
                current_time = time.time()

                # Check if it's time for next capture
                if current_time - last_capture_time < self.frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue

                # Skip if paused
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                # Capture frame
                success = self._capture_frame()

                if success:
                    self.stats.images_captured += 1
                    last_capture_time = current_time
                    self.stats.last_capture_time = current_time
                else:
                    self.stats.errors += 1

                # Update statistics
                self.stats.total_duration = current_time - session_start_time
                if self.stats.total_duration > 0:
                    self.stats.average_fps = (
                        self.stats.images_captured / self.stats.total_duration
                    )

            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                self.stats.errors += 1
                time.sleep(0.1)  # Brief pause on error

        # Clean up MSS instance
        try:
            if self.mss_instance:
                self.mss_instance.close()
                self.mss_instance = None
        except Exception as e:
            self.logger.error(f"Error closing MSS instance: {e}")

    def _capture_frame(self) -> bool:
        """Capture a single frame.

        Returns:
            True if frame was captured successfully, False otherwise.
        """
        try:
            if self.target_window:
                # Capture specific window
                image = self._capture_window()
            else:
                # Capture full screen
                image = self._capture_screen()

            if image is None:
                return False

            # Process image
            processed_image = self.image_processor.process_image(image, self.resolution)

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%")[:-3]  # milliseconds
            filename = (
                f"{self.session_id}_{timestamp}.{self.config.image_format.lower()}"
            )
            filepath = self.output_dir / filename

            # Save with appropriate format
            if self.config.image_format.upper() == "JPEG":
                processed_image.save(filepath, "JPEG", quality=self.config.jpeg_quality)
            else:
                processed_image.save(filepath, self.config.image_format.upper())

            return True

        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return False

    def _capture_window(self) -> Optional[Image.Image]:
        """Capture specific window.

        Returns:
            Captured image or None if failed.
        """
        if not self.target_window or not self.mss_instance:
            return None

        try:
            bbox = self.target_window.bbox

            # Use MSS for better performance
            monitor = {
                "left": bbox[0],
                "top": bbox[1],
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1],
            }

            screenshot = self.mss_instance.grab(monitor)
            image = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )

            return image

        except Exception as e:
            self.logger.error(f"Error capturing window: {e}")
            return None

    def _capture_screen(self) -> Optional[Image.Image]:
        """Capture full screen.

        Returns:
            Captured image or None if failed.
        """
        if not self.mss_instance:
            return None

        try:
            # Use MSS for full screen capture
            monitor = self.mss_instance.monitors[1]  # Primary monitor
            screenshot = self.mss_instance.grab(monitor)
            image = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )

            return image

        except Exception as e:
            self.logger.error(f"Error capturing screen: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get capture session statistics.

        Returns:
            Dictionary containing session statistics.
        """
        return {
            "session_id": self.session_id,
            "is_active": self.is_active,
            "is_paused": self.is_paused,
            "images_captured": self.stats.images_captured,
            "total_duration": self.stats.total_duration,
            "average_fps": self.stats.average_fps,
            "target_fps": self.fps,
            "resolution": self.resolution,
            "target_window": self.target_window.title if self.target_window else None,
            "errors": self.stats.errors,
        }


class WindowCaptureSystem:
    """Main window capture system."""

    def __init__(self, config: CaptureConfig, hardware_detector: Any = None) -> None:
        """Initialize window capture system.

        Args:
            config: Capture configuration.
            hardware_detector: Optional hardware detector instance.
        """
        self.config = config
        self.hardware_detector = hardware_detector
        self.logger = get_component_logger("window_capture")

        # Initialize window detector
        self.window_detector = WindowDetector()

        # Active sessions
        self.active_sessions: Dict[str, CaptureSession] = {}

        self.logger.info("Window capture system initialized")

    def get_available_windows(
        self, include_minimized: bool = False
    ) -> List[WindowInfo]:
        """Get list of available windows.

        Args:
            include_minimized: Whether to include minimized windows.

        Returns:
            List of available windows.
        """
        return self.window_detector.get_available_windows(include_minimized)

    def find_window(
        self, title: str, partial_match: bool = True
    ) -> Optional[WindowInfo]:
        """Find window by title.

        Args:
            title: Window title to search for.
            partial_match: Whether to use partial matching.

        Returns:
            Window info if found, None otherwise.
        """
        return self.window_detector.find_window_by_title(title, partial_match)

    def start_session(
        self,
        session_id: str,
        output_dir: Path,
        window_title: Optional[str] = None,
        fps: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> CaptureSession:
        """Start a new capture session.

        Args:
            session_id: Unique session identifier.
            output_dir: Directory to save captured images.
            window_title: Optional window title to capture.
            fps: Optional frames per second.
            resolution: Optional target resolution.

        Returns:
            Created capture session.

        Raises:
            ValueError: If session already exists.
        """
        if session_id in self.active_sessions:
            raise ValueError(f"Session '{session_id}' already exists")

        # Create session
        session = CaptureSession(session_id, output_dir, self.config)

        # Configure session
        if fps is not None:
            session.set_fps(fps)
        if resolution is not None:
            session.set_resolution(resolution)

        # Set target window if specified
        if window_title is not None:
            window = self.find_window(window_title)
            if window:
                session.set_target_window(window)
            else:
                self.logger.warning(
                    f"Window '{window_title}' not found, using full screen"
                )

        # Start capture
        session.start_capture()

        # Add to active sessions
        self.active_sessions[session_id] = session

        self.logger.info(f"Started capture session: {session_id}")
        return session

    def stop_session(self, session_id: str) -> None:
        """Stop a capture session.

        Args:
            session_id: Session identifier to stop.

        Raises:
            ValueError: If session not found.
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session '{session_id}' not found")

        session = self.active_sessions[session_id]
        session.stop_capture()

        del self.active_sessions[session_id]

        self.logger.info(f"Stopped capture session: {session_id}")

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a capture session.

        Args:
            session_id: Session identifier.

        Returns:
            Session status dictionary.
        """
        if session_id not in self.active_sessions:
            return {"status": "not_found"}

        session = self.active_sessions[session_id]
        return session.get_stats()

    def pause_session(self, session_id: str) -> None:
        """Pause a capture session.

        Args:
            session_id: Session identifier to pause.
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id].pause_capture()

    def resume_session(self, session_id: str) -> None:
        """Resume a capture session.

        Args:
            session_id: Session identifier to resume.
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id].resume_capture()

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status.

        Returns:
            System status dictionary.
        """
        return {
            "active_sessions": len(self.active_sessions),
            "available_windows": len(self.get_available_windows()),
            "sessions": {
                session_id: session.get_stats()
                for session_id, session in self.active_sessions.items()
            },
        }

    def shutdown(self) -> None:
        """Shutdown the capture system."""
        self.logger.info("Shutting down window capture system...")

        # Stop all active sessions
        for session_id in list(self.active_sessions.keys()):
            try:
                self.stop_session(session_id)
            except Exception as e:
                self.logger.error(f"Error stopping session {session_id}: {e}")

        self.logger.info("Window capture system shutdown complete")


def main() -> int:
    """Provide the main entry point for the window capture system.

    Provides an interactive demo and testing interface.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="YOLO Vision Studio - Window Capture System"
    )
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument(
        "--list-windows", action="store_true", help="List available windows"
    )
    parser.add_argument("--fps", type=int, default=5, help="Capture FPS (default: 5)")
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Demo duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/temp",
        help="Output directory for captures",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger("window_capture")
    logger = get_component_logger("window_capture_main")

    try:
        # Initialize capture config
        config = CaptureConfig()

        # Initialize window capture system
        capture_system = WindowCaptureSystem(config)

        if args.list_windows:
            # List available windows
            print("\nAvailable Windows:")
            print("=" * 50)
            windows = capture_system.get_available_windows()
            if not windows:
                print("No windows found")
            else:
                for i, window in enumerate(windows, 1):
                    print(f"{i:2d}. {window.title}")
                    print(f"    Handle: {window.handle}, PID: {window.pid}")
                    print(f"    Position: {window.bbox}")
                    print(
                        f"    Visible: {window.is_visible}, "
                        f"Minimized: {window.is_minimized}"
                    )
                    print()
            return 0

        if args.demo:
            # Run interactive demo
            print("\nYOLO Vision Studio - Window Capture Demo")
            print("=" * 50)

            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # List available windows
            windows = capture_system.get_available_windows()
            if windows:
                print(f"\nFound {len(windows)} available windows:")
                for i, window in enumerate(windows[:10], 1):  # Show first 10
                    print(f"  {i}. {window.title}")

            print("\nStarting capture demo...")
            print(f"   FPS: {args.fps}")
            print(f"   Duration: {args.duration} seconds")
            print(f"   Output: {output_dir}")

            # Start capture session
            session_id = f"demo_{int(time.time())}"
            session = capture_system.start_session(
                session_id=session_id, output_dir=output_dir, fps=args.fps
            )

            try:
                # Monitor capture
                start_time = time.time()
                last_status_time = start_time

                while (time.time() - start_time) < args.duration:
                    time.sleep(1)

                    # Show status every 2 seconds
                    if (time.time() - last_status_time) >= 2:
                        stats = session.get_stats()
                        elapsed = time.time() - start_time
                        print(
                            f"   Captured: {stats['images_captured']} images, "
                            f"Elapsed: {elapsed:.1f}s, "
                            f"FPS: {stats['average_fps']:.1f}"
                        )
                        last_status_time = time.time()

                # Stop capture
                capture_system.stop_session(session_id)

                # Final stats
                final_stats = session.get_stats()
                print("\nDemo completed!")
                print(f"   Total images captured: {final_stats['images_captured']}")
                print(f"   Average FPS: {final_stats['average_fps']:.2f}")
                print(f"   Total duration: {final_stats['total_duration']:.2f}s")
                print(f"   Output directory: {output_dir}")

            except KeyboardInterrupt:
                print("\nDemo interrupted by user")
                capture_system.stop_session(session_id)

        else:
            # Default: show help and system status
            print("\nYOLO Vision Studio - Window Capture System")
            print("=" * 50)
            print("System initialized and ready!")
            print("\nAvailable options:")
            print("  --demo              Run interactive capture demo")
            print("  --list-windows      List all available windows")
            print("  --fps N             Set capture FPS (default: 5)")
            print("  --duration N        Set demo duration in seconds (default: 10)")
            print("  --output-dir PATH   Set output directory (default: data/temp)")

            # Show system status
            status = capture_system.get_system_status()
            print("\nSystem Status:")
            print(f"  Available windows: {status['available_windows']}")
            print(f"  Active sessions: {status['active_sessions']}")

        # Cleanup
        capture_system.shutdown()

    except Exception as e:
        logger.error(f"Error in window capture main: {e}")
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
