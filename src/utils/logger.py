"""
Logging utilities for DMS.

Provides comprehensive logging setup with file rotation, console output,
and structured logging for different components.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import colorama
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Rich console logging support
try:
    RICH_AVAILABLE = True
    install_rich_traceback()
except ImportError:
    RICH_AVAILABLE = False

# Colorama for Windows color support
try:
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color support."""
        if COLORAMA_AVAILABLE or os.name != "nt":  # Not Windows or colorama available
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class StudioLogger:
    """Main logger class for YOLO Vision Studio."""

    def __init__(self, name: str = "yolo_vision_studio") -> None:
        """Initialize the studio logger.

        Args:
            name: Logger name.
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_logger()

    def _setup_logger(self) -> None:
        """Set up logger with file and console handlers."""
        self.logger.setLevel(logging.DEBUG)

        # File handler with rotation
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)

        # File formatter
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        if RICH_AVAILABLE:
            console_handler: logging.Handler = RichHandler(
                console=Console(stderr=True),
                show_path=False,
                show_time=True,
                rich_tracebacks=True,
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            console_handler.setFormatter(console_formatter)

        console_handler.setLevel(logging.INFO)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger.

        Returns:
            Configured logger instance.
        """
        return self.logger


class ComponentLogger:
    """Specialized logger for different components."""

    def __init__(
        self, component: str, parent_logger: str = "yolo_vision_studio"
    ) -> None:
        """Initialize component logger.

        Args:
            component: Component name.
            parent_logger: Parent logger name.
        """
        self.component = component
        self.logger_name = f"{parent_logger}.{component}"
        self.logger = logging.getLogger(self.logger_name)
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Setup component-specific file handler
        if not any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in self.logger.handlers
        ):
            self._setup_component_handler()

    def _setup_component_handler(self) -> None:
        """Set up component-specific file handler."""
        log_file = self.log_dir / f"{self.component}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"  # 50MB
        )
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Get the component logger.

        Returns:
            Component logger instance.
        """
        return self.logger


class PerformanceLogger:
    """Performance and metrics logger."""

    def __init__(self, component: str = "performance") -> None:
        """Initialize performance logger.

        Args:
            component: Component name.
        """
        self.component = component
        self.logger = logging.getLogger(f"yolo_vision_studio.{component}")
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Setup performance-specific handler
        self._setup_performance_handler()

    def _setup_performance_handler(self) -> None:
        """Set up performance logging handler."""
        log_file = self.log_dir / f"{self.component}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=25 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 25MB
        )
        file_handler.setLevel(logging.INFO)

        # JSON-like formatter for structured logging
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def log_metric(
        self, metric_name: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a performance metric.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            metadata: Additional metadata.
        """
        metadata = metadata or {}
        log_entry = {
            "metric": metric_name,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        }
        self.logger.info(str(log_entry))

    def log_training_step(
        self, epoch: int, step: int, loss: float, metrics: Dict[str, float], lr: float
    ) -> None:
        """Log training step information.

        Args:
            epoch: Current epoch.
            step: Current step.
            loss: Loss value.
            metrics: Training metrics.
            lr: Learning rate.
        """
        log_entry = {
            "type": "training_step",
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self.logger.info(str(log_entry))

    def log_annotation_session(
        self,
        session_id: str,
        images_annotated: int,
        time_taken: float,
        quality_score: float,
    ) -> None:
        """Log annotation session information.

        Args:
            session_id: Session identifier.
            images_annotated: Number of images annotated.
            time_taken: Time taken in seconds.
            quality_score: Quality score.
        """
        log_entry = {
            "type": "annotation_session",
            "session_id": session_id,
            "images_annotated": images_annotated,
            "time_taken_seconds": time_taken,
            "quality_score": quality_score,
            "images_per_minute": (
                (images_annotated / time_taken) * 60 if time_taken > 0 else 0
            ),
            "timestamp": datetime.now().isoformat(),
        }
        self.logger.info(str(log_entry))


# Global logger instances
_main_logger: Optional[StudioLogger] = None
_component_loggers: Dict[str, ComponentLogger] = {}
_performance_logger: Optional[PerformanceLogger] = None


def setup_logger(name: str = "yolo_vision_studio") -> logging.Logger:
    """Set up and return the main studio logger.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    global _main_logger
    if _main_logger is None or _main_logger.name != name:
        _main_logger = StudioLogger(name)
    return _main_logger.get_logger()


def get_logger(name: str = "yolo_vision_studio") -> logging.Logger:
    """Get existing logger or create new one.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def get_component_logger(component: str) -> logging.Logger:
    """Get or create a component-specific logger.

    Args:
        component: Component name.

    Returns:
        Component logger instance.
    """
    if component not in _component_loggers:
        _component_loggers[component] = ComponentLogger(component)
    return _component_loggers[component].get_logger()


def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger instance.

    Returns:
        Performance logger instance.
    """
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def set_log_level(level: str) -> None:
    """Set log level for all loggers.

    Args:
        level: Log level name.

    Raises:
        ValueError: If invalid log level is provided.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Set level for all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith("yolo_vision_studio"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)


def cleanup_logs(days_to_keep: int = 30) -> None:
    """Cleanup old log files.

    Args:
        days_to_keep: Number of days to keep logs.
    """
    log_dir = Path("logs")
    if not log_dir.exists():
        return

    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)

    for log_file in log_dir.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                print(f"Deleted old log file: {log_file}")
            except OSError as e:
                print(f"Failed to delete {log_file}: {e}")


def get_log_stats() -> Dict[str, Any]:
    """Get logging statistics.

    Returns:
        Dictionary containing log statistics.
    """
    log_dir = Path("logs")
    if not log_dir.exists():
        return {"total_logs": 0, "total_size": 0}

    log_files = list(log_dir.glob("*.log*"))
    total_size = sum(f.stat().st_size for f in log_files)

    return {
        "total_logs": len(log_files),
        "total_size": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "log_files": [f.name for f in log_files],
    }
