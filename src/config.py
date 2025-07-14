import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from pydantic import BaseModel, validator

"""
Configuration Management System

Centralized configuration handling with validation and environment support.
"""

# Optional pydantic import for validation
try:
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    def validator(*args, **kwargs):
        return lambda func: func

logger = logging.getLogger(__name__)

@dataclass
class HardwareConfig:
    """Hardware-specific configuration"""

    device: str = "auto"  # auto, cuda, directml, cpu
    batch_size: int = -1  # -1 for auto-detection
    workers: int = -1  # -1 for auto-detection
    mixed_precision: bool = True
    memory_fraction: float = 0.8

@dataclass
class CaptureConfig:
    """Screen capture configuration"""

    fps: int = 5
    monitor: int = 0
    window_name: str = ""
    resolution: tuple = (640, 640)
    quality: int = 95

@dataclass
class TrainingConfig:
    """Training configuration"""

    epochs: int = 100
    patience: int = 10
    save_period: int = 10
    model_architecture: str = "yolov8n"  # yolov5s, yolov8n, etc.
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0005

@dataclass
class AnnotationConfig:
    """Annotation configuration"""

    auto_save: bool = True
    confidence_threshold: float = 0.6
    review_threshold: float = 0.2
    max_annotations_per_image: int = 50
    default_class: str = "object"

@dataclass
class ProjectConfig:
    """Project-specific configuration"""

    name: str = "default_project"
    description: str = ""
    classes: list = field(default_factory=lambda: ["object"])
    data_path: Path = field(default_factory=lambda: Path("data"))
    output_path: Path = field(default_factory=lambda: Path("output"))

class Config:
    """Main configuration manager"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = (
            Path(config_path) if config_path else self._get_default_config_path()
        )

        # Initialize default configurations
        self.hardware = HardwareConfig()
        self.capture = CaptureConfig()
        self.training = TrainingConfig()
        self.annotation = AnnotationConfig()
        self.project = ProjectConfig()

        # Load configuration if exists
        if self.config_path.exists():
            self.load()
        else:
            self.save()  # Create default config

    def _get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        # Try user config directory first
        config_dir = Path.home() / ".yolo_vision_studio"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.yaml"

    def load(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from YAML file"""
        if config_path:
            self.config_path = Path(config_path)

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning("Empty configuration file, using defaults")
                return

            # Update configurations
            if "hardware" in data:
                self._update_dataclass(self.hardware, data["hardware"])
            if "capture" in data:
                self._update_dataclass(self.capture, data["capture"])
            if "training" in data:
                self._update_dataclass(self.training, data["training"])
            if "annotation" in data:
                self._update_dataclass(self.annotation, data["annotation"])
            if "project" in data:
                self._update_dataclass(self.project, data["project"])

            logger.info("Configuration loaded from {self.config_path}")

        except Exception as e:
            logger.error("Failed to load configuration: {e}")
            logger.info("Using default configuration")

    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to YAML file"""
        if config_path:
            self.config_path = Path(config_path)

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                "hardware": self._dataclass_to_dict(self.hardware),
                "capture": self._dataclass_to_dict(self.capture),
                "training": self._dataclass_to_dict(self.training),
                "annotation": self._dataclass_to_dict(self.annotation),
                "project": self._dataclass_to_dict(self.project),
            }

            with open(self.config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)

            logger.info("Configuration saved to {self.config_path}")

        except Exception as e:
            logger.error("Failed to save configuration: {e}")

    def _update_dataclass(self, obj, data: Dict[str, Any]) -> None:
        """Update dataclass with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                # Handle Path objects
                if isinstance(getattr(obj, key), Path):
                    setattr(obj, key, Path(value))
                else:
                    setattr(obj, key, value)

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        model_configs = {
            "yolov5s": {"img_size": 640, "batch_size": 16},
            "yolov5m": {"img_size": 640, "batch_size": 12},
            "yolov5l": {"img_size": 640, "batch_size": 8},
            "yolov5x": {"img_size": 640, "batch_size": 4},
            "yolov8n": {"img_size": 640, "batch_size": 32},
            "yolov8s": {"img_size": 640, "batch_size": 16},
            "yolov8m": {"img_size": 640, "batch_size": 12},
            "yolov8l": {"img_size": 640, "batch_size": 8},
            "yolov8x": {"img_size": 640, "batch_size": 4},
        }
        return model_configs.get(model_name, model_configs["yolov8n"])

    def validate(self) -> bool:
        """Validate configuration"""
        try:
            # Validate hardware config
            if self.hardware.batch_size < -1:
                return False
            if self.hardware.workers < -1:
                return False

            # Validate capture config
            if self.capture.fps <= 0:
                return False
            if self.capture.quality < 1 or self.capture.quality > 100:
                return False

            # Validate training config
            if self.training.epochs <= 0:
                return False
            if self.training.learning_rate <= 0:
                return False

            # Validate annotation config
            if not 0 <= self.annotation.confidence_threshold <= 1:
                return False
            if not 0 <= self.annotation.review_threshold <= 1:
                return False

            return True

        except Exception as e:
            logger.error("Configuration validation failed: {e}")
            return False

    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        env_mappings = {
            "YOLO_DEVICE": ("hardware", "device"),
            "YOLO_BATCH_SIZE": ("hardware", "batch_size", int),
            "YOLO_WORKERS": ("hardware", "workers", int),
            "YOLO_FPS": ("capture", "fps", int),
            "YOLO_EPOCHS": ("training", "epochs", int),
            "YOLO_LR": ("training", "learning_rate", float),
        }

        for env_var, (section, attr, *type_info) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if type_info:
                        value = type_info[0](value)
                    setattr(getattr(self, section), attr, value)
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid environment variable {env_var}: {e}")

# Alias for backward compatibility
YOLOVisionConfig = Config
