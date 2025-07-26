"""Configuration Management System.

Centralized configuration handling with validation and environment support.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import yaml


# Custom YAML constructor for tuples
def tuple_constructor(loader: Any, node: Any) -> tuple[Any, ...]:
    """Construct a tuple from YAML node.

    Args:
        loader: YAML loader instance.
        node: YAML node to construct tuple from.

    Returns:
        Tuple constructed from the node.
    """
    return tuple(loader.construct_sequence(node))


# Register the constructor for both old and new formats
yaml.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)
yaml.add_constructor("!!python/tuple", tuple_constructor)

# Optional pydantic import for validation
try:
    from pydantic import BaseModel, validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    # Create a dummy BaseModel class
    class BaseModel:  # type: ignore
        """Dummy BaseModel class when pydantic is not available."""

        pass

    def validator(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        """Create a dummy validator function when pydantic is not available."""
        return lambda func: func


logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Hardware-specific configuration.

    Attributes:
        device: Device to use for inference (auto, cuda, directml, cpu).
        batch_size: Batch size for training/inference (-1 for auto-detection).
        workers: Number of worker processes (-1 for auto-detection).
        mixed_precision: Whether to use mixed precision training.
        memory_fraction: Fraction of GPU memory to use.
    """

    device: str = "auto"  # auto, cuda, directml, cpu
    batch_size: int = -1  # -1 for auto-detection
    workers: int = -1  # -1 for auto-detection
    mixed_precision: bool = True
    memory_fraction: float = 0.8


@dataclass
class CaptureConfig:
    """Screen capture configuration.

    Attributes:
        fps: Frames per second for capture.
        monitor: Monitor index to capture.
        window_name: Name of window to capture (empty for full screen).
        resolution: Capture resolution as (width, height).
        quality: JPEG quality for captured images (1-100).
    """

    fps: int = 30
    monitor: int = 0
    window_name: str = ""
    resolution: Tuple[int, int] = (640, 640)
    quality: int = 95
    show_preview: bool = False


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        epochs: Number of training epochs.
        patience: Early stopping patience.
        save_period: How often to save checkpoints.
        model_architecture: YOLO model architecture to use.
        optimizer: Optimizer to use for training.
        learning_rate: Learning rate for training.
        weight_decay: Weight decay for regularization.
    """

    epochs: int = 100
    patience: int = 50
    save_period: int = 10
    model_architecture: str = "yolov8n"  # yolov5s, yolov8n, etc.
    model: str = "yolov8n"
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0005


@dataclass
class AnnotationConfig:
    """Annotation configuration.

    Attributes:
        auto_save: Whether to automatically save annotations.
        confidence_threshold: Minimum confidence for auto-annotations.
        review_threshold: Threshold for manual review of annotations.
        max_annotations_per_image: Maximum annotations per image.
        default_class: Default class name for new annotations.
    """

    auto_save: bool = True
    confidence_threshold: float = 0.6
    review_threshold: float = 0.2
    max_annotations_per_image: int = 50
    default_class: str = "object"
    use_acc_framework: bool = True


@dataclass
class ProjectConfig:
    """Project-specific configuration.

    Attributes:
        name: Project name.
        description: Project description.
        classes: List of class names for the project.
        data_path: Path to project data directory.
        output_path: Path to project output directory.
    """

    name: str = ""
    description: str = ""
    classes: Dict[str, str] = field(
        default_factory=lambda: {"object": "object"}
    )  # Changed to dict to match test expectation
    data_path: Path = field(default_factory=lambda: Path("data"))
    output_path: Path = field(default_factory=lambda: Path("output"))
    version: str = "1.0.0"  # Added for compatibility


class Config:
    """Main configuration manager.

    Provides centralized configuration management with validation,
    environment variable support, and YAML file persistence.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file.
                If None, uses default location.
        """
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
            self._load_config()
        else:
            self.save()  # Create default config

    def _get_default_config_path(self) -> Path:
        """Get default configuration file path.

        Returns:
            Path to the default configuration file.
        """
        # Try user config directory first
        config_dir = Path.home() / ".yolo_vision_studio"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.yaml"

    def load(self) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Optional path to configuration file.
                If None, uses the current config_path.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
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

                logger.info("Configuration loaded from %s", self.config_path)

            except Exception as e:
                logger.error("Failed to load configuration: %s", e)
                logger.info("Using default configuration")

    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to YAML file.

        Args:
            config_path: Optional path to save configuration file.
                If None, uses the current config_path.
        """
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

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)

            logger.info("Configuration saved to %s", self.config_path)

        except Exception as e:
            logger.error("Failed to save configuration: %s", e)

    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """Update dataclass with dictionary data.

        Args:
            obj: Dataclass instance to update.
            data: Dictionary containing new values.
        """
        for key, value in data.items():
            if hasattr(obj, key):
                # Handle Path objects
                if isinstance(getattr(obj, key), Path):
                    setattr(obj, key, Path(value))
                else:
                    setattr(obj, key, value)

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary.

        Args:
            obj: Dataclass instance to convert.

        Returns:
            Dictionary representation of the dataclass.
        """
        result: Dict[str, Any] = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                # Convert tuples to lists for YAML compatibility
                result[key] = list(value)
            else:
                result[key] = value
        return result

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration.

        Args:
            model_name: Name of the YOLO model.

        Returns:
            Dictionary containing model-specific configuration.
        """
        model_configs: Dict[str, Dict[str, Any]] = {
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
        """Validate configuration.

        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            # Validate hardware config
            if self.hardware.batch_size < -1:
                return False
            if self.hardware.workers < -1:
                return False

            # Validate device
            valid_devices = ["auto", "cuda", "directml", "cpu"]
            if self.hardware.device not in valid_devices:
                return False

            # Validate capture config
            if self.capture.fps <= 0 or self.capture.fps > 60:
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

            # Validate threshold relationship
            if self.annotation.confidence_threshold < self.annotation.review_threshold:
                return False

            return True

        except Exception as e:
            logger.error("Configuration validation failed: %s", e)
            return False

    def update_from_env(self) -> None:
        """Update configuration from environment variables.

        Reads environment variables and updates corresponding
        configuration values if they are set.
        """
        env_mappings: Dict[str, Union[Tuple[str, str], Tuple[str, str, str]]] = {
            "YOLO_DEVICE": ("hardware", "device"),
            "YOLO_BATCH_SIZE": ("hardware", "batch_size", "int"),
            "YOLO_WORKERS": ("hardware", "workers", "int"),
            "YOLO_FPS": ("capture", "fps", "int"),
            "YOLO_EPOCHS": ("training", "epochs", "int"),
            "YOLO_LR": ("training", "learning_rate", "float"),
        }

        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    section, attr, *type_info = mapping
                    current_obj = getattr(self, section)

                    # Convert value based on type info and current attribute type
                    if type_info and type_info[0] == "int":
                        converted_value: Union[int, float, str] = int(value)
                    elif type_info and type_info[0] == "float":
                        converted_value = float(value)
                    else:
                        converted_value = value

                    setattr(current_obj, attr, converted_value)
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid environment variable %s: %s", env_var, e)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "hardware": self._dataclass_to_dict(self.hardware),
            "capture": self._dataclass_to_dict(self.capture),
            "training": self._dataclass_to_dict(self.training),
            "annotation": self._dataclass_to_dict(self.annotation),
            "project": self._dataclass_to_dict(self.project),
        }

    def _from_dict(self, data: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
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

    def update(self, data: Dict[str, Any]) -> None:
        """Update configuration with new data."""
        self._from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()
        config._from_dict(data)
        return config

    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data:
                if "project" in data:
                    self._update_dataclass(self.project, data["project"])

            logger.info("Configuration loaded from %s", self.config_path)

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)

    def _load_from_env(self) -> None:
        """Load configuration from environment variables (alias for update_from_env)."""
        self.update_from_env()

    def merge(self, other: "Config") -> None:
        """Merge another configuration into this one."""
        if other.hardware:
            self._update_dataclass(
                self.hardware, self._dataclass_to_dict(other.hardware)
            )
        if other.capture:
            self._update_dataclass(self.capture, self._dataclass_to_dict(other.capture))
        if other.training:
            self._update_dataclass(
                self.training, self._dataclass_to_dict(other.training)
            )
        if other.annotation:
            self._update_dataclass(
                self.annotation, self._dataclass_to_dict(other.annotation)
            )
        if other.project:
            self._update_dataclass(self.project, self._dataclass_to_dict(other.project))

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        # Support nested keys like "hardware.device"
        if "." in key:
            section, attr = key.split(".", 1)
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, attr):
                    return getattr(section_obj, attr)
        return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        # Support nested keys like "hardware.device"
        if "." in key:
            section, attr = key.split(".", 1)
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, attr):
                    setattr(section_obj, attr, value)
        else:
            # Try to set on main config object
            if hasattr(self, key):
                setattr(self, key, value)


# Alias for backward compatibility
YOLOVisionConfig = Config
