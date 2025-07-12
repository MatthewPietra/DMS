"""
Configuration Management Module

Handles loading, validation, and management of configuration files
for YOLO Vision Studio components.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
import logging
from copy import deepcopy

try:
    from omegaconf import OmegaConf, DictConfig

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


@dataclass
class StudioConfig:
    """Main studio configuration."""

    name: str = "YOLO Vision Studio"
    version: str = "1.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"
    max_concurrent_projects: int = 5
    auto_save_interval: int = 300


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    auto_detect_gpu: bool = True
    preferred_device: str = "auto"
    gpu_memory_fraction: float = 0.8
    cpu_threads: int = -1

    # GPU-specific settings (optional, from YAML config)
    cuda: Optional[Dict[str, Any]] = None
    directml: Optional[Dict[str, Any]] = None
    cpu: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Set defaults for nested configurations if not provided
        if self.cuda is None:
            self.cuda = {"enabled": True, "device_ids": [0], "mixed_precision": True}
        if self.directml is None:
            self.directml = {"enabled": True, "device_id": 0, "force_fp16": False}
        if self.cpu is None:
            self.cpu = {"num_workers": 4, "optimization_level": "O2"}


@dataclass
class CaptureConfig:
    """Capture system configuration."""

    default_fps: int = 5
    min_fps: int = 1
    max_fps: int = 10
    default_resolution: List[int] = None
    min_resolution: List[int] = None
    max_resolution: List[int] = None
    image_format: str = "PNG"
    jpeg_quality: int = 95

    # Additional settings from YAML config (optional)
    window_detection: Optional[Dict[str, Any]] = None
    compression_level: int = 6
    preview: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.default_resolution is None:
            self.default_resolution = [640, 640]
        if self.min_resolution is None:
            self.min_resolution = [320, 320]
        if self.max_resolution is None:
            self.max_resolution = [1920, 1920]

        # Set defaults for nested configurations if not provided
        if self.window_detection is None:
            self.window_detection = {
                "cross_platform": True,
                "refresh_interval": 1.0,
                "include_minimized": False,
            }
        if self.preview is None:
            self.preview = {
                "enabled": True,
                "max_preview_size": [320, 320],
                "update_interval": 0.1,
            }


@dataclass
class AnnotationConfig:
    """Annotation interface configuration."""

    theme: str = "dark"
    font_size: int = 12
    zoom_sensitivity: float = 0.1
    pan_sensitivity: float = 1.0
    enable_validation: bool = True
    min_box_size: int = 10
    max_overlap_threshold: float = 0.8

    # Additional settings from YAML config (optional)
    ui: Optional[Dict[str, Any]] = None
    tools: Optional[Dict[str, Any]] = None
    shortcuts: Optional[Dict[str, Any]] = None
    classes: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Set defaults for nested configurations if not provided
        if self.ui is None:
            self.ui = {
                "theme": self.theme,
                "font_size": self.font_size,
                "zoom_sensitivity": self.zoom_sensitivity,
                "pan_sensitivity": self.pan_sensitivity,
            }
        if self.tools is None:
            self.tools = {
                "bounding_box": True,
                "polygon": True,
                "point": True,
                "line": False,
                "circle": False,
            }
        if self.shortcuts is None:
            self.shortcuts = {
                "save": "Ctrl+S",
                "undo": "Ctrl+Z",
                "redo": "Ctrl+Y",
                "delete": "Delete",
                "next_image": "Right",
                "prev_image": "Left",
                "zoom_in": "Ctrl+Plus",
                "zoom_out": "Ctrl+Minus",
                "fit_to_window": "Ctrl+0",
            }
        if self.classes is None:
            self.classes = {
                "default_colors": [
                    "#FF0000",
                    "#00FF00",
                    "#0000FF",
                    "#FFFF00",
                    "#FF00FF",
                    "#00FFFF",
                ],
                "max_classes": 100,
                "auto_assign_colors": True,
            }
        if self.quality is None:
            self.quality = {
                "enable_validation": self.enable_validation,
                "min_box_size": self.min_box_size,
                "max_overlap_threshold": self.max_overlap_threshold,
                "require_all_objects": True,
            }


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    epochs: int = 100
    batch_size: int = -1
    image_size: int = 640
    patience: int = 10
    save_period: int = 10
    min_map50: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80

    # Additional settings from YAML config (optional)
    supported_models: Optional[List[str]] = None
    defaults: Optional[Dict[str, Any]] = None
    data_splits: Optional[Dict[str, Any]] = None
    augmentation: Optional[Dict[str, Any]] = None
    thresholds: Optional[Dict[str, Any]] = None
    hyperopt: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Set defaults for nested configurations if not provided
        if self.supported_models is None:
            self.supported_models = [
                "yolov5n",
                "yolov5s",
                "yolov5m",
                "yolov5l",
                "yolov5x",
                "yolov8n",
                "yolov8s",
                "yolov8m",
                "yolov8l",
                "yolov8x",
                "yolov9c",
                "yolov9e",
                "yolov10n",
                "yolov10s",
                "yolov10m",
                "yolov10l",
                "yolov10x",
                "yolov11n",
                "yolov11s",
                "yolov11m",
                "yolov11l",
                "yolov11x",
            ]
        if self.defaults is None:
            self.defaults = {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "image_size": self.image_size,
                "patience": self.patience,
                "save_period": self.save_period,
            }
        if self.data_splits is None:
            self.data_splits = {
                "train": 0.7,
                "val": 0.2,
                "test": 0.1,
                "stratified": True,
            }
        if self.augmentation is None:
            self.augmentation = {
                "mosaic": 1.0,
                "mixup": 0.1,
                "copy_paste": 0.1,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
            }
        if self.thresholds is None:
            self.thresholds = {
                "min_map50": self.min_map50,
                "min_precision": self.min_precision,
                "min_recall": self.min_recall,
            }
        if self.hyperopt is None:
            self.hyperopt = {
                "enabled": False,
                "iterations": 100,
                "search_space": {
                    "lr0": [0.001, 0.1],
                    "momentum": [0.8, 0.95],
                    "weight_decay": [0.0001, 0.001],
                },
            }


@dataclass
class AutoAnnotationConfig:
    """Auto-annotation system configuration."""

    auto_accept_threshold: float = 0.60
    human_review_threshold: float = 0.20
    auto_reject_threshold: float = 0.20
    min_dataset_size: int = 100
    min_model_performance: float = 0.70
    min_class_examples: int = 50
    batch_size: int = 32

    # Additional settings from YAML config (optional)
    thresholds: Optional[Dict[str, Any]] = None
    quality_control: Optional[Dict[str, Any]] = None
    activation: Optional[Dict[str, Any]] = None
    processing: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Set defaults for nested configurations if not provided
        if self.thresholds is None:
            self.thresholds = {
                "auto_accept": self.auto_accept_threshold,
                "human_review": self.human_review_threshold,
                "auto_reject": self.auto_reject_threshold,
            }
        if self.quality_control is None:
            self.quality_control = {
                "enable_acc_framework": True,
                "accuracy_threshold": 0.90,
                "credibility_threshold": 0.85,
                "consistency_threshold": 0.80,
            }
        if self.activation is None:
            self.activation = {
                "min_dataset_size": self.min_dataset_size,
                "min_model_performance": self.min_model_performance,
                "min_class_examples": self.min_class_examples,
                "min_acceptance_rate": 0.90,
            }
        if self.processing is None:
            self.processing = {
                "batch_size": self.batch_size,
                "max_concurrent_batches": 2,
                "timeout_per_image": 30,
            }


class ConfigManager:
    """
    Configuration manager for YOLO Vision Studio.

    Handles loading, validation, and management of configuration files
    with support for environment variables and runtime overrides.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.logger = logging.getLogger(__name__)

        # Default config paths
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)

        self.main_config_path = config_path or self.config_dir / "studio_config.yaml"

        # Configuration storage
        self._config: Dict[str, Any] = {}
        self._config_objects: Dict[str, Any] = {}

        # Load configuration
        self._load_default_config()
        if self.main_config_path.exists():
            self.load_config(self.main_config_path)
        else:
            self.logger.info(
                f"Config file not found at {self.main_config_path}, using defaults"
            )
            self.save_config()  # Save default config

    def _load_default_config(self):
        """Load default configuration values."""
        self._config = {
            "studio": asdict(StudioConfig()),
            "hardware": asdict(HardwareConfig()),
            "capture": asdict(CaptureConfig()),
            "annotation": asdict(AnnotationConfig()),
            "training": asdict(TrainingConfig()),
            "auto_annotation": asdict(AutoAnnotationConfig()),
            "logging": {
                "level": "INFO",
                "files": {
                    "main": "logs/studio.log",
                    "training": "logs/training.log",
                    "annotation": "logs/annotation.log",
                    "capture": "logs/capture.log",
                },
                "rotation": {"max_size": "100MB", "backup_count": 5},
            },
            "data": {
                "formats": {
                    "primary": "COCO",
                    "export_formats": ["COCO", "YOLO", "Pascal VOC", "TensorFlow"],
                },
                "storage": {
                    "max_project_size": "50GB",
                    "auto_cleanup": True,
                    "backup_enabled": True,
                    "backup_interval": 86400,
                },
            },
        }

        # Create config objects
        self._config_objects = {
            "studio": StudioConfig(**self._config["studio"]),
            "hardware": HardwareConfig(**self._config["hardware"]),
            "capture": CaptureConfig(**self._config["capture"]),
            "annotation": AnnotationConfig(**self._config["annotation"]),
            "training": TrainingConfig(**self._config["training"]),
            "auto_annotation": AutoAnnotationConfig(**self._config["auto_annotation"]),
        }

    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix.lower() == ".json":
                    loaded_config = json.load(f)
                elif config_path.suffix.lower() in [".yaml", ".yml"]:
                    loaded_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            # Merge with defaults
            self._merge_config(loaded_config)

            # Update config objects
            self._update_config_objects()

            self.logger.info(f"Configuration loaded from {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise

    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing config."""

        def merge_dicts(base: Dict, update: Dict):
            for key, value in update.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value

        merge_dicts(self._config, new_config)

    def _update_config_objects(self):
        """Update configuration objects from loaded config."""
        try:
            self._config_objects["studio"] = StudioConfig(**self._config["studio"])

            # Handle configs with potential extra fields from YAML
            for config_key, config_class in [
                ("hardware", HardwareConfig),
                ("capture", CaptureConfig),
                ("annotation", AnnotationConfig),
                ("training", TrainingConfig),
                ("auto_annotation", AutoAnnotationConfig),
            ]:
                try:
                    config_data = self._config[config_key].copy()
                    self._config_objects[config_key] = config_class(**config_data)
                except TypeError as e:
                    # Handle unexpected keyword arguments gracefully
                    self.logger.warning(f"Config mismatch for {config_key}: {e}")
                    # Try with only the fields that the dataclass expects
                    expected_fields = set(config_class.__dataclass_fields__.keys())
                    filtered_config = {
                        k: v for k, v in config_data.items() if k in expected_fields
                    }
                    self._config_objects[config_key] = config_class(**filtered_config)

        except Exception as e:
            self.logger.error(f"Failed to update config objects: {e}")
            # Keep existing objects if update fails

    def save_config(self, config_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file."""
        config_path = Path(config_path) if config_path else self.main_config_path

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)."""
        keys = key.split(".")
        config = self._config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

        # Update config objects if needed
        if keys[0] in self._config_objects:
            try:
                self._update_config_objects()
            except Exception as e:
                self.logger.warning(
                    f"Failed to update config object for {keys[0]}: {e}"
                )

    def get_studio_config(self) -> StudioConfig:
        """Get studio configuration object."""
        return self._config_objects["studio"]

    def get_hardware_config(self) -> HardwareConfig:
        """Get hardware configuration object."""
        return self._config_objects["hardware"]

    def get_capture_config(self) -> CaptureConfig:
        """Get capture configuration object."""
        return self._config_objects["capture"]

    def get_annotation_config(self) -> AnnotationConfig:
        """Get annotation configuration object."""
        return self._config_objects["annotation"]

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration object."""
        return self._config_objects["training"]

    def get_auto_annotation_config(self) -> AutoAnnotationConfig:
        """Get auto-annotation configuration object."""
        return self._config_objects["auto_annotation"]

    def get_full_config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return deepcopy(self._config)

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate studio config
        studio_config = self._config_objects["studio"]
        if studio_config.max_concurrent_projects < 1:
            issues.append("studio.max_concurrent_projects must be >= 1")
        if studio_config.auto_save_interval < 60:
            issues.append("studio.auto_save_interval must be >= 60 seconds")

        # Validate hardware config
        hardware_config = self._config_objects["hardware"]
        if not 0.1 <= hardware_config.gpu_memory_fraction <= 1.0:
            issues.append("hardware.gpu_memory_fraction must be between 0.1 and 1.0")

        # Validate capture config
        capture_config = self._config_objects["capture"]
        if (
            not capture_config.min_fps
            <= capture_config.default_fps
            <= capture_config.max_fps
        ):
            issues.append(
                "capture fps values must satisfy: min_fps <= default_fps <= max_fps"
            )
        if capture_config.jpeg_quality < 1 or capture_config.jpeg_quality > 100:
            issues.append("capture.jpeg_quality must be between 1 and 100")

        # Validate training config
        training_config = self._config_objects["training"]
        if training_config.epochs < 1:
            issues.append("training.epochs must be >= 1")
        if training_config.patience < 1:
            issues.append("training.patience must be >= 1")
        if not 0.0 <= training_config.min_map50 <= 1.0:
            issues.append("training.min_map50 must be between 0.0 and 1.0")

        # Validate auto-annotation config
        auto_config = self._config_objects["auto_annotation"]
        if not 0.0 <= auto_config.auto_accept_threshold <= 1.0:
            issues.append(
                "auto_annotation.auto_accept_threshold must be between 0.0 and 1.0"
            )
        if auto_config.min_dataset_size < 10:
            issues.append("auto_annotation.min_dataset_size must be >= 10")

        return issues

    def apply_environment_overrides(self):
        """Apply configuration overrides from environment variables."""
        env_prefix = "YOLO_STUDIO_"

        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix) :].lower().replace("_", ".")

                # Try to parse as JSON first, then as string
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value

                self.set(config_key, parsed_value)
                self.logger.info(
                    f"Applied environment override: {config_key} = {parsed_value}"
                )

    def create_project_config(self, project_name: str, **kwargs) -> Dict[str, Any]:
        """Create project-specific configuration."""
        project_config = {
            "name": project_name,
            "created_at": str(Path.cwd()),
            "version": self.get_studio_config().version,
            "capture": asdict(self.get_capture_config()),
            "annotation": asdict(self.get_annotation_config()),
            "training": asdict(self.get_training_config()),
            "auto_annotation": asdict(self.get_auto_annotation_config()),
        }

        # Apply any overrides
        for key, value in kwargs.items():
            if "." in key:
                self._set_nested_value(project_config, key, value)
            else:
                project_config[key] = value

        return project_config

    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """Set nested value in configuration dictionary."""
        keys = key.split(".")
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value


# Global configuration manager
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Union[str, Path]) -> ConfigManager:
    """Load configuration and return manager instance."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager
