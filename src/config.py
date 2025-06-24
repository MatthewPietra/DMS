"""
Configuration Management System

Centralized configuration handling with validation and environment support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, validator
import logging

logger = logging.getLogger(__name__)

@dataclass
class HardwareConfig:
    """Hardware-specific configuration"""
    device: str = "auto"  # auto, cuda, directml, cpu
    batch_size: int = -1  # -1 for auto-detection
    workers: int = -1     # -1 for auto-detection
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
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        
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
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data:
                logger.warning("Empty configuration file, using defaults")
                return
            
            # Update configurations
            if 'hardware' in data:
                self._update_dataclass(self.hardware, data['hardware'])
            if 'capture' in data:
                self._update_dataclass(self.capture, data['capture'])
            if 'training' in data:
                self._update_dataclass(self.training, data['training'])
            if 'annotation' in data:
                self._update_dataclass(self.annotation, data['annotation'])
            if 'project' in data:
                self._update_dataclass(self.project, data['project'])
                
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to YAML file"""
        if config_path:
            self.config_path = Path(config_path)
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                'hardware': self._dataclass_to_dict(self.hardware),
                'capture': self._dataclass_to_dict(self.capture),
                'training': self._dataclass_to_dict(self.training),
                'annotation': self._dataclass_to_dict(self.annotation),
                'project': self._dataclass_to_dict(self.project),
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
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
            'yolov5s': {'img_size': 640, 'batch_size': 16},
            'yolov5m': {'img_size': 640, 'batch_size': 12},
            'yolov5l': {'img_size': 640, 'batch_size': 8},
            'yolov5x': {'img_size': 640, 'batch_size': 4},
            'yolov8n': {'img_size': 640, 'batch_size': 32},
            'yolov8s': {'img_size': 640, 'batch_size': 16},
            'yolov8m': {'img_size': 640, 'batch_size': 12},
            'yolov8l': {'img_size': 640, 'batch_size': 8},
            'yolov8x': {'img_size': 640, 'batch_size': 4},
        }
        
        return model_configs.get(model_name, {'img_size': 640, 'batch_size': 8})
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate hardware settings
            if self.hardware.device not in ['auto', 'cuda', 'directml', 'cpu']:
                logger.error(f"Invalid device: {self.hardware.device}")
                return False
            
            # Validate capture settings
            if not (1 <= self.capture.fps <= 60):
                logger.error(f"Invalid FPS: {self.capture.fps}")
                return False
            
            # Validate training settings
            if self.training.epochs <= 0:
                logger.error(f"Invalid epochs: {self.training.epochs}")
                return False
            
            # Validate paths
            if not self.project.data_path.parent.exists():
                logger.warning(f"Data path parent does not exist: {self.project.data_path.parent}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        env_mappings = {
            'YOLO_DEVICE': ('hardware', 'device'),
            'YOLO_BATCH_SIZE': ('hardware', 'batch_size'),
            'YOLO_FPS': ('capture', 'fps'),
            'YOLO_EPOCHS': ('training', 'epochs'),
            'YOLO_LEARNING_RATE': ('training', 'learning_rate'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                section_obj = getattr(self, section)
                
                # Type conversion
                current_value = getattr(section_obj, key)
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                elif isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes')
                
                setattr(section_obj, key, value)
                logger.info(f"Updated {section}.{key} from environment: {value}")

# Global configuration instance
config = Config() 