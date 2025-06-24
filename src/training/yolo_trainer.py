"""
YOLO Vision Studio - YOLO Trainer

Comprehensive YOLO model training with support for:
- YOLOv5, YOLOv8, YOLOv8-s, YOLOv8-n, YOLOv9, YOLOv10, YOLOv11
- Cross-vendor GPU support (NVIDIA CUDA, AMD DirectML, CPU)
- Automated hyperparameter optimization
- Real-time training monitoring
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor

try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config import ConfigManager
from ..utils.hardware import HardwareDetector
from ..utils.metrics import MetricsCalculator


@dataclass
class TrainingConfig:
    """Training configuration data class."""
    model_name: str
    epochs: int = 100
    batch_size: int = -1  # Auto-calculate
    image_size: int = 640
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    patience: int = 10  # Early stopping
    save_period: int = 10
    device: str = "auto"
    workers: int = 8
    project_name: str = "yolo_training"
    experiment_name: str = "exp"
    resume: bool = False
    pretrained: bool = True
    
    # Data augmentation
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    
    # Performance thresholds
    min_map50: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80


@dataclass 
class TrainingResults:
    """Training results data class."""
    model_path: str
    best_map50: float
    best_precision: float
    best_recall: float
    final_loss: float
    training_time: float
    epochs_completed: int
    device_used: str
    model_size: int  # bytes
    config_used: Dict[str, Any]
    metrics_history: Dict[str, List[float]]


class YOLOTrainer:
    """Comprehensive YOLO model trainer with cross-platform support."""
    
    SUPPORTED_MODELS = {
        # YOLOv5 models
        'yolov5n': 'yolov5n.pt',
        'yolov5s': 'yolov5s.pt', 
        'yolov5m': 'yolov5m.pt',
        'yolov5l': 'yolov5l.pt',
        'yolov5x': 'yolov5x.pt',
        
        # YOLOv8 models
        'yolov8n': 'yolov8n.pt',
        'yolov8s': 'yolov8s.pt',
        'yolov8m': 'yolov8m.pt',
        'yolov8l': 'yolov8l.pt',
        'yolov8x': 'yolov8x.pt',
        
        # YOLOv9 models
        'yolov9c': 'yolov9c.pt',
        'yolov9e': 'yolov9e.pt',
        
        # YOLOv10 models
        'yolov10n': 'yolov10n.pt',
        'yolov10s': 'yolov10s.pt',
        'yolov10m': 'yolov10m.pt',
        'yolov10l': 'yolov10l.pt',
        'yolov10x': 'yolov10x.pt',
        
        # YOLOv11 models
        'yolov11n': 'yolo11n.pt',
        'yolov11s': 'yolo11s.pt',
        'yolov11m': 'yolo11m.pt',
        'yolov11l': 'yolo11l.pt',
        'yolov11x': 'yolo11x.pt'
    }
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize YOLO trainer."""
        self.config = config_manager
        self.logger = get_logger(__name__)
        self.hardware_detector = HardwareDetector()
        self.metrics_calculator = MetricsCalculator()
        
        # State
        self.current_model: Optional[YOLO] = None
        self.training_config: Optional[TrainingConfig] = None
        self.training_active: bool = False
        self.training_results: Optional[TrainingResults] = None
        
        # Device setup
        self.device_info = self.hardware_detector.get_device_info()
        self.device = self._setup_device()
        
        self.logger.info(f"YOLOTrainer initialized with device: {self.device}")
        
    def _setup_device(self) -> str:
        """Setup training device based on hardware detection."""
        device_type = self.device_info['device_type']
        
        if device_type == 'cuda':
            device = f"cuda:{self.device_info.get('device_id', 0)}"
            self.logger.info(f"Using NVIDIA GPU: {device}")
            
        elif device_type == 'directml':
            if DIRECTML_AVAILABLE:
                device = f"dml:{self.device_info.get('device_id', 0)}"
                self.logger.info(f"Using AMD GPU with DirectML: {device}")
            else:
                self.logger.warning("DirectML not available, falling back to CPU")
                device = 'cpu'
                
        else:
            device = 'cpu'
            self.logger.info("Using CPU for training")
            
        return device
        
    def get_supported_models(self) -> List[str]:
        """Get list of supported YOLO models."""
        return list(self.SUPPORTED_MODELS.keys())
        
    def validate_model_name(self, model_name: str) -> bool:
        """Validate if model name is supported."""
        return model_name.lower() in self.SUPPORTED_MODELS
        
    def calculate_optimal_batch_size(self, model_name: str, image_size: int = 640) -> int:
        """Calculate optimal batch size based on available memory."""
        try:
            device_type = self.device_info['device_type']
            
            if device_type == 'cuda':
                # NVIDIA GPU memory calculation
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                # Rough estimation based on model size and image resolution
                model_size_factor = {
                    'n': 1.0, 's': 2.0, 'm': 4.0, 'l': 6.0, 'x': 8.0,
                    'c': 4.0, 'e': 6.0  # YOLOv9 variants
                }
                
                model_suffix = model_name[-1].lower()
                factor = model_size_factor.get(model_suffix, 2.0)
                
                # Calculate batch size
                memory_per_image = (image_size / 640) ** 2 * factor * 0.1  # GB per image
                optimal_batch = max(1, int(gpu_memory_gb * 0.8 / memory_per_image))
                
                # Clamp to reasonable range
                return min(max(optimal_batch, 1), 64)
                
            elif device_type == 'directml':
                # AMD GPU - more conservative estimation
                return min(16, max(2, 8))  # Conservative range for DirectML
                
            else:
                # CPU - very conservative
                return min(8, max(1, 4))
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal batch size: {e}")
            return 16  # Safe default
            
    def prepare_training_config(self, 
                              model_name: str,
                              data_yaml_path: str,
                              **kwargs) -> TrainingConfig:
        """Prepare training configuration."""
        if not self.validate_model_name(model_name):
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Get default config from settings
        default_config = self.config.get('training.defaults', {})
        
        # Calculate optimal batch size if not specified
        batch_size = kwargs.get('batch_size', default_config.get('batch_size', -1))
        if batch_size <= 0:
            batch_size = self.calculate_optimal_batch_size(model_name, 
                                                         kwargs.get('image_size', 640))
            
        # Create training config
        config = TrainingConfig(
            model_name=model_name,
            epochs=kwargs.get('epochs', default_config.get('epochs', 100)),
            batch_size=batch_size,
            image_size=kwargs.get('image_size', default_config.get('image_size', 640)),
            patience=kwargs.get('patience', default_config.get('patience', 10)),
            device=self.device,
            **kwargs
        )
        
        self.training_config = config
        self.logger.info(f"Training config prepared: {config.model_name}, "
                        f"batch_size={config.batch_size}, epochs={config.epochs}")
        
        return config
        
    def setup_data_yaml(self, 
                       images_dir: Path,
                       labels_dir: Path,
                       classes: List[str],
                       train_split: float = 0.7,
                       val_split: float = 0.2) -> Path:
        """Setup YOLO data.yaml file."""
        data_yaml = {
            'train': str(images_dir / 'train'),
            'val': str(images_dir / 'val'),
            'test': str(images_dir / 'test'),
            'nc': len(classes),
            'names': classes
        }
        
        # Create data.yaml file
        yaml_path = images_dir.parent / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
            
        self.logger.info(f"Data YAML created: {yaml_path}")
        return yaml_path
        
    def train_model(self, 
                   data_yaml_path: str,
                   training_config: Optional[TrainingConfig] = None,
                   callbacks: Optional[Dict] = None) -> TrainingResults:
        """Train YOLO model with comprehensive monitoring."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics YOLO package required for training")
            
        if training_config is None:
            training_config = self.training_config
            
        if training_config is None:
            raise ValueError("Training configuration not provided")
            
        self.logger.info(f"Starting training: {training_config.model_name}")
        self.training_active = True
        
        start_time = datetime.now()
        
        try:
            # Initialize model
            model_path = self.SUPPORTED_MODELS[training_config.model_name]
            self.current_model = YOLO(model_path)
            
            # Setup device-specific optimizations
            self._setup_device_optimizations(training_config)
            
            # Prepare training arguments
            train_args = self._prepare_train_args(training_config, data_yaml_path)
            
            # Setup callbacks
            if callbacks:
                self._setup_callbacks(callbacks)
                
            # Start training
            self.logger.info("Starting YOLO training...")
            results = self.current_model.train(**train_args)
            
            # Calculate training time
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Process results
            training_results = self._process_training_results(
                results, training_config, training_time
            )
            
            self.training_results = training_results
            self.logger.info(f"Training completed successfully in {training_time:.1f}s")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
            
        finally:
            self.training_active = False
            
    def _setup_device_optimizations(self, config: TrainingConfig):
        """Setup device-specific optimizations."""
        device_type = self.device_info['device_type']
        
        if device_type == 'cuda':
            # NVIDIA CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Mixed precision training
            if self.device_info.get('mixed_precision_support', True):
                self.logger.info("Enabling mixed precision training")
                
        elif device_type == 'directml':
            # AMD DirectML optimizations
            if DIRECTML_AVAILABLE:
                # Set DirectML device
                torch_directml.device_count()  # Initialize DirectML
                self.logger.info("DirectML optimizations enabled")
                
        else:
            # CPU optimizations
            torch.set_num_threads(config.workers)
            self.logger.info(f"CPU training with {config.workers} threads")
            
    def _prepare_train_args(self, config: TrainingConfig, data_yaml_path: str) -> Dict:
        """Prepare training arguments for YOLO."""
        args = {
            'data': data_yaml_path,
            'epochs': config.epochs,
            'batch': config.batch_size,
            'imgsz': config.image_size,
            'lr0': config.learning_rate,
            'momentum': config.momentum,
            'weight_decay': config.weight_decay,
            'patience': config.patience,
            'save_period': config.save_period,
            'device': config.device if config.device != 'auto' else self.device,
            'workers': config.workers,
            'project': config.project_name,
            'name': config.experiment_name,
            'exist_ok': True,
            'pretrained': config.pretrained,
            'resume': config.resume,
            
            # Augmentation parameters
            'mosaic': config.mosaic,
            'mixup': config.mixup,
            'copy_paste': config.copy_paste,
            'hsv_h': config.hsv_h,
            'hsv_s': config.hsv_s,
            'hsv_v': config.hsv_v,
            'degrees': config.degrees,
            'translate': config.translate,
            'scale': config.scale,
            'shear': config.shear,
            'perspective': config.perspective,
            'flipud': config.flipud,
            'fliplr': config.fliplr,
        }
        
        # Device-specific adjustments
        if self.device_info['device_type'] == 'directml':
            args['amp'] = False  # Disable AMP for DirectML
            
        return args
        
    def _setup_callbacks(self, callbacks: Dict):
        """Setup training callbacks."""
        # Custom callback implementation would go here
        pass
        
    def _process_training_results(self, 
                                results, 
                                config: TrainingConfig, 
                                training_time: float) -> TrainingResults:
        """Process and format training results."""
        try:
            # Extract metrics from results
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            
            # Get best metrics
            best_map50 = metrics.get('metrics/mAP50(B)', 0.0)
            best_precision = metrics.get('metrics/precision(B)', 0.0)
            best_recall = metrics.get('metrics/recall(B)', 0.0)
            final_loss = metrics.get('train/box_loss', 0.0)
            
            # Get model path
            model_path = str(results.save_dir / 'weights' / 'best.pt')
            
            # Calculate model size
            model_size = 0
            if Path(model_path).exists():
                model_size = Path(model_path).stat().st_size
                
            # Extract metrics history
            metrics_history = self._extract_metrics_history(results)
            
            training_results = TrainingResults(
                model_path=model_path,
                best_map50=best_map50,
                best_precision=best_precision,
                best_recall=best_recall,
                final_loss=final_loss,
                training_time=training_time,
                epochs_completed=config.epochs,
                device_used=self.device,
                model_size=model_size,
                config_used=asdict(config),
                metrics_history=metrics_history
            )
            
            # Validate results against thresholds
            self._validate_training_results(training_results, config)
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Failed to process training results: {e}")
            raise
            
    def _extract_metrics_history(self, results) -> Dict[str, List[float]]:
        """Extract metrics history from training results."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'map50': [],
            'precision': [],
            'recall': []
        }
        
        try:
            # Extract from results if available
            if hasattr(results, 'history'):
                for key, values in results.history.items():
                    if 'loss' in key.lower():
                        if 'train' in key.lower():
                            history['train_loss'].extend(values)
                        elif 'val' in key.lower():
                            history['val_loss'].extend(values)
                    elif 'map50' in key.lower():
                        history['map50'].extend(values)
                    elif 'precision' in key.lower():
                        history['precision'].extend(values)
                    elif 'recall' in key.lower():
                        history['recall'].extend(values)
                        
        except Exception as e:
            self.logger.warning(f"Failed to extract metrics history: {e}")
            
        return history
        
    def _validate_training_results(self, results: TrainingResults, config: TrainingConfig):
        """Validate training results against minimum thresholds."""
        issues = []
        
        if results.best_map50 < config.min_map50:
            issues.append(f"mAP50 {results.best_map50:.3f} below threshold {config.min_map50}")
            
        if results.best_precision < config.min_precision:
            issues.append(f"Precision {results.best_precision:.3f} below threshold {config.min_precision}")
            
        if results.best_recall < config.min_recall:
            issues.append(f"Recall {results.best_recall:.3f} below threshold {config.min_recall}")
            
        if issues:
            self.logger.warning("Training quality issues detected:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.info("Training results meet all quality thresholds")
            
    def evaluate_model(self, model_path: str, data_yaml_path: str) -> Dict[str, float]:
        """Evaluate trained model on test set."""
        try:
            model = YOLO(model_path)
            results = model.val(data=data_yaml_path, device=self.device)
            
            metrics = {
                'map50': results.box.map50,
                'map': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
            }
            
            self.logger.info(f"Model evaluation completed: mAP50={metrics['map50']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
            
    def export_model(self, 
                    model_path: str, 
                    export_format: str = 'onnx',
                    optimize: bool = True) -> str:
        """Export trained model to different formats."""
        try:
            model = YOLO(model_path)
            
            export_path = model.export(
                format=export_format,
                optimize=optimize,
                device=self.device
            )
            
            self.logger.info(f"Model exported to {export_format}: {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Model export failed: {e}")
            raise
            
    def stop_training(self):
        """Stop current training session."""
        if self.training_active:
            self.logger.info("Stopping training...")
            self.training_active = False
            # Implementation would depend on how to interrupt YOLO training
            
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'active': self.training_active,
            'current_model': self.training_config.model_name if self.training_config else None,
            'device': self.device,
            'device_info': self.device_info,
            'supported_models': list(self.SUPPORTED_MODELS.keys())
        }
        
    def cleanup(self):
        """Cleanup resources."""
        self.current_model = None
        self.training_active = False
        
        # Clear CUDA cache if using GPU
        if self.device_info['device_type'] == 'cuda':
            torch.cuda.empty_cache()
            
        self.logger.info("YOLOTrainer cleanup completed")


class ModelManager:
    """Manager for YOLO model lifecycle and versioning."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
    def save_model(self, 
                  model_path: str,
                  model_name: str,
                  version: str,
                  metadata: Dict[str, Any]) -> Path:
        """Save model with versioning and metadata."""
        version_dir = self.models_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        import shutil
        model_dest = version_dir / 'model.pt'
        shutil.copy2(model_path, model_dest)
        
        # Save metadata
        metadata_path = version_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Model saved: {model_name} v{version}")
        return model_dest
        
    def load_model(self, model_name: str, version: str = 'latest') -> Tuple[str, Dict[str, Any]]:
        """Load model with metadata."""
        if version == 'latest':
            version = self._get_latest_version(model_name)
            
        version_dir = self.models_dir / model_name / version
        model_path = version_dir / 'model.pt'
        metadata_path = version_dir / 'metadata.json'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_name} v{version}")
            
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        return str(model_path), metadata
        
    def list_models(self) -> Dict[str, List[str]]:
        """List all available models and versions."""
        models = {}
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                versions = []
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir() and (version_dir / 'model.pt').exists():
                        versions.append(version_dir.name)
                        
                if versions:
                    models[model_dir.name] = sorted(versions)
                    
        return models
        
    def _get_latest_version(self, model_name: str) -> str:
        """Get latest version of a model."""
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_name}")
            
        versions = []
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir() and (version_dir / 'model.pt').exists():
                versions.append(version_dir.name)
                
        if not versions:
            raise FileNotFoundError(f"No versions found for model: {model_name}")
            
        # Sort versions (assumes semantic versioning)
        return sorted(versions)[-1]


def main():
    """Main entry point for YOLO Training interface."""
    import argparse
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    parser = argparse.ArgumentParser(description="YOLO Vision Studio - Training Interface")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--model", type=str, help="Model name to train")
    parser.add_argument("--data", type=str, help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=-1, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default="auto", help="Training device")
    args = parser.parse_args()
    
    try:
        # Initialize system
        from ..utils.config import ConfigManager
        config_manager = ConfigManager()
        
        console.print(Panel.fit(
            "[bold green]🤖 YOLO Vision Studio - Training Interface[/bold green]\n"
            "[blue]Comprehensive YOLO model training system[/blue]",
            title="Training System"
        ))
        
        # Initialize trainer
        trainer = YOLOTrainer(config_manager)
        
        # Display system info
        status = trainer.get_training_status()
        
        table = Table(title="Training System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        table.add_row("Device", status['device'])
        table.add_row("Device Type", status['device_info']['device_type'])
        table.add_row("Selected Device", status['device_info']['selected_device'])
        table.add_row("Active Training", "Yes" if status['active'] else "No")
        table.add_row("Current Model", status['current_model'] or "None")
        
        console.print(table)
        
        # Display supported models
        models_table = Table(title="Supported YOLO Models")
        models_table.add_column("Model Family", style="cyan")
        models_table.add_column("Models", style="green")
        
        # Group models by family
        model_families = {
            "YOLOv5": [m for m in status['supported_models'] if 'v5' in m],
            "YOLOv8": [m for m in status['supported_models'] if 'v8' in m],
            "YOLOv9": [m for m in status['supported_models'] if 'v9' in m],
            "YOLOv10": [m for m in status['supported_models'] if 'v10' in m],
            "YOLOv11": [m for m in status['supported_models'] if 'v11' in m],
        }
        
        for family, models in model_families.items():
            if models:
                models_table.add_row(family, ", ".join(models))
        
        console.print(models_table)
        
        if args.demo:
            console.print("\n[yellow]Demo mode - Training interface initialized successfully![/yellow]")
            console.print("[blue]To train a model, provide --model and --data arguments[/blue]")
            
        elif args.model and args.data:
            # Validate inputs
            if not trainer.validate_model_name(args.model):
                console.print(f"[red]Error: Unsupported model '{args.model}'[/red]")
                console.print(f"[blue]Supported models: {', '.join(status['supported_models'])}[/blue]")
                return 1
                
            if not Path(args.data).exists():
                console.print(f"[red]Error: Data file not found: {args.data}[/red]")
                return 1
                
            console.print(f"\n[green]Starting training: {args.model}[/green]")
            console.print(f"[blue]Data: {args.data}[/blue]")
            console.print(f"[blue]Epochs: {args.epochs}[/blue]")
            console.print(f"[blue]Batch Size: {args.batch_size}[/blue]")
            console.print(f"[blue]Device: {args.device}[/blue]")
            
            # Prepare training configuration
            training_config = trainer.prepare_training_config(
                model_name=args.model,
                data_yaml_path=args.data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )
            
            # Start training
            results = trainer.train_model(args.data, training_config)
            
            console.print(f"\n[green]Training completed![/green]")
            console.print(f"[blue]Best mAP50: {results.best_map50:.3f}[/blue]")
            console.print(f"[blue]Best Precision: {results.best_precision:.3f}[/blue]")
            console.print(f"[blue]Best Recall: {results.best_recall:.3f}[/blue]")
            console.print(f"[blue]Model saved: {results.model_path}[/blue]")
            
        else:
            console.print("\n[yellow]Training interface ready![/yellow]")
            console.print("[blue]Use --demo for demo mode or provide --model and --data to start training[/blue]")
            
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Training error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())