#!/usr/bin/env python3
"""
DMS - Main Studio Interface.

Provides the main entry point for the DMS (Detection Model Suite) pipeline.
Integrates all components: capture, annotation, training, and auto-annotation.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import core components with graceful handling
try:
    from .annotation.annotation_interface import AnnotationInterface

    ANNOTATION_AVAILABLE = True
except ImportError:
    ANNOTATION_AVAILABLE = False
    AnnotationInterface = None  # type: ignore

try:
    from .annotation.coco_exporter import COCOExporter

    COCO_EXPORTER_AVAILABLE = True
except ImportError:
    COCO_EXPORTER_AVAILABLE = False
    COCOExporter = None  # type: ignore

try:
    from .auto_annotation.auto_annotator import AutoAnnotator

    AUTO_ANNOTATION_AVAILABLE = True
except ImportError:
    AUTO_ANNOTATION_AVAILABLE = False
    AutoAnnotator = None  # type: ignore

try:
    from .capture.window_capture import WindowCaptureSystem

    CAPTURE_AVAILABLE = True
except ImportError:
    CAPTURE_AVAILABLE = False
    WindowCaptureSystem = None  # type: ignore

try:
    from .utils.config import ConfigManager

    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Error importing DMS core components: {e}")
    print("Please ensure all dependencies are installed.")
    CONFIG_AVAILABLE = False
    ConfigManager = None  # type: ignore

try:
    from .training.yolo_trainer import YOLOTrainer

    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    YOLOTrainer = None  # type: ignore

try:
    from .utils.hardware import HardwareDetector

    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    HardwareDetector = None  # type: ignore

try:
    from .utils.logger import setup_logger

    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    setup_logger = None  # type: ignore


class DMS:
    """
    Main DMS (Detection Model Suite) class that integrates all components.

    Provides a unified interface for:
    - Project management
    - Screen capture
    - Model training
    - Annotation
    - Auto-annotation
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize DMS with configuration.

        Args:
            config_path: Optional path to configuration file.

        Raises:
            ImportError: If core components are not available.
        """
        if not CONFIG_AVAILABLE or ConfigManager is None:
            raise ImportError(
                "DMS core components not available. Please install dependencies."
            )

        if not LOGGER_AVAILABLE or setup_logger is None:
            raise ImportError("Logger component not available.")

        self.logger = setup_logger("dms")
        self.logger.info("Initializing DMS...")

        # Load configuration
        self.config = ConfigManager(config_path)

        # Initialize components with proper type annotations
        self.hardware_detector: Optional[Any] = None
        self.capture_system: Optional[Any] = None
        self.trainer: Optional[Any] = None
        self.auto_annotator: Optional[Any] = None

        # Current project state
        self.current_project: Optional[Dict[str, Any]] = None
        self.current_project_path: Optional[Path] = None

        # Initialize components
        self._initialize_components()

        self.logger.info("DMS initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all DMS components."""
        try:
            # Hardware detection
            if HARDWARE_AVAILABLE and HardwareDetector is not None:
                self.hardware_detector = HardwareDetector()  # type: ignore
                if self.hardware_detector is not None:
                    device_type = self.hardware_detector.get_device_type()
                    self.logger.info(f"Hardware detected: {device_type}")
            else:
                self.hardware_detector = None
                self.logger.warning("Hardware detection not available")

            # Capture system
            if (
                CAPTURE_AVAILABLE
                and WindowCaptureSystem is not None
                and self.hardware_detector is not None
            ):
                # Create capture config from the config manager
                capture_config = self.config.get_capture_config()

                self.capture_system = WindowCaptureSystem(
                    capture_config, self.hardware_detector
                )
            else:
                self.capture_system = None
                self.logger.warning("Capture system not available")

            # Training system
            if TRAINING_AVAILABLE and YOLOTrainer is not None:
                self.trainer = YOLOTrainer(self.config)
            else:
                self.trainer = None
                self.logger.warning("Training system not available")

            # Auto-annotation system
            if AUTO_ANNOTATION_AVAILABLE and AutoAnnotator is not None:
                self.auto_annotator = AutoAnnotator(self.config)
            else:
                self.auto_annotator = None
                self.logger.warning("Auto-annotation system not available")

            self.logger.info("Components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def create_project(
        self, name: str, description: str = "", classes: Optional[List[str]] = None
    ) -> Path:
        """
        Create a new DMS project.

        Args:
            name: Project name.
            description: Project description.
            classes: List of class names for annotation.

        Returns:
            Path to the created project.

        Raises:
            OSError: If project directory creation fails.
        """
        if classes is None:
            classes = ["object"]

        # Create project directory structure
        project_path = Path("data/projects") / name
        project_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (project_path / "images").mkdir(exist_ok=True)
        (project_path / "annotations").mkdir(exist_ok=True)
        (project_path / "models").mkdir(exist_ok=True)
        (project_path / "exports").mkdir(exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)

        # Create project configuration
        project_config = {
            "name": name,
            "description": description,
            "classes": classes,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
        }

        # Save project config
        config_file = project_path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(project_config, f, indent=2)

        # Save classes file
        classes_file = project_path / "classes.txt"
        with open(classes_file, "w", encoding="utf-8") as f:
            for class_name in classes:
                f.write(f"{class_name}\n")

        self.logger.info(f"Created project: {name} at {project_path}")
        return project_path

    def load_project(self, project_path: Union[str, Path]) -> None:
        """
        Load an existing project.

        Args:
            project_path: Path to the project directory.

        Raises:
            FileNotFoundError: If project or configuration not found.
        """
        project_path_obj = Path(project_path)
        if not project_path_obj.exists():
            raise FileNotFoundError(f"Project not found: {project_path}")

        # Load project configuration
        config_file = project_path_obj / "config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                project_config = json.load(f)

            self.current_project = project_config
            self.current_project_path = project_path_obj
            self.logger.info(f"Loaded project: {project_config['name']}")
        else:
            raise FileNotFoundError(f"Project configuration not found: {config_file}")

    def start_capture(
        self, duration: Optional[int] = None, output_dir: str = "data/captured"
    ) -> Dict[str, Any]:
        """
        Start a screen capture session.

        Args:
            duration: Capture duration in seconds (None for manual stop).
            output_dir: Output directory for captured images.

        Returns:
            Capture session results.

        Raises:
            RuntimeError: If capture system is not available or no windows found.
        """
        if self.capture_system is None:
            raise RuntimeError("Capture system not available")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get available windows
        windows = self.capture_system.get_available_windows()

        if not windows:
            raise RuntimeError("No windows found for capture")

        # Start capture session
        session_id = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.capture_system.start_session(
            session_id=session_id,
            output_dir=output_path,
            window_title=windows[0].title,  # Use first available window
            fps=5,
        )

        self.logger.info(f"Started capture session: {session_id}")

        # Run capture for specified duration or until stopped
        if duration:
            time.sleep(duration)
            self.capture_system.stop_session(session_id)

        return {
            "session_id": session_id,
            "output_dir": str(output_path),
            "windows_found": len(windows),
        }

    def train_model(
        self, data_path: str, model_name: str = "yolov8n", epochs: int = 100
    ) -> Dict[str, Any]:
        """
        Train a YOLO model.

        Args:
            data_path: Path to training data.
            model_name: YOLO model name (e.g., yolov8n, yolov8s).
            epochs: Number of training epochs.

        Returns:
            Training results.

        Raises:
            RuntimeError: If training system is not available.
        """
        if self.trainer is None:
            raise RuntimeError("Training system not available")

        self.logger.info(f"Starting training: {model_name} for {epochs} epochs")

        # Prepare training configuration
        training_config = self.trainer.prepare_training_config(
            model_name=model_name, data_yaml_path=data_path, epochs=epochs
        )

        # Start training
        results = self.trainer.train_model(
            data_yaml_path=data_path, training_config=training_config
        )

        self.logger.info(f"Training completed: {results.model_path}")

        return {
            "model_path": results.model_path,
            "best_map50": results.best_map50,
            "training_time": results.training_time,
            "epochs_completed": results.epochs_completed,
        }

    def start_annotation(self, data_path: str, auto_annotate: bool = False) -> None:
        """
        Start the annotation interface.

        Args:
            data_path: Path to images for annotation.
            auto_annotate: Whether to enable auto-annotation.

        Raises:
            RuntimeError: If annotation system is not available.
        """
        if AnnotationInterface is None:
            raise RuntimeError("Annotation system not available")

        self.logger.info(f"Starting annotation interface: {data_path}")

        # Create annotation interface
        annotation_interface = AnnotationInterface(self.config)

        # Load images
        image_path = Path(data_path)
        if image_path.is_file():
            # Single image
            annotation_interface.load_image(str(image_path))
        elif image_path.is_dir():
            # Directory of images
            jpg_files = list(image_path.glob("*.jpg"))
            png_files = list(image_path.glob("*.png"))
            image_files = jpg_files + png_files
            if image_files:
                annotation_interface.load_images([str(f) for f in image_files])

        # Launch interface
        annotation_interface.show()

    def auto_annotate(
        self, data_path: str, model_path: str, output_path: str = "data/auto_annotated"
    ) -> Dict[str, Any]:
        """
        Run auto-annotation on images.

        Args:
            data_path: Path to images.
            model_path: Path to trained model.
            output_path: Output directory for annotations.

        Returns:
            Auto-annotation results.

        Raises:
            RuntimeError: If auto-annotation system is not available.
            FileNotFoundError: If no images found in data_path.
        """
        if self.auto_annotator is None:
            raise RuntimeError("Auto-annotation system not available")

        self.logger.info(f"Starting auto-annotation: {data_path}")

        # Get image files
        image_path = Path(data_path)
        if image_path.is_file():
            image_files = [str(image_path)]
        else:
            jpg_files = list(image_path.glob("*.jpg"))
            png_files = list(image_path.glob("*.png"))
            image_files = [str(f) for f in jpg_files + png_files]

        if not image_files:
            raise FileNotFoundError(f"No images found in: {data_path}")

        # Run auto-annotation
        results = self.auto_annotator.batch_annotate(
            image_paths=image_files, output_dir=output_path
        )

        self.logger.info(f"Auto-annotation completed: {len(results)} images processed")

        return {
            "images_processed": len(results),
            "output_path": output_path,
            "results": results,
        }

    def export_dataset(
        self, data_path: str, output_path: str, format: str = "coco"
    ) -> Dict[str, Any]:
        """
        Export dataset in specified format.

        Args:
            data_path: Path to dataset.
            output_path: Output directory.
            format: Export format (coco, yolo, pascal).

        Returns:
            Export results.

        Raises:
            RuntimeError: If export system is not available or export fails.
        """
        if not COCO_EXPORTER_AVAILABLE or COCOExporter is None:
            raise RuntimeError("Export system not available")

        self.logger.info(f"Exporting dataset: {data_path} to {format}")

        # Import exporter
        exporter = COCOExporter()  # type: ignore
        success = exporter.export_dataset(
            project_path=Path(data_path),
            output_path=Path(output_path),
            export_format=format.upper(),
        )

        if success:
            self.logger.info(f"Dataset exported successfully: {output_path}")
            return {"success": True, "output_path": output_path}
        else:
            raise RuntimeError(f"Failed to export dataset to {format}")

    def _basic_annotation(self, data_path: str) -> None:
        """
        Run basic annotation workflow for demonstration.

        Args:
            data_path: Path to data for annotation.
        """
        self.logger.info("Starting basic annotation workflow")

        # Create project
        project_path = self.create_project(
            name="demo_project",
            description="Demo project for basic annotation",
            classes=["person", "car", "bike"],
        )

        # Start capture
        self.start_capture(duration=10)

        # Start annotation
        self.start_annotation(str(project_path / "images"))

        self.logger.info("Basic annotation workflow completed")

    def get_project_status(self) -> Dict[str, Any]:
        """
        Get current project status.

        Returns:
            Dictionary containing project status information.
        """
        if not self.current_project or self.current_project_path is None:
            return {"status": "no_project_loaded"}

        project_path = self.current_project_path

        # Count files
        jpg_files = list((project_path / "images").glob("*.jpg"))
        png_files = list((project_path / "images").glob("*.png"))
        image_count = len(jpg_files + png_files)
        annotation_count = len(list((project_path / "annotations").glob("*.json")))
        model_count = len(list((project_path / "models").glob("*.pt")))

        return {
            "project_name": self.current_project["name"],
            "project_path": str(project_path),
            "images": image_count,
            "annotations": annotation_count,
            "models": model_count,
            "classes": self.current_project.get("classes", []),
            "created_at": self.current_project.get("created_at", ""),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up DMS resources")

        if self.capture_system is not None:
            if hasattr(self.capture_system, "shutdown"):
                self.capture_system.shutdown()

        if self.trainer is not None:
            if hasattr(self.trainer, "cleanup"):
                self.trainer.cleanup()

        if self.auto_annotator is not None:
            if hasattr(self.auto_annotator, "cleanup"):
                self.auto_annotator.cleanup()


def main() -> None:
    """Run main entry point for DMS."""
    parser = argparse.ArgumentParser(description="DMS - Detection Model Suite")
    parser.add_argument("--demo", action="store_true", help="Run demo workflow")
    parser.add_argument("--project", type=str, help="Project name")
    parser.add_argument("--capture", type=int, help="Capture duration in seconds")
    parser.add_argument("--train", type=str, help="Train model with data path")
    parser.add_argument("--annotate", type=str, help="Start annotation with data path")

    args = parser.parse_args()

    # Initialize DMS
    dms = DMS()

    try:
        if args.demo:
            # Run demo workflow
            dms._basic_annotation("data/demo")

        elif args.capture:
            # Run capture
            results = dms.start_capture(duration=args.capture)
            print(f"Capture completed: {results}")

        elif args.train:
            # Run training
            results = dms.train_model(args.train)
            print(f"Training completed: {results}")

        elif args.annotate:
            # Start annotation
            dms.start_annotation(args.annotate)

        else:
            # Interactive mode
            print("DMS - Detection Model Suite")
            print("Use --help for command line options")
            print("Or run without arguments for interactive mode")

            # Start annotation interface
            dms.start_annotation("data/images")

    finally:
        dms.cleanup()


if __name__ == "__main__":
    main()
