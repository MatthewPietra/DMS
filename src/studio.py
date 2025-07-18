import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .annotation.annotation_interface import AnnotationInterface
from .annotation.coco_exporter import COCOExporter
from .auto_annotation.auto_annotator import AutoAnnotator
from .capture.window_capture import CaptureConfig, WindowCaptureSystem
from .config import Config
from .training.yolo_trainer import YOLOTrainer
from .utils.hardware import HardwareDetector
from .utils.logger import setup_logger

#!/usr/bin/env python3
"""
DMS - Main Studio Interface

Provides the main entry point for the DMS (Detection Model Suite) pipeline.
Integrates all components: capture, annotation, training, and auto-annotation.
"""

# Import core components with graceful handling
try:
    CONFIG_AVAILABLE = True
except ImportError as _e:
    print("Error importing DMS core components: {e}")
    print("Please ensure all dependencies are installed.")
    CONFIG_AVAILABLE = False

# Try to import optional components
try:
    ANNOTATION_AVAILABLE = True
except ImportError:
    ANNOTATION_AVAILABLE = False
    AnnotationInterface = None

try:
    AUTO_ANNOTATION_AVAILABLE = True
except ImportError:
    AUTO_ANNOTATION_AVAILABLE = False
    AutoAnnotator = None

try:
    CAPTURE_AVAILABLE = True
except ImportError:
    CAPTURE_AVAILABLE = False
    CaptureConfig = None
    WindowCaptureSystem = None

try:
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    YOLOTrainer = None

try:
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    HardwareDetector = None


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

    def __init__(self, config_path: Optional[str] = None):
        """Initialize DMS with configuration."""
        if not CONFIG_AVAILABLE:
            raise ImportError(
                "DMS core components not available. Please install dependencies."
            )

        self.logger = setup_logger("dms")
        self.logger.info("Initializing DMS...")

        # Load configuration
        self.config = Config(config_path)

        # Initialize components
        self._initialize_components()

        # Current project state
        self.current_project = None
        self.current_project_path = None

        self.logger.info("DMS initialized successfully")

    def _initialize_components(self):
        """Initialize all DMS components."""
        try:
            # Hardware detection
            if HARDWARE_AVAILABLE:
                self.hardware_detector = HardwareDetector()
                self.logger.info(
                    "Hardware detected: {self.hardware_detector.get_device_type()}"
                )
            else:
                self.hardware_detector = None
                self.logger.warning("Hardware detection not available")

            # Capture system
            if CAPTURE_AVAILABLE and self.hardware_detector:
                capture_config = CaptureConfig()
                self.capture_system = WindowCaptureSystem(
                    capture_config, self.hardware_detector
                )
            else:
                self.capture_system = None
                self.logger.warning("Capture system not available")

            # Training system
            if TRAINING_AVAILABLE:
                self.trainer = YOLOTrainer(self.config)
            else:
                self.trainer = None
                self.logger.warning("Training system not available")

            # Auto-annotation system
            if AUTO_ANNOTATION_AVAILABLE:
                self.auto_annotator = AutoAnnotator(self.config)
            else:
                self.auto_annotator = None
                self.logger.warning("Auto-annotation system not available")

            self.logger.info("Components initialized successfully")

        except Exception as _e:
            self.logger.error("Failed to initialize components: {e}")
            raise

    def create_project(
        self, name: str, description: str = "", classes: List[str] = None
    ) -> Path:
        """
        Create a new DMS project.

        Args:
            name: Project name
            description: Project description
            classes: List of class names for annotation

        Returns:
            Path to the created project
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
        with open(project_path / "config.json", "w") as f:
            json.dump(project_config, f, indent=2)

        # Save classes file
        with open(project_path / "classes.txt", "w") as f:
            for class_name in classes:
                f.write("{class_name}\n")

        self.logger.info("Created project: {name} at {project_path}")
        return project_path

    def load_project(self, project_path: str):
        """Load an existing project."""
        project_path = Path(project_path)
        if not project_path.exists():
            raise FileNotFoundError("Project not found: {project_path}")

        # Load project configuration
        config_file = project_path / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                project_config = json.load(f)

            self.current_project = project_config
            self.current_project_path = project_path
            self.logger.info("Loaded project: {project_config['name']}")
        else:
            raise FileNotFoundError("Project configuration not found: {config_file}")

    def start_capture(
        self, duration: Optional[int] = None, output_dir: str = "data/captured"
    ) -> Dict[str, Any]:
        """
        Start a screen capture session.

        Args:
            duration: Capture duration in seconds (None for manual stop)
            output_dir: Output directory for captured images

        Returns:
            Capture session results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get available windows
        windows = self.capture_system.get_available_windows()

        if not windows:
            raise RuntimeError("No windows found for capture")

        # Start capture session
        session_id = "capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        _session = self.capture_system.start_session(
            session_id=session_id,
            output_dir=output_path,
            window_title=windows[0].title,  # Use first available window
            fps=5,
        )

        self.logger.info("Started capture session: {session_id}")

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
            data_path: Path to training data
            model_name: YOLO model name (e.g., yolov8n, yolov8s)
            epochs: Number of training epochs

        Returns:
            Training results
        """
        self.logger.info("Starting training: {model_name} for {epochs} epochs")

        # Prepare training configuration
        training_config = self.trainer.prepare_training_config(
            model_name=model_name, data_yaml_path=data_path, epochs=epochs
        )

        # Start training
        results = self.trainer.train_model(
            data_yaml_path=data_path, training_config=training_config
        )

        self.logger.info("Training completed: {results.model_path}")

        return {
            "model_path": results.model_path,
            "best_map50": results.best_map50,
            "training_time": results.training_time,
            "epochs_completed": results.epochs_completed,
        }

    def start_annotation(self, data_path: str, auto_annotate: bool = False):
        """
        Start the annotation interface.

        Args:
            data_path: Path to images for annotation
            auto_annotate: Whether to enable auto-annotation
        """
        self.logger.info("Starting annotation interface: {data_path}")

        # Create annotation interface
        annotation_interface = AnnotationInterface(self.config)

        # Load images
        image_path = Path(data_path)
        if image_path.is_file():
            # Single image
            annotation_interface.load_image(str(image_path))
        elif image_path.is_dir():
            # Directory of images
            image_files = list(image_path.glob("*.jpg")) + list(
                image_path.glob("*.png")
            )
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
            data_path: Path to images
            model_path: Path to trained model
            output_path: Output directory for annotations

        Returns:
            Auto-annotation results
        """
        self.logger.info("Starting auto-annotation: {data_path}")

        # Get image files
        image_path = Path(data_path)
        if image_path.is_file():
            image_files = [str(image_path)]
        else:
            image_files = [
                str(f) for f in image_path.glob("*.jpg") + image_path.glob("*.png")
            ]

        if not image_files:
            raise FileNotFoundError("No images found in: {data_path}")

        # Run auto-annotation
        results = self.auto_annotator.batch_annotate(
            image_paths=image_files, output_dir=output_path
        )

        self.logger.info("Auto-annotation completed: {len(results)} images processed")

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
            data_path: Path to dataset
            output_path: Output directory
            format: Export format (coco, yolo, pascal)

        Returns:
            Export results
        """
        self.logger.info("Exporting dataset: {data_path} to {format}")

        # Import exporter
        exporter = COCOExporter()
        success = exporter.export_dataset(
            project_path=Path(data_path),
            output_path=Path(output_path),
            export_format=format.upper(),
        )

        if success:
            self.logger.info("Dataset exported successfully: {output_path}")
            return {"success": True, "output_path": output_path}
        else:
            raise RuntimeError("Failed to export dataset to {format}")

    def _basic_annotation(self, data_path: str):
        """Basic annotation workflow for demonstration."""
        self.logger.info("Starting basic annotation workflow")

        # Create project
        project_path = self.create_project(
            name="demo_project",
            description="Demo project for basic annotation",
            classes=["person", "car", "bike"],
        )

        # Start capture
        _capture_results = self.start_capture(duration=10)

        # Start annotation
        self.start_annotation(str(project_path / "images"))

        self.logger.info("Basic annotation workflow completed")

    def get_project_status(self) -> Dict[str, Any]:
        """Get current project status."""
        if not self.current_project:
            return {"status": "no_project_loaded"}

        project_path = self.current_project_path

        # Count files
        image_count = len(
            list((project_path / "images").glob("*.jpg"))
            + list((project_path / "images").glob("*.png"))
        )
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

    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up DMS resources")

        if hasattr(self, "capture_system"):
            self.capture_system.shutdown()

        if hasattr(self, "trainer"):
            self.trainer.cleanup()

        if hasattr(self, "auto_annotator"):
            self.auto_annotator.cleanup()


def main():
    """Main entry point for DMS."""
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
            print("Capture completed: {results}")

        elif args.train:
            # Run training
            _results = dms.train_model(args.train)
            print("Training completed: {results}")

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
