"""Integration Tests

End-to-end workflow testing for YOLO Vision Studio.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from src.cli import create_project, main
from src.config import YOLOVisionConfig
from src.studio import DMS
from src.utils.hardware import HardwareDetector
from src.utils.metrics import BoundingBox
from src.utils.performance import MemoryManager, PerformanceMonitor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestProjectWorkflow(unittest.TestCase):
    """Test complete project workflow"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "test_project"

        # Create test configuration
        self.config = YOLOVisionConfig()
        self.config.project.name = "test_project"
        self.config.project.classes = {"0": "person", "1": "car"}

        # Initialize studio
        self.studio = YOLOVisionStudio(config=self.config)

    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_project_creation_workflow(self):
        """Test complete project creation workflow"""
        # Create project
        success = self.studio.create_project(
            project_path=self.project_path,
            project_name="test_project",
            classes=["person", "car"],
        )

        self.assertTrue(success)
        self.assertTrue(self.project_path.exists())

        # Verify project structure
        expected_dirs = [
            "images",
            "annotations",
            "models",
            "exports",
            "logs",
            "config",
            "cache",
        ]

        for dir_name in expected_dirs:
            dir_path = self.project_path / dir_name
            self.assertTrue(dir_path.exists(), "Directory {dir_name} should exist")

        # Verify configuration file
        config_file = self.project_path / "config" / "project_config.yaml"
        self.assertTrue(config_file.exists())

        # Verify classes file
        classes_file = self.project_path / "config" / "classes.json"
        self.assertTrue(classes_file.exists())

        with open(classes_file, "r") as f:
            classes = json.load(f)

        self.assertEqual(classes, {"0": "person", "1": "car"})

    def test_project_loading_workflow(self):
        """Test project loading workflow"""
        # First create a project
        self.studio.create_project(
            project_path=self.project_path,
            project_name="test_project",
            classes=["person", "car"],
        )

        # Create new studio instance and load project
        new_studio = YOLOVisionStudio()
        success = new_studio.load_project(self.project_path)

        self.assertTrue(success)
        self.assertEqual(new_studio.project_path, self.project_path)
        self.assertIsNotNone(new_studio.config)

    @patch("src.capture.window_capture.WindowCapture.capture_sequence")
    def test_capture_workflow(self, mock_capture):
        """Test image capture workflow"""
        # Setup mock
        mock_capture.return_value = ["image1.jpg", "image2.jpg"]

        # Create project
        self.studio.create_project(
            project_path=self.project_path,
            project_name="test_project",
            classes=["person", "car"],
        )

        # Test capture
        captured_images = self.studio.capture_images(
            duration=1.0, fps=1, window_title=None
        )

        self.assertIsNotNone(captured_images)
        mock_capture.assert_called_once()

    def test_annotation_workflow(self):
        """Test annotation workflow"""
        # Create project
        self.studio.create_project(
            project_path=self.project_path,
            project_name="test_project",
            classes=["person", "car"],
        )

        # Create sample image file
        images_dir = self.project_path / "images"
        sample_image = images_dir / "test.jpg"

        # Create a dummy image file (just for testing)
        sample_image.touch()

        # Create sample annotations
        annotations = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=1.0),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1, confidence=1.0),
        ]

        # Save annotations
        annotation_file = self.project_path / "annotations" / "test.txt"
        with open(annotation_file, "w") as f:
            for ann in annotations:
                center_x = (ann.x1 + ann.x2) / 2
                center_y = (ann.y1 + ann.y2) / 2
                width = ann.x2 - ann.x1
                height = ann.y2 - ann.y1
                f.write("{ann.class_id} {center_x} {center_y} {width} {height}\n")

        # Verify annotation file exists
        self.assertTrue(annotation_file.exists())

        # Verify annotation content
        with open(annotation_file, "r") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)

    @patch("src.training.yolo_trainer.YOLOTrainer.train")
    def test_training_workflow(self, mock_train):
        """Test training workflow"""
        # Setup mock
        mock_train.return_value = {
            "success": True,
            "model_path": "test_model.pt",
            "metrics": {"mAP": 0.8},
        }

        # Create project with annotations
        self.studio.create_project(
            project_path=self.project_path,
            project_name="test_project",
            classes=["person", "car"],
        )

        # Create sample training data
        self._create_sample_training_data()

        # Test training
        result = self.studio.train_model(model_name="yolov8n", epochs=1, batch_size=2)

        self.assertIsNotNone(result)
        mock_train.assert_called_once()

    def test_export_workflow(self):
        """Test dataset export workflow"""
        # Create project
        self.studio.create_project(
            project_path=self.project_path,
            project_name="test_project",
            classes=["person", "car"],
        )

        # Create sample data
        self._create_sample_training_data()

        # Test export
        export_path = self.project_path / "exports" / "coco_export"

        success = self.studio.export_dataset(
            format="coco", output_path=export_path, train_split=0.8, val_split=0.2
        )

        self.assertTrue(success)
        self.assertTrue(export_path.exists())

    def _create_sample_training_data(self):
        """Create sample training data for testing"""
        images_dir = self.project_path / "images"
        annotations_dir = self.project_path / "annotations"

        # Create sample images and annotations
        for i in range(3):
            # Create dummy image file
            image_file = images_dir / "sample_{i}.jpg"
            image_file.touch()

            # Create corresponding annotation
            annotation_file = annotations_dir / "sample_{i}.txt"
            with open(annotation_file, "w") as f:
                f.write("0 0.5 0.5 0.2 0.3\n")
                f.write("1 0.7 0.3 0.1 0.2\n")

class TestAutoAnnotationWorkflow(unittest.TestCase):
    """Test auto-annotation workflow"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "auto_annotation_project"

        self.config = YOLOVisionConfig()
        self.config.project.name = "auto_annotation_project"
        self.config.project.classes = {"0": "person", "1": "car"}

        self.studio = YOLOVisionStudio(config=self.config)

    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.auto_annotation.auto_annotator.AutoAnnotator.annotate_batch")
    def test_auto_annotation_workflow(self, mock_annotate):
        """Test auto-annotation workflow"""
        # Setup mock
        mock_annotate.return_value = {
            "annotations": [
                BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.8)
            ],
            "confidence_scores": [0.8],
            "processing_time": 1.0,
        }

        # Create project
        self.studio.create_project(
            project_path=self.project_path,
            project_name="auto_annotation_project",
            classes=["person", "car"],
        )

        # Create sample images
        images_dir = self.project_path / "images"
        for i in range(3):
            image_file = images_dir / "auto_{i}.jpg"
            image_file.touch()

        # Test auto-annotation
        result = self.studio.auto_annotate(
            confidence_threshold=0.7, review_threshold=0.3, batch_size=2
        )

        self.assertIsNotNone(result)
        mock_annotate.assert_called()

class TestHardwareIntegration(unittest.TestCase):
    """Test hardware detection and optimization integration"""

    def setUp(self):
        """Setup test fixtures"""
        self.detector = HardwareDetector()

    def test_hardware_detection_integration(self):
        """Test hardware detection integration"""
        # Test hardware detection
        specs = self.detector.detect_hardware()

        self.assertIsNotNone(specs)
        self.assertGreater(specs.cpu_count, 0)
        self.assertGreater(specs.memory_total, 0)

        # Test device selection
        device = self.detector.get_optimal_device()
        self.assertIn(device, ["cuda", "directml", "cpu"])

        # Test batch size optimization
        batch_size = self.detector.get_optimal_batch_size()
        self.assertGreater(batch_size, 0)

    def test_configuration_hardware_integration(self):
        """Test configuration system with hardware detection"""
        config = YOLOVisionConfig()

        # Test that hardware settings can be applied
        config.hardware.device = "auto"
        config.hardware.batch_size = -1  # Auto-detect
        config.hardware.workers = -1  # Auto-detect

        # Validate configuration
        self.assertTrue(config.validate())

        # Test hardware optimization
        specs = self.detector.detect_hardware()
        if specs.gpus:
            optimal_device = (
                "cuda"
                if any(gpu.device_type == "CUDA" for gpu in specs.gpus)
                else "directml"
            )
        else:
            optimal_device = "cpu"

        self.assertIn(optimal_device, ["cuda", "directml", "cpu"])

class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_command_integration(self):
        """Test CLI command integration"""
        # Test hardware command
        with patch("sys.argv", ["yolo-vision", "hardware"]):
            with patch("src.cli.show_hardware_info") as mock_hardware:
                try:
                    main()
                    mock_hardware.assert_called_once()
                except SystemExit:
                    pass  # CLI commands often exit

    def test_cli_project_creation(self):
        """Test CLI project creation"""
        project_path = Path(self.temp_dir) / "cli_project"

        # Test project creation via CLI function
        success = create_project(
            project_path=str(project_path),
            project_name="cli_project",
            classes=["person", "car"],
        )

        self.assertTrue(success)
        self.assertTrue(project_path.exists())

class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = YOLOVisionConfig()
        self.studio = YOLOVisionStudio(config=self.config)

    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_project_path_handling(self):
        """Test handling of invalid project paths"""
        invalid_path = Path("/invalid/path/that/does/not/exist")

        # Should handle gracefully
        success = self.studio.load_project(invalid_path)
        self.assertFalse(success)

    def test_missing_dependencies_handling(self):
        """Test handling of missing dependencies"""
        # Test with mock missing dependency
        with patch("src.utils.hardware.torch", None):
            detector = HardwareDetector()

            # Should not crash
            specs = detector.detect_hardware()
            self.assertIsNotNone(specs)

    def test_configuration_error_handling(self):
        """Test configuration error handling"""
        config = YOLOVisionConfig()

        # Test invalid configuration
        config.training.epochs = -1
        config.capture.fps = 0

        # Should detect invalid configuration
        self.assertFalse(config.validate())

    def test_file_permission_error_handling(self):
        """Test file permission error handling"""
        # Create read-only directory
        readonly_dir = Path(self.temp_dir) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        try:
            # Should handle permission errors gracefully
            success = self.studio.create_project(
                project_path=readonly_dir / "project",
                project_name="test",
                classes=["person"],
            )

            # Should fail gracefully
            self.assertFalse(success)

        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

class TestPerformanceIntegration(unittest.TestCase):
    """Test performance optimization integration"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_optimization_integration(self):
        """Test memory optimization integration"""
        memory_manager = MemoryManager()

        # Test memory information
        gpu_info = memory_manager.get_gpu_memory_info()
        sys_info = memory_manager.get_system_memory_info()

        self.assertIsInstance(gpu_info, dict)
        self.assertIsInstance(sys_info, dict)

        # Test batch size optimization
        optimized_batch = memory_manager.optimize_batch_size(
            base_batch_size=32, memory_per_item_mb=100, device="cpu"
        )

        self.assertGreater(optimized_batch, 0)
        self.assertLessEqual(optimized_batch, 32)

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        monitor = PerformanceMonitor(monitoring_interval=0.1)

        # Test monitoring
        monitor.start_monitoring()

        # Get metrics
        current_metrics = monitor.get_current_metrics()
        self.assertIsNotNone(current_metrics)

        # Stop monitoring
        monitor.stop_monitoring()

if __name__ == "__main__":
    unittest.main()
