"""
Configuration System Tests

Test suite for configuration management and validation.
"""

import unittest
import tempfile
import os
import yaml
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    YOLOVisionConfig,
    HardwareConfig,
    CaptureConfig,
    TrainingConfig,
    AnnotationConfig,
    ProjectConfig,
)


class TestConfigurationClasses(unittest.TestCase):
    """Test configuration dataclasses"""

    def test_hardware_config(self):
        """Test hardware configuration"""
        config = HardwareConfig()

        # Test default values
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.batch_size, -1)
        self.assertEqual(config.workers, -1)

        # Test custom values
        custom_config = HardwareConfig(device="cuda", batch_size=16, workers=4)

        self.assertEqual(custom_config.device, "cuda")
        self.assertEqual(custom_config.batch_size, 16)
        self.assertEqual(custom_config.workers, 4)

    def test_capture_config(self):
        """Test capture configuration"""
        config = CaptureConfig()

        # Test default values
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.quality, 95)
        self.assertFalse(config.show_preview)

        # Test validation
        config.fps = 60
        self.assertEqual(config.fps, 60)

    def test_training_config(self):
        """Test training configuration"""
        config = TrainingConfig()

        # Test default values
        self.assertEqual(config.model, "yolov8n")
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.patience, 50)

        # Test model validation
        self.assertIn(
            config.model, ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        )

    def test_annotation_config(self):
        """Test annotation configuration"""
        config = AnnotationConfig()

        # Test default values
        self.assertEqual(config.confidence_threshold, 0.6)
        self.assertEqual(config.review_threshold, 0.2)
        self.assertTrue(config.use_acc_framework)

    def test_project_config(self):
        """Test project configuration"""
        config = ProjectConfig()

        # Test default values
        self.assertEqual(config.name, "")
        self.assertEqual(config.version, "1.0.0")
        self.assertIsInstance(config.classes, dict)


class TestYOLOVisionConfig(unittest.TestCase):
    """Test main configuration class"""

    def setUp(self):
        """Setup test fixtures"""
        self.config = YOLOVisionConfig()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_initialization(self):
        """Test configuration initialization"""
        self.assertIsInstance(self.config.hardware, HardwareConfig)
        self.assertIsInstance(self.config.capture, CaptureConfig)
        self.assertIsInstance(self.config.training, TrainingConfig)
        self.assertIsInstance(self.config.annotation, AnnotationConfig)
        self.assertIsInstance(self.config.project, ProjectConfig)

    def test_save_and_load_config(self):
        """Test configuration save and load"""
        config_path = Path(self.temp_dir) / "test_config.yaml"

        # Modify some values
        self.config.hardware.device = "cuda"
        self.config.training.epochs = 200
        self.config.project.name = "test_project"

        # Save configuration
        self.config.save(config_path)
        self.assertTrue(config_path.exists())

        # Load configuration
        loaded_config = YOLOVisionConfig.load(config_path)

        # Verify values
        self.assertEqual(loaded_config.hardware.device, "cuda")
        self.assertEqual(loaded_config.training.epochs, 200)
        self.assertEqual(loaded_config.project.name, "test_project")

    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        self.assertTrue(self.config.validate())

        # Test invalid values
        self.config.training.epochs = -1
        self.assertFalse(self.config.validate())

        # Reset to valid
        self.config.training.epochs = 100
        self.assertTrue(self.config.validate())

    def test_environment_variables(self):
        """Test environment variable support"""
        # Set environment variable
        os.environ["YOLO_DEVICE"] = "cpu"
        os.environ["YOLO_BATCH_SIZE"] = "8"

        try:
            # Load configuration with environment variables
            config = YOLOVisionConfig()
            config._load_from_env()

            # Environment variables should override defaults
            self.assertEqual(config.hardware.device, "cpu")
            self.assertEqual(config.hardware.batch_size, 8)

        finally:
            # Cleanup environment variables
            del os.environ["YOLO_DEVICE"]
            del os.environ["YOLO_BATCH_SIZE"]

    def test_config_update(self):
        """Test configuration updates"""
        updates = {
            "hardware": {"device": "directml", "batch_size": 12},
            "training": {"model": "yolov8s", "epochs": 150},
        }

        self.config.update(updates)

        self.assertEqual(self.config.hardware.device, "directml")
        self.assertEqual(self.config.hardware.batch_size, 12)
        self.assertEqual(self.config.training.model, "yolov8s")
        self.assertEqual(self.config.training.epochs, 150)

    def test_config_to_dict(self):
        """Test configuration to dictionary conversion"""
        config_dict = self.config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn("hardware", config_dict)
        self.assertIn("capture", config_dict)
        self.assertIn("training", config_dict)
        self.assertIn("annotation", config_dict)
        self.assertIn("project", config_dict)

        # Test nested structure
        self.assertIsInstance(config_dict["hardware"], dict)
        self.assertIn("device", config_dict["hardware"])

    def test_config_from_dict(self):
        """Test configuration from dictionary creation"""
        config_dict = {
            "hardware": {"device": "cuda", "batch_size": 16},
            "training": {"model": "yolov8m", "epochs": 80},
        }

        config = YOLOVisionConfig.from_dict(config_dict)

        self.assertEqual(config.hardware.device, "cuda")
        self.assertEqual(config.hardware.batch_size, 16)
        self.assertEqual(config.training.model, "yolov8m")
        self.assertEqual(config.training.epochs, 80)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation logic"""

    def setUp(self):
        """Setup test fixtures"""
        self.config = YOLOVisionConfig()

    def test_hardware_validation(self):
        """Test hardware configuration validation"""
        # Valid devices
        for device in ["auto", "cuda", "directml", "cpu"]:
            self.config.hardware.device = device
            self.assertTrue(self.config.validate())

        # Invalid device
        self.config.hardware.device = "invalid"
        self.assertFalse(self.config.validate())

        # Reset
        self.config.hardware.device = "auto"

    def test_training_validation(self):
        """Test training configuration validation"""
        # Valid epochs
        self.config.training.epochs = 100
        self.assertTrue(self.config.validate())

        # Invalid epochs
        self.config.training.epochs = 0
        self.assertFalse(self.config.validate())

        self.config.training.epochs = -10
        self.assertFalse(self.config.validate())

        # Reset
        self.config.training.epochs = 100

    def test_capture_validation(self):
        """Test capture configuration validation"""
        # Valid FPS
        for fps in [1, 30, 60]:
            self.config.capture.fps = fps
            self.assertTrue(self.config.validate())

        # Invalid FPS
        for fps in [0, -1, 61]:
            self.config.capture.fps = fps
            self.assertFalse(self.config.validate())

        # Reset
        self.config.capture.fps = 30

    def test_annotation_validation(self):
        """Test annotation configuration validation"""
        # Valid thresholds
        self.config.annotation.confidence_threshold = 0.5
        self.config.annotation.review_threshold = 0.1
        self.assertTrue(self.config.validate())

        # Invalid thresholds (out of range)
        self.config.annotation.confidence_threshold = 1.5
        self.assertFalse(self.config.validate())

        self.config.annotation.confidence_threshold = -0.1
        self.assertFalse(self.config.validate())

        # Invalid threshold relationship
        self.config.annotation.confidence_threshold = 0.3
        self.config.annotation.review_threshold = 0.5
        self.assertFalse(self.config.validate())

        # Reset
        self.config.annotation.confidence_threshold = 0.6
        self.config.annotation.review_threshold = 0.2


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration with other components"""

    def setUp(self):
        """Setup test fixtures"""
        self.config = YOLOVisionConfig()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_yaml_compatibility(self):
        """Test YAML format compatibility"""
        config_path = Path(self.temp_dir) / "config.yaml"

        # Save configuration
        self.config.save(config_path)

        # Load with PyYAML directly
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        self.assertIsInstance(yaml_data, dict)
        self.assertIn("hardware", yaml_data)
        self.assertIn("training", yaml_data)

    def test_config_merge(self):
        """Test configuration merging"""
        base_config = YOLOVisionConfig()
        base_config.hardware.device = "cuda"
        base_config.training.epochs = 100

        override_config = {"training": {"epochs": 200, "model": "yolov8s"}}

        base_config.update(override_config)

        # Should keep unchanged values
        self.assertEqual(base_config.hardware.device, "cuda")

        # Should update changed values
        self.assertEqual(base_config.training.epochs, 200)
        self.assertEqual(base_config.training.model, "yolov8s")


if __name__ == "__main__":
    unittest.main()
