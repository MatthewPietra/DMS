import sys
import unittest
from pathlib import Path
from src.utils.hardware import GPUInfo, HardwareDetector
            from src.config import YOLOVisionConfig
            from src.training.yolo_trainer import YOLOTrainer

"""
Hardware Detection Tests

Test suite for hardware detection and optimization components.
"""

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestHardwareDetector(unittest.TestCase):
    """Test hardware detection functionality"""

    def setUp(self):
        """Setup test fixtures"""
        self.detector = HardwareDetector()

    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsInstance(self.detector, HardwareDetector)
        self.assertIsNotNone(self.detector.system_info)

    def test_detect_hardware(self):
        """Test hardware detection"""
        specs = self.detector.detect_hardware()

        # Check basic structure
        self.assertIsNotNone(specs)
        self.assertTrue(hasattr(specs, "cpu_count"))
        self.assertTrue(hasattr(specs, "memory_total"))
        self.assertTrue(hasattr(specs, "gpus"))

        # Validate CPU count
        self.assertGreater(specs.cpu_count, 0)

        # Validate memory
        self.assertGreater(specs.memory_total, 0)

        # Validate GPU list
        self.assertIsInstance(specs.gpus, list)

    def test_get_optimal_device(self):
        """Test optimal device selection"""
        device = self.detector.get_optimal_device()

        # Should return a valid device string
        self.assertIsInstance(device, str)
        self.assertIn(device, ["cuda", "directml", "cpu"])

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        batch_size = self.detector.get_optimal_batch_size()

        # Should return a positive integer
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 64)  # Reasonable upper bound

    def test_get_optimal_workers(self):
        """Test optimal worker count calculation"""
        workers = self.detector.get_optimal_workers()

        # Should return a positive integer
        self.assertIsInstance(workers, int)
        self.assertGreater(workers, 0)
        self.assertLessEqual(workers, 16)  # Reasonable upper bound

    def test_gpu_info_structure(self):
        """Test GPU info structure"""
        specs = self.detector.detect_hardware()

        for gpu in specs.gpus:
            self.assertIsInstance(gpu, GPUInfo)
            self.assertIsInstance(gpu.name, str)
            self.assertIsInstance(gpu.device_type, str)
            self.assertIsInstance(gpu.memory_total, int)
            self.assertIsInstance(gpu.memory_available, int)
            self.assertIsInstance(gpu.compute_capability, (str, type(None)))

    def test_validate_device(self):
        """Test device validation"""
        # Test valid devices
        self.assertTrue(self.detector.validate_device("cpu"))

        # Test cuda validation (depends on system)
        cuda_valid = self.detector.validate_device("cuda")
        self.assertIsInstance(cuda_valid, bool)

        # Test directml validation (depends on system)
        directml_valid = self.detector.validate_device("directml")
        self.assertIsInstance(directml_valid, bool)

        # Test invalid device
        self.assertFalse(self.detector.validate_device("invalid_device"))

    def test_memory_estimation(self):
        """Test memory estimation functions"""
        # Test batch size estimation
        batch_size = self.detector._estimate_batch_size_from_memory(4096)  # 4GB
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

        # Test with different memory sizes
        small_batch = self.detector._estimate_batch_size_from_memory(1024)  # 1GB
        large_batch = self.detector._estimate_batch_size_from_memory(16384)  # 16GB

        self.assertLessEqual(small_batch, large_batch)

class TestHardwareIntegration(unittest.TestCase):
    """Test hardware integration with other components"""

    def setUp(self):
        """Setup test fixtures"""
        self.detector = HardwareDetector()

    def test_configuration_integration(self):
        """Test integration with configuration system"""
        try:
            config = YOLOVisionConfig()
            specs = self.detector.detect_hardware()

            # Test that hardware specs can be used in configuration
            self.assertIsNotNone(specs)

        except ImportError:
            self.skipTest("Configuration module not available")

    def test_training_integration(self):
        """Test integration with training system"""
        try:
            # Test that hardware detector can be used by trainer
            device = self.detector.get_optimal_device()
            batch_size = self.detector.get_optimal_batch_size()

            self.assertIsInstance(device, str)
            self.assertIsInstance(batch_size, int)

        except ImportError:
            self.skipTest("Training module not available")

if __name__ == "__main__":
    unittest.main()
