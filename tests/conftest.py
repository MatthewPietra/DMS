import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    import PyQt5
except ImportError:
    PyQt5 = None

try:
    import PyQt6
except ImportError:
    PyQt6 = None

try:
    import PySide6
except ImportError:
    PySide6 = None

"""Pytest configuration and fixtures for DMS test suite.

This module provides common fixtures and configuration for all DMS tests.
"""

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Fixture providing path to test data directory.

    Returns:
        Path to test data directory
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Fixture providing temporary directory for tests.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Fixture providing mock configuration.

    Returns:
        Mock configuration dictionary
    """
    return {
        "hardware": {
            "device": "cpu",
            "batch_size": 16,
            "num_workers": 4,
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
            "patience": 5,
        },
        "capture": {
            "fps": 5,
            "quality": "medium",
            "format": "jpg",
        },
        "annotation": {
            "auto_save": True,
            "backup_interval": 300,
        },
    }


@pytest.fixture
def mock_project_config() -> Dict[str, Any]:
    """Fixture providing mock project configuration.

    Returns:
        Mock project configuration dictionary
    """
    return {
        "name": "test_project",
        "description": "Test project for unit tests",
        "classes": ["person", "car", "bicycle"],
        "version": "1.0.0",
        "created_at": "2024-01-01T00:00:00",
    }


@pytest.fixture
def mock_project_dir(temp_dir: Path, mock_project_config: Dict[str, Any]) -> Path:
    """Fixture providing mock project directory structure.

    Args:
        temp_dir: Temporary directory fixture
        mock_project_config: Mock project configuration

    Returns:
        Path to mock project directory
    """
    project_dir = temp_dir / "test_project"
    project_dir.mkdir()

    # Create project structure
    (project_dir / "images").mkdir()
    (project_dir / "annotations").mkdir()
    (project_dir / "models").mkdir()
    (project_dir / "exports").mkdir()

    # Create config file
    with open(project_dir / "config.json", "w") as f:
        json.dump(mock_project_config, f, indent=2)

    # Create classes file
    with open(project_dir / "classes.txt", "w") as f:
        for class_name in mock_project_config["classes"]:
            f.write("{class_name}\n")

    return project_dir


@pytest.fixture
def mock_image_files(temp_dir: Path) -> list[Path]:
    """Fixture providing mock image files.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        List of paths to mock image files
    """
    image_files = []
    for i in range(3):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        img_path = temp_dir / "test_image_{i}.jpg"
        img.save(img_path)
        image_files.append(img_path)

    return image_files


@pytest.fixture
def mock_annotation_files(temp_dir: Path, mock_image_files: list[Path]) -> list[Path]:
    """Fixture providing mock annotation files.

    Args:
        temp_dir: Temporary directory fixture
        mock_image_files: Mock image files fixture

    Returns:
        List of paths to mock annotation files
    """
    annotation_files = []

    for img_path in mock_image_files:
        # Create corresponding annotation file
        annotation_path = temp_dir / "{img_path.stem}.json"

        annotation_data = {
            "image_path": str(img_path),
            "annotations": [
                {
                    "class": "person",
                    "bbox": [10, 10, 50, 80],
                    "confidence": 0.95,
                },
                {
                    "class": "car",
                    "bbox": [60, 20, 90, 60],
                    "confidence": 0.87,
                },
            ],
        }

        with open(annotation_path, "w") as f:
            json.dump(annotation_data, f, indent=2)

        annotation_files.append(annotation_path)

    return annotation_files


@pytest.fixture
def mock_model_file(temp_dir: Path) -> Path:
    """Fixture providing mock model file.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to mock model file
    """
    model_path = temp_dir / "mock_model.pt"

    # Create a dummy model file
    model_data = {
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "epoch": 100,
        "loss": 0.05,
        "accuracy": 0.95,
    }

    torch.save(model_data, model_path)

    return model_path


@pytest.fixture
def mock_hardware_detector():
    """Fixture providing mock hardware detector.

    Returns:
        Mock hardware detector instance
    """
    mock_detector = Mock()
    mock_detector.get_device_type.return_value = "cpu"
    mock_detector.get_device_name.return_value = "Mock CPU"
    mock_detector.get_memory_info.return_value = {"total": 8192, "available": 4096}
    mock_detector.is_gpu_available.return_value = False
    mock_detector.get_recommended_batch_size.return_value = 16

    return mock_detector


@pytest.fixture
def mock_logger():
    """Fixture providing mock logger.

    Returns:
        Mock logger instance
    """
    mock_log = Mock()
    mock_log.info = Mock()
    mock_log.warning = Mock()
    mock_log.error = Mock()
    mock_log.debug = Mock()

    return mock_log


@pytest.fixture
def mock_keyauth_config() -> Dict[str, Any]:
    """Fixture providing mock KeyAuth configuration.

    Returns:
        Mock KeyAuth configuration dictionary
    """
    return {
        "application": {
            "name": "Test Application",
            "ownerid": "test_owner",
            "secret": "test_secret",
            "version": "1.0",
        },
        "settings": {
            "session_duration_hours": 24,
            "auto_cleanup_sessions": True,
        },
        "ui": {
            "window_title": "Test Authentication",
            "theme": "dark",
        },
    }


@pytest.fixture
def mock_keyauth_api():
    """Fixture providing mock KeyAuth API.

    Returns:
        Mock KeyAuth API instance
    """
    mock_api = Mock()
    mock_api.initialized = True
    mock_api.license.return_value = True
    mock_api.login.return_value = True
    mock_api.register.return_value = True

    # Mock user data
    mock_user_data = Mock()
    mock_user_data.username = "test_user"
    mock_user_data.expires = "2025-12-31"
    mock_user_data.subscription = "premium"
    mock_user_data.hwid = "test_hwid"
    mock_api.user_data = mock_user_data

    return mock_api


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Fixture that mocks heavy dependencies for faster testing.

    This fixture automatically mocks common heavy imports to speed up tests.
    """
    with patch.dict(
        "sys.modules",
        {
            "torch": Mock(),
            "torchvision": Mock(),
            "ultralytics": Mock(),
            "cv2": Mock(),
            "wmi": Mock(),
            "win32security": Mock(),
        },
    ):
        yield


@pytest.fixture
def skip_if_no_gpu():
    """Fixture to skip tests if no GPU is available.

    Usage:
        @pytest.mark.gpu
        def test_gpu_functionality(skip_if_no_gpu):
            # Test code that requires GPU
    """
    try:
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def skip_if_no_gui():
    """Fixture to skip tests if GUI is not available.

    Usage:
        @pytest.mark.gui
        def test_gui_functionality(skip_if_no_gui):
            # Test code that requires GUI
    """
    try:
        import PyQt5
    except ImportError:
        try:
            import PyQt6
        except ImportError:
            try:
                import PySide6
            except ImportError:
                pytest.skip("No GUI framework available")


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "gui: mark test as requiring GUI")
    config.addinivalue_line("markers", "auth: mark test as requiring authentication")


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add slow marker for tests with "slow" in name
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)

        # Add markers based on test name patterns
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)

        if "gui" in item.name or "qt" in item.name:
            item.add_marker(pytest.mark.gui)

        if "auth" in item.name or "keyauth" in item.name:
            item.add_marker(pytest.mark.auth)
