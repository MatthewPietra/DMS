# Contributing to DMS - Detection Model Suite

Thank you for your interest in contributing to DMS! This document provides guidelines and information for contributors.

## 🤝 Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](https://www.contributor-covenant.org/). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of computer vision and machine learning concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/dms-detection-suite.git
   cd dms-detection-suite
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Setup**
   ```bash
   pytest
   black --check src/ tests/
   mypy src/
   ```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code following our style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` code style changes
- `refactor:` code refactoring
- `test:` test additions or changes
- `chore:` maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request through GitHub.

## 🎯 Code Standards

### Python Style Guide

We follow PEP 8 with these specific requirements:

- **Line Length**: 88 characters (Black default)
- **Import Sorting**: Use isort with Black profile
- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style docstrings
- **Testing**: pytest with minimum 80% coverage

### Code Quality Tools

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Security scanning
bandit -r src/

# Run all checks
pre-commit run --all-files
```

### Example Code Style

```python
"""Module docstring describing the purpose."""

from typing import List, Optional, Dict, Any
import logging

from dms.utils.logger import setup_logger


class ExampleClass:
    """Example class following DMS standards.
    
    This class demonstrates proper code style including:
    - Type hints
    - Docstrings
    - Error handling
    - Logging
    
    Args:
        name: The name of the example
        config: Optional configuration dictionary
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}
        self.logger = setup_logger(self.__class__.__name__)
        
    def process_data(self, data: List[str]) -> Dict[str, int]:
        """Process a list of strings and return statistics.
        
        Args:
            data: List of strings to process
            
        Returns:
            Dictionary containing processing statistics
            
        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Data cannot be empty")
            
        self.logger.info(f"Processing {len(data)} items")
        
        result = {
            "total_items": len(data),
            "total_length": sum(len(item) for item in data),
            "average_length": sum(len(item) for item in data) / len(data),
        }
        
        return result
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_auth/
│   ├── test_capture/
│   └── test_training/
├── integration/            # Integration tests
├── fixtures/              # Test fixtures
└── conftest.py            # Pytest configuration
```

### Writing Tests

```python
"""Test module for ExampleClass."""

import pytest
from unittest.mock import Mock, patch

from dms.example import ExampleClass


class TestExampleClass:
    """Test cases for ExampleClass."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        example = ExampleClass("test")
        assert example.name == "test"
        assert example.config == {}
        
    def test_process_data_success(self):
        """Test successful data processing."""
        example = ExampleClass("test")
        data = ["hello", "world", "test"]
        
        result = example.process_data(data)
        
        assert result["total_items"] == 3
        assert result["total_length"] == 14
        assert result["average_length"] == pytest.approx(4.67, abs=0.01)
        
    def test_process_data_empty_raises_error(self):
        """Test that empty data raises ValueError."""
        example = ExampleClass("test")
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            example.process_data([])
            
    @patch('dms.example.setup_logger')
    def test_logger_setup(self, mock_logger):
        """Test logger is properly configured."""
        ExampleClass("test")
        mock_logger.assert_called_once_with("ExampleClass")
```

### Test Categories

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Unit test."""
    pass

@pytest.mark.integration
def test_integration_workflow():
    """Integration test."""
    pass

@pytest.mark.slow
def test_long_running_process():
    """Slow test that takes >5 seconds."""
    pass

@pytest.mark.gpu
def test_gpu_functionality():
    """Test requiring GPU."""
    pass
```

## 📚 Documentation

### Docstring Standards

We use Google-style docstrings:

```python
def train_model(
    data_path: str,
    model_name: str = "yolov8n",
    epochs: int = 100,
    device: str = "auto"
) -> TrainingResults:
    """Train a YOLO model on the provided dataset.
    
    This function handles the complete training pipeline including
    data validation, model initialization, training loop, and
    result collection.
    
    Args:
        data_path: Path to the training dataset directory
        model_name: Name of the YOLO model to use. Must be one of:
            yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        epochs: Number of training epochs (default: 100)
        device: Device to use for training. Options:
            - "auto": Automatically detect best device
            - "cpu": Force CPU usage
            - "cuda": Use NVIDIA GPU
            - "mps": Use Apple Silicon GPU
            
    Returns:
        TrainingResults object containing:
            - model_path: Path to the trained model
            - best_map50: Best mAP@0.5 achieved during training
            - training_time: Total training time in seconds
            - epochs_completed: Number of epochs completed
            
    Raises:
        FileNotFoundError: If data_path doesn't exist
        ValueError: If model_name is not supported
        RuntimeError: If training fails
        
    Example:
        >>> results = train_model(
        ...     data_path="./dataset",
        ...     model_name="yolov8n",
        ...     epochs=50
        ... )
        >>> print(f"Best mAP: {results.best_map50:.3f}")
        Best mAP: 0.847
    """
```

### README Updates

When adding new features:

1. Update the main README.md
2. Add usage examples
3. Update the feature list
4. Add any new dependencies

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Test with the latest version
3. Verify it's not a configuration issue

### Bug Report Template

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) and include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Relevant logs or screenshots

## 💡 Feature Requests

### Before Requesting

1. Check existing feature requests
2. Consider if it fits the project scope
3. Think about implementation complexity

### Feature Request Template

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml) and include:

- Problem description
- Proposed solution
- Use case details
- Priority level

## 🔄 Pull Request Process

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] PR description is clear and detailed

### PR Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Reviewers may test the changes locally
4. **Approval**: PR is approved and merged by a maintainer

### PR Guidelines

- Keep PRs focused and atomic
- Write clear commit messages
- Include tests for new features
- Update documentation
- Respond to review feedback promptly

## 🏗️ Project Structure

```
dms-detection-suite/
├── src/dms/                    # Main package
│   ├── __init__.py
│   ├── auth/                   # Authentication system
│   ├── capture/                # Screen capture
│   ├── annotation/             # Annotation tools
│   ├── training/               # Model training
│   ├── auto_annotation/        # Auto-annotation
│   ├── gui/                    # GUI components
│   ├── cli/                    # CLI tools
│   └── utils/                  # Utilities
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── config/                     # Configuration files
├── .github/                    # GitHub templates
├── requirements/               # Dependency files
└── tools/                      # Development tools
```

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Publish to PyPI (automated)

## 🤔 Questions?

- **General Questions**: Use GitHub Discussions
- **Bug Reports**: Create an issue
- **Feature Requests**: Create an issue
- **Security Issues**: Email security@dms-detection.com

## 📞 Contact

- **Maintainers**: @maintainer1, @maintainer2
- **Email**: contribute@dms-detection.com
- **Discord**: [DMS Community](https://discord.gg/dms-community)

---

Thank you for contributing to DMS! Your help makes this project better for everyone. 🎉 