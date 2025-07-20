# DMS - Developer Guide

## Architecture Overview

DMS (Detection Model Suite) is built with a modular architecture:

```
src/
├── capture/          # Screen capture and image acquisition
├── annotation/       # Interactive annotation interface
├── training/         # Multi-YOLO model training pipeline
├── auto_annotation/  # Intelligent auto-labeling system
├── gui/              # 🆕 Modern GUI system
│   ├── main_window.py        # Main application window
│   ├── components/           # GUI components
│   │   ├── dashboard.py      # Dashboard widget
│   │   ├── project_manager.py # Project management
│   │   ├── training.py       # Training interface
│   │   ├── annotation.py     # Annotation interface
│   │   ├── capture.py        # Screen capture
│   │   ├── system_monitor.py # System monitoring
│   │   └── settings.py       # Settings interface
│   └── utils/               # GUI utilities
│       ├── styles.py        # Theme styling
│       └── icons.py         # Icon management
└── utils/            # Shared utilities and hardware detection
    ├── bug_fixes.py              # Bug fixes and workarounds
    ├── system_optimizer.py        # System optimization for production
    └── production_validator.py    # Production readiness validation
```

## GUI Development

### 🖥️ **GUI Architecture**

The GUI system is built using PySide6 (with fallbacks to PyQt6/PyQt5) and follows a modular component-based architecture:

#### Main Window (`DMSMainWindow`)
- **Purpose**: Central application window with navigation
- **Features**: Sidebar navigation, content stacking, menu system
- **Location**: `src/gui/main_window.py`

#### Component System
Each major feature has its own widget component:
- **DashboardWidget**: Overview and quick actions
- **ProjectManagerWidget**: Project creation and management
- **TrainingWidget**: Model training interface
- **AnnotationWidget**: Data annotation tools
- **CaptureWidget**: Screen capture interface
- **SystemMonitorWidget**: Resource monitoring
- **SettingsWidget**: Configuration management

#### Utility System
- **IconManager**: Centralized icon management with Unicode fallbacks
- **Style System**: Dark/light theme support with CSS-like styling
- **Configuration**: Integration with existing config system

### **Adding New GUI Components**

#### 1. Create Component Widget
```python
# src/gui/components/my_component.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

class MyComponentWidget(QWidget):
    """My custom component widget."""
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Add your UI elements here
        label = QLabel("My Custom Component")
        layout.addWidget(label)
        
    def cleanup(self):
        """Cleanup resources."""
        pass
```

#### 2. Add to Main Window
```python
# In src/gui/main_window.py
from .components.my_component import MyComponentWidget

class DMSMainWindow(QMainWindow):
    def create_main_content(self):
        # ... existing code ...
        
        # Add your component
        self.pages["my_component"] = MyComponentWidget(self)
        self.content_stack.addWidget(self.pages["my_component"])
```

#### 3. Add Navigation
```python
# In src/gui/main_window.py
def create_navigation_buttons(self, layout):
    # ... existing code ...
    
    nav_items = [
        # ... existing items ...
        ("my_component", "My Component", "icon_name", "Description"),
    ]
```

#### 4. Update Imports
```python
# In src/gui/components/__init__.py
from .my_component import MyComponentWidget

__all__ = [
    # ... existing components ...
    'MyComponentWidget'
]
```

### Project Manager GUI

The ProjectManagerWidget provides a full-featured interface for managing DMS projects:
- Project list (auto-detected from data/projects)
- Create new project (dialog for name, description, classes)
- Open existing project (sets current project in main window)
- Project settings (edit name, description, classes)
- Import/export projects (import config.json, export using COCOExporter)

#### Dialogs
- **ProjectDialog**: Used for both creating and editing project details.

#### Export
- Uses `COCOExporter` (see `src/annotation/coco_exporter.py`) for exporting projects in COCO format.

### **Customizing Themes**

#### Adding New Themes
```python
# In src/gui/utils/styles.py
def get_custom_theme():
    """Get custom theme stylesheet."""
    return """
    /* Custom theme styles */
    QMainWindow {
        background-color: #your_color;
        color: #your_text_color;
    }
    /* ... more styles ... */
    """
```

#### Theme Integration
```python
# In src/gui/main_window.py
def apply_styling(self):
    """Apply styling to the application."""
    theme = self.config.get("annotation", {}).get("ui", {}).get("theme", "dark")
    
    if theme == "dark":
        self.setStyleSheet(get_dark_style())
    elif theme == "light":
        self.setStyleSheet(get_light_style())
    elif theme == "custom":
        self.setStyleSheet(get_custom_theme())
```

### **Icon Management**

#### Adding Custom Icons
```python
# In src/gui/utils/icons.py
class IconManager:
    _builtin_icons = {
        # ... existing icons ...
        "my_icon": "🎯",  # Unicode symbol
    }
    
    @classmethod
    def add_custom_icon(cls, icon_name: str, icon_path: str):
        """Add a custom icon to the cache."""
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            cls._icon_cache[icon_name] = icon
        else:
            raise FileNotFoundError(f"Icon file not found: {icon_path}")
```

#### Using Icons in Components
```python
from ..utils.icons import IconManager

# In your component
btn = QPushButton("My Button")
btn.setIcon(IconManager.get_icon("my_icon"))
```

### **GUI Testing**

#### Unit Testing GUI Components
```python
# tests/test_gui_components.py
import pytest
from PySide6.QtWidgets import QApplication
from src.gui.components.dashboard import DashboardWidget

class TestDashboardWidget:
    @pytest.fixture
    def app(self):
        return QApplication([])
    
    @pytest.fixture
    def main_window(self):
        # Mock main window
        return MockMainWindow()
    
    def test_dashboard_creation(self, app, main_window):
        widget = DashboardWidget(main_window)
        assert widget is not None
        assert widget.main_window == main_window
```

#### Integration Testing
```python
# tests/test_gui_integration.py
def test_gui_launch():
    """Test GUI launcher functionality."""
    from gui_launcher import main
    
    # Test GUI launch
    result = main()
    assert result == 0
```

## Core API Reference

### DMS Class

The main entry point for the DMS API.

```python
from dms import DMS

studio = DMS(config_path="config/studio_config.yaml")
```

#### Methods

**Project Management**
```python
# Create new project
project_path = studio.create_project(
    name="my_project",
    description="Object detection project",
    classes=["person", "car", "bike"]
)

# Load existing project
studio.load_project("path/to/project")
```

**Capture System**
```python
# Start capture session
capture_results = studio.start_capture(
    duration=60,  # seconds
    output_dir="data/captured",
    fps=5,
    window_title="Game Window"
)
```

**Training Pipeline**
```python
# Train model
training_results = studio.train_model(
    data_path="data/train",
    model_name="yolov8n",
    epochs=100,
    batch_size=16
)
```

**Auto-Annotation**
```python
# Auto-annotate images
auto_results = studio.auto_annotate(
    data_path="data/images",
    model_path="models/best.pt",
    output_path="data/auto_annotated"
)
```

**Export System**
```python
# Export dataset
export_results = studio.export_dataset(
    data_path="data/annotated",
    output_path="exports/coco",
    format="coco"
)
```

### Configuration Management

```python
from dms.utils.config import ConfigManager

# Initialize config manager
config = ConfigManager("config/studio_config.yaml")

# Get specific configurations
hardware_config = config.get_hardware_config()
training_config = config.get_training_config()
capture_config = config.get_capture_config()

# Update configurations
config.set("hardware.device", "cuda")
config.set("training.epochs", 200)
config.save()
```

### Hardware Detection

```python
from dms.utils.hardware import HardwareDetector

detector = HardwareDetector()

# Get device information
device_type = detector.get_device_type()  # CUDA, DIRECTML, CPU
device = detector.get_device()  # "cuda:0", "directml:0", "cpu"
gpu_info = detector.get_gpu_info()

# Calculate optimal batch size
batch_size = detector.get_optimal_batch_size(
    image_size=640,
    model_size="n"
)
```

## Advanced Usage

### Custom Training Workflows

#### Multi-Stage Training
```python
from dms.training.yolo_trainer import YOLOTrainer

trainer = YOLOTrainer(config)

# Stage 1: Initial training with frozen backbone
config.training.freeze_backbone = True
config.training.epochs = 50
config.training.lr0 = 0.001

stage1_results = trainer.train_model(
    data_yaml_path="data.yaml",
    training_config=config
)

# Stage 2: Fine-tuning with unfrozen backbone
config.training.freeze_backbone = False
config.training.epochs = 100
config.training.lr0 = 0.0001

stage2_results = trainer.resume_training(
    model_path=stage1_results.model_path,
    training_config=config
)
```

#### Transfer Learning
```python
# Transfer learning setup
transfer_config = {
    'source_model': 'path/to/pretrained/model.pt',
    'freeze_layers': ['backbone.conv1', 'backbone.layer1'],
    'learning_rates': {
        'backbone': 0.0001,
        'neck': 0.001,
        'head': 0.01
    }
}

results = trainer.transfer_learning_train(
    data_yaml_path="data.yaml",
    transfer_config=transfer_config,
    epochs=100
)
```

### Custom Auto-Annotation

#### Confidence Management
```python
from dms.auto_annotation.confidence_manager import ConfidenceManager

# Custom confidence thresholds per class
class_thresholds = {
    'person': 0.8,      # High confidence for person detection
    'car': 0.6,         # Medium confidence for cars
    'bicycle': 0.7      # Medium-high for bicycles
}

confidence_manager = ConfidenceManager()
confidence_manager.set_class_thresholds(class_thresholds)
```

#### Quality-Based Auto-Annotation
```python
from dms.auto_annotation.auto_annotator import AutoAnnotator

annotator = AutoAnnotator(config)

# Custom quality assessment
quality_config = {
    'min_confidence': 0.6,
    'max_overlap': 0.8,
    'min_box_size': 10,
    'quality_metrics': ['accuracy', 'credibility', 'consistency']
}

results = annotator.annotate_with_quality(
    data_path="data/images",
    model_path="models/best.pt",
    quality_config=quality_config
)
```

### GUI Integration

#### Custom GUI Components
```python
# Create custom training widget
class CustomTrainingWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.trainer = YOLOTrainer(main_window.config)
        self.init_ui()
    
    def init_ui(self):
        # Create training interface
        self.create_model_selection()
        self.create_parameter_inputs()
        self.create_training_controls()
        self.create_progress_display()
    
    def start_training(self):
        # Start training with custom parameters
        config = self.get_training_config()
        self.trainer.train_model(config)
```

#### Real-time Updates
```python
# In your GUI component
from PySide6.QtCore import QTimer

class TrainingWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_training_progress)
        self.update_timer.start(1000)  # Update every second
    
    def update_training_progress(self):
        # Update progress bars, charts, etc.
        if self.training_active:
            progress = self.trainer.get_progress()
            self.progress_bar.setValue(progress.epoch)
            self.loss_chart.add_point(progress.loss)
```

## Performance Optimization

### GUI Performance

#### Efficient Updates
```python
# Use QTimer for periodic updates instead of continuous polling
self.update_timer = QTimer()
self.update_timer.timeout.connect(self.update_data)
self.update_timer.start(1000)  # Update every second

# Batch updates to reduce UI redraws
def update_data(self):
    with self.batch_update():
        self.update_statistics()
        self.update_progress()
        self.update_charts()
```

#### Memory Management
```python
# Clean up resources in components
def cleanup(self):
    """Cleanup resources."""
    if hasattr(self, 'update_timer'):
        self.update_timer.stop()
    
    # Clear large data structures
    self.image_cache.clear()
    self.model_cache.clear()
```

### Training Performance

#### GPU Optimization
```python
# Optimize GPU memory usage
config.training.batch_size = detector.get_optimal_batch_size()
config.training.mixed_precision = True
config.training.gradient_accumulation = 4

# Use gradient checkpointing for large models
config.training.gradient_checkpointing = True
```

#### Data Pipeline Optimization
```python
# Optimize data loading
config.training.num_workers = 4
config.training.prefetch_factor = 2
config.training.pin_memory = True

# Use memory-efficient data formats
config.data.format = "parquet"
config.data.compression = "lz4"
```

## Testing & Quality Assurance

### Unit Testing

#### Testing Core Components
```python
# tests/test_core.py
import pytest
from dms import DMS

def test_dms_initialization():
    studio = DMS()
    assert studio is not None
    assert studio.config is not None

def test_project_creation():
    studio = DMS()
    project_path = studio.create_project("test_project")
    assert project_path.exists()
```

#### Testing GUI Components
```python
# tests/test_gui.py
import pytest
from PySide6.QtWidgets import QApplication
from src.gui.main_window import DMSMainWindow

@pytest.fixture
def app():
    return QApplication([])

def test_main_window_creation(app):
    window = DMSMainWindow()
    assert window is not None
    assert window.project_root.exists()
```

### Integration Testing

#### End-to-End Testing
```python
# tests/test_integration.py
def test_complete_workflow():
    """Test complete DMS workflow."""
    studio = DMS()
    
    # Create project
    project = studio.create_project("test_workflow")
    
    # Capture data
    capture_results = studio.start_capture(duration=10)
    assert len(capture_results.images) > 0
    
    # Train model
    training_results = studio.train_model(
        data_path=capture_results.output_dir,
        model_name="yolov8n",
        epochs=5
    )
    assert training_results.model_path.exists()
    
    # Auto-annotate
    auto_results = studio.auto_annotate(
        data_path=capture_results.output_dir,
        model_path=training_results.model_path
    )
    assert len(auto_results.annotations) > 0
```

### Performance Testing

#### Benchmark Testing
```python
# tests/test_performance.py
import time

def test_training_performance():
    """Test training performance benchmarks."""
    start_time = time.time()
    
    studio = DMS()
    results = studio.train_model(
        data_path="test_data",
        model_name="yolov8n",
        epochs=10
    )
    
    training_time = time.time() - start_time
    assert training_time < 300  # Should complete within 5 minutes
```

## Deployment & Production

### Production Configuration

#### System Optimization
```python
from dms.utils.system_optimizer import optimize_system_for_production

# Apply production optimizations
optimize_system_for_production()
```

#### Production Validation
```python
from dms.utils.production_validator import validate_production_readiness

# Validate production readiness
report = validate_production_readiness()
if not report.is_ready:
    print("Production validation failed:", report.issues)
```

### Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/requirements_nvidia.txt /tmp/
RUN pip install -r /tmp/requirements_nvidia.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose ports
EXPOSE 8080

# Run application
CMD ["python", "gui_launcher.py"]
```

#### Docker Compose
```yaml
version: '3.8'
services:
  dms:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - YOLO_DEVICE=cuda
      - YOLO_BATCH_SIZE=16
```

## Contributing

### Development Setup

#### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd dms

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/Mac
# or
dev_env\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements/requirements_dev.txt
```

#### Code Style
```bash
# Run linting
flake8 src/ tests/
black src/ tests/
isort src/ tests/

# Run type checking
mypy src/
```

#### Testing
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_gui.py
pytest tests/test_core.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=src tests/
```

### Pull Request Guidelines

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Make changes**: Follow code style guidelines
4. **Add tests**: Include unit and integration tests
5. **Update documentation**: Update relevant docs
6. **Submit PR**: Include description and test results

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is adequate
- [ ] Documentation is updated
- [ ] No breaking changes (or documented)
- [ ] Performance impact is considered
- [ ] Security implications are reviewed

## Support & Community

### Getting Help

- **Documentation**: Check this guide and [User Guide](USER_GUIDE.md)
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Discord**: Real-time help and discussions

### Contributing Guidelines

- **Bug Reports**: Include system info and reproduction steps
- **Feature Requests**: Describe use case and benefits
- **Code Contributions**: Follow development guidelines
- **Documentation**: Help improve docs and examples

### Community Resources

- **GitHub Repository**: Main development hub
- **Discord Server**: Community discussions and help
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Sample projects and use cases

---
