# DMS - Complete Object Detection Pipeline

## 🚀 Overview

DMS (Detection Model Suite) is a comprehensive ML object detection annotation tool that automates the entire workflow from data collection to model deployment. The system integrates image capture, annotation management, model training, and auto-annotation capabilities into a unified platform optimized for YOLO architectures.

**🎯 NEW: Modern GUI Interface** - DMS now features a beautiful, intuitive graphical user interface that makes object detection accessible to everyone!

## ✨ Key Features

### 🎯 Core Capabilities
- **Automated Image Acquisition**: Configurable window capture (1-10 fps)
- **Intuitive Annotation Interface**: Collaborative features with real-time collaboration
- **Multi-YOLO Model Training**: Support for YOLOv5, YOLOv8, YOLOv8-s, YOLOv8-n, YOLOv9, YOLOv10, YOLOv11
- **Intelligent Auto-Annotation**: Quality control with confidence thresholds
- **Project Management**: Dataset versioning and backup systems
- **Real-time Monitoring**: Training progress and evaluation metrics

### 🖥️ **NEW: Modern GUI Interface**
- **Beautiful Dark/Light Themes**: Professional appearance with customizable themes
- **Dashboard Overview**: Real-time statistics, quick actions, and system monitoring
- **Intuitive Navigation**: Sidebar-based navigation with clear sections
- **Project Manager**: Visual project creation and management
- **Training Interface**: Real-time training progress with charts and metrics
- **Annotation Tools**: Visual annotation interface with drawing tools
- **System Monitor**: Live resource monitoring (CPU, GPU, memory)
- **Settings Panel**: Comprehensive configuration management

### 🔧 Technical Features
- **Cross-Platform GPU Support**: NVIDIA CUDA and AMD DirectML
- **Advanced Quality Metrics**: ACC framework (Accuracy, Credibility, Consistency)
- **Active Learning Pipeline**: Continuous improvement workflow
- **COCO Format Support**: Industry-standard annotation format
- **Docker Deployment**: Container-based deployment ready

### 🚀 **Central Launcher Features**
- **Zero-Configuration Setup**: No manual dependency management
- **Lazy Installation**: Install features only when needed
- **Hardware Auto-Detection**: Automatic GPU/CPU configuration
- **Interactive Menu System**: User-friendly interface with help
- **Self-Verification**: Confirms all installations work correctly
- **First-Run Wizard**: Guided setup for new users
- **System Diagnostics**: Built-in troubleshooting tools

## 🏗️ Architecture

```
dms/
├── src/
│   ├── capture/          # Screen capture module
│   ├── annotation/       # Annotation interface
│   ├── training/         # Model training pipeline
│   ├── auto_annotation/  # Auto-labeling system
│   ├── gui/              # 🆕 Modern GUI system
│   │   ├── main_window.py        # Main application window
│   │   ├── components/           # GUI components
│   │   │   ├── dashboard.py      # Dashboard widget
│   │   │   ├── project_manager.py # Project management
│   │   │   ├── training.py       # Training interface
│   │   │   ├── annotation.py     # Annotation interface
│   │   │   ├── capture.py        # Screen capture
│   │   │   ├── system_monitor.py # System monitoring
│   │   │   └── settings.py       # Settings interface
│   │   └── utils/               # GUI utilities
│   │       ├── styles.py        # Theme styling
│   │       └── icons.py         # Icon management
│   └── utils/            # Shared utilities
│       ├── bug_fixes.py              # Bug fixes and workarounds
│       ├── system_optimizer.py        # System optimization for production
│       └── production_validator.py    # Production readiness validation
├── data/
│   ├── projects/         # Individual project folders
│   ├── models/           # Trained model storage
│   └── temp/             # Temporary processing files
├── config/               # Configuration files
├── tests/                # Test suites
└── docs/                 # Documentation
```

## 🚀 Quick Start

### 🎯 **NEW: GUI Launcher** (Recommended)

The easiest way to get started is with our **Modern GUI Launcher** - beautiful interface with zero setup!

```bash
# Windows: Double-click launch_gui.bat
# Linux/Mac: ./launch_gui.sh

# Or run directly:
python gui_launcher.py
```

**The GUI launcher will:**
- ✅ Automatically detect your hardware (NVIDIA/AMD/CPU)
- ✅ Install GUI dependencies (PySide6, PyYAML, psutil)
- ✅ Launch beautiful modern interface
- ✅ Provide intuitive navigation and real-time monitoring
- ✅ Fallback to console if GUI unavailable

### 🎯 **Console Launcher** (Alternative)

For users who prefer console-based interaction:

```bash
# Simply run the launcher
python main.py

# Or use convenient scripts:
# Windows: Double-click launch.bat
# Linux/Mac: ./launch.sh
```

**The console launcher will:**
- ✅ Automatically detect your hardware (NVIDIA/AMD/CPU)
- ✅ Install only the dependencies you need
- ✅ Guide you through first-time setup
- ✅ Provide an intuitive menu for all features
- ✅ Self-verify all installations

### 📋 Manual Installation (Advanced Users)

If you prefer manual control:

#### Prerequisites
- Python 3.8+
- NVIDIA GPU (CUDA 11.8+) or AMD GPU (DirectML)
- 8GB+ RAM recommended
- 50GB+ free disk space

#### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd dms
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
# For GUI (recommended)
pip install PySide6 PyYAML psutil

# For NVIDIA GPUs
pip install -r requirements/requirements_nvidia.txt

# For AMD GPUs
pip install -r requirements/requirements_amd.txt

# For CPU only
pip install -r requirements/requirements_cpu.txt
```

4. **Initialize the studio**
```bash
python -m src.cli init
```

### Quick Start Example

```python
from dms import DMS

# Initialize studio
studio = DMS()

# Create new project
project = studio.create_project("my_detection_project")

# Start capture session
capture_session = project.start_capture(
    window_title="Game Window",
    fps=5,
    resolution=(640, 640)
)

# Launch annotation interface
studio.launch_annotation_ui(project)
```

## 🖥️ GUI Features

### **Modern Interface**
- **Dark/Light Themes**: Choose your preferred visual style
- **Responsive Design**: Adapts to different screen sizes
- **Intuitive Navigation**: Sidebar-based navigation with clear sections
- **Real-time Updates**: Live system monitoring and status updates

### **Dashboard**
- **Quick Actions**: One-click access to common tasks
- **Statistics**: Real-time project and system statistics
- **Recent Activity**: Track your recent actions
- **System Overview**: Monitor CPU, memory, and GPU usage

### **Modular Components**
- **Project Manager**: Create, open, and manage projects
- **Screen Capture**: Automated data collection from windows
- **Annotation Interface**: Label and manage datasets
- **Training Interface**: Configure and monitor model training
- **System Monitor**: Real-time resource monitoring
- **Settings**: Configure all aspects of the system

### **Smart Features**
- **Hardware Detection**: Automatic GPU/CPU detection
- **Dependency Management**: Automatic installation of required packages
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Error Handling**: Graceful fallbacks and helpful error messages

## 🔧 Utilities & Diagnostics

DMS includes advanced utilities for diagnostics, optimization, and production validation:

### Bug Fixes & Workarounds
- **Location:** `src/utils/bug_fixes.py`
- **Purpose:** Apply known bug fixes and workarounds for PyTorch, OpenCV, NumPy, and platform-specific issues.
- **Usage Example:**
```python
from dms.utils.bug_fixes import apply_all_bug_fixes
apply_all_bug_fixes()  # Applies all known bug fixes for your environment
```

### System Optimizer
- **Location:** `src/utils/system_optimizer.py`
- **Purpose:** Optimize system settings for production (memory, CPU, GPU, filesystem, network, process priority).
- **Usage Example:**
```python
from dms.utils.system_optimizer import optimize_system_for_production
optimize_system_for_production()  # Applies system-wide optimizations
```

### Production Readiness Validator
- **Location:** `src/utils/production_validator.py`
- **Purpose:** Validate that your system and dependencies are ready for production use.
- **Usage Example:**
```python
from dms.utils.production_validator import validate_production_readiness
report = validate_production_readiness()
print(report)
```

## 📊 GPU Support Matrix

| GPU Type | Training | Inference | Auto-Detection | GUI Support |
|----------|----------|-----------|----------------|-------------|
| NVIDIA RTX/GTX | ✅ CUDA | ✅ CUDA | ✅ | ✅ |
| AMD RX Series | ✅ DirectML | ✅ DirectML | ✅ | ✅ |
| Intel Arc | ✅ DirectML | ✅ DirectML | ✅ | ✅ |
| CPU Fallback | ✅ | ✅ | ✅ | ✅ |

## 🎯 Performance Benchmarks

### Training Performance (YOLOv8n, 1000 images)
- **RTX 4090**: ~15 minutes
- **RTX 3080**: ~25 minutes
- **RX 6800 XT**: ~35 minutes (DirectML)
- **RX 5700 XT**: ~45 minutes (DirectML)

### Auto-Annotation Speed
- **GPU Accelerated**: 100+ images/minute
- **CPU Fallback**: 20+ images/minute

### GUI Performance
- **Startup Time**: <5 seconds
- **Memory Usage**: ~200MB base
- **Responsiveness**: 60fps interface updates

## 📖 Documentation

### 📚 **Comprehensive Documentation Structure**

We've streamlined our documentation into focused, comprehensive guides:

- **[User Guide](docs/USER_GUIDE.md)** - Complete user manual with GUI and console interfaces
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - API reference, advanced usage, and integration examples
- **[GUI Guide](GUI_README.md)** - Detailed GUI-specific documentation

### 📋 **What's Included**

**User Guide** covers:
- Quick start with GUI and console launchers
- Manual installation for advanced users
- Core features (capture, annotation, training, auto-annotation)
- Project management and configuration
- Hardware support and troubleshooting
- Performance benchmarks

**Developer Guide** covers:
- Architecture overview and API reference
- Advanced usage patterns and custom workflows
- Performance optimization and quality assurance
- Integration examples (MLflow, W&B, etc.)
- Testing and contributing guidelines
- Utilities and diagnostics

**GUI Guide** covers:
- GUI installation and setup
- Interface navigation and features
- Customization and theming
- Troubleshooting GUI issues
- Development and extension

### 🎯 **Benefits of New Structure**
- **Reduced complexity**: Fewer files to navigate
- **Focused content**: Each guide serves a specific audience
- **Better organization**: Logical flow from basic to advanced
- **Easier maintenance**: Centralized, well-structured documentation
- **Improved discoverability**: Clear separation of user vs developer content
- **GUI-focused**: Dedicated documentation for the new interface

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv5/v8/v9/v10/v11 teams for the excellent models
- PyTorch team for DirectML support
- OpenCV community for computer vision tools
- Qt/PySide6 team for the GUI framework
- The open-source ML community

## TODO
