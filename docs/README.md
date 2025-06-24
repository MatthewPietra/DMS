# DMS - Complete Object Detection Pipeline

## 🚀 Overview

DMS (Detection Model Suite) is a comprehensive ML object detection annotation tool that automates the entire workflow from data collection to model deployment. The system integrates image capture, annotation management, model training, and auto-annotation capabilities into a unified platform optimized for YOLO architectures.

## ✨ Key Features

### 🎯 Core Capabilities
- **Automated Image Acquisition**: Configurable window capture (1-10 fps)
- **Intuitive Annotation Interface**: Collaborative features with real-time collaboration
- **Multi-YOLO Model Training**: Support for YOLOv5, YOLOv8, YOLOv8-s, YOLOv8-n, YOLOv9, YOLOv10, YOLOv11
- **Intelligent Auto-Annotation**: Quality control with confidence thresholds
- **Project Management**: Dataset versioning and backup systems
- **Real-time Monitoring**: Training progress and evaluation metrics

### 🔧 Technical Features
- **Cross-Platform GPU Support**: NVIDIA CUDA and AMD DirectML
- **Advanced Quality Metrics**: ACC framework (Accuracy, Credibility, Consistency)
- **Active Learning Pipeline**: Continuous improvement workflow
- **COCO Format Support**: Industry-standard annotation format
- **Docker Deployment**: Container-based deployment ready

### 🚀 **NEW: Central Launcher Features**
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

### 🎯 **NEW: One-Click Launcher** (Recommended)

The easiest way to get started is with our **Central Launcher** - no manual setup required!

```bash
# Simply run the launcher
python main.py

# Or use convenient scripts:
# Windows: Double-click launch.bat
# Linux/Mac: ./launch.sh
```

**The launcher will:**
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

| GPU Type | Training | Inference | Auto-Detection |
|----------|----------|-----------|----------------|
| NVIDIA RTX/GTX | ✅ CUDA | ✅ CUDA | ✅ |
| AMD RX Series | ✅ DirectML | ✅ DirectML | ✅ |
| Intel Arc | ✅ DirectML | ✅ DirectML | ✅ |
| CPU Fallback | ✅ | ✅ | ✅ |

## 🎯 Performance Benchmarks

### Training Performance (YOLOv8n, 1000 images)
- **RTX 4090**: ~15 minutes
- **RTX 3080**: ~25 minutes
- **RX 6800 XT**: ~35 minutes (DirectML)
- **RX 5700 XT**: ~45 minutes (DirectML)

### Auto-Annotation Speed
- **GPU Accelerated**: 100+ images/minute
- **CPU Fallback**: 20+ images/minute

## 📖 Documentation

### 📚 **Consolidated Documentation Structure**

We've streamlined our documentation into focused, comprehensive guides:

- **[User Guide](docs/USER_GUIDE.md)** - Complete user manual with quick start, features, and troubleshooting
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - API reference, advanced usage, and integration examples

### 📋 **What's Included**

**User Guide** covers:
- Quick start with one-click launcher
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

### 🎯 **Benefits of New Structure**
- **Reduced complexity**: Fewer files to navigate
- **Focused content**: Each guide serves a specific audience
- **Better organization**: Logical flow from basic to advanced
- **Easier maintenance**: Centralized, well-structured documentation
- **Improved discoverability**: Clear separation of user vs developer content

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv5/v8/v9/v10/v11 teams for the excellent models
- PyTorch team for DirectML support
- OpenCV community for computer vision tools
- The open-source ML community

## 🆘 Support

- 📧 Email: support@dms.com
- 💬 Discord: [Join our community](https://discord.gg/dms)
- 📋 Issues: [GitHub Issues](https://github.com/your-repo/dms/issues)
- 📚 Wiki: [Project Wiki](https://github.com/your-repo/dms/wiki) 