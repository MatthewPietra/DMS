# DMS - User Guide

## Quick Start

### 🚀 **NEW: GUI Launcher** (Recommended)

The easiest way to get started - beautiful modern interface with zero setup!

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

### 📋 Manual Installation (Advanced Users)

#### Prerequisites
- Python 3.8+
- NVIDIA GPU (CUDA 11.8+) or AMD GPU (DirectML)
- 8GB+ RAM recommended
- 50GB+ free disk space

#### Installation Steps

1. **Clone and setup**
```bash
git clone <repository-url>
cd dms
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. **Install dependencies**
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

## 🖥️ GUI Interface

### **Getting Started with GUI**

1. **Launch the GUI**
   ```bash
   # Windows
   launch_gui.bat
   
   # Linux/macOS
   ./launch_gui.sh
   ```

2. **First Launch Setup**
   - The system will automatically detect your hardware
   - Install required dependencies
   - Create necessary directories
   - Launch the main interface

3. **Navigation**
   - **Dashboard**: Main overview with quick actions
   - **Projects**: Create and manage projects
   - **Capture**: Screen capture and data collection
   - **Annotation**: Label and annotate datasets
   - **Training**: Configure and monitor training
   - **System**: Monitor system resources
   - **Settings**: Configure application settings

### **Dashboard Features**

#### Quick Actions
- **New Project**: Create a new object detection project
- **Screen Capture**: Start capturing screen data
- **Annotate Data**: Open the annotation interface
- **Train Model**: Start model training
- **System Monitor**: View system resources

#### Statistics Panel
- **Project Count**: Number of active projects
- **Image Count**: Total images in projects
- **Model Count**: Trained models available
- **Annotation Count**: Total annotations created

#### System Overview
- **CPU Usage**: Real-time CPU utilization
- **Memory Usage**: RAM usage monitoring
- **GPU Status**: GPU detection and availability
- **System Information**: Python version, GUI framework, project root

#### Recent Activity
- **Activity Log**: Track recent actions
- **Timestamps**: When actions occurred
- **Clear History**: Option to clear activity log

### **Project Management**

#### Creating Projects
1. Navigate to **Projects** section
2. Click **New Project** button
3. Enter project details:
   - Project name
   - Description
   - Classes to detect
4. Click **Create**

#### Managing Projects
- **Open Project**: Load existing project
- **Project Settings**: Configure project parameters
- **Export Project**: Export to various formats
- **Delete Project**: Remove project (with confirmation)

### **Screen Capture**

#### Capture Settings
- **Window Selection**: Choose target window
- **FPS Configuration**: Set capture rate (1-10 fps)
- **Resolution**: Set capture resolution
- **Duration**: Set capture duration

#### Capture Controls
- **Start Capture**: Begin capturing
- **Pause/Resume**: Control capture flow
- **Stop Capture**: End capture session
- **Preview**: Live preview of capture

### **Annotation Interface**

#### Annotation Tools
- **Bounding Box**: Draw rectangular annotations
- **Polygon**: Create polygonal annotations
- **Point**: Mark specific points
- **Line**: Draw line annotations

#### Class Management
- **Add Classes**: Create new object classes
- **Class Colors**: Assign colors to classes
- **Class Properties**: Configure class settings

#### Keyboard Shortcuts
- **Ctrl+S**: Save annotations
- **Ctrl+Z**: Undo last action
- **Ctrl+Y**: Redo action
- **Delete**: Remove selected annotation
- **Arrow Keys**: Navigate between images

### **Training Interface**

#### Model Selection
- **YOLO Models**: Choose from YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11
- **Model Size**: Select n, s, m, l, x variants
- **Custom Models**: Load pre-trained models

#### Training Configuration
- **Epochs**: Number of training epochs
- **Batch Size**: Training batch size
- **Learning Rate**: Initial learning rate
- **Image Size**: Input image resolution
- **Data Augmentation**: Configure augmentation settings

#### Training Monitoring
- **Real-time Progress**: Live training progress
- **Loss Charts**: Visualize training loss
- **Metrics**: mAP, precision, recall
- **GPU Usage**: Monitor GPU utilization

### **System Monitor**

#### Resource Monitoring
- **CPU Usage**: Real-time CPU utilization
- **Memory Usage**: RAM usage and availability
- **GPU Usage**: GPU utilization and memory
- **Disk Usage**: Storage space monitoring

#### Performance Metrics
- **System Health**: Overall system status
- **Performance Alerts**: Notifications for issues
- **Resource History**: Historical usage data

### **Settings Panel**

#### General Settings
- **Theme Selection**: Dark or light theme
- **Language**: Interface language
- **Auto-save**: Configure auto-save intervals

#### Hardware Settings
- **GPU Configuration**: CUDA or DirectML
- **Memory Management**: RAM allocation
- **Performance Mode**: Optimize for speed or quality

#### Training Settings
- **Default Models**: Set default model preferences
- **Training Parameters**: Default training settings
- **Export Formats**: Preferred export formats

## Core Features

### 🎯 Image Capture System
- **Real-time window capture** (1-10 fps configurable)
- **Multi-monitor support** with automatic detection
- **Quality control** with image validation
- **Batch processing** for large datasets

```python
from dms import DMS

studio = DMS()
capture_session = studio.start_capture(
    window_title="Game Window",
    fps=5,
    duration=60  # seconds
)
```

### ✏️ Annotation Interface
- **Professional PyQt GUI** with collaboration features
- **Multiple annotation tools**: bounding boxes, polygons, points
- **Keyboard shortcuts** for efficient workflow
- **Quality validation** with built-in checks

```bash
# Launch annotation interface
python -m src.annotation

# Or use CLI
python -m src.cli annotate data/images --project my_project
```

### 🤖 Model Training
- **Multi-YOLO support**: YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11
- **Cross-platform GPU acceleration**: NVIDIA CUDA, AMD DirectML
- **Automatic hyperparameter optimization**
- **Real-time training monitoring**

```bash
# Train a model
python -m src.cli train data/dataset --model yolov8n --epochs 100

# Or use Python API
results = studio.train_model(
    data_path="data/train",
    model_name="yolov8n",
    epochs=100
)
```

### 🎯 Auto-Annotation
- **AI-powered annotation** with quality control
- **Confidence-based filtering** (accept/review/reject)
- **ACC framework** for quality assessment
- **Ensemble predictions** for improved accuracy

```bash
# Auto-annotate images
python -m src.cli auto-annotate data/images models/best.pt --confidence 0.6
```

## Project Management

### Project Structure
```
my_project/
├── config.yaml              # Project configuration
├── classes.txt              # Class definitions
├── data/                    # Dataset storage
│   ├── raw/                 # Original images
│   ├── processed/           # Processed images
│   ├── annotated/           # Annotated data
│   ├── train/               # Training split
│   └── val/                 # Validation split
├── models/                  # Trained models
├── exports/                 # Dataset exports
└── logs/                    # Training logs
```

### Creating Projects
```bash
# Create new project
python -m src.cli create my_project --classes "person,car,bike"

# Or use Python API
project_path = studio.create_project(
    name="my_detection_project",
    classes=["person", "car", "bike"]
)
```

## Hardware Support

### Supported Configurations

| Hardware | Status | Performance | GUI Support |
|----------|--------|-------------|-------------|
| NVIDIA RTX 40xx | ✅ Excellent | Native CUDA | ✅ Full |
| NVIDIA RTX 30xx | ✅ Excellent | Native CUDA | ✅ Full |
| NVIDIA GTX 16xx | ✅ Good | CUDA 7.5+ | ✅ Full |
| AMD RX 6000/7000 | ✅ Good | DirectML | ✅ Full |
| AMD RX 5000 | ✅ Good | DirectML | ✅ Full |
| Intel Arc | 🚧 Limited | DirectML | ✅ Full |
| CPU (Intel/AMD) | ✅ Basic | OpenMP | ✅ Full |

### Check Your Hardware
```bash
python -m src.cli hardware
```

## Configuration

### View Current Config
```bash
python -m src.cli config --show
```

### Modify Settings
```bash
python -m src.cli config --set hardware.device=cuda
python -m src.cli config --set capture.fps=15
python -m src.cli config --set training.epochs=200
```

### Environment Variables
```bash
export YOLO_DEVICE=cuda
export YOLO_BATCH_SIZE=16
export YOLO_FPS=10
```

## Export & Integration

### Export Formats
```bash
# COCO format
python -m src.cli export data/annotated --format coco --output exports/coco

# YOLO format
python -m src.cli export data/annotated --format yolo --output exports/yolo

# Pascal VOC format
python -m src.cli export data/annotated --format pascal --output exports/pascal
```

## Troubleshooting

### GUI Issues

#### GUI Won't Start
1. **Check Dependencies**: Ensure PySide6, PyYAML, psutil are installed
2. **Python Version**: Verify Python 3.8+ is installed
3. **Virtual Environment**: Make sure you're using the correct environment
4. **Fallback**: Use console launcher if GUI fails

#### GUI Performance Issues
1. **System Resources**: Check CPU and memory usage
2. **GPU Drivers**: Update GPU drivers
3. **Theme Issues**: Try switching between dark/light themes
4. **Restart**: Restart the application

#### Missing GUI Elements
1. **Dependency Issues**: Reinstall PySide6
2. **Theme Problems**: Reset to default theme
3. **Configuration**: Check GUI configuration settings

### Console Issues

#### Launcher Problems
1. **Python Path**: Ensure Python is in PATH
2. **Virtual Environment**: Activate correct environment
3. **Dependencies**: Install required packages
4. **Permissions**: Run as administrator if needed

#### Hardware Detection Issues
1. **GPU Drivers**: Update to latest drivers
2. **CUDA Installation**: Verify CUDA installation
3. **DirectML**: Check DirectML support
4. **Fallback**: Use CPU mode if GPU unavailable

### General Issues

#### Performance Problems
1. **System Resources**: Monitor CPU, memory, GPU usage
2. **Background Processes**: Close unnecessary applications
3. **Storage Space**: Ensure sufficient disk space
4. **Network**: Check internet connection for downloads

#### Training Issues
1. **Data Quality**: Verify dataset integrity
2. **Model Compatibility**: Check model version compatibility
3. **Hardware**: Ensure sufficient GPU memory
4. **Configuration**: Review training parameters

## Support

### Getting Help
1. **Documentation**: Check this user guide and [GUI Guide](GUI_README.md)
2. **Console Fallback**: Use console interface if GUI has issues
3. **Community**: Join Discord community for help
4. **Issues**: Report problems on GitHub

### Contact Information
- 📧 Email: support@dms.com
- 💬 Discord: [Join our community](https://discord.gg/dms)
- 📋 Issues: [GitHub Issues](https://github.com/your-repo/dms/issues)
- 📚 Wiki: [Project Wiki](https://github.com/your-repo/dms/wiki)
---