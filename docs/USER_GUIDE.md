# DMS - User Guide

## Quick Start

### 🚀 One-Click Launcher (Recommended)

The easiest way to get started - no manual setup required!

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
# For NVIDIA GPUs
pip install -r requirements/requirements_nvidia.txt

# For AMD GPUs
pip install -r requirements/requirements_amd.txt

# For CPU only
pip install -r requirements/requirements_cpu.txt
```

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

| Hardware | Status | Performance |
|----------|--------|-------------|
| NVIDIA RTX 40xx | ✅ Excellent | Native CUDA |
| NVIDIA RTX 30xx | ✅ Excellent | Native CUDA |
| NVIDIA GTX 16xx | ✅ Good | CUDA 7.5+ |
| AMD RX 6000/7000 | ✅ Good | DirectML |
| AMD RX 5000 | ✅ Good | DirectML |
| Intel Arc | 🚧 Limited | DirectML |
| CPU (Intel/AMD) | ✅ Basic | OpenMP |

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

### Python API Integration
```python
from dms import DMS

# Initialize studio
studio = DMS()

# Create project
project_path = studio.create_project(
    name="my_project",
    classes=["person", "car", "bike"]
)

# Start capture
results = studio.start_capture(duration=60)

# Train model
training_results = studio.train_model(
    data_path="data/train",
    model_name="yolov8n",
    epochs=100
)

# Auto-annotate
auto_results = studio.auto_annotate(
    data_path="data/images",
    model_path="models/best.pt"
)
```

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check hardware
python -m src.cli hardware

# Force device
python -m src.cli config --set hardware.device=cuda
```

#### Out of Memory
```bash
# Reduce batch size
python -m src.cli config --set hardware.batch_size=4

# Use smaller model
python -m src.cli train data --model yolov8n
```

#### Slow Capture
```bash
# Reduce FPS
python -m src.cli config --set capture.fps=5

# Check monitor setup
python -m src.cli capture --monitor 0
```

### Debug Mode
```bash
# Enable verbose logging
python -m src.cli --verbose train data/dataset
```

## Performance Benchmarks

### Training Performance (YOLOv8n, 1000 images)
- **RTX 4090**: ~15 minutes
- **RTX 3080**: ~25 minutes
- **RX 6800 XT**: ~35 minutes (DirectML)
- **RX 5700 XT**: ~45 minutes (DirectML)

### Auto-Annotation Speed
- **GPU Accelerated**: 100+ images/minute
- **CPU Fallback**: 20+ images/minute
---