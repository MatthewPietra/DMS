# DMS - Detection Model Suite

**A comprehensive object detection pipeline with integrated KeyAuth authentication**

## 🚀 Quick Start

### Windows
```bash
# Double-click or run from command line
launch.bat
```

### Linux/Mac
```bash
# Make executable (first time only)
chmod +x launch.sh

# Run the launcher
./launch.sh
```

## ✨ Features

- **🔐 Integrated KeyAuth Authentication** - Secure license verification
- **🖥️ GUI/CLI Mode Selection** - Choose your preferred interface
- **📦 Automatic Dependency Management** - Zero-configuration setup
- **🔧 Hardware Auto-Detection** - Optimized for NVIDIA, AMD, and CPU
- **🎯 Cross-Platform Support** - Windows, Linux, and macOS
- **📊 Project Management** - Complete workflow from capture to training

## 🎮 How It Works

1. **First Launch**: Choose between GUI or CLI mode (remembered for future)
2. **Authentication**: Enter your KeyAuth license key and login credentials
3. **Mode Selection**: Enjoy either the graphical interface or command-line tools
4. **Full Pipeline**: Capture, annotate, train, and deploy your models

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, Linux, or macOS
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space minimum
- **GPU**: NVIDIA/AMD GPU recommended (CPU-only supported)

## 🛠️ Installation

### Automatic Installation
The launcher handles all dependencies automatically:

1. Download/clone the DMS repository
2. Run the launcher script for your platform
3. Follow the on-screen instructions

### Manual Installation
If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements/requirements_base.txt

# Run the unified launcher
python unified_launcher.py
```

## 🔑 KeyAuth Setup

1. **Obtain License**: Get your KeyAuth license key from the provider
2. **First Launch**: Enter your license key when prompted
3. **Create Account**: Register with a username and password
4. **Automatic Sessions**: Future launches will remember your authentication

## 🎯 Usage Modes

### GUI Mode (Recommended)
- **User-friendly interface** with visual components
- **Drag-and-drop functionality** for easy file management
- **Real-time preview** of captures and annotations
- **Interactive training progress** with graphs and metrics

### CLI Mode
- **Command-line interface** for advanced users
- **Scriptable operations** for automation
- **Lightweight resource usage** for remote servers
- **Batch processing capabilities** for large datasets

## 📁 Project Structure

```
DMS/
├── launch.bat                    # 🪟 Windows unified launcher
├── launch.sh                     # 🐧 Linux/Mac unified launcher
├── unified_launcher.py           # 🚀 Main launcher with KeyAuth
├── src/                          # 📦 Core application modules
│   ├── auth/                     # 🔐 Authentication system
│   ├── gui/                      # 🖥️ GUI components
│   ├── capture/                  # 📸 Screen capture system
│   ├── annotation/               # 🏷️ Annotation tools
│   ├── training/                 # 🤖 Model training
│   └── auto_annotation/          # 🔄 Auto-annotation
├── config/                       # ⚙️ Configuration files
├── data/                         # 💾 User data and projects
├── logs/                         # 📝 Application logs
└── docs/                         # 📚 Additional documentation
```

## 🔧 Configuration

### Launcher Preferences
Stored in `config/launcher_preferences.json`:
- **Default mode**: GUI or CLI
- **Show mode dialog**: Enable/disable first-time popup
- **User choices**: Saved preferences

### KeyAuth Configuration
Stored in `config/keyauth_config.json`:
- **Application settings**: Name, version, credentials
- **Security settings**: Password requirements, session duration
- **UI preferences**: Theme, window size, behavior

## 🚨 Troubleshooting

### Common Issues

**Authentication Failed**
- Verify your KeyAuth license key is valid
- Check internet connection for license verification
- Ensure system time is synchronized

**GUI Not Loading**
- Install Qt framework: `pip install PyQt5` or `pip install PySide6`
- Try CLI mode as fallback
- Check logs in `logs/` directory

**Dependencies Missing**
- Run the launcher script to auto-install dependencies
- Manually install: `pip install -r requirements/requirements_base.txt`
- Ensure Python 3.8+ is installed

**Virtual Environment Issues**
- Delete `venv` folder and re-run launcher
- Ensure `python -m venv` is available
- Try system Python if virtual environment fails

### Getting Help

1. **Check Logs**: Look in `logs/` directory for error details
2. **Documentation**: Review files in `docs/` directory
3. **GitHub Issues**: Report bugs on the project repository
4. **Support**: Contact the development team

## 📊 Features Overview

### Screen Capture System
- **Multi-monitor support** with automatic detection
- **Region selection** for targeted captures
- **Real-time preview** with adjustable quality
- **Batch capture** for dataset generation

### Annotation Interface
- **Intuitive tools** for bounding box creation
- **Class management** with custom categories
- **Keyboard shortcuts** for efficient workflow
- **Export formats** including COCO, YOLO, and Pascal VOC

### Model Training
- **YOLO integration** with latest versions
- **Hardware optimization** for NVIDIA/AMD GPUs
- **Progress monitoring** with real-time metrics
- **Automatic validation** and model selection

### Auto-Annotation
- **AI-powered suggestions** for faster labeling
- **Confidence thresholds** for quality control
- **Batch processing** for large datasets
- **Human-in-the-loop** verification workflow

## 🎨 Customization

### Themes and UI
- **Dark/Light themes** with system integration
- **Customizable shortcuts** for power users
- **Workspace layouts** for different workflows
- **Plugin system** for extensions

### Hardware Optimization
- **Automatic GPU detection** (NVIDIA/AMD)
- **Memory management** for large datasets
- **Performance profiling** and optimization
- **Batch size optimization** for training

## 🔄 Updates and Maintenance

### Automatic Updates
- **Dependency management** with version checking
- **Security updates** for KeyAuth integration
- **Feature updates** through the launcher
- **Rollback capability** for stability

### Manual Maintenance
- **Log cleanup** in `logs/` directory
- **Cache management** for temporary files
- **Database optimization** for user data
- **Backup procedures** for important projects

## 📈 Performance Tips

1. **Use GPU acceleration** when available
2. **Optimize batch sizes** for your hardware
3. **Regular cleanup** of temporary files
4. **Monitor system resources** during training
5. **Use SSD storage** for better I/O performance

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines in the `docs/` directory.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **KeyAuth** for authentication services
- **Ultralytics** for YOLO implementation
- **Qt Framework** for GUI components
- **OpenCV** for computer vision utilities

---

**Need help?** Check the `docs/` directory for detailed guides or contact our support team. 