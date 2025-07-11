# DMS GUI - Modern Graphical Interface

## Overview

The DMS GUI provides a modern, intuitive graphical user interface for the Detection Model Suite (DMS). It replaces the console-based interface with a comprehensive GUI that makes the object detection pipeline more accessible and user-friendly.

## Features

### 🎯 **Modern Interface**
- **Dark/Light Themes**: Choose your preferred visual style
- **Responsive Design**: Adapts to different screen sizes
- **Intuitive Navigation**: Sidebar-based navigation with clear sections
- **Real-time Updates**: Live system monitoring and status updates

### 📊 **Dashboard**
- **Quick Actions**: One-click access to common tasks
- **Statistics**: Real-time project and system statistics
- **Recent Activity**: Track your recent actions
- **System Overview**: Monitor CPU, memory, and GPU usage

### 🏗️ **Modular Components**
- **Project Manager**: Create, open, and manage projects
- **Screen Capture**: Automated data collection from windows
- **Annotation Interface**: Label and manage datasets
- **Training Interface**: Configure and monitor model training
- **System Monitor**: Real-time resource monitoring
- **Settings**: Configure all aspects of the system

### 🔧 **Smart Features**
- **Hardware Detection**: Automatic GPU/CPU detection
- **Dependency Management**: Automatic installation of required packages
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Error Handling**: Graceful fallbacks and helpful error messages

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Quick Start

#### Windows
```bash
# Clone the repository
git clone <repository-url>
cd DMS

# Launch GUI (recommended)
launch.bat

# Or launch console version
launch.bat
```

#### Linux/macOS
```bash
# Clone the repository
git clone <repository-url>
cd DMS

# Make script executable
chmod +x launch.sh

# Launch unified launcher (GUI/CLI mode selection)
./launch.sh
```

### Manual Installation

If you prefer to install dependencies manually:

```bash
# Create virtual environment
python -m venv dms_venv310

# Activate virtual environment
# Windows:
dms_venv310\Scripts\activate
# Linux/macOS:
source dms_venv310/bin/activate

# Install GUI dependencies
pip install PySide6 PyYAML psutil

# Launch GUI
python gui_launcher.py
```

## Usage

### First Launch
1. Run the appropriate launch script for your platform
2. The system will automatically:
   - Create a virtual environment
   - Install required dependencies
   - Detect your hardware configuration
   - Launch the GUI interface

### Navigation
- **Dashboard**: Main overview and quick actions
- **Projects**: Manage your object detection projects
- **Capture**: Screen capture and data collection
- **Annotation**: Label and annotate your datasets
- **Training**: Train YOLO models
- **System**: Monitor system resources
- **Settings**: Configure application settings

### Quick Actions
- **New Project**: Create a new object detection project
- **Screen Capture**: Start capturing screen data
- **Annotate Data**: Open the annotation interface
- **Train Model**: Start model training
- **System Monitor**: View system resources

## Configuration

### Themes
The GUI supports both dark and light themes. You can change the theme in the Settings panel.

### Hardware Configuration
The system automatically detects your hardware and configures itself accordingly:
- **NVIDIA GPU**: Automatic CUDA detection and setup
- **AMD GPU**: DirectML support for Windows/Linux
- **CPU Only**: Fallback mode for systems without GPU

### Project Structure
```
DMS/
├── data/
│   ├── projects/          # Your annotation projects
│   ├── models/           # Trained models
│   └── temp/             # Temporary files
├── config/               # Configuration files
├── logs/                 # System logs
├── exports/              # Exported data
└── src/gui/              # GUI source code
```

## Troubleshooting

### GUI Won't Start
1. **Check Python Version**: Ensure you have Python 3.8+
2. **Install Dependencies**: Run `pip install PySide6 PyYAML psutil`
3. **Virtual Environment**: Make sure you're using the correct virtual environment
4. **Fallback**: Use the console launcher (`launcher.py`) if GUI fails

### Missing Dependencies
The launch scripts automatically install required dependencies. If you encounter issues:

```bash
# Manual installation
pip install PySide6 PyYAML psutil

# Alternative GUI frameworks
pip install PyQt6  # or
pip install PyQt5
```

### Performance Issues
1. **GPU Detection**: Check if your GPU is properly detected
2. **Memory Usage**: Monitor system memory usage
3. **Background Processes**: Close unnecessary applications
4. **Virtual Environment**: Ensure you're using the dedicated environment

### Common Errors

#### "No module named 'PySide6'"
```bash
pip install PySide6
```

#### "Permission denied" (Linux/macOS)
```bash
chmod +x launch_gui.sh
```

#### "Python not found"
Install Python 3.8+ from [python.org](https://python.org)

## Development

### GUI Architecture
```
src/gui/
├── main_window.py        # Main application window
├── components/           # Individual GUI components
│   ├── dashboard.py      # Dashboard widget
│   ├── project_manager.py # Project management
│   ├── training.py       # Training interface
│   ├── annotation.py     # Annotation interface
│   ├── capture.py        # Screen capture
│   ├── system_monitor.py # System monitoring
│   └── settings.py       # Settings interface
└── utils/               # Utility functions
    ├── styles.py        # Theme styling
    └── icons.py         # Icon management
```

### Adding New Components
1. Create a new widget in `src/gui/components/`
2. Inherit from `QWidget`
3. Add to the main window navigation
4. Update the imports in `__init__.py` files

### Customizing Themes
Edit `src/gui/utils/styles.py` to modify the appearance:
- `get_dark_style()`: Dark theme stylesheet
- `get_light_style()`: Light theme stylesheet

## Support

### Getting Help
1. **Documentation**: Check the main README.md
2. **Issues**: Report problems on the project repository
3. **Console Fallback**: Use `launcher.py` if GUI has issues

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Note**: The GUI system is designed to work alongside the existing console-based system. You can use either interface depending on your preferences and needs. 