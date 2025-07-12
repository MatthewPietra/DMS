# DMS Import Fixes

This document explains how to fix the import errors in the DMS project.

## Quick Fix

Run the dependency installer:
```bash
python install_dependencies.py
```

## Manual Installation

### Core Dependencies
```bash
pip install PySide6 numpy Pillow PyYAML requests psutil
```

### Optional AI Features
```bash
pip install torch opencv-python ultralytics
```

### Development/Testing
```bash
pip install pytest
```

## Import Errors Fixed

### 1. Authentication Module Imports
- ✅ Fixed relative imports in `src/auth/auth_gui.py`
- ✅ Fixed signal definitions for different Qt versions
- ✅ Fixed imports in `test_auth.py`

### 2. GUI Framework Imports
- ✅ Already using fallback system (PyQt5 → PyQt6 → PySide6)
- ✅ Install at least one GUI framework to resolve errors

### 3. Test File Imports
- ✅ Fixed numpy/PIL imports in `tests/conftest.py`
- ✅ Added proper error handling for missing dependencies

### 4. Launcher Script Imports
- ✅ Fixed relative imports in `scripts/unified_launcher.py`
- ✅ Fixed CLI imports in `src/cli.py`

### 5. Missing Configuration Files
- ✅ Created `config/keyauth_config.json`

## Common Issues

### "No module named 'PySide6'"
Install a GUI framework:
```bash
pip install PySide6
# OR
pip install PyQt6
# OR
pip install PyQt5
```

### "No module named 'numpy'"
Install numpy:
```bash
pip install numpy
```

### "No module named 'PIL'"
Install Pillow:
```bash
pip install Pillow
```

### "No module named 'yaml'"
Install PyYAML:
```bash
pip install PyYAML
```

### "No module named 'torch'"
Install PyTorch (optional):
```bash
pip install torch
```

### "No module named 'cv2'"
Install OpenCV (optional):
```bash
pip install opencv-python
```

## Project Structure

The import fixes maintain the proper project structure:
```
DMS/
├── src/
│   ├── auth/           # Authentication modules
│   ├── gui/            # GUI components
│   ├── cli.py          # Command-line interface
│   └── ...
├── tests/              # Test files
├── scripts/            # Launcher scripts
└── config/             # Configuration files
```

## Running the Project

After fixing imports, you can run:
```bash
# GUI mode
python scripts/unified_launcher.py

# CLI mode
python -m src.cli --help

# Test authentication
python test_auth.py
```

## Notes

- The project uses a fallback system for GUI frameworks
- Authentication system is optional but recommended
- Some dependencies are optional (torch, opencv, etc.)
- All import paths have been corrected for proper module resolution 