#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS GUI Launcher

A modern GUI-based launcher for the Detection Model Suite (DMS).
Replaces the console-based launcher with an intuitive graphical interface.
"""

import sys
import os
from pathlib import Path

# Ensure DMSMainWindow is always defined
DMSMainWindow = None

# Add the src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

GUI_AVAILABLE = False
try:
    from PySide6.QtWidgets import QApplication
    from gui.main_window import DMSMainWindow
    GUI_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import DMSMainWindow
        GUI_AVAILABLE = True
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
            from gui.main_window import DMSMainWindow
            GUI_AVAILABLE = True
        except ImportError:
            GUI_AVAILABLE = False


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    # Check for GUI framework
    if not GUI_AVAILABLE:
        missing_deps.append("PySide6, PyQt6, or PyQt5")
    
    # Check for other dependencies
    try:
        import yaml
    except ImportError:
        missing_deps.append("PyYAML")
    
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    return missing_deps


def install_dependencies():
    """Install missing dependencies."""
    print("Installing missing dependencies...")
    
    # Try to install GUI framework
    if not GUI_AVAILABLE:
        print("Installing PySide6...")
        os.system(f"{sys.executable} -m pip install PySide6")
    
    # Install other dependencies
    deps = ["PyYAML", "psutil"]
    for dep in deps:
        try:
            __import__(dep.lower())
        except ImportError:
            print(f"Installing {dep}...")
            os.system(f"{sys.executable} -m pip install {dep}")


def show_console_fallback():
    """Show console fallback when GUI is not available."""
    print("=" * 60)
    print("DMS - Detection Model Suite")
    print("=" * 60)
    print()
    print("GUI dependencies not available. Please install one of:")
    print("  - PySide6 (recommended)")
    print("  - PyQt6")
    print("  - PyQt5")
    print()
    print("To install dependencies, run:")
    print("  pip install PySide6 PyYAML psutil")
    print()
    print("Or use the console launcher:")
    print("  python launcher.py")
    print()
    input("Press Enter to exit...")


def main():
    """Main entry point for the GUI launcher."""
    # Check dependencies
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        
        # Try to install automatically
        try:
            install_dependencies()
            # Re-check after installation
            missing_deps = check_dependencies()
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
        
        if missing_deps:
            show_console_fallback()
            return
    
    # Create Qt application
    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print(f"Failed to create Qt application: {e}")
        show_console_fallback()
        return
    
    # Set application properties
    app.setApplicationName("DMS")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("DMS Team")
    
    # Create and show main window
    try:
        if DMSMainWindow is None:
            print("DMSMainWindow class not available. GUI cannot be started.")
            show_console_fallback()
            return
        window = DMSMainWindow()
        window.show()
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        show_console_fallback()


if __name__ == "__main__":
    main() 