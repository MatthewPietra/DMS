#!/usr/bin/env python3
"""
DMS Dependency Installer

This script helps install missing dependencies for the DMS project.
"""

import subprocess
import sys
import importlib
from pathlib import Path

def check_module(module_name, package_name=None):
    """Check if a module is available."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔍 DMS Dependency Checker")
    print("=" * 50)
    
    # Core dependencies
    dependencies = [
        ("numpy", "numpy>=1.21.0"),
        ("PIL", "Pillow>=9.0.0"),
        ("yaml", "PyYAML>=6.0.0"),
        ("requests", "requests>=2.28.0"),
        ("psutil", "psutil>=5.9.0"),
    ]
    
    # GUI dependencies (at least one needed)
    gui_dependencies = [
        ("PySide6", "PySide6>=6.5.0"),
        ("PyQt6", "PyQt6>=6.5.0"),
        ("PyQt5", "PyQt5>=5.15.0"),
    ]
    
    # Optional dependencies
    optional_dependencies = [
        ("torch", "torch>=2.0.0"),
        ("cv2", "opencv-python>=4.8.0"),
        ("ultralytics", "ultralytics>=8.0.0"),
        ("pytest", "pytest>=7.0.0"),
    ]
    
    # Check core dependencies
    print("\n📦 Core Dependencies:")
    missing_core = []
    for module, package in dependencies:
        if check_module(module):
            print(f"✅ {module}")
        else:
            print(f"❌ {module} (missing)")
            missing_core.append(package)
    
    # Check GUI dependencies
    print("\n🖥️ GUI Dependencies (at least one needed):")
    gui_available = False
    missing_gui = []
    for module, package in gui_dependencies:
        if check_module(module):
            print(f"✅ {module}")
            gui_available = True
        else:
            print(f"❌ {module} (missing)")
            missing_gui.append(package)
    
    # Check optional dependencies
    print("\n🔧 Optional Dependencies:")
    for module, package in optional_dependencies:
        if check_module(module):
            print(f"✅ {module}")
        else:
            print(f"⚠️ {module} (optional)")
    
    # Install missing dependencies
    if missing_core or not gui_available:
        print(f"\n🚀 Installing Missing Dependencies...")
        
        # Install core dependencies
        for package in missing_core:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
        
        # Install GUI framework (prefer PySide6)
        if not gui_available:
            print("Installing PySide6 (GUI framework)...")
            if install_package("PySide6>=6.5.0"):
                print("✅ PySide6 installed successfully")
            else:
                print("❌ Failed to install PySide6")
                print("💡 Try installing manually: pip install PySide6")
    
    else:
        print("\n🎉 All required dependencies are available!")
    
    print(f"\n📋 Installation Summary:")
    print("=" * 50)
    print("To manually install dependencies, run:")
    print("pip install PySide6 numpy Pillow PyYAML requests psutil")
    print("\nFor optional AI features:")
    print("pip install torch opencv-python ultralytics")
    print("\nFor development/testing:")
    print("pip install pytest")

if __name__ == "__main__":
    main() 