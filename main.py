#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS - Central Launcher Menu System

A comprehensive, zero-dependency launcher that provides a single entry point
for the complete DMS (Detection Model Suite) pipeline. Features automatic hardware
detection, modular dependency installation, and an intuitive menu interface.

This launcher can run independently without requiring any non-standard
dependencies initially installed, and intelligently installs features
only when accessed.

Author: DMS Team
License: MIT
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
import json
import time
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime


class FeatureStatus(Enum):
    """Status of each feature/component."""
    NOT_INSTALLED = "❌ Not Installed"
    INSTALLING = "🔄 Installing..."
    INSTALLED = "✅ Installed"
    ERROR = "⚠️  Error"
    CHECKING = "🔍 Checking..."
    READY = "🚀 Ready"


class DependencyGroup(Enum):
    """Dependency groups for modular installation."""
    BASE = "base"
    CAPTURE = "capture"
    ANNOTATION = "annotation"
    TRAINING = "training"
    AUTO_ANNOTATION = "auto_annotation"
    AMD = "amd"
    CUDA = "cuda"
    GUI = "gui"


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CentralLauncher:
    """
    Central launcher for DMS with comprehensive menu system.
    
    Features:
    - Zero external dependencies initially
    - Smart dependency management
    - Hardware auto-detection
    - Modular feature installation
    - Integration with existing components
    - Comprehensive error handling
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_path = self.project_root / "venv"
        self.python_exe = self._get_python_executable()
        self.pip_exe = self._get_pip_executable()
        
        # Feature status tracking
        self.feature_status = {
            'system': FeatureStatus.NOT_INSTALLED,
            'hardware': FeatureStatus.NOT_INSTALLED,
            'capture': FeatureStatus.NOT_INSTALLED,
            'annotation': FeatureStatus.NOT_INSTALLED,
            'training': FeatureStatus.NOT_INSTALLED,
            'auto_annotation': FeatureStatus.NOT_INSTALLED,
            'studio': FeatureStatus.NOT_INSTALLED
        }
        
        # Hardware detection results
        self.hardware_info = None
        self.installation_strategy = None
        
        # Dependency mapping
        self.dependency_groups = {
            DependencyGroup.BASE: [
                "PyYAML>=6.0",
                "pathlib2>=2.3.0",
                "rich>=13.0.0",
                "tqdm>=4.64.0"
            ],
            DependencyGroup.CAPTURE: [
                "mss>=9.0.0",
                "pygetwindow>=0.0.9",
                "pyautogui>=0.9.54",
                "opencv-python>=4.8.0",
                "Pillow>=9.5.0"
            ],
            DependencyGroup.ANNOTATION: [
                "PyQt5>=5.15.0",
                "PySide6>=6.5.0"
            ],
            DependencyGroup.TRAINING: [
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "ultralytics>=8.0.0",
                "numpy>=1.21.0",
                "scipy>=1.9.0",
                "pandas>=1.5.0",
                "scikit-learn>=1.3.0"
            ],
            DependencyGroup.AUTO_ANNOTATION: [
                "torch>=2.0.0",
                "ultralytics>=8.0.0",
                "opencv-python>=4.8.0"
            ],
            DependencyGroup.AMD: [
                "torch-directml>=0.2.0.dev240914",
                "onnxruntime-directml>=1.16.0"
            ],
            DependencyGroup.CUDA: [
                "torch>=2.0.0",
                "torchvision>=0.15.0"
            ],
            DependencyGroup.GUI: [
                "PyQt5>=5.15.0",
                "PySide6>=6.5.0",
                "customtkinter>=5.2.0"
            ]
        }
        
        # Setup logging
        self._setup_logging()
        
        # Initialize system
        self._initialize_system()
    
    def _setup_logging(self):
        """Setup logging system."""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "launcher.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("dms_launcher")
    
    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable path."""
        if self.venv_path.exists():
            if platform.system() == "Windows":
                return str(self.venv_path / "Scripts" / "python.exe")
            else:
                return str(self.venv_path / "bin" / "python")
        return sys.executable
    
    def _get_pip_executable(self) -> str:
        """Get the appropriate pip executable path."""
        if self.venv_path.exists():
            if platform.system() == "Windows":
                return str(self.venv_path / "Scripts" / "pip.exe")
            else:
                return str(self.venv_path / "bin" / "pip")
        return "pip"
    
    def _initialize_system(self):
        """Initialize the launcher system."""
        self.logger.info("Initializing DMS Launcher...")
        
        # Create necessary directories
        directories = [
            "data/projects",
            "data/models",
            "data/temp",
            "logs",
            "config",
            "exports"
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        # Detect hardware
        self.detect_hardware()
        
        # Check existing installations
        self._check_existing_installations()
        
        self.logger.info("Launcher initialization complete")
    
    def run_command(self, command: List[str], capture_output: bool = True, 
                   show_output: bool = False) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            if show_output and not capture_output:
                result = subprocess.run(command, check=False)
                return result.returncode == 0, ""
            else:
                result = subprocess.run(
                    command, 
                    capture_output=capture_output, 
                    text=True, 
                    check=False
                )
                output = result.stdout + result.stderr if capture_output else ""
                if show_output and output:
                    print(output)
                return result.returncode == 0, output
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return False, str(e)
    
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect system hardware configuration."""
        self.feature_status['hardware'] = FeatureStatus.CHECKING
        
        print(f"{Colors.OKBLUE}🔍 Detecting system hardware...{Colors.ENDC}")
        
        hardware_info = {
            "system": {
                "os": platform.system(),
                "version": platform.version(),
                "architecture": platform.machine(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            },
            "gpu": {
                "nvidia": False,
                "amd": False,
                "details": []
            },
            "recommended_strategy": "cpu"
        }
        
        # Check for NVIDIA GPU
        nvidia_success, nvidia_output = self.run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        )
        if nvidia_success and nvidia_output.strip():
            hardware_info["gpu"]["nvidia"] = True
            hardware_info["recommended_strategy"] = "nvidia"
            for line in nvidia_output.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split(', ')
                    if len(parts) >= 2:
                        hardware_info["gpu"]["details"].append({
                            "type": "NVIDIA",
                            "name": parts[0],
                            "memory": f"{parts[1]} MB"
                        })
        
        # Check for AMD GPU (Windows)
        if platform.system() == "Windows" and not hardware_info["gpu"]["nvidia"]:
            wmic_success, wmic_output = self.run_command(
                ["wmic", "path", "win32_VideoController", "get", "name"]
            )
            if wmic_success and ("AMD" in wmic_output or "Radeon" in wmic_output):
                hardware_info["gpu"]["amd"] = True
                hardware_info["recommended_strategy"] = "amd"
                hardware_info["gpu"]["details"].append({
                    "type": "AMD",
                    "name": "AMD/Radeon GPU detected",
                    "memory": "Unknown"
                })
        
        # Check for AMD GPU (Linux)
        elif platform.system() == "Linux" and not hardware_info["gpu"]["nvidia"]:
            lspci_success, lspci_output = self.run_command(["lspci"])
            if lspci_success and ("AMD" in lspci_output or "Radeon" in lspci_output):
                hardware_info["gpu"]["amd"] = True
                hardware_info["recommended_strategy"] = "amd"
                hardware_info["gpu"]["details"].append({
                    "type": "AMD",
                    "name": "AMD/Radeon GPU detected",
                    "memory": "Unknown"
                })
        
        self.hardware_info = hardware_info
        self.installation_strategy = hardware_info["recommended_strategy"]
        self.feature_status['hardware'] = FeatureStatus.INSTALLED
        
        self.logger.info(f"Hardware detection complete: {hardware_info['recommended_strategy']}")
        return hardware_info
    
    def _check_existing_installations(self):
        """Check for existing installations and update status."""
        self.logger.info("Checking existing installations...")
        
        # Check if virtual environment exists
        if self.venv_path.exists():
            self.feature_status['system'] = FeatureStatus.INSTALLED
        
        # Check for existing components
        self._verify_installation('capture', self._verify_capture_deps)
        self._verify_installation('annotation', self._verify_annotation_deps)
        self._verify_installation('training', self._verify_training_deps)
        self._verify_installation('auto_annotation', self._verify_auto_annotation_deps)
    
    def _verify_installation(self, feature: str, verification_func: Callable) -> bool:
        """Verify installation of a specific feature."""
        try:
            if verification_func():
                self.feature_status[feature] = FeatureStatus.INSTALLED
                return True
            else:
                self.feature_status[feature] = FeatureStatus.NOT_INSTALLED
                return False
        except Exception as e:
            self.logger.error(f"Verification failed for {feature}: {e}")
            self.feature_status[feature] = FeatureStatus.ERROR
            return False
    
    def _verify_capture_deps(self) -> bool:
        """Verify capture dependencies."""
        try:
            import mss
            import cv2
            import PIL
            return True
        except ImportError:
            return False
    
    def _verify_annotation_deps(self) -> bool:
        """Verify annotation dependencies."""
        try:
            # Try PyQt5 first, then PySide6
            try:
                import PyQt5
                return True
            except ImportError:
                import PySide6
                return True
        except ImportError:
            return False
    
    def _verify_training_deps(self) -> bool:
        """Verify training dependencies."""
        try:
            import torch
            import ultralytics
            return True
        except ImportError:
            return False
    
    def _verify_auto_annotation_deps(self) -> bool:
        """Verify auto-annotation dependencies."""
        try:
            import torch
            import ultralytics
            import cv2
            return True
        except ImportError:
            return False
    
    def install_dependencies(self, dependency_group: DependencyGroup) -> bool:
        """Install dependencies for a specific group."""
        if dependency_group not in self.dependency_groups:
            self.logger.error(f"Unknown dependency group: {dependency_group}")
            return False
        
        print(f"{Colors.OKBLUE}📦 Installing {dependency_group.value} dependencies...{Colors.ENDC}")
        
        dependencies = self.dependency_groups[dependency_group]
        
        # Add hardware-specific dependencies
        if dependency_group == DependencyGroup.TRAINING:
            if self.installation_strategy == "amd":
                dependencies.extend(self.dependency_groups[DependencyGroup.AMD])
            elif self.installation_strategy == "nvidia":
                dependencies.extend(self.dependency_groups[DependencyGroup.CUDA])
        
        # Install dependencies
        for dep in dependencies:
            print(f"   Installing: {dep}")
            success, output = self.run_command([self.pip_exe, "install", dep])
            if not success:
                self.logger.error(f"Failed to install {dep}: {output}")
                return False
        
        print(f"{Colors.OKGREEN}✅ {dependency_group.value} dependencies installed successfully{Colors.ENDC}")
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment if it doesn't exist."""
        if self.venv_path.exists():
            print(f"{Colors.OKGREEN}✅ Virtual environment already exists{Colors.ENDC}")
            return True
        
        print(f"{Colors.OKBLUE}🔧 Creating virtual environment...{Colors.ENDC}")
        
        success, output = self.run_command([sys.executable, "-m", "venv", str(self.venv_path)])
        if not success:
            self.logger.error(f"Failed to create virtual environment: {output}")
            return False
        
        # Upgrade pip in the virtual environment
        success, output = self.run_command([self.pip_exe, "install", "--upgrade", "pip"])
        if not success:
            self.logger.warning(f"Failed to upgrade pip: {output}")
        
        self.feature_status['system'] = FeatureStatus.INSTALLED
        print(f"{Colors.OKGREEN}✅ Virtual environment created successfully{Colors.ENDC}")
        return True
    
    def launch_feature(self, feature: str) -> bool:
        """Launch a specific feature with dependency management."""
        self.logger.info(f"Launching feature: {feature}")
        
        # Ensure virtual environment exists
        if not self.create_virtual_environment():
            return False
        
        # Install dependencies based on feature
        if feature == "capture":
            if self.feature_status['capture'] != FeatureStatus.INSTALLED:
                if not self.install_dependencies(DependencyGroup.CAPTURE):
                    return False
                self.feature_status['capture'] = FeatureStatus.INSTALLED
        
        elif feature == "annotation":
            if self.feature_status['annotation'] != FeatureStatus.INSTALLED:
                if not self.install_dependencies(DependencyGroup.ANNOTATION):
                    return False
                self.feature_status['annotation'] = FeatureStatus.INSTALLED
        
        elif feature == "training":
            if self.feature_status['training'] != FeatureStatus.INSTALLED:
                if not self.install_dependencies(DependencyGroup.TRAINING):
                    return False
                self.feature_status['training'] = FeatureStatus.INSTALLED
        
        elif feature == "auto_annotation":
            if self.feature_status['auto_annotation'] != FeatureStatus.INSTALLED:
                if not self.install_dependencies(DependencyGroup.AUTO_ANNOTATION):
                    return False
                self.feature_status['auto_annotation'] = FeatureStatus.INSTALLED
        
        # Launch the feature using existing components
        return self._launch_feature_module(feature)
    
    def _launch_feature_module(self, feature: str) -> bool:
        """Launch feature using existing module system."""
        try:
            if feature == "capture":
                # Launch capture system
                capture_script = self.project_root / "src" / "capture" / "__main__.py"
                if capture_script.exists():
                    return self.run_command([self.python_exe, str(capture_script)])[0]
                else:
                    print(f"{Colors.WARNING}⚠️  Capture module not found{Colors.ENDC}")
                    return False
            
            elif feature == "annotation":
                # Launch annotation interface
                annotation_script = self.project_root / "src" / "annotation" / "__main__.py"
                if annotation_script.exists():
                    return self.run_command([self.python_exe, str(annotation_script)])[0]
                else:
                    print(f"{Colors.WARNING}⚠️  Annotation module not found{Colors.ENDC}")
                    return False
            
            elif feature == "training":
                # Launch training system
                training_script = self.project_root / "src" / "training" / "__main__.py"
                if training_script.exists():
                    return self.run_command([self.python_exe, str(training_script)])[0]
                else:
                    print(f"{Colors.WARNING}⚠️  Training module not found{Colors.ENDC}")
                    return False
            
            elif feature == "auto_annotation":
                # Launch auto-annotation system
                auto_annotation_script = self.project_root / "src" / "auto_annotation" / "__main__.py"
                if auto_annotation_script.exists():
                    return self.run_command([self.python_exe, str(auto_annotation_script)])[0]
                else:
                    print(f"{Colors.WARNING}⚠️  Auto-annotation module not found{Colors.ENDC}")
                    return False
            
            elif feature == "studio":
                # Launch full studio
                studio_script = self.project_root / "src" / "studio.py"
                if studio_script.exists():
                    return self.run_command([self.python_exe, str(studio_script)])[0]
                else:
                    print(f"{Colors.WARNING}⚠️  Studio module not found{Colors.ENDC}")
                    return False
            
            elif feature == "cli":
                # Launch CLI interface
                cli_script = self.project_root / "src" / "cli.py"
                if cli_script.exists():
                    return self.run_command([self.python_exe, str(cli_script), "--help"])[0]
                else:
                    print(f"{Colors.WARNING}⚠️  CLI module not found{Colors.ENDC}")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to launch {feature}: {e}")
            return False
    
    def display_hardware_info(self):
        """Display detected hardware information."""
        if not self.hardware_info:
            return
        
        print(f"\n{Colors.OKBLUE}📊 System Information:{Colors.ENDC}")
        print(f"   OS: {self.hardware_info['system']['os']} {self.hardware_info['system']['version']}")
        print(f"   Architecture: {self.hardware_info['system']['architecture']}")
        print(f"   Python: {self.hardware_info['system']['python_version']}")
        
        print(f"\n{Colors.OKBLUE}🔧 Hardware Configuration:{Colors.ENDC}")
        print(f"   Strategy: {self.hardware_info['recommended_strategy'].upper()}")
        
        if self.hardware_info['gpu']['details']:
            print(f"\n{Colors.OKBLUE}🎮 GPU Information:{Colors.ENDC}")
            for gpu in self.hardware_info['gpu']['details']:
                print(f"   {gpu['type']}: {gpu['name']} ({gpu['memory']})")
        else:
            print(f"\n{Colors.WARNING}💻 No GPU detected - using CPU mode{Colors.ENDC}")
    
    def display_feature_status(self):
        """Display current feature installation status."""
        print(f"\n{Colors.OKBLUE}📋 Feature Status:{Colors.ENDC}")
        for feature, status in self.feature_status.items():
            print(f"   {feature.replace('_', ' ').title()}: {status.value}")
    
    def show_main_menu(self):
        """Display the main launcher menu."""
        while True:
            self.clear_screen()
            self.print_banner()
            self.display_hardware_info()
            self.display_feature_status()
            
            print(f"\n{Colors.HEADER}{Colors.BOLD}Main Menu:{Colors.ENDC}")
            print("1. 🚀 Quick Start (Full Studio)")
            print("2. 📸 Image Capture System")
            print("3. ✏️  Annotation Interface")
            print("4. 🎯 Model Training")
            print("5. 🤖 Auto-Annotation")
            print("6. 💻 Command Line Interface")
            print("7. ⚙️  System Setup & Configuration")
            print("8. 📊 System Information")
            print("9. ❓ Help & Documentation")
            print("0. 🚪 Exit")
            
            choice = input(f"\n{Colors.OKCYAN}Select an option (0-9): {Colors.ENDC}").strip()
            
            if not self.handle_menu_choice(choice):
                break
    
    def handle_menu_choice(self, choice: str) -> bool:
        """Handle main menu choice."""
        if choice == "0":
            print(f"\n{Colors.OKGREEN}👋 Thank you for using DMS!{Colors.ENDC}")
            return False
        
        elif choice == "1":
            return self.launch_feature("studio")
        
        elif choice == "2":
            return self.launch_feature("capture")
        
        elif choice == "3":
            return self.launch_feature("annotation")
        
        elif choice == "4":
            return self.launch_feature("training")
        
        elif choice == "5":
            return self.launch_feature("auto_annotation")
        
        elif choice == "6":
            return self.launch_feature("cli")
        
        elif choice == "7":
            return self.show_setup_menu()
        
        elif choice == "8":
            return self.show_system_info()
        
        elif choice == "9":
            return self.show_help()
        
        else:
            print(f"{Colors.WARNING}⚠️  Invalid choice. Please try again.{Colors.ENDC}")
            time.sleep(1)
            return True
    
    def show_setup_menu(self) -> bool:
        """Show system setup menu."""
        while True:
            self.clear_screen()
            print(f"{Colors.HEADER}{Colors.BOLD}System Setup & Configuration{Colors.ENDC}")
            print("1. 🔧 Create Virtual Environment")
            print("2. 📦 Install Base Dependencies")
            print("3. 🎮 Install GPU Dependencies")
            print("4. 🖥️  Install GUI Dependencies")
            print("5. 🔍 Verify Installations")
            print("6. 🧹 Clean Installation")
            print("0. ↩️  Back to Main Menu")
            
            choice = input(f"\n{Colors.OKCYAN}Select an option (0-6): {Colors.ENDC}").strip()
            
            if choice == "0":
                return True
            
            elif choice == "1":
                self.create_virtual_environment()
                input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == "2":
                self.install_dependencies(DependencyGroup.BASE)
                input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == "3":
                if self.installation_strategy == "amd":
                    self.install_dependencies(DependencyGroup.AMD)
                elif self.installation_strategy == "nvidia":
                    self.install_dependencies(DependencyGroup.CUDA)
                else:
                    print(f"{Colors.WARNING}⚠️  No GPU detected for GPU dependencies{Colors.ENDC}")
                input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == "4":
                self.install_dependencies(DependencyGroup.GUI)
                input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == "5":
                self._check_existing_installations()
                print(f"{Colors.OKGREEN}✅ Installation verification complete{Colors.ENDC}")
                input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
            
            elif choice == "6":
                if input(f"{Colors.WARNING}⚠️  Are you sure you want to clean the installation? (y/N): {Colors.ENDC}").lower() == 'y':
                    self._clean_installation()
                input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
            
            else:
                print(f"{Colors.WARNING}⚠️  Invalid choice. Please try again.{Colors.ENDC}")
                time.sleep(1)
    
    def show_system_info(self) -> bool:
        """Show detailed system information."""
        self.clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}System Information{Colors.ENDC}")
        
        self.display_hardware_info()
        self.display_feature_status()
        
        # Additional system info
        print(f"\n{Colors.OKBLUE}📁 Project Structure:{Colors.ENDC}")
        print(f"   Project Root: {self.project_root}")
        print(f"   Virtual Environment: {self.venv_path}")
        print(f"   Python Executable: {self.python_exe}")
        print(f"   Pip Executable: {self.pip_exe}")
        
        # Check directory structure
        directories = ["data", "config", "logs", "src"]
        print(f"\n{Colors.OKBLUE}📂 Directory Status:{Colors.ENDC}")
        for directory in directories:
            dir_path = self.project_root / directory
            status = "✅" if dir_path.exists() else "❌"
            print(f"   {status} {directory}/")
        
        input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
        return True
    
    def show_help(self) -> bool:
        """Show help and documentation."""
        self.clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}Help & Documentation{Colors.ENDC}")
        
        help_text = """
🚀 DMS - Complete Object Detection Pipeline

📖 Quick Start Guide:
1. Run this launcher: python main.py
2. Select "Quick Start" for full studio experience
3. Or choose specific features to install and run

🔧 Features Available:
• Image Capture: Screen and window capture with real-time processing
• Annotation Interface: Professional annotation tools with PyQt GUI
• Model Training: Multi-YOLO architecture training (YOLOv5-v11)
• Auto-Annotation: AI-powered annotation with quality control
• Command Line Interface: Full CLI for automation and scripting

🎮 Hardware Support:
• NVIDIA CUDA: Full GPU acceleration
• AMD DirectML: DirectML acceleration for AMD GPUs
• CPU Fallback: Optimized CPU processing

📦 Smart Dependency Management:
• Zero initial dependencies required
• Features install automatically when accessed
• Hardware-specific optimizations
• Virtual environment isolation

🔍 Troubleshooting:
• Check system information for hardware detection
• Use setup menu for manual dependency installation
• Verify installations before running features
• Check logs for detailed error information

📚 Documentation:
• README.md: Complete installation and usage guide
• docs/: Comprehensive documentation
• examples/: Usage examples and tutorials

💡 Tips:
• First run will take longer due to dependency installation
• GPU features require appropriate drivers
• Use virtual environment for clean isolation
• Check hardware compatibility before installation
        """
        
        print(help_text)
        input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
        return True
    
    def _clean_installation(self):
        """Clean the current installation."""
        print(f"{Colors.WARNING}🧹 Cleaning installation...{Colors.ENDC}")
        
        # Remove virtual environment
        if self.venv_path.exists():
            import shutil
            shutil.rmtree(self.venv_path)
            print("   Removed virtual environment")
        
        # Reset feature status
        for feature in self.feature_status:
            self.feature_status[feature] = FeatureStatus.NOT_INSTALLED
        
        print(f"{Colors.OKGREEN}✅ Installation cleaned successfully{Colors.ENDC}")
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if platform.system() == 'Windows' else 'clear')
    
    def print_banner(self):
        """Print the main launcher banner."""
        banner = f"""
{Colors.HEADER}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                              DMS Launcher                                    ║
║                     Complete Object Detection Pipeline                       ║
║                                                                              ║
║  🚀 Zero-Configuration Setup    🔧 Hardware Auto-Detection                  ║
║  📦 Modular Dependencies        🎯 One-Click Experience                     ║
║  🔍 Self-Verification          ✨ Intuitive Interface                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
        """
        print(banner)


def main():
    """Main entry point for DMS Launcher."""
    try:
        launcher = CentralLauncher()
        launcher.show_main_menu()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️  Launcher interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Launcher error: {e}{Colors.ENDC}")
        logging.error(f"Launcher error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 