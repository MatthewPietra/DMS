#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS - Central Launcher

A zero-dependency launcher that provides a single entry point for the complete
DMS (Detection Model Suite) pipeline. Features automatic hardware detection, modular
dependency installation, and an intuitive menu system.

No external dependencies required - uses only Python standard library.
"""

import os
import sys
import subprocess
import platform
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FeatureStatus(Enum):
    """Status of each feature/component."""
    NOT_INSTALLED = "❌ Not Installed"
    INSTALLING = "🔄 Installing..."
    INSTALLED = "✅ Installed"
    ERROR = "⚠️  Error"
    CHECKING = "🔍 Checking..."


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


class LauncherCore:
    """Core launcher functionality with zero external dependencies."""
    
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
            'studio': FeatureStatus.NOT_INSTALLED
        }
        
        # Hardware detection results
        self.hardware_info = None
        self.installation_strategy = None
    
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
            return False, str(e)
    
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
    
    def detect_hardware(self) -> Dict[str, any]:
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
        nvidia_success, nvidia_output = self.run_command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
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
            wmic_success, wmic_output = self.run_command(["wmic", "path", "win32_VideoController", "get", "name"])
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
        
        return hardware_info
    
    def display_hardware_info(self):
        """Display detected hardware information."""
        if not self.hardware_info:
            return
        
        print(f"\n{Colors.OKBLUE}📊 System Information:{Colors.ENDC}")
        print(f"   OS: {self.hardware_info['system']['os']} {self.hardware_info['system']['version']}")
        print(f"   Architecture: {self.hardware_info['system']['architecture']}")
        print(f"   Python: {self.hardware_info['system']['python_version']}")
        
        print(f"\n{Colors.OKBLUE}🎮 GPU Configuration:{Colors.ENDC}")
        if self.hardware_info["gpu"]["details"]:
            for gpu in self.hardware_info["gpu"]["details"]:
                print(f"   ✅ {gpu['name']} ({gpu['memory']})")
        else:
            print(f"   ℹ️  CPU-only mode (no GPU detected)")
        
        print(f"\n{Colors.OKGREEN}🔧 Recommended Strategy: {self.installation_strategy.upper()}{Colors.ENDC}")
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment if it doesn't exist."""
        if self.venv_path.exists():
            return True
        
        print(f"{Colors.OKBLUE}📦 Creating virtual environment...{Colors.ENDC}")
        success, output = self.run_command([sys.executable, "-m", "venv", str(self.venv_path)])
        
        if success:
            print(f"{Colors.OKGREEN}✅ Virtual environment created{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}❌ Failed to create virtual environment: {output}{Colors.ENDC}")
            return False
    
    def install_base_dependencies(self) -> bool:
        """Install base dependencies required for the launcher."""
        print(f"{Colors.OKBLUE}📦 Installing base dependencies...{Colors.ENDC}")
        
        base_packages = [
            "pip>=23.0.0",
            "setuptools>=65.0.0",
            "wheel>=0.38.0"
        ]
        
        for package in base_packages:
            success, output = self.run_command([self.pip_exe, "install", "--upgrade", package])
            if not success:
                print(f"{Colors.FAIL}❌ Failed to install {package}: {output}{Colors.ENDC}")
                return False
        
        print(f"{Colors.OKGREEN}✅ Base dependencies installed{Colors.ENDC}")
        return True
    
    def install_feature_dependencies(self, feature: str) -> bool:
        """Install dependencies for a specific feature."""
        self.feature_status[feature] = FeatureStatus.INSTALLING
        
        requirements_map = {
            'capture': ['mss>=9.0.0', 'pygetwindow>=0.0.9', 'pyautogui>=0.9.54', 'Pillow>=10.0.0', 'psutil>=5.9.0', 'GPUtil>=1.4.0', 'WMI>=1.5.1; sys_platform=="win32"'],
            'annotation': ['PyQt5>=5.15.0', 'opencv-python>=4.8.0', 'numpy>=1.24.0', 'psutil>=5.9.0'],
            'training': self._get_training_requirements(),
            'studio': ['PyYAML>=6.0', 'rich>=13.0.0', 'loguru>=0.7.0', 'tqdm>=4.65.0', 'psutil>=5.9.0', 'click>=8.0.0']
        }
        
        if feature not in requirements_map:
            self.feature_status[feature] = FeatureStatus.ERROR
            return False
        
        print(f"{Colors.OKBLUE}📦 Installing {feature} dependencies...{Colors.ENDC}")
        
        packages = requirements_map[feature]
        for package in packages:
            print(f"   Installing {package}...")
            
            # Special handling for torch-directml with pre-release flag
            if 'torch-directml' in package:
                success, output = self.run_command([self.pip_exe, "install", "--pre", package])
                if not success:
                    print(f"{Colors.WARNING}⚠️  DirectML pre-release not available, trying standard version...{Colors.ENDC}")
                    success, output = self.run_command([self.pip_exe, "install", package])
                    if not success:
                        print(f"{Colors.WARNING}⚠️  DirectML not available for this configuration, skipping: {package}{Colors.ENDC}")
                        print(f"   DirectML requires Windows 10+ and DirectX 12 compatible GPU")
                        continue
            else:
                success, output = self.run_command([self.pip_exe, "install", package])
                if not success:
                    print(f"{Colors.FAIL}❌ Failed to install {package}: {output}{Colors.ENDC}")
                    self.feature_status[feature] = FeatureStatus.ERROR
                    return False
        
        self.feature_status[feature] = FeatureStatus.INSTALLED
        print(f"{Colors.OKGREEN}✅ {feature.capitalize()} dependencies installed{Colors.ENDC}")
        return True
    
    def _get_training_requirements(self) -> List[str]:
        """Get training requirements based on hardware strategy."""
        base_requirements = [
            'numpy>=1.24.0',
            'opencv-python>=4.8.0',
            'Pillow>=10.0.0',
            'PyYAML>=6.0',
            'tqdm>=4.65.0',
            'matplotlib>=3.7.0',
            'ultralytics>=8.0.0',
            'psutil>=5.9.0',
            'GPUtil>=1.4.0',
            'WMI>=1.5.1; sys_platform=="win32"'
        ]
        
        if self.installation_strategy == "nvidia":
            base_requirements.extend([
                'torch>=2.2.0',
                'torchvision>=0.17.0',
                '--extra-index-url https://download.pytorch.org/whl/cu118'
            ])
        elif self.installation_strategy == "amd":
            # DirectML requires specific PyTorch versions (≤2.3.1) and Python ≤3.12
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if sys.version_info >= (3, 13):
                print(f"⚠️  DirectML requires Python ≤3.12, current: {python_version}")
                print("   Installing CPU version as fallback...")
                base_requirements.extend([
                    'torch>=2.0.0',
                    'torchvision>=0.15.0'
                ])
            else:
                print(f"✅ Installing DirectML for AMD GPU acceleration (Python {python_version})")
                base_requirements.extend([
                    'torch-directml'  # This will install compatible PyTorch version
                ])
        else:
            base_requirements.extend([
                'torch>=2.0.0',
                'torchvision>=0.15.0'
            ])
        
        return base_requirements
    
    def _get_training_verification_command(self) -> List[str]:
        """Get training verification command based on installation strategy."""
        if self.installation_strategy == "amd":
            # For AMD, verify DirectML is working
            verification_code = """
import torch, ultralytics
try:
    import torch_directml
    device = torch_directml.device()
    print(f'Training OK - DirectML: {device}')
except ImportError:
    print('Training OK - CPU fallback')
"""
        elif self.installation_strategy == "nvidia":
            # For NVIDIA, verify CUDA is working
            verification_code = """
import torch, ultralytics
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0
print(f'Training OK - CUDA: {cuda_available}, Devices: {device_count}')
"""
        else:
            # For CPU
            verification_code = "import torch, ultralytics; print('Training OK - CPU')"
        
        return [self.python_exe, "-c", verification_code]
    
    def verify_installation(self, feature: str) -> bool:
        """Verify that a feature is properly installed."""
        verification_commands = {
            'system': [self.python_exe, "-c", "import sys; print('System OK')"],
            'hardware': [self.python_exe, "-c", "import platform; print('Hardware detection OK')"],
            'capture': [self.python_exe, "-c", "import mss, pygetwindow; print('Capture OK')"],
            'annotation': [self.python_exe, "-c", "import cv2, numpy; print('Annotation OK')"],
            'training': self._get_training_verification_command(),
            'studio': [self.python_exe, "-c", "import yaml, rich; print('Studio OK')"]
        }
        
        if feature not in verification_commands:
            return False
        
        success, output = self.run_command(verification_commands[feature])
        return success and "OK" in output
    
    def get_feature_status_display(self, feature: str) -> str:
        """Get formatted status display for a feature."""
        status = self.feature_status.get(feature, FeatureStatus.NOT_INSTALLED)
        return status.value
    
    def launch_studio_component(self, component: str) -> bool:
        """Launch a specific studio component."""
        component_commands = {
            'studio': [self.python_exe, "-m", "src", "--demo"],
            'cli': [self.python_exe, "-m", "src.cli"],
            'capture': [self.python_exe, "-m", "src.capture", "--demo"],
            'annotation': [self.python_exe, "-m", "src.annotation"],
            'training': [self.python_exe, "-m", "src.training", "--demo"],
            'auth': [self.python_exe, "-m", "src.auth"]
        }
        
        if component not in component_commands:
            print(f"{Colors.FAIL}❌ Unknown component: {component}{Colors.ENDC}")
            return False
        
        # For authentication components, ensure dependencies are available
        if component == "auth":
            try:
                print(f"{Colors.OKBLUE}🔍 Ensuring authentication dependencies...{Colors.ENDC}")
                # Add src directory to path for import
                src_path = self.project_root / "src"
                sys.path.insert(0, str(src_path))
                
                from auth.dependency_manager import ensure_auth_dependencies
                ensure_auth_dependencies()
                print(f"{Colors.OKGREEN}✅ Authentication dependencies ready{Colors.ENDC}")
            except ImportError as e:
                print(f"{Colors.WARNING}⚠️  Authentication dependency manager not available: {e}{Colors.ENDC}")
                print(f"{Colors.WARNING}   Authentication features may not work properly{Colors.ENDC}")
        
        print(f"{Colors.OKBLUE}🚀 Launching {component}...{Colors.ENDC}")
        
        # Change to project directory
        os.chdir(self.project_root)
        
        success, output = self.run_command(component_commands[component], capture_output=False, show_output=True)
        
        if not success:
            print(f"{Colors.FAIL}❌ Failed to launch {component}{Colors.ENDC}")
            if output:
                print(f"Error: {output}")
        
        return success


class LauncherUI:
    """User interface for the launcher."""
    
    def __init__(self, core: LauncherCore):
        self.core = core
        self.first_run = not (self.core.project_root / "data").exists()
    
    def show_main_menu(self):
        """Display the main launcher menu."""
        self.core.clear_screen()
        self.core.print_banner()
        
        if self.core.hardware_info:
            self.core.display_hardware_info()
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}📋 MAIN MENU{Colors.ENDC}")
        print(f"   Current Directory: {self.core.project_root}")
        print(f"   Python: {self.core.python_exe}")
        
        menu_options = [
            ("1", "🔧 System Setup & Verification", "system"),
            ("2", "🔍 Hardware Detection", "hardware"),
            ("3", "📸 Screen Capture System", "capture"),
            ("4", "🏷️  Annotation Interface", "annotation"),
            ("5", "🤖 Model Training", "training"),
            ("6", "📁 Project Management", "studio"),
            ("7", "🚀 Launch Full Studio", "studio"),
            ("8", "❓ Help & Documentation", "help"),
            ("9", "🚪 Exit", "exit")
        ]
        
        print(f"\n{Colors.OKBLUE}Available Options:{Colors.ENDC}")
        for key, description, feature in menu_options:
            if feature in ['help', 'exit']:
                status = ""
            else:
                status = f" {self.core.get_feature_status_display(feature)}"
            print(f"   {key}. {description}{status}")
        
        if self.first_run:
            print(f"\n{Colors.WARNING}🆕 First run detected! Recommend starting with option 1 (System Setup).{Colors.ENDC}")
    
    def handle_menu_choice(self, choice: str) -> bool:
        """Handle user menu choice. Returns False to exit."""
        choice = choice.strip().lower()
        
        if choice == '1':
            self.setup_system()
        elif choice == '2':
            self.detect_hardware()
        elif choice == '3':
            self.setup_capture()
        elif choice == '4':
            self.setup_annotation()
        elif choice == '5':
            self.setup_training()
        elif choice == '6':
            self.launch_project_management()
        elif choice == '7':
            self.launch_full_studio()
        elif choice == '8':
            self.show_help()
        elif choice == '9':
            return False
        else:
            print(f"{Colors.FAIL}❌ Invalid choice. Please select 1-9.{Colors.ENDC}")
            input("\nPress Enter to continue...")
        
        return True
    
    def setup_system(self):
        """Set up the base system requirements."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔧 SYSTEM SETUP{Colors.ENDC}")
        
        self.core.feature_status['system'] = FeatureStatus.INSTALLING
        
        # Create virtual environment
        if not self.core.create_virtual_environment():
            input("\nPress Enter to continue...")
            return
        
        # Install base dependencies
        if not self.core.install_base_dependencies():
            input("\nPress Enter to continue...")
            return
        
        # Create basic directory structure
        directories = ["data", "data/projects", "data/models", "data/temp", "logs", "config"]
        for directory in directories:
            (self.core.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        # Verify system setup
        if self.core.verify_installation('system'):
            self.core.feature_status['system'] = FeatureStatus.INSTALLED
            print(f"\n{Colors.OKGREEN}✅ System setup completed successfully!{Colors.ENDC}")
            self.first_run = False
        else:
            self.core.feature_status['system'] = FeatureStatus.ERROR
            print(f"\n{Colors.FAIL}❌ System setup verification failed{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def detect_hardware(self):
        """Run hardware detection."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔍 HARDWARE DETECTION{Colors.ENDC}")
        
        self.core.detect_hardware()
        self.core.display_hardware_info()
        
        print(f"\n{Colors.OKGREEN}✅ Hardware detection completed!{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def setup_capture(self):
        """Set up screen capture system."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}📸 SCREEN CAPTURE SETUP{Colors.ENDC}")
        
        if not self._ensure_system_ready():
            return
        
        if self.core.install_feature_dependencies('capture'):
            if self.core.verify_installation('capture'):
                print(f"\n{Colors.OKGREEN}✅ Screen capture system ready!{Colors.ENDC}")
                
                # Option to test capture
                test = input(f"\n{Colors.OKBLUE}Would you like to test the capture system? (y/n): {Colors.ENDC}").lower()
                if test == 'y':
                    self.core.launch_studio_component('capture')
            else:
                print(f"\n{Colors.FAIL}❌ Capture system verification failed{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def setup_annotation(self):
        """Set up annotation interface."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}🏷️  ANNOTATION INTERFACE SETUP{Colors.ENDC}")
        
        if not self._ensure_system_ready():
            return
        
        if self.core.install_feature_dependencies('annotation'):
            if self.core.verify_installation('annotation'):
                print(f"\n{Colors.OKGREEN}✅ Annotation interface ready!{Colors.ENDC}")
                
                # Option to launch annotation interface
                launch = input(f"\n{Colors.OKBLUE}Would you like to launch the annotation interface? (y/n): {Colors.ENDC}").lower()
                if launch == 'y':
                    self.core.launch_studio_component('annotation')
            else:
                print(f"\n{Colors.FAIL}❌ Annotation interface verification failed{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def setup_training(self):
        """Set up model training system."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}🤖 MODEL TRAINING SETUP{Colors.ENDC}")
        
        if not self._ensure_system_ready():
            return
        
        # Ensure hardware is detected for optimal training setup
        if not self.core.hardware_info:
            print(f"{Colors.WARNING}⚠️  Running hardware detection first...{Colors.ENDC}")
            self.core.detect_hardware()
            self.core.display_hardware_info()
        
        # Check for AMD DirectML compatibility issue
        if self.core.installation_strategy == "amd" and sys.version_info >= (3, 13):
            print(f"\n{Colors.WARNING}⚠️  DirectML Compatibility Issue{Colors.ENDC}")
            print(f"   Current Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
            print(f"   DirectML requires: Python ≤3.12")
            print(f"\n{Colors.OKBLUE}We have a DirectML-compatible environment ready!{Colors.ENDC}")
            print(f"   Located at: venv_directml (Python 3.11)")
            print(f"   DirectML is already installed and tested")
            
            choice = input(f"\n{Colors.OKBLUE}Continue with CPU fallback in current environment? (y/n): {Colors.ENDC}").strip().lower()
            if choice != 'y':
                print(f"\n{Colors.OKGREEN}To use DirectML:{Colors.ENDC}")
                print(f"   1. Run: venv_directml\\Scripts\\python.exe dms\\launcher.py")
                print(f"   2. Or activate DirectML venv and run training there")
                input("\nPress Enter to continue...")
                return
        
        print(f"\n{Colors.OKBLUE}Installing training dependencies for {self.core.installation_strategy.upper()} configuration...{Colors.ENDC}")
        
        if self.core.install_feature_dependencies('training'):
            if self.core.verify_installation('training'):
                print(f"\n{Colors.OKGREEN}✅ Model training system ready!{Colors.ENDC}")
                print(f"   Strategy: {self.core.installation_strategy.upper()}")
                
                # Option to launch training interface
                launch = input(f"\n{Colors.OKBLUE}Would you like to launch the training interface? (y/n): {Colors.ENDC}").lower()
                if launch == 'y':
                    self.core.launch_studio_component('training')
            else:
                print(f"\n{Colors.FAIL}❌ Training system verification failed{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def launch_project_management(self):
        """Launch project management interface."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}📁 PROJECT MANAGEMENT{Colors.ENDC}")
        
        if not self._ensure_system_ready():
            return
        
        if self.core.install_feature_dependencies('studio'):
            print(f"\n{Colors.OKGREEN}✅ Launching project management...{Colors.ENDC}")
            self.core.launch_studio_component('cli')
        
        input("\nPress Enter to continue...")
    
    def launch_full_studio(self):
        """Launch the complete DMS."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}🚀 LAUNCHING FULL STUDIO{Colors.ENDC}")
        
        # Check if all major components are ready
        required_features = ['system', 'studio']
        missing_features = []
        
        for feature in required_features:
            if self.core.feature_status.get(feature) != FeatureStatus.INSTALLED:
                missing_features.append(feature)
        
        if missing_features:
            print(f"{Colors.WARNING}⚠️  Some components need setup first:{Colors.ENDC}")
            for feature in missing_features:
                print(f"   - {feature}: {self.core.get_feature_status_display(feature)}")
            
            setup = input(f"\n{Colors.OKBLUE}Would you like to set up missing components now? (y/n): {Colors.ENDC}").lower()
            if setup == 'y':
                for feature in missing_features:
                    if feature == 'system':
                        self.setup_system()
                    elif feature == 'studio':
                        self.core.install_feature_dependencies('studio')
        
        # Launch the studio
        if self.core.install_feature_dependencies('studio'):
            print(f"\n{Colors.OKGREEN}✅ Launching DMS...{Colors.ENDC}")
            self.core.launch_studio_component('studio')
        
        input("\nPress Enter to continue...")
    
    def show_help(self):
        """Show help and documentation."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}❓ HELP & DOCUMENTATION{Colors.ENDC}")
        
        help_text = f"""
{Colors.OKBLUE}🎯 Getting Started:{Colors.ENDC}
   1. Run "System Setup" first (option 1)
   2. Let the system detect your hardware (option 2)
   3. Set up the features you need (options 3-5)
   4. Launch the full studio (option 7)

{Colors.OKBLUE}📋 Feature Overview:{Colors.ENDC}
   • Screen Capture: Automated image collection from windows
   • Annotation Interface: Label and manage your datasets
   • Model Training: Train YOLO models with your data
   • Project Management: Organize and manage multiple projects

{Colors.OKBLUE}🔧 Hardware Support:{Colors.ENDC}
   • NVIDIA GPUs: Automatic CUDA detection and setup
   • AMD GPUs: DirectML support for Windows/Linux
   • CPU Fallback: Works on any system without GPU

{Colors.OKBLUE}📁 Project Structure:{Colors.ENDC}
   • data/projects/: Your annotation projects
   • data/models/: Trained models
   • config/: Configuration files
   • logs/: System logs

{Colors.OKBLUE}🆘 Troubleshooting:{Colors.ENDC}
   • If installation fails, try running as administrator
   • Check that Python 3.8+ is installed
   • Ensure internet connection for package downloads
   • Check logs/ directory for detailed error information

{Colors.OKBLUE}📚 Documentation:{Colors.ENDC}
   • README.md: Complete project overview
   • docs/: Detailed documentation
   • GitHub Issues: Report problems and get help
        """
        
        print(help_text)
        input("\nPress Enter to continue...")
    
    def _ensure_system_ready(self) -> bool:
        """Ensure system is set up before proceeding with features."""
        if self.core.feature_status.get('system') != FeatureStatus.INSTALLED:
            print(f"{Colors.WARNING}⚠️  System setup required first.{Colors.ENDC}")
            setup = input(f"{Colors.OKBLUE}Would you like to run system setup now? (y/n): {Colors.ENDC}").lower()
            if setup == 'y':
                self.setup_system()
                return self.core.feature_status.get('system') == FeatureStatus.INSTALLED
            else:
                input("\nPress Enter to continue...")
                return False
        return True


def main():
    """Main launcher entry point."""
    try:
        # Initialize launcher
        core = LauncherCore()
        ui = LauncherUI(core)
        
        # Main loop
        while True:
            ui.show_main_menu()
            
            try:
                choice = input(f"\n{Colors.OKBLUE}Select an option (1-9): {Colors.ENDC}")
                
                if not ui.handle_menu_choice(choice):
                    break
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.WARNING}👋 Goodbye!{Colors.ENDC}")
                break
            except Exception as e:
                print(f"\n{Colors.FAIL}❌ An error occurred: {e}{Colors.ENDC}")
                input("Press Enter to continue...")
    
    except Exception as e:
        print(f"{Colors.FAIL}❌ Fatal error: {e}{Colors.ENDC}")
        sys.exit(1)
    
    print(f"\n{Colors.OKGREEN}👋 Thank you for using DMS!{Colors.ENDC}")


if __name__ == "__main__":
    main() 