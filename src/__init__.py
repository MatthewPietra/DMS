"""
DMS

A comprehensive computer vision toolkit for object detection, annotation, and training
with multi-YOLO architecture support and cross-platform GPU optimization.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

# Version information
__version__ = "1.0.0"
__author__ = "DMS Team"
__license__ = "MIT"

# Package metadata
__all__ = [
    "DMS",
    "DMSConfig", 
    "YOLOTrainer",
    "AutoAnnotator",
    "AnnotationInterface",
    "WindowCapture",
    "HardwareDetector",
    "Logger",
    "__version__"
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def _apply_startup_fixes():
    """Apply critical startup fixes"""
    try:
        # Import and apply bug fixes
        from .utils.bug_fixes import apply_all_bug_fixes
        apply_all_bug_fixes()
        
        # Apply system optimizations
        from .utils.system_optimizer import optimize_system_for_production
        optimize_system_for_production()
        
    except ImportError as e:
        logger.warning(f"Could not apply startup optimizations: {e}")
    except Exception as e:
        logger.error(f"Error during startup optimization: {e}")

def _validate_environment():
    """Validate runtime environment"""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        # Check critical dependencies
        critical_deps = ['torch', 'torchvision', 'ultralytics', 'opencv-python', 'numpy']
        missing_deps = []
        
        for dep in critical_deps:
            try:
                __import__(dep.replace('-', '_'))
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"Missing critical dependencies: {', '.join(missing_deps)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

# Apply startup fixes and validation
try:
    if _validate_environment():
        _apply_startup_fixes()
        logger.info("DMS initialized successfully")
    else:
        logger.warning("Environment validation failed - some features may not work")
except Exception as e:
    logger.error(f"Initialization error: {e}")

# Safe imports with error handling
try:
    from .studio import DMS
except ImportError as e:
    logger.error(f"Failed to import DMS: {e}")
    DMS = None

try:
    from .config import DMSConfig
except ImportError as e:
    logger.warning(f"Failed to import DMSConfig: {e}")
    DMSConfig = None

try:
    from .training.yolo_trainer import YOLOTrainer
except ImportError as e:
    logger.warning(f"Failed to import YOLOTrainer: {e}")
    YOLOTrainer = None

try:
    from .auto_annotation.auto_annotator import AutoAnnotator
except ImportError as e:
    logger.warning(f"Failed to import AutoAnnotator: {e}")
    AutoAnnotator = None

try:
    from .annotation.annotation_interface import AnnotationInterface
except ImportError as e:
    logger.warning(f"Failed to import AnnotationInterface: {e}")
    AnnotationInterface = None

try:
    from .capture.window_capture import WindowCapture
except ImportError as e:
    logger.warning(f"Failed to import WindowCapture: {e}")
    WindowCapture = None

try:
    from .utils.hardware import HardwareDetector
except ImportError as e:
    logger.warning(f"Failed to import HardwareDetector: {e}")
    HardwareDetector = None

try:
    from .utils.logger import Logger
except ImportError as e:
    logger.warning(f"Failed to import Logger: {e}")
    Logger = None

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

def get_version():
    """Get package version"""
    return __version__

def check_installation():
    """Check installation completeness"""
    
    print(f"\n🔍 DMS v{__version__} - Installation Check")
    print("=" * 60)
    
    # Check core components
    components = {
        'Studio': DMS,
        'Config': DMSConfig,
        'Trainer': YOLOTrainer,
        'Auto Annotator': AutoAnnotator,
        'Annotation Interface': AnnotationInterface,
        'Window Capture': WindowCapture,
        'Hardware Detector': HardwareDetector,
        'Logger': Logger
    }
    
    available_count = 0
    for name, component in components.items():
        status = "✅" if component is not None else "❌"
        print(f"  {status} {name}")
        if component is not None:
            available_count += 1
    
    # Check dependencies
    print(f"\n📦 Dependencies:")
    deps = {
        'PyTorch': 'torch',
        'Torchvision': 'torchvision', 
        'Ultralytics': 'ultralytics',
        'OpenCV': 'cv2',
        'NumPy': 'numpy',
        'Pillow': 'PIL',
        'PyYAML': 'yaml',
        'tqdm': 'tqdm',
        'psutil': 'psutil'
    }
    
    available_deps = 0
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
            available_deps += 1
        except ImportError:
            print(f"  ❌ {name}")
    
    # Overall status
    component_score = available_count / len(components) * 100
    dependency_score = available_deps / len(deps) * 100
    overall_score = (component_score + dependency_score) / 2
    
    print(f"\n📊 Installation Status:")
    print(f"  Components: {available_count}/{len(components)} ({component_score:.1f}%)")
    print(f"  Dependencies: {available_deps}/{len(deps)} ({dependency_score:.1f}%)")
    print(f"  Overall: {overall_score:.1f}%")
    
    if overall_score >= 90:
        print(f"  🎉 Excellent! Installation is complete.")
    elif overall_score >= 70:
        print(f"  ⚠️  Good installation with some missing components.")
    else:
        print(f"  ❌ Installation needs attention.")
    
    return overall_score

def run_production_validation():
    """Run production readiness validation"""
    try:
        from .utils.production_validator import validate_production_readiness
        return validate_production_readiness()
    except ImportError:
        logger.error("Production validator not available")
        return None

# Package information
def print_info():
    """Print package information"""
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                         DMS                                  ║
║                         v{__version__}                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🎯 Multi-YOLO Architecture Support                         ║
║     • YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11             ║
║                                                              ║
║  🖥️  Cross-Platform GPU Optimization                        ║
║     • NVIDIA CUDA, AMD DirectML, CPU                        ║
║                                                              ║
║  🏷️  Professional Annotation Tools                          ║
║     • Manual annotation interface                           ║
║     • Intelligent auto-annotation                           ║
║     • Multi-format export (COCO, YOLO, Pascal VOC)         ║
║                                                              ║
║  📸 Advanced Screen Capture                                  ║
║     • Multi-monitor support                                 ║
║     • Window targeting                                       ║
║     • Configurable frame rates                              ║
║                                                              ║
║  🚀 Production-Ready Features                                ║
║     • Comprehensive testing suite                           ║
║     • Performance optimization                               ║
║     • Error handling and recovery                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    print_info()
    check_installation() 