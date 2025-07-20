"""Bug Fixes and Workarounds for DMS.

This module contains common bug fixes and workarounds for known issues
in the DMS ecosystem, including PyTorch, OpenCV, and
platform-specific problems.
"""

import logging
import os
import platform
import shutil
import sys
import warnings
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Known bug fixes registry
BUG_FIXES: Dict[str, bool] = {
    "torch_cuda_memory": True,
    "opencv_threading": True,
    "numpy_deprecation": True,
    "windows_path_issues": platform.system() == "Windows",
    "linux_permissions": platform.system() == "Linux",
    "macos_metal": platform.system() == "Darwin",
}


def apply_all_bug_fixes() -> List[str]:
    """Apply all known bug fixes and workarounds.

    This function applies various fixes for common issues:
    - PyTorch CUDA memory management
    - OpenCV threading issues
    - NumPy deprecation warnings
    - Platform-specific path issues
    - Permission and compatibility problems

    Returns:
        List[str]: Names of successfully applied bug fixes.
    """
    logger.info("Applying bug fixes and workarounds...")

    fixes_applied: List[str] = []

    # Apply PyTorch CUDA memory fixes
    if BUG_FIXES["torch_cuda_memory"]:
        if apply_torch_cuda_fixes():
            fixes_applied.append("torch_cuda_memory")

    # Apply OpenCV threading fixes
    if BUG_FIXES["opencv_threading"]:
        if apply_opencv_fixes():
            fixes_applied.append("opencv_threading")

    # Apply NumPy deprecation fixes
    if BUG_FIXES["numpy_deprecation"]:
        if apply_numpy_fixes():
            fixes_applied.append("numpy_deprecation")

    # Apply platform-specific fixes
    if BUG_FIXES["windows_path_issues"]:
        if apply_windows_fixes():
            fixes_applied.append("windows_path_issues")

    if BUG_FIXES["linux_permissions"]:
        if apply_linux_fixes():
            fixes_applied.append("linux_permissions")

    if BUG_FIXES["macos_metal"]:
        if apply_macos_fixes():
            fixes_applied.append("macos_metal")

    logger.info(
        f"Applied {len(fixes_applied)} bug fixes: {', '.join(fixes_applied)}"
    )
    return fixes_applied


def apply_torch_cuda_fixes() -> bool:
    """Apply PyTorch CUDA-related bug fixes.

    Returns:
        bool: True if fixes were applied successfully, False otherwise.
    """
    try:
        # Fix CUDA memory fragmentation
        if torch.cuda.is_available():
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)

            # Enable memory efficient attention if available
            if hasattr(torch.backends, "cuda") and hasattr(
                torch.backends.cuda, "enable_flash_sdp"
            ):
                torch.backends.cuda.enable_flash_sdp(True)

            # Set deterministic algorithms for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            logger.debug("Applied PyTorch CUDA fixes")
            return True

    except ImportError:
        logger.debug("PyTorch not available, skipping CUDA fixes")
    except Exception as exc:
        logger.warning(f"Failed to apply PyTorch CUDA fixes: {exc}")

    return False


def apply_opencv_fixes() -> bool:
    """Apply OpenCV-related bug fixes.

    Returns:
        bool: True if fixes were applied successfully, False otherwise.
    """
    try:
        # Fix threading issues
        cv2.setNumThreads(0)  # Use all available threads

        # Set OpenCV to use optimized algorithms
        cv2.setUseOptimized(True)

        # Fix potential memory leaks
        if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)  # Disable OpenCL to prevent issues

        logger.debug("Applied OpenCV fixes")
        return True

    except ImportError:
        logger.debug("OpenCV not available, skipping fixes")
    except Exception as exc:
        logger.warning(f"Failed to apply OpenCV fixes: {exc}")

    return False


def apply_numpy_fixes() -> bool:
    """Apply NumPy-related bug fixes and warnings suppression.

    Returns:
        bool: True if fixes were applied successfully, False otherwise.
    """
    try:
        # Suppress deprecation warnings
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="numpy"
        )
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module="numpy"
        )

        # Set NumPy to use optimized BLAS if available
        if hasattr(np, "set_printoptions"):
            np.set_printoptions(precision=4, suppress=True)

        logger.debug("Applied NumPy fixes")
        return True

    except ImportError:
        logger.debug("NumPy not available, skipping fixes")
    except Exception as exc:
        logger.warning(f"Failed to apply NumPy fixes: {exc}")

    return False


def apply_windows_fixes() -> bool:
    """Apply Windows-specific bug fixes.

    Returns:
        bool: True if fixes were applied successfully, False otherwise.
    """
    try:
        # Fix path length issues
        if hasattr(os, "path") and hasattr(os.path, "abspath"):
            # Ensure long path support
            os.environ["PYTHONIOENCODING"] = "utf-8"

        # Fix multiprocessing issues on Windows
        if hasattr(os, "environ"):
            os.environ["PYTHONHASHSEED"] = "0"

        # Fix potential DLL loading issues
        if "PATH" in os.environ:
            # Add current directory to PATH for DLL loading
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in os.environ["PATH"]:
                os.environ["PATH"] = current_dir + os.pathsep + os.environ["PATH"]

        logger.debug("Applied Windows-specific fixes")
        return True

    except Exception as exc:
        logger.warning(f"Failed to apply Windows fixes: {exc}")
        return False


def apply_linux_fixes() -> bool:
    """Apply Linux-specific bug fixes.

    Returns:
        bool: True if fixes were applied successfully, False otherwise.
    """
    try:
        # Fix permission issues
        if hasattr(os, "umask"):
            os.umask(0o022)  # Set default file permissions

        # Fix potential shared memory issues
        if hasattr(os, "environ"):
            os.environ["OMP_NUM_THREADS"] = "1"  # Prevent OpenMP conflicts

        # Fix potential display issues
        if "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = ":0"

        logger.debug("Applied Linux-specific fixes")
        return True

    except Exception as exc:
        logger.warning(f"Failed to apply Linux fixes: {exc}")
        return False


def apply_macos_fixes() -> bool:
    """Apply macOS-specific bug fixes.

    Returns:
        bool: True if fixes were applied successfully, False otherwise.
    """
    try:
        # Fix Metal performance issues
        if hasattr(os, "environ"):
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Fix potential display scaling issues
        if hasattr(os, "environ") and "QT_MAC_WANTS_LAYER" not in os.environ:
            os.environ["QT_MAC_WANTS_LAYER"] = "1"

        logger.debug("Applied macOS-specific fixes")
        return True

    except Exception as exc:
        logger.warning(f"Failed to apply macOS fixes: {exc}")
        return False


def get_bug_fix_status() -> Dict[str, bool]:
    """Get the status of all bug fixes.

    Returns:
        Dict[str, bool]: Dictionary mapping bug fix names to their enabled status.
    """
    return BUG_FIXES.copy()


def enable_bug_fix(fix_name: str, enable: bool = True) -> None:
    """Enable or disable a specific bug fix.

    Args:
        fix_name: Name of the bug fix to enable/disable.
        enable: Whether to enable (True) or disable (False) the bug fix.
    """
    if fix_name in BUG_FIXES:
        BUG_FIXES[fix_name] = enable
        status = "Enabled" if enable else "Disabled"
        logger.info(f"{status} bug fix: {fix_name}")
    else:
        logger.warning(f"Unknown bug fix: {fix_name}")


def list_available_fixes() -> List[str]:
    """List all available bug fixes.

    Returns:
        List[str]: List of all available bug fix names.
    """
    return list(BUG_FIXES.keys())


def check_system_compatibility() -> Dict[str, Any]:
    """Check system compatibility and suggest fixes.

    Returns:
        Dict[str, Any]: Compatibility report with platform info, issues, and
            recommendations.
    """
    compatibility_report: Dict[str, Any] = {
        "platform": platform.system(),
        "python_version": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "issues": [],
        "recommendations": [],
    }

    # Check Python version
    if sys.version_info < (3, 8):
        compatibility_report["issues"].append("Python version < 3.8")
        compatibility_report["recommendations"].append(
            "Upgrade to Python 3.8 or higher"
        )

    # Check for common issues
    try:
        if torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3:  # 4GB
                compatibility_report["issues"].append("Low GPU memory")
                compatibility_report["recommendations"].append(
                    "Consider using smaller batch sizes"
                )
    except ImportError:
        compatibility_report["issues"].append("PyTorch not available")
        compatibility_report["recommendations"].append(
            "Install PyTorch for GPU acceleration"
        )

    # Check disk space
    try:
        total, used, free = shutil.disk_usage(".")
        if free < 10 * 1024**3:  # 10GB
            compatibility_report["issues"].append("Low disk space")
            compatibility_report["recommendations"].append(
                "Free up disk space for training data"
            )
    except Exception as exc:
        logger.debug(f"Error checking disk space: {exc}")

    return compatibility_report
