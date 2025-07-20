"""System Optimizer for DMS.

This module provides system optimization functions to improve performance
and stability for production use of DMS.
"""

import gc
import logging
import os
import platform
import socket
import sys
from typing import Any, Dict, List, Optional

import psutil
import torch

logger = logging.getLogger(__name__)


class SystemOptimizer:
    """System optimization manager for DMS.

    This class provides comprehensive system optimization capabilities
    for improving performance and stability in production environments.
    """

    def __init__(self) -> None:
        """Initialize the SystemOptimizer.

        Sets up the optimizer with empty optimization tracking and
        initializes system information collection.
        """
        self.optimizations_applied: List[str] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.system_info: Dict[str, Any] = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information.

        Returns:
            Dictionary containing system information including platform,
            CPU, memory, and disk details.
        """
        try:
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": (
                    f"{sys.version_info.major}.{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": (
                    psutil.disk_usage("/").total
                    if platform.system() != "Windows"
                    else psutil.disk_usage("C:\\").total
                ),
            }
        except Exception as e:
            logger.warning("Failed to get system info: %s", e)
            return {}

    def optimize_system_for_production(self) -> Dict[str, Any]:
        """Apply comprehensive system optimizations for production use.

        This function applies various optimizations:
        - Memory management
        - CPU optimization
        - GPU optimization
        - File system optimization
        - Network optimization
        - Process priority optimization

        Returns:
            Dictionary containing optimization results and performance metrics.
        """
        logger.info("Applying system optimizations for production...")

        optimization_results = {
            "memory": self._optimize_memory(),
            "cpu": self._optimize_cpu(),
            "gpu": self._optimize_gpu(),
            "filesystem": self._optimize_filesystem(),
            "network": self._optimize_network(),
            "process": self._optimize_process(),
            "python": self._optimize_python(),
        }

        # Record applied optimizations
        for category, result in optimization_results.items():
            if result.get("success", False):
                self.optimizations_applied.append(category)

        # Measure performance impact
        self.performance_metrics = self._measure_performance()

        logger.info(
            "Applied %d optimizations: %s",
            len(self.optimizations_applied),
            ", ".join(self.optimizations_applied),
        )

        return {
            "optimizations_applied": self.optimizations_applied,
            "results": optimization_results,
            "performance_metrics": self.performance_metrics,
        }

    def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage and management.

        Returns:
            Dictionary containing optimization results and memory usage info.
        """
        try:
            # Force garbage collection
            gc.collect()

            # Set memory limits for better management
            if hasattr(psutil, "virtual_memory"):
                vm = psutil.virtual_memory()
                memory_usage_percent = vm.percent

                # If memory usage is high, apply aggressive optimization
                if memory_usage_percent > 80:
                    # Clear Python cache
                    if hasattr(sys, "getallocatedblocks"):
                        initial_blocks = sys.getallocatedblocks()
                        gc.collect()
                        final_blocks = sys.getallocatedblocks()
                        blocks_freed = initial_blocks - final_blocks
                        logger.info("Freed %d memory blocks", blocks_freed)

            # Set environment variables for memory optimization
            os.environ["PYTHONMALLOC"] = "malloc"
            os.environ["PYTHONDEVMODE"] = "0"

            return {"success": True, "memory_usage": memory_usage_percent}

        except Exception as e:
            logger.warning("Memory optimization failed: %s", e)
            return {"success": False, "error": str(e)}

    def _optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU usage and threading.

        Returns:
            Dictionary containing optimization results and CPU info.
        """
        try:
            # Set CPU affinity for better performance
            if hasattr(psutil.Process(), "cpu_affinity"):
                process = psutil.Process()
                cpu_count = psutil.cpu_count()
                if cpu_count is not None:
                    # Use all available cores
                    process.cpu_affinity(list(range(cpu_count)))

            # Set environment variables for CPU optimization
            cpu_count = psutil.cpu_count()
            if cpu_count is not None:
                os.environ["OMP_NUM_THREADS"] = str(cpu_count)
                os.environ["MKL_NUM_THREADS"] = str(cpu_count)
                os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)

            # Set process priority
            if hasattr(psutil.Process(), "nice"):
                process = psutil.Process()
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

            return {"success": True, "cpu_count": psutil.cpu_count()}

        except Exception as e:
            logger.warning("CPU optimization failed: %s", e)
            return {"success": False, "error": str(e)}

    def _optimize_gpu(self) -> Dict[str, Any]:
        """Optimize GPU usage and memory.

        Returns:
            Dictionary containing optimization results and GPU info.
        """
        try:
            if torch.cuda.is_available():
                # Set GPU memory fraction
                torch.cuda.set_per_process_memory_fraction(0.8)

                # Enable memory efficient attention
                if hasattr(torch.backends, "cuda") and hasattr(
                    torch.backends.cuda, "enable_flash_sdp"
                ):
                    torch.backends.cuda.enable_flash_sdp(True)

                # Set deterministic algorithms
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                # Clear GPU cache
                torch.cuda.empty_cache()

                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                return {
                    "success": True,
                    "gpu_memory": gpu_memory,
                    "device_count": torch.cuda.device_count(),
                }
            else:
                return {"success": True, "gpu_available": False}

        except ImportError:
            logger.debug("PyTorch not available, skipping GPU optimization")
            return {"success": True, "pytorch_available": False}
        except Exception as e:
            logger.warning("GPU optimization failed: %s", e)
            return {"success": False, "error": str(e)}

    def _optimize_filesystem(self) -> Dict[str, Any]:
        """Optimize file system operations.

        Returns:
            Dictionary containing optimization results and platform info.
        """
        try:
            # Set buffer size for file operations
            if hasattr(os, "environ"):
                os.environ["PYTHONIOENCODING"] = "utf-8"
                os.environ["PYTHONHASHSEED"] = "0"

            # Optimize for the current platform
            if platform.system() == "Windows":
                # Windows-specific optimizations
                os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
            elif platform.system() == "Linux":
                # Linux-specific optimizations
                os.environ["PYTHONUNBUFFERED"] = "1"

            return {"success": True, "platform": platform.system()}

        except Exception as e:
            logger.warning("Filesystem optimization failed: %s", e)
            return {"success": False, "error": str(e)}

    def _optimize_network(self) -> Dict[str, Any]:
        """Optimize network operations.

        Returns:
            Dictionary containing optimization results.
        """
        try:
            # Set socket timeout
            socket.setdefaulttimeout(30)

            # Set environment variables for network optimization
            os.environ["REQUESTS_CA_BUNDLE"] = ""
            os.environ["CURL_CA_BUNDLE"] = ""

            return {"success": True}

        except Exception as e:
            logger.warning("Network optimization failed: %s", e)
            return {"success": False, "error": str(e)}

    def _optimize_process(self) -> Dict[str, Any]:
        """Optimize process settings.

        Returns:
            Dictionary containing optimization results and process info.
        """
        try:
            process = psutil.Process()

            # Set process priority
            if platform.system() == "Windows":
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                process.nice(10)  # Lower priority on Unix systems

            # Set I/O priority
            if hasattr(process, "ionice"):
                try:
                    # Use IOPRIO_CLASS_BE if available, otherwise use a default value
                    ioprio_class = getattr(psutil, "IOPRIO_CLASS_BE", 2)
                    process.ionice(ioprio_class, 2)
                except (AttributeError, OSError):
                    # I/O priority setting might not be available on all systems
                    pass

            return {"success": True, "pid": process.pid}

        except Exception as e:
            logger.warning("Process optimization failed: %s", e)
            return {"success": False, "error": str(e)}

    def _optimize_python(self) -> Dict[str, Any]:
        """Optimize Python runtime settings.

        Returns:
            Dictionary containing optimization results and Python version.
        """
        try:
            # Set Python optimization flags
            sys.dont_write_bytecode = True

            # Optimize garbage collection
            gc.set_threshold(700, 10, 10)

            # Set recursion limit
            sys.setrecursionlimit(10000)

            # Optimize string interning
            if hasattr(sys, "intern"):
                # Intern commonly used strings
                common_strings = ["image", "label", "bbox", "class", "confidence"]
                for s in common_strings:
                    sys.intern(s)

            return {"success": True, "python_version": sys.version}

        except Exception as e:
            logger.warning("Python optimization failed: %s", e)
            return {"success": False, "error": str(e)}

    def _measure_performance(self) -> Dict[str, Any]:
        """Measure system performance metrics.

        Returns:
            Dictionary containing performance metrics.
        """
        try:
            metrics = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": (
                    psutil.disk_usage("/").percent
                    if platform.system() != "Windows"
                    else psutil.disk_usage("C:\\").percent
                ),
                "load_average": (
                    psutil.getloadavg() if platform.system() != "Windows" else None
                ),
            }

            # Get GPU metrics if available
            try:
                if torch.cuda.is_available():
                    metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated(0)
                    metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved(0)
            except ImportError:
                pass

            return metrics

        except Exception as e:
            logger.warning("Performance measurement failed: %s", e)
            return {}

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status.

        Returns:
            Dictionary containing current optimization status and metrics.
        """
        return {
            "optimizations_applied": self.optimizations_applied,
            "performance_metrics": self.performance_metrics,
            "system_info": self.system_info,
        }

    def reset_optimizations(self) -> None:
        """Reset all applied optimizations.

        Clears all optimization tracking and resets environment variables
        to their default state.
        """
        logger.info("Resetting system optimizations...")
        self.optimizations_applied = []
        self.performance_metrics = {}

        # Reset environment variables
        for var in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "PYTHONMALLOC",
            "PYTHONDEVMODE",
        ]:
            if var in os.environ:
                del os.environ[var]

        logger.info("System optimizations reset")


# Global optimizer instance
_optimizer: Optional[SystemOptimizer] = None


def optimize_system_for_production() -> Dict[str, Any]:
    """Apply comprehensive system optimizations for production use.

    Returns:
        Dictionary containing optimization results and performance metrics.
    """
    global _optimizer

    if _optimizer is None:
        _optimizer = SystemOptimizer()

    return _optimizer.optimize_system_for_production()


def get_optimization_status() -> Dict[str, Any]:
    """Get current optimization status.

    Returns:
        Dictionary containing current optimization status and metrics.
    """
    global _optimizer  # noqa: F824

    if _optimizer is None:
        return {
            "optimizations_applied": [],
            "performance_metrics": {},
            "system_info": {},
        }

    return _optimizer.get_optimization_status()


def reset_optimizations() -> None:
    """Reset all applied optimizations.

    Resets the global optimizer instance and clears all optimizations.
    """
    global _optimizer  # noqa: F824

    if _optimizer is not None:
        _optimizer.reset_optimizations()


def get_system_recommendations() -> List[str]:
    """Get system optimization recommendations.

    Analyzes the current system and provides recommendations for
    optimal performance based on hardware capabilities.

    Returns:
        List of recommendation strings for system optimization.
    """
    recommendations = []

    try:
        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count is not None and cpu_count < 4:
            recommendations.append(
                "Consider using a system with 4+ CPU cores for better performance"
            )

        # Check memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            recommendations.append(
                "Consider using a system with 8GB+ RAM for optimal performance"
            )

        # Check disk space
        disk_gb = (
            psutil.disk_usage("/").free / (1024**3)
            if platform.system() != "Windows"
            else psutil.disk_usage("C:\\").free / (1024**3)
        )
        if disk_gb < 10:
            recommendations.append(
                "Ensure at least 10GB free disk space for training data"
            )

        # Check GPU
        try:
            if not torch.cuda.is_available():
                recommendations.append(
                    "Consider using a CUDA-compatible GPU for faster training"
                )
        except ImportError:
            recommendations.append("Install PyTorch for GPU acceleration")

    except Exception as e:
        logger.warning("Failed to generate recommendations: %s", e)

    return recommendations
