"""
Hardware Detection and Management Module

Provides automatic detection and configuration for different hardware backends:
- NVIDIA GPUs with CUDA support
- AMD GPUs with DirectML support
- CPU fallback for systems without GPU acceleration

Supports automatic backend selection and optimal configuration.
"""

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import psutil

# Lazy import torch to avoid dependency issues
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import wmi

    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False


class DeviceType(Enum):
    """Enumeration of supported device types."""

    CUDA = "cuda"
    DIRECTML = "directml"
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """GPU information container."""

    name: str
    memory_total: int  # MB
    memory_free: int  # MB
    driver_version: str
    device_type: DeviceType
    device_id: int = 0
    compute_capability: Optional[Tuple[int, int]] = None
    directml_supported: bool = False


@dataclass
class SystemInfo:
    """System information container."""

    os_name: str
    os_version: str
    cpu_count: int
    cpu_name: str
    memory_total: int  # GB
    memory_available: int  # GB
    python_version: str
    pytorch_version: str


class HardwareDetector:
    """
    Hardware detection and management class.

    Automatically detects available hardware and configures optimal settings
    for YOLO training and inference.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._system_info: Optional[SystemInfo] = None
        self._gpu_info: List[GPUInfo] = []
        self._device_type: DeviceType = DeviceType.UNKNOWN
        self._selected_device: Optional[str] = None

        # Initialize detection
        self._detect_system()
        self._detect_gpus()
        self._select_optimal_device()

    def _detect_system(self) -> None:
        """Detect system information."""
        try:
            # Get system info
            memory_info = psutil.virtual_memory()

            self._system_info = SystemInfo(
                os_name=platform.system(),
                os_version=platform.version(),
                cpu_count=psutil.cpu_count(logical=True),
                cpu_name=platform.processor(),
                memory_total=round(memory_info.total / (1024**3)),  # GB
                memory_available=round(memory_info.available / (1024**3)),  # GB
                python_version=sys.version.split()[0],
                pytorch_version=(
                    torch.__version__ if TORCH_AVAILABLE else "Not installed"
                ),
            )

            self.logger.info(
                f"System detected: {self._system_info.os_name} {self._system_info.os_version}"
            )
            self.logger.info(
                f"CPU: {self._system_info.cpu_name} ({self._system_info.cpu_count} cores)"
            )
            self.logger.info(
                f"Memory: {self._system_info.memory_available}GB available / {self._system_info.memory_total}GB total"
            )

        except Exception as e:
            self.logger.error(f"System detection failed: {e}")
            # Create minimal system info
            self._system_info = SystemInfo(
                os_name="Unknown",
                os_version="Unknown",
                cpu_count=1,
                cpu_name="Unknown",
                memory_total=8,
                memory_available=4,
                python_version=sys.version.split()[0],
                pytorch_version=(
                    torch.__version__ if TORCH_AVAILABLE else "Not installed"
                ),
            )

    def _detect_cuda_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA CUDA GPUs."""
        cuda_gpus = []

        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self.logger.info("CUDA not available")
            return cuda_gpus

        try:
            cuda_device_count = torch.cuda.device_count()
            self.logger.info(f"Found {cuda_device_count} CUDA device(s)")

            for i in range(cuda_device_count):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory // (1024**2)  # MB
                memory_free = memory_total - (
                    torch.cuda.memory_allocated(i) // (1024**2)
                )

                gpu_info = GPUInfo(
                    name=props.name,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    driver_version=torch.version.cuda or "Unknown",
                    device_type=DeviceType.CUDA,
                    device_id=i,
                    compute_capability=(props.major, props.minor),
                )

                cuda_gpus.append(gpu_info)
                self.logger.info(f"CUDA GPU {i}: {gpu_info.name} ({memory_total}MB)")

        except Exception as e:
            self.logger.error(f"CUDA GPU detection failed: {e}")

        return cuda_gpus

    def _detect_directml_gpus(self) -> List[GPUInfo]:
        """Detect AMD/Intel GPUs with DirectML support."""
        directml_gpus = []

        # Check if DirectML is available
        try:
            import torch_directml

            self.logger.info("torch-directml successfully imported")

            # Test DirectML device creation
            try:
                device = torch_directml.device()
                self.logger.info(f"DirectML device created: {device}")
            except Exception as e:
                self.logger.error(f"DirectML device creation failed: {e}")
                return directml_gpus

        except ImportError as e:
            self.logger.info(f"torch-directml not installed: {e}")
            return directml_gpus

        try:
            # Try to get DirectML device count
            device_count = torch_directml.device_count()
            self.logger.info(f"Found {device_count} DirectML device(s)")

            for i in range(device_count):
                # Get device name (this might require additional methods)
                device_name = f"DirectML Device {i}"

                # Try to get more detailed info on Windows
                if WMI_AVAILABLE and platform.system() == "Windows":
                    try:
                        c = wmi.WMI()
                        for gpu in c.Win32_VideoController():
                            if gpu.Name and "AMD" in gpu.Name or "Radeon" in gpu.Name:
                                device_name = gpu.Name
                                break
                    except Exception as e:
                        self.logger.debug(f"Error querying GPU info via WMI: {e}")

                # Estimate memory (DirectML doesn't provide direct access)
                estimated_memory = 4096  # Default 4GB estimate

                gpu_info = GPUInfo(
                    name=device_name,
                    memory_total=estimated_memory,
                    memory_free=estimated_memory,  # Assume free for now
                    driver_version="DirectML",
                    device_type=DeviceType.DIRECTML,
                    device_id=i,
                    directml_supported=True,
                )

                directml_gpus.append(gpu_info)
                self.logger.info(f"DirectML GPU {i}: {gpu_info.name}")

        except Exception as e:
            self.logger.error(f"DirectML GPU detection failed: {e}")

        return directml_gpus

    def _detect_gpus(self) -> None:
        """Detect all available GPUs."""
        self._gpu_info = []

        # Detect CUDA GPUs first (preferred)
        cuda_gpus = self._detect_cuda_gpus()
        self._gpu_info.extend(cuda_gpus)

        # If no CUDA GPUs, try DirectML
        if not cuda_gpus:
            directml_gpus = self._detect_directml_gpus()
            self._gpu_info.extend(directml_gpus)

        if not self._gpu_info:
            self.logger.info("No GPU acceleration available, using CPU")
        else:
            self.logger.info(f"Total GPUs detected: {len(self._gpu_info)}")

    def _select_optimal_device(self) -> None:
        """Select the optimal device for training and inference."""
        if not self._gpu_info:
            self._device_type = DeviceType.CPU
            self._selected_device = "cpu"
            self.logger.info("Selected device: CPU")
            return

        # Prefer CUDA over DirectML
        cuda_gpus = [
            gpu for gpu in self._gpu_info if gpu.device_type == DeviceType.CUDA
        ]
        if cuda_gpus:
            # Select GPU with most memory
            best_gpu = max(cuda_gpus, key=lambda x: x.memory_free)
            self._device_type = DeviceType.CUDA
            self._selected_device = f"cuda:{best_gpu.device_id}"
            self.logger.info(
                f"Selected device: {self._selected_device} ({best_gpu.name})"
            )
            return

        # Use DirectML if available
        directml_gpus = [
            gpu for gpu in self._gpu_info if gpu.device_type == DeviceType.DIRECTML
        ]
        if directml_gpus:
            best_gpu = directml_gpus[0]  # Use first DirectML device
            self._device_type = DeviceType.DIRECTML
            self._selected_device = f"directml:{best_gpu.device_id}"
            self.logger.info(
                f"Selected device: {self._selected_device} ({best_gpu.name})"
            )
            return

        # Fallback to CPU
        self._device_type = DeviceType.CPU
        self._selected_device = "cpu"
        self.logger.info("Selected device: CPU (fallback)")

    def get_device_type(self) -> DeviceType:
        """Get the selected device type."""
        return self._device_type

    def get_device(self) -> str:
        """Get the selected device string."""
        return self._selected_device or "cpu"

    def get_torch_device(self):
        """Get PyTorch device object."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, cannot create device")
            return None

        if self._device_type == DeviceType.CUDA:
            return torch.device(self._selected_device)
        elif self._device_type == DeviceType.DIRECTML:
            try:
                import torch_directml

                return torch_directml.device()
            except ImportError:
                self.logger.warning("torch-directml not available, falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    def get_optimal_batch_size(
        self, image_size: int = 640, model_size: str = "n"
    ) -> int:
        """Calculate optimal batch size based on available hardware."""
        if self._device_type == DeviceType.CPU:
            return 4  # Conservative for CPU

        if not self._gpu_info:
            return 4

        # Get GPU with most memory
        best_gpu = max(self._gpu_info, key=lambda x: x.memory_free)
        memory_gb = best_gpu.memory_free / 1024  # Convert to GB

        # Rough estimates based on model size and image resolution
        size_multiplier = {
            "n": 1.0,  # nano
            "s": 1.5,  # small
            "m": 2.5,  # medium
            "l": 4.0,  # large
            "x": 6.0,  # extra large
        }.get(model_size.lower(), 1.0)

        # Base batch size calculation
        if image_size <= 320:
            base_batch = 64
        elif image_size <= 640:
            base_batch = 32
        elif image_size <= 1024:
            base_batch = 16
        else:
            base_batch = 8

        # Adjust for model size and available memory
        adjusted_batch = int(base_batch * (memory_gb / 8.0) / size_multiplier)

        # Ensure minimum and maximum bounds
        return max(1, min(adjusted_batch, 128))

    def get_system_info(self) -> SystemInfo:
        """Get system information."""
        return self._system_info

    def get_gpu_info(self) -> List[GPUInfo]:
        """Get GPU information list."""
        return self._gpu_info.copy()

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        return {
            "device_type": self._device_type.value,
            "selected_device": self._selected_device,
            "system_info": {
                "os": f"{self._system_info.os_name} {self._system_info.os_version}",
                "cpu": f"{self._system_info.cpu_name} ({self._system_info.cpu_count} cores)",
                "memory": f"{self._system_info.memory_available}GB / {self._system_info.memory_total}GB",
                "python": self._system_info.python_version,
                "pytorch": self._system_info.pytorch_version,
            },
            "gpu_info": [
                {
                    "name": gpu.name,
                    "memory": f"{gpu.memory_free}MB / {gpu.memory_total}MB",
                    "type": gpu.device_type.value,
                    "device_id": gpu.device_id,
                }
                for gpu in self._gpu_info
            ],
        }

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._device_type == DeviceType.CUDA

    def is_directml_available(self) -> bool:
        """Check if DirectML is available."""
        return self._device_type == DeviceType.DIRECTML

    def configure_torch_settings(self) -> None:
        """Configure PyTorch settings for optimal performance."""
        if not TORCH_AVAILABLE:
            self.logger.warning(
                "PyTorch not available, skipping optimization configuration"
            )
            return

        try:
            if self._device_type == DeviceType.CUDA:
                # CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                if self._system_info.memory_total >= 16:  # 16GB+ RAM
                    torch.backends.cudnn.enabled = True

                self.logger.info("CUDA optimizations applied")

            elif self._device_type == DeviceType.DIRECTML:
                # DirectML optimizations
                try:
                    import torch_directml

                    torch_directml.set_default_device()
                    self.logger.info("DirectML optimizations applied")
                except ImportError:
                    self.logger.warning("torch-directml not available for optimization")

            else:
                # CPU optimizations
                torch.set_num_threads(self._system_info.cpu_count)
                if hasattr(torch.backends, "mkl") and torch.backends.mkl.is_available():
                    torch.backends.mkl.enabled = True

                self.logger.info(
                    f"CPU optimizations applied ({self._system_info.cpu_count} threads)"
                )

        except Exception as e:
            self.logger.error(f"Failed to configure PyTorch settings: {e}")


# Global hardware detector instance
_hardware_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """Get global hardware detector instance."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector
