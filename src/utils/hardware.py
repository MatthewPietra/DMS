"""Hardware Detection and Management Module.

Provides automatic detection and configuration for different hardware backends:
- NVIDIA GPUs with CUDA support
- AMD GPUs with DirectML support
- CPU fallback for systems without GPU acceleration

Supports automatic backend selection and optimal configuration.
"""

import logging
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any, List, Optional, Tuple, cast

import psutil

# Lazy imports to avoid dependency issues
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = cast(ModuleType, None)

try:
    import GPUtil  # type: ignore

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = cast(ModuleType, None)

try:
    import wmi

    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False
    wmi = cast(ModuleType, None)

try:
    import torch_directml  # type: ignore

    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False
    torch_directml = cast(ModuleType, None)


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
    memory_available: int  # MB
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
    """Hardware detection and management class.

    Automatically detects available hardware and configures optimal settings
    for YOLO training and inference.
    """

    def __init__(self) -> None:
        """Initialize the hardware detector."""
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
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count is None:
                cpu_count = 1

            self._system_info = SystemInfo(
                os_name=platform.system(),
                os_version=platform.version(),
                cpu_count=cpu_count,
                cpu_name=platform.processor(),
                memory_total=round(memory_info.total / (1024**3)),  # GB
                memory_available=round(memory_info.available / (1024**3)),  # GB
                python_version=sys.version.split()[0],
                pytorch_version=(
                    torch.__version__ if TORCH_AVAILABLE and torch else "Not installed"
                ),
            )

            self.logger.info(
                f"System detected: {self._system_info.os_name} "
                f"{self._system_info.os_version}"
            )
            self.logger.info(
                f"CPU: {self._system_info.cpu_name} "
                f"({self._system_info.cpu_count} cores)"
            )
            self.logger.info(
                f"Memory: {self._system_info.memory_available}GB available / "
                f"{self._system_info.memory_total}GB total"
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
                    torch.__version__ if TORCH_AVAILABLE and torch else "Not installed"
                ),
            )

    def _detect_cuda_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA CUDA GPUs."""
        cuda_gpus: List[GPUInfo] = []

        if not TORCH_AVAILABLE or not torch or not torch.cuda.is_available():
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
                    memory_available=memory_free,
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
        directml_gpus: List[GPUInfo] = []

        # First, try to detect AMD GPUs via WMI even without DirectML
        if WMI_AVAILABLE and platform.system() == "Windows" and wmi:
            try:
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    if gpu.Name and ("AMD" in gpu.Name or "Radeon" in gpu.Name):
                        # Get memory in MB, handle invalid values
                        memory_mb = 0
                        if hasattr(gpu, "AdapterRAM") and gpu.AdapterRAM:
                            try:
                                memory_mb = gpu.AdapterRAM // (1024**2)
                                if memory_mb <= 0:  # Invalid memory value
                                    memory_mb = 8192  # Default 8GB for RX 5700 XT
                            except (ValueError, TypeError):
                                memory_mb = 8192  # Default 8GB for RX 5700 XT
                        else:
                            memory_mb = 8192  # Default 8GB for RX 5700 XT

                        gpu_info = GPUInfo(
                            name=gpu.Name,
                            memory_total=memory_mb,
                            memory_free=memory_mb,  # Assume free for now
                            memory_available=memory_mb,
                            driver_version="AMD Driver",
                            device_type=DeviceType.DIRECTML,
                            device_id=len(directml_gpus),
                            directml_supported=False,  # Will be True if DirectML is installed
                        )

                        directml_gpus.append(gpu_info)
                        self.logger.info(
                            f"AMD GPU detected: {gpu_info.name} ({memory_mb}MB)"
                        )
            except Exception as e:
                self.logger.debug(f"Error querying GPU info via WMI: {e}")

        # Check if DirectML is available
        try:
            self.logger.info("torch-directml successfully imported")

            # Test DirectML device creation
            try:
                if torch_directml:
                    device = torch_directml.device()
                    self.logger.info(f"DirectML device created: {device}")

                    # Update existing GPUs to show DirectML support
                    for gpu in directml_gpus:
                        gpu.directml_supported = True
                        gpu.driver_version = "DirectML"

            except Exception as e:
                self.logger.error(f"DirectML device creation failed: {e}")
                return directml_gpus

        except ImportError as e:
            self.logger.info(f"torch-directml not installed: {e}")
            return directml_gpus

        try:
            # Try to get DirectML device count
            if torch_directml:
                device_count = torch_directml.device_count()
                self.logger.info(f"Found {device_count} DirectML device(s)")

                # If we don't have any GPUs from WMI, create generic ones
                if not directml_gpus:
                    for i in range(device_count):
                        device_name = f"DirectML Device {i}"

                        gpu_info = GPUInfo(
                            name=device_name,
                            memory_total=4096,  # Default 4GB estimate
                            memory_free=4096,
                            memory_available=4096,
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

    def get_torch_device(self) -> Optional[Any]:
        """Get PyTorch device object."""
        if not TORCH_AVAILABLE or not torch:
            self.logger.warning("PyTorch not available, cannot create device")
            return None

        if self._device_type == DeviceType.CUDA:
            return torch.device(self._selected_device or "cpu")
        elif self._device_type == DeviceType.DIRECTML:
            try:
                if torch_directml:
                    return torch_directml.device()
                else:
                    self.logger.warning(
                        "torch-directml not available, falling back to CPU"
                    )
                    return torch.device("cpu")
            except ImportError:
                self.logger.warning("torch-directml not available, falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    def get_optimal_batch_size(
        self, image_size: int = 640, model_size: str = "n"
    ) -> int:
        """Calculate optimal batch size based on available hardware.

        Args:
            image_size: Input image size in pixels.
            model_size: Model size identifier (n, s, m, l, x).

        Returns:
            Optimal batch size for the given hardware configuration.
        """
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

    def get_system_info(self) -> Optional[SystemInfo]:
        """Get system information."""
        return self._system_info

    @property
    def system_info(self) -> Optional[SystemInfo]:
        """Get system information (property for compatibility)."""
        return self._system_info

    def get_gpu_info(self) -> List[GPUInfo]:
        """Get GPU information list."""
        return self._gpu_info.copy()

    def get_device_info(self) -> dict[str, Any]:
        """Get comprehensive device information."""
        if not self._system_info:
            return {
                "device_type": (
                    self._device_type.value
                    if isinstance(self._device_type, Enum)
                    else self._device_type
                ),
                "selected_device": self._selected_device,
                "system_info": {
                    "os": "Unknown",
                    "cpu": "Unknown",
                    "memory": "Unknown",
                    "python": "Unknown",
                    "pytorch": "Unknown",
                },
                "gpu_info": [
                    {
                        "name": gpu.name,
                        "memory": f"{gpu.memory_free}MB / {gpu.memory_total}MB",
                        "type": (
                            gpu.device_type.value
                            if isinstance(gpu.device_type, Enum)
                            else gpu.device_type
                        ),
                        "device_id": gpu.device_id,
                    }
                    for gpu in self._gpu_info
                ],
            }

        return {
            "device_type": (
                self._device_type.value
                if isinstance(self._device_type, Enum)
                else self._device_type
            ),
            "selected_device": self._selected_device,
            "system_info": {
                "os": f"{self._system_info.os_name} {self._system_info.os_version}",
                "cpu": f"{self._system_info.cpu_name} "
                f"({self._system_info.cpu_count} cores)",
                "memory": f"{self._system_info.memory_available}GB / "
                f"{self._system_info.memory_total}GB",
                "python": self._system_info.python_version,
                "pytorch": self._system_info.pytorch_version,
            },
            "gpu_info": [
                {
                    "name": gpu.name,
                    "memory": f"{gpu.memory_free}MB / {gpu.memory_total}MB",
                    "type": (
                        gpu.device_type.value
                        if isinstance(gpu.device_type, Enum)
                        else gpu.device_type
                    ),
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
        if not TORCH_AVAILABLE or not torch:
            self.logger.warning(
                "PyTorch not available, skipping optimization configuration"
            )
            return

        try:
            if self._device_type == DeviceType.CUDA:
                # CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                if self._system_info and self._system_info.memory_total >= 16:
                    # 16GB+ RAM
                    torch.backends.cudnn.enabled = True

                self.logger.info("CUDA optimizations applied")

            elif self._device_type == DeviceType.DIRECTML:
                # DirectML optimizations
                if torch_directml and hasattr(torch_directml, "set_default_device"):
                    torch_directml.set_default_device()
                self.logger.info("DirectML optimizations applied")

            else:
                # CPU optimizations
                if self._system_info:
                    torch.set_num_threads(self._system_info.cpu_count)
                mkl_available = (
                    hasattr(torch.backends, "mkl")
                    and torch.backends.mkl.is_available()  # type: ignore
                )
                if mkl_available:
                    # Check if enabled attribute exists
                    if hasattr(torch.backends.mkl, "enabled"):
                        torch.backends.mkl.enabled = True

                cpu_count = self._system_info.cpu_count if self._system_info else 1
                self.logger.info(f"CPU optimizations applied ({cpu_count} threads)")

        except Exception as e:
            self.logger.error(f"Failed to configure PyTorch settings: {e}")

    def detect_hardware(self) -> Any:
        """Detect and return hardware specifications."""
        return self.get_hardware_specs()

    def get_hardware_specs(self) -> Any:
        """Get hardware specifications as a structured object.

        Returns:
            HardwareSpecs object containing system information.
        """

        class HardwareSpecs:
            def __init__(self, detector: "HardwareDetector") -> None:
                self.cpu_count = (
                    detector._system_info.cpu_count if detector._system_info else 0
                )
                self.gpus = detector._gpu_info
                self.device_type = detector._device_type.value
                self.optimal_device = detector._selected_device
                self.memory_total = (
                    detector._system_info.memory_total if detector._system_info else 0
                )

        return HardwareSpecs(self)

    def get_optimal_device(self) -> str:
        """Get the optimal device for training/inference."""
        # Return just the device type, not the full device string
        if self._device_type == DeviceType.CUDA:
            return "cuda"
        elif self._device_type == DeviceType.DIRECTML:
            return "directml"
        else:
            return "cpu"

    def get_optimal_workers(self) -> int:
        """Get optimal number of workers for data loading."""
        if self._system_info:
            return min(self._system_info.cpu_count, 8)
        return 4

    def validate_device(self, device: str) -> bool:
        """Validate if a device is available."""
        valid_devices = ["cpu", "cuda", "directml"]
        return device.lower() in valid_devices

    def _estimate_batch_size_from_memory(self, memory_mb: int) -> int:
        """Estimate optimal batch size based on available memory."""
        # Simple estimation: 1GB per batch item for YOLO
        estimated_batch_size = max(1, memory_mb // 1024)
        return min(estimated_batch_size, 32)  # Cap at 32


# Global hardware detector instance
_hardware_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """Get global hardware detector instance."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector
