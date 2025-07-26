"""Secure Subprocess Utilities.

Provides safe subprocess execution with input validation and proper error handling.
This module centralizes all subprocess operations to ensure security best practices.
"""

import logging
import subprocess  # nosec
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SecureSubprocess:
    """Secure subprocess execution utilities."""

    @staticmethod
    def run_command(
        cmd: Union[str, List[str]],
        timeout: int = 300,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str, str]:
        """Execute a command securely.

        Args:
            cmd: Command to execute (string or list)
            timeout: Timeout in seconds
            cwd: Working directory
            env: Environment variables

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            # Validate command
            if isinstance(cmd, str):
                cmd = [cmd]

            if not cmd or not all(isinstance(arg, str) for arg in cmd):
                return False, "", "Invalid command format"

            # Execute command
            result = subprocess.run(  # nosec
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
            )

            return result.returncode == 0, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except FileNotFoundError:
            return False, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, "", f"Subprocess error: {str(e)}"

    @staticmethod
    def run_python_module(
        module: str, args: Optional[List[str]] = None, timeout: int = 300
    ) -> Tuple[bool, str, str]:
        """Execute a Python module securely.

        Args:
            module: Python module to execute
            args: Additional arguments
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        cmd = [sys.executable, "-m", module]
        if args:
            cmd.extend(args)

        return SecureSubprocess.run_command(cmd, timeout=timeout)

    @staticmethod
    def run_pip_install(package: str, timeout: int = 300) -> Tuple[bool, str, str]:
        """Install a package using pip securely.

        Args:
            package: Package to install
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        return SecureSubprocess.run_python_module("pip", ["install", package], timeout)

    @staticmethod
    def get_system_info() -> Tuple[bool, str, str]:
        """Get system information securely.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            import platform
            import psutil

            info_lines = []
            info_lines.append(f"OS: {platform.system()} {platform.release()}")
            info_lines.append(f"Python: {sys.version.split()[0]}")
            info_lines.append(f"CPU: {platform.processor()}")
            info_lines.append(f"CPU Cores: {psutil.cpu_count()}")
            info_lines.append(
                f"Memory: {psutil.virtual_memory().total // (1024**3)} GB"
            )

            # Try to get GPU info
            try:
                import torch

                if torch.cuda.is_available():
                    info_lines.append(f"CUDA: Available ({torch.version.cuda})")
                    info_lines.append(f"GPU Count: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        info_lines.append(f"GPU {i}: {props.name}")
                else:
                    info_lines.append("CUDA: Not Available")
            except ImportError:
                info_lines.append("PyTorch: Not Installed")

            # Try to get AMD GPU info on Windows
            if sys.platform == "win32":
                try:
                    import wmi

                    c = wmi.WMI()
                    amd_gpus = [
                        gpu
                        for gpu in c.Win32_VideoController()
                        if gpu.Name and ("AMD" in gpu.Name or "Radeon" in gpu.Name)
                    ]
                    if amd_gpus:
                        info_lines.append(f"AMD GPUs: {len(amd_gpus)} found")
                        for i, gpu in enumerate(amd_gpus):
                            info_lines.append(f"AMD GPU {i}: {gpu.Name}")
                except ImportError:
                    pass
                except Exception:
                    pass

            return True, "\n".join(info_lines), ""

        except Exception as e:
            return False, f"Error getting system info: {e}", ""


def run_command(cmd: Union[str, List[str]], **kwargs: Any) -> Tuple[bool, str, str]:
    """Run commands using the secure subprocess utilities.

    Args:
        cmd: Command to execute
        **kwargs: Additional arguments passed to SecureSubprocess.run_command

    Returns:
        Tuple of (success, stdout, stderr)
    """
    return SecureSubprocess.run_command(cmd, **kwargs)


def run_python_module(
    module: str, args: Optional[List[str]] = None, **kwargs: Any
) -> Tuple[bool, str, str]:
    """Run Python modules using the secure subprocess utilities.

    Args:
        module: Python module to execute
        args: Additional arguments
        **kwargs: Additional arguments passed to SecureSubprocess.run_python_module

    Returns:
        Tuple of (success, stdout, stderr)
    """
    return SecureSubprocess.run_python_module(module, args, **kwargs)


def run_pip_install(package: str, **kwargs: Any) -> Tuple[bool, str, str]:
    """Install packages using pip with secure subprocess utilities.

    Args:
        package: Package to install
        **kwargs: Additional arguments passed to SecureSubprocess.run_pip_install

    Returns:
        Tuple of (success, stdout, stderr)
    """
    return SecureSubprocess.run_pip_install(package, **kwargs)


def get_system_info() -> Tuple[bool, str, str]:
    """Get system information using secure subprocess utilities.

    Returns:
        Tuple of (success, stdout, stderr)
    """
    return SecureSubprocess.get_system_info()
