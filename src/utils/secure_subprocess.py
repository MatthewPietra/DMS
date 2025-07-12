"""
Secure Subprocess Utilities

Provides safe subprocess execution with input validation and proper error handling.
This module centralizes all subprocess operations to ensure security best practices.
"""

import logging
import subprocess  # nosec
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SecureSubprocess:
    """Secure subprocess execution utilities."""

    @staticmethod
    def run_command(
        cmd: Union[str, List[str]],
        timeout: int = 300,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
    ) -> Tuple[bool, str, str]:
        """
        Execute a command securely.

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
        module: str, args: List[str] = None, timeout: int = 300
    ) -> Tuple[bool, str, str]:
        """
        Execute a Python module securely.

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
        """
        Install a package using pip securely.

        Args:
            package: Package to install
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        return SecureSubprocess.run_python_module("pip", ["install", package], timeout)

    @staticmethod
    def get_system_info() -> Tuple[bool, str, str]:
        """
        Get system information securely.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        if sys.platform == "darwin":
            return SecureSubprocess.run_command(["ioreg", "-l"], timeout=10)
        elif sys.platform == "win32":
            return SecureSubprocess.run_command(
                ["wmic", "csproduct", "get", "uuid"], timeout=10
            )
        else:
            return SecureSubprocess.run_command(["cat", "/etc/machine-id"], timeout=10)


def run_command(cmd: Union[str, List[str]], **kwargs) -> Tuple[bool, str, str]:
    """Convenience function for running commands."""
    return SecureSubprocess.run_command(cmd, **kwargs)


def run_python_module(
    module: str, args: List[str] = None, **kwargs
) -> Tuple[bool, str, str]:
    """Convenience function for running Python modules."""
    return SecureSubprocess.run_python_module(module, args, **kwargs)


def run_pip_install(package: str, **kwargs) -> Tuple[bool, str, str]:
    """Convenience function for pip install."""
    return SecureSubprocess.run_pip_install(package, **kwargs)


def get_system_info() -> Tuple[bool, str, str]:
    """Convenience function for getting system info."""
    return SecureSubprocess.get_system_info()
