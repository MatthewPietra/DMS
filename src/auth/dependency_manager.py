"""Dependency Manager for DMS Authentication.

Automatically checks for and installs missing authentication dependencies
to ensure seamless user experience without manual intervention.
"""

import importlib
import platform
import sys
from pathlib import Path
from typing import List, Tuple

from ..utils.secure_subprocess import run_pip_install


class AuthenticationDependencyManager:
    """Manages authentication dependencies automatically."""

    # Core authentication dependencies
    AUTH_DEPENDENCIES = {
        "requests": "requests>=2.31.0",
        "cryptography": "cryptography>=41.0.0",
        "psutil": "psutil>=5.9.0",
        "PyQt5": "PyQt5>=5.15.0",
    }

    # Windows-specific dependencies
    WINDOWS_DEPENDENCIES = {
        "wmi": "wmi>=1.5.1",
        "win32security": "pywin32>=306",  # win32security is part of pywin32
    }

    def __init__(self) -> None:
        """Initialize the dependency manager."""
        self.project_root = Path(__file__).parent.parent.parent
        self.python_exe = self._get_python_executable()
        self.pip_exe = self._get_pip_executable()
        self.installation_log: List[str] = []

    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable path.

        Returns:
            str: Path to the Python executable.
        """
        # Check for virtual environment
        venv_path = self.project_root / "venv"
        if venv_path.exists():
            if platform.system() == "Windows":
                return str(venv_path / "Scripts" / "python.exe")
            else:
                return str(venv_path / "bin" / "python")
        return sys.executable

    def _get_pip_executable(self) -> str:
        """Get the appropriate pip executable path.

        Returns:
            str: Path to the pip executable.
        """
        return f"{self.python_exe} -m pip"

    def check_dependency(self, module_name: str) -> bool:
        """Check if a module is available.

        Args:
            module_name: Name of the module to check.

        Returns:
            bool: True if module is available, False otherwise.
        """
        try:
            # Special handling for wmi module
            if module_name == "wmi" and platform.system() != "Windows":
                return True  # Skip wmi check on non-Windows systems

            # Special handling for win32security module
            if module_name == "win32security" and platform.system() != "Windows":
                return True  # Skip win32security check on non-Windows systems

            # Special handling for cryptography module
            if module_name == "cryptography":
                try:
                    importlib.import_module("cryptography.hazmat.primitives.ciphers")
                    importlib.import_module("cryptography.hazmat.primitives.hashes")
                    importlib.import_module("cryptography.hazmat.primitives.padding")
                    return True
                except ImportError:
                    return False

            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing authentication dependencies.

        Returns:
            List[str]: List of missing package specifications.
        """
        missing = []

        # Check core dependencies
        for module, package in self.AUTH_DEPENDENCIES.items():
            if not self.check_dependency(module):
                missing.append(package)

        # Check Windows-specific dependencies
        if platform.system() == "Windows":
            for module, package in self.WINDOWS_DEPENDENCIES.items():
                if not self.check_dependency(module):
                    missing.append(package)

        return missing

    def install_dependency(self, package: str) -> Tuple[bool, str]:
        """Install a single dependency.

        Args:
            package: Package specification to install.

        Returns:
            Tuple[bool, str]: Success status and message.
        """
        try:
            print(f"Installing: {package}")

            # Use secure subprocess utility
            success, stdout, stderr = run_pip_install(package, timeout=300)

            if success:
                self.installation_log.append(
                    f"SUCCESS: Successfully installed {package}"
                )
                return True, f"Successfully installed {package}"
            else:
                error_msg = f"Failed to install {package}: {stderr}"
                self.installation_log.append(f"ERROR: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = f"Error installing {package}: {str(e)}"
            self.installation_log.append(f"ERROR: {error_msg}")
            return False, error_msg

    def install_missing_dependencies(self) -> Tuple[bool, List[str]]:
        """Install all missing authentication dependencies.

        Returns:
            Tuple[bool, List[str]]: Success status and installation log.
        """
        missing_deps = self.get_missing_dependencies()

        if not missing_deps:
            return True, ["All authentication dependencies are already installed"]

        print("Installing missing authentication dependencies...")
        print("=" * 60)

        success_count = 0
        failed_packages = []

        for package in missing_deps:
            success, message = self.install_dependency(package)
            if success:
                success_count += 1
            else:
                failed_packages.append(package)

        print("=" * 60)
        print(
            f"Installation complete: "
            f"{success_count}/{len(missing_deps)} packages installed"
        )

        if failed_packages:
            print(f"Failed packages: {', '.join(failed_packages)}")
            return False, self.installation_log
        else:
            print("All authentication dependencies installed successfully!")
            return True, self.installation_log

    def ensure_authentication_ready(self) -> Tuple[bool, List[str]]:
        """Verify all authentication dependencies are available.

        Returns:
            Tuple[bool, List[str]]: Success status and installation log.
        """
        print("Checking authentication dependencies...")

        # Check if requests module is available (most critical)
        if not self.check_dependency("requests"):
            print(
                "WARNING: 'requests' module not found. "
                "Installing authentication dependencies..."
            )
            return self.install_missing_dependencies()

        # Check other critical dependencies
        critical_deps = ["cryptography", "psutil"]
        missing_critical = [
            dep for dep in critical_deps if not self.check_dependency(dep)
        ]

        if missing_critical:
            print(
                f"WARNING: Missing critical dependencies: "
                f"{','.join(missing_critical)}"
            )
            return self.install_missing_dependencies()

        print("All authentication dependencies are available")
        return True, ["All dependencies are already installed"]

    def get_installation_summary(self) -> str:
        """Retrieve a summary of the installation process.

        Returns:
            str: Installation summary.
        """
        if not self.installation_log:
            return "No installation performed"

        return "\n".join(self.installation_log)


def ensure_auth_dependencies() -> bool:
    """Ensure authentication dependencies are available.

    Returns:
        bool: True if all dependencies are available, False otherwise.
    """
    manager = AuthenticationDependencyManager()
    success, _ = manager.ensure_authentication_ready()
    return success


if __name__ == "__main__":
    # Test the dependency manager
    manager = AuthenticationDependencyManager()
    success, log = manager.ensure_authentication_ready()

    print("\nInstallation Summary:")
    print("=" * 40)
    for entry in log:
        print(entry)

    if success:
        print("\nAuthentication system is ready!")
    else:
        print("\nSome dependencies failed to install.")
