#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DMS Authentication Module Entry Point.

Provides a command-line interface for testing and managing authentication.
"""

import sys
from pathlib import Path

from .auth_manager import AuthenticationManager
from .dependency_manager import ensure_auth_dependencies

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> int:
    """Provide the main entry point for authentication module."""
    print("DMS Authentication System")
    print("=" * 40)

    try:
        # First, ensure dependencies are available
        print("Checking authentication dependencies...")
        success = ensure_auth_dependencies()

        if not success:
            print("Failed to install authentication dependencies")
            print("Please run the authentication dependency installer manually.")
            return 1

        # Import authentication components
        print("Authentication system ready")
        print("\nAvailable authentication features:")
        print("  - KeyAuth license verification")
        print("  - User registration and login")
        print("  - Session management")
        print("  - GUI authentication dialog")

        # Test authentication system
        print("\nTesting authentication system...")
        AuthenticationManager()

        print("Authentication system initialized successfully")
        print(
            "\nThe authentication system is ready for use in the main DMS application."
        )

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all authentication dependencies are installed.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
