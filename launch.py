#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS Simple Launcher.

A simple launcher that:
1. Shows clean login/register screen
2. After successful auth, shows interface selection
3. Launches the selected interface

Usage: python launch.py
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def main() -> int:
    """Main launcher function.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("[LAUNCHER] DMS Launcher - Starting Authentication...")

    try:
        # Import clean authentication GUI
        from auth.auth_gui import show_clean_authentication_dialog  # type: ignore

        # Show authentication dialog with clean interface
        auth_result = show_clean_authentication_dialog()

        if not auth_result:
            print("[ERROR] Authentication failed or cancelled")
            return 1

        user_data = auth_result["user_data"]
        interface = auth_result["interface"]

        print(
            f"[SUCCESS] Authentication successful! Welcome, "
            f"{user_data.get('username', 'User')}"
        )
        print(f"[INFO] Selected interface: {interface.upper()}")

        # Launch the selected interface
        if interface == "gui":
            print("[GUI] Launching GUI...")
            return launch_gui()
        elif interface == "cli":
            print("[CLI] Launching CLI...")
            return launch_cli()
        else:
            print("[ERROR] Unknown interface selected")
            return 1

    except ImportError as e:
        print(f"[ERROR] Error importing authentication: {e}")
        print("Please ensure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1


def launch_gui() -> int:
    """Launch the GUI interface.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        from gui.main_window import main as gui_main  # type: ignore

        gui_main()
        return 0
    except ImportError as e:
        print(f"[ERROR] Error importing GUI: {e}")
        print(
            "Please install GUI dependencies: "
            "pip install -r requirements/requirements_gui.txt"
        )
        return 1
    except Exception as e:
        print(f"[ERROR] Error launching GUI: {e}")
        return 1


def launch_cli() -> int:
    """Launch the CLI interface.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        from cli import main as cli_main  # type: ignore

        # Launch CLI with help to show available commands
        sys.argv = ["cli", "--help"]
        cli_main()
        return 0
    except ImportError as e:
        print(f"[ERROR] Error importing CLI: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Error launching CLI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
