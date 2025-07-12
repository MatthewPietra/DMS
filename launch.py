#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS Simple Launcher

A simple launcher that:
1. Shows clean login/register screen
2. After successful auth, shows interface selection  
3. Launches the selected interface

Usage: python launch.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """Main launcher function."""
    print("🚀 DMS Launcher - Starting Authentication...")
    
    try:
        # Import clean authentication GUI
        from auth.clean_auth_gui import show_clean_authentication_dialog
        
        # Show authentication dialog with clean interface
        auth_result = show_clean_authentication_dialog()
        
        if not auth_result:
            print("❌ Authentication failed or cancelled")
            return 1
        
        user_data = auth_result['user_data']
        interface = auth_result['interface']
        
        print(f"✅ Authentication successful! Welcome, {user_data.get('username', 'User')}")
        print(f"🎯 Selected interface: {interface.upper()}")
        
        # Launch the selected interface
        if interface == "gui":
            print("🖥️  Launching GUI...")
            return launch_gui()
        elif interface == "cli":
            print("💻 Launching CLI...")
            return launch_cli()
        else:
            print("❌ Unknown interface selected")
            return 1
    
    except ImportError as e:
        print(f"❌ Error importing authentication: {e}")
        print("Please ensure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

def launch_gui():
    """Launch the GUI interface."""
    try:
        from gui.main_window import main as gui_main
        gui_main()
        return 0
    except ImportError as e:
        print(f"❌ Error importing GUI: {e}")
        print("Please install GUI dependencies: pip install -r requirements/requirements_gui.txt")
        return 1
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        return 1

def launch_cli():
    """Launch the CLI interface."""
    try:
        from cli import main as cli_main
        # Launch CLI with help to show available commands
        sys.argv = ["cli", "--help"]
        cli_main()
        return 0
    except ImportError as e:
        print(f"❌ Error importing CLI: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error launching CLI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 