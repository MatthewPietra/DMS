#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS Unified Launcher

A single, comprehensive launcher for the Detection Model Suite (DMS) that:
- Integrates KeyAuth authentication seamlessly
- Defaults to GUI mode with optional CLI mode selection
- Provides one-time popup for mode preference with "don't show again" option
- Supports both Windows and Linux platforms
- Consolidates all previous launcher functionality

Author: DMS Team
Version: 2.0.0
"""

import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import authentication system with automatic dependency management
AUTH_AVAILABLE = False
try:
    # First, ensure authentication dependencies are available
    from auth.dependency_manager import ensure_auth_dependencies

    print("🔍 Checking authentication dependencies...")
    ensure_auth_dependencies()

    # Now import authentication components
    from auth.auth_gui import show_authentication_dialog
    from auth.auth_manager import AuthenticationManager

    AUTH_AVAILABLE = True
    print("✅ Authentication system ready")
except ImportError as e:
    print(f"Warning: Authentication system not available: {e}")
    print("The launcher will continue without authentication features.")

# Import GUI system
GUI_AVAILABLE = False
try:
    from gui.main_window import DMSMainWindow

    GUI_AVAILABLE = True
except ImportError:
    try:
        # Try different Qt frameworks
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QDialog,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
        )

        GUI_AVAILABLE = True
    except ImportError:
        try:
            from PyQt6.QtCore import Qt
            from PyQt6.QtWidgets import (
                QApplication,
                QCheckBox,
                QDialog,
                QHBoxLayout,
                QLabel,
                QMessageBox,
                QPushButton,
                QVBoxLayout,
            )

            GUI_AVAILABLE = True
        except ImportError:
            try:
                from PySide6.QtCore import Qt
                from PySide6.QtWidgets import (
                    QApplication,
                    QCheckBox,
                    QDialog,
                    QHBoxLayout,
                    QLabel,
                    QMessageBox,
                    QPushButton,
                    QVBoxLayout,
                )

                GUI_AVAILABLE = True
            except ImportError:
                pass

# Import CLI launcher
CLI_AVAILABLE = False
try:
    from main import CentralLauncher

    CLI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CLI launcher not available: {e}")


class LauncherPreferences:
    """Manages launcher preferences and settings."""

    def __init__(self):
        self.config_dir = project_root / "config"
        self.config_file = self.config_dir / "launcher_preferences.json"
        self.config_dir.mkdir(exist_ok=True)
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> Dict[str, Any]:
        """Load launcher preferences from file."""
        default_preferences = {
            "default_mode": "gui",  # gui or cli
            "show_mode_selection": True,
            "last_updated": datetime.now().isoformat(),
            "user_choices": {"preferred_mode": None, "skip_mode_dialog": False},
        }

        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    loaded_prefs = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_preferences.update(loaded_prefs)
                    return default_preferences
        except Exception as e:
            print(f"Error loading preferences: {e}")

        return default_preferences

    def save_preferences(self) -> bool:
        """Save preferences to file."""
        try:
            self.preferences["last_updated"] = datetime.now().isoformat()
            with open(self.config_file, "w") as f:
                json.dump(self.preferences, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving preferences: {e}")
            return False

    def get_preferred_mode(self) -> str:
        """Get the user's preferred launcher mode."""
        return self.preferences.get("user_choices", {}).get("preferred_mode", "gui")

    def set_preferred_mode(self, mode: str, skip_dialog: bool = False):
        """Set the user's preferred launcher mode."""
        self.preferences["user_choices"]["preferred_mode"] = mode
        self.preferences["user_choices"]["skip_mode_dialog"] = skip_dialog
        self.save_preferences()

    def should_show_mode_selection(self) -> bool:
        """Check if mode selection dialog should be shown."""
        return self.preferences.get(
            "show_mode_selection", True
        ) and not self.preferences.get("user_choices", {}).get(
            "skip_mode_dialog", False
        )


class ModeSelectionDialog:
    """Dialog for selecting launcher mode (GUI/CLI) with 'don't show again' option."""

    def __init__(self):
        self.selected_mode = "gui"
        self.dont_show_again = False
        self.result = None

    def show_console_dialog(self) -> Tuple[str, bool]:
        """Show console-based mode selection dialog."""
        print("\n" + "=" * 60)
        print("DMS Launcher - Mode Selection")
        print("=" * 60)
        print("\nWelcome to the Detection Model Suite!")
        print("\nPlease choose your preferred launcher mode:")
        print("  1. GUI Mode (Recommended) - Graphical interface")
        print("  2. CLI Mode - Command-line interface")
        print("\nThis choice will be remembered for future launches.")

        while True:
            try:
                choice = input("\nEnter your choice (1 or 2): ").strip()
                if choice == "1":
                    self.selected_mode = "gui"
                    break
                elif choice == "2":
                    self.selected_mode = "cli"
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except (KeyboardInterrupt, EOFError):
                print("\nUsing default GUI mode.")
                self.selected_mode = "gui"
                break

        # Ask about future prompts
        while True:
            try:
                skip_choice = (
                    input("\nDon't show this dialog again? (y/n): ").strip().lower()
                )
                if skip_choice in ["y", "yes"]:
                    self.dont_show_again = True
                    break
                elif skip_choice in ["n", "no"]:
                    self.dont_show_again = False
                    break
                else:
                    print("Please enter 'y' or 'n'.")
            except (KeyboardInterrupt, EOFError):
                self.dont_show_again = False
                break

        return self.selected_mode, self.dont_show_again

    def show_gui_dialog(self) -> Tuple[str, bool]:
        """Show GUI-based mode selection dialog."""
        if not GUI_AVAILABLE:
            return self.show_console_dialog()

        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)

            dialog = QDialog()
            dialog.setWindowTitle("DMS Launcher - Mode Selection")
            dialog.setFixedSize(400, 300)
            dialog.setModal(True)

            # Center the dialog
            screen = app.primaryScreen()
            if screen:
                screen_geometry = screen.availableGeometry()
                x = (screen_geometry.width() - dialog.width()) // 2
                y = (screen_geometry.height() - dialog.height()) // 2
                dialog.move(x, y)

            layout = QVBoxLayout(dialog)
            layout.setSpacing(20)

            # Title
            title_label = QLabel("Welcome to DMS!")
            title_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #4a9eff;"
            )
            layout.addWidget(title_label)

            # Description
            desc_label = QLabel("Please choose your preferred launcher mode:")
            desc_label.setStyleSheet("font-size: 12px; color: #666666;")
            layout.addWidget(desc_label)

            # Mode selection buttons
            button_layout = QVBoxLayout()

            gui_button = QPushButton("🖥️ GUI Mode (Recommended)")
            gui_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #4a9eff;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 15px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #357abd;
                }
            """
            )
            gui_button.clicked.connect(lambda: self._set_mode_and_close(dialog, "gui"))
            button_layout.addWidget(gui_button)

            cli_button = QPushButton("💻 CLI Mode")
            cli_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #666666;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 15px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #777777;
                }
            """
            )
            cli_button.clicked.connect(lambda: self._set_mode_and_close(dialog, "cli"))
            button_layout.addWidget(cli_button)

            layout.addLayout(button_layout)

            # Don't show again checkbox
            checkbox_layout = QHBoxLayout()
            self.dont_show_checkbox = QCheckBox("Don't show this dialog again")
            self.dont_show_checkbox.setStyleSheet("color: #888888;")
            checkbox_layout.addWidget(self.dont_show_checkbox)
            checkbox_layout.addStretch()
            layout.addLayout(checkbox_layout)

            # Set dark theme
            dialog.setStyleSheet(
                """
                QDialog {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                QLabel {
                    color: #ffffff;
                }
            """
            )

            # Show dialog
            dialog.exec_()

            self.dont_show_again = self.dont_show_checkbox.isChecked()
            return self.selected_mode, self.dont_show_again

        except Exception as e:
            print(f"Error showing GUI dialog: {e}")
            return self.show_console_dialog()

    def _set_mode_and_close(self, dialog, mode: str):
        """Set the selected mode and close the dialog."""
        self.selected_mode = mode
        dialog.accept()


class UnifiedLauncher:
    """
    Unified launcher that integrates KeyAuth authentication and provides
    both GUI and CLI modes with user preference management.
    """

    def __init__(self):
        self.project_root = project_root
        self.preferences = LauncherPreferences()
        self.auth_manager = None
        self.authenticated_user = None
        self.current_mode = "gui"  # Default to GUI mode

        # Initialize authentication if available
        if AUTH_AVAILABLE:
            self.auth_manager = AuthenticationManager()

    def run(self) -> int:
        """Main entry point for the unified launcher."""
        try:
            print("=" * 60)
            print("DMS - Detection Model Suite")
            print("Unified Launcher v2.0.0")
            print("=" * 60)

            # Determine launcher mode
            self.current_mode = self._determine_launcher_mode()

            print(f"\nLauncher Mode: {self.current_mode.upper()}")

            # Perform authentication if available
            if AUTH_AVAILABLE:
                if not self._authenticate_user():
                    print("\n❌ Authentication failed. Exiting.")
                    return 1
            else:
                print(
                    "\n⚠️  Authentication system not available. Proceeding without authentication."
                )

            # Launch the appropriate interface
            if self.current_mode == "gui":
                return self._launch_gui_mode()
            else:
                return self._launch_cli_mode()

        except KeyboardInterrupt:
            print("\n\n⚠️  Launcher interrupted by user.")
            return 1
        except Exception as e:
            print(f"\n❌ Launcher error: {e}")
            return 1
        finally:
            self._cleanup()

    def _determine_launcher_mode(self) -> str:
        """Determine which launcher mode to use based on user preferences."""
        # Check if user has a saved preference
        if not self.preferences.should_show_mode_selection():
            preferred_mode = self.preferences.get_preferred_mode()
            if preferred_mode:
                return preferred_mode

        # Show mode selection dialog
        try:
            dialog = ModeSelectionDialog()

            # Try GUI dialog first, fall back to console
            if GUI_AVAILABLE:
                selected_mode, dont_show_again = dialog.show_gui_dialog()
            else:
                selected_mode, dont_show_again = dialog.show_console_dialog()

            # Save user preferences
            self.preferences.set_preferred_mode(selected_mode, dont_show_again)

            return selected_mode

        except Exception as e:
            print(f"Error in mode selection: {e}")
            return "gui"  # Default to GUI mode

    def _authenticate_user(self) -> bool:
        """Authenticate the user using KeyAuth."""
        try:
            # Check for existing session
            if self._check_existing_session():
                return True

            # Show authentication dialog
            if self.current_mode == "gui" and GUI_AVAILABLE:
                return self._gui_authentication()
            else:
                return self._cli_authentication()

        except Exception as e:
            print(f"Authentication error: {e}")
            return False

    def _check_existing_session(self) -> bool:
        """Check for existing valid session."""
        try:
            result = self.auth_manager.load_session_from_file()

            if result["success"]:
                self.authenticated_user = result["data"]
                print(f"✅ Welcome back, {self.authenticated_user['username']}!")

                # Check KeyAuth expiry
                expiry_check = self.auth_manager.check_keyauth_expiry()
                if expiry_check["valid"]:
                    if "days_left" in expiry_check:
                        print(f"   License expires in {expiry_check['days_left']} days")
                    return True
                else:
                    print(f"⚠️  License issue: {expiry_check['error']}")
                    return False

            return False

        except Exception as e:
            print(f"Error checking session: {e}")
            return False

    def _gui_authentication(self) -> bool:
        """Perform GUI-based authentication."""
        try:
            result = show_authentication_dialog()

            if result and result.get("user_data"):
                self.authenticated_user = result["user_data"]

                # Save session
                if self.auth_manager:
                    self.auth_manager.current_user = self.authenticated_user
                    self.auth_manager.current_session = self.authenticated_user.get(
                        "session_token"
                    )
                    self.auth_manager.is_authenticated = True
                    self.auth_manager.save_session_to_file()

                print(
                    f"✅ Authentication successful! Welcome, {self.authenticated_user['username']}"
                )
                return True

            return False

        except Exception as e:
            print(f"GUI authentication error: {e}")
            return False

    def _cli_authentication(self) -> bool:
        """Perform CLI-based authentication."""
        print("\n🔐 Authentication Required")
        print("Please provide your KeyAuth license key and login credentials.")

        try:
            # KeyAuth verification
            license_key = input("KeyAuth License Key: ").strip()
            if not license_key:
                print("❌ License key is required.")
                return False

            print("Verifying license key...")
            keyauth_result = self.auth_manager.verify_keyauth_license(license_key)

            if not keyauth_result["success"]:
                print(f"❌ KeyAuth verification failed: {keyauth_result['error']}")
                return False

            print("✅ License key verified!")

            # User login
            username = input("Username: ").strip()
            password = input("Password: ").strip()

            if not username or not password:
                print("❌ Username and password are required.")
                return False

            # Try login first
            login_result = self.auth_manager.login_user(username, password)

            if login_result["success"]:
                self.authenticated_user = login_result["data"]
                print(f"✅ Login successful! Welcome, {username}")
                return True
            else:
                # Try registration
                print("User not found. Creating new account...")
                register_result = self.auth_manager.register_user(username, password)

                if register_result["success"]:
                    self.authenticated_user = register_result["data"]
                    print(f"✅ Registration successful! Welcome, {username}")
                    return True
                else:
                    print(f"❌ Registration failed: {register_result['error']}")
                    return False

        except (KeyboardInterrupt, EOFError):
            print("\n❌ Authentication cancelled.")
            return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False

    def _launch_gui_mode(self) -> int:
        """Launch the GUI mode."""
        try:
            if not GUI_AVAILABLE:
                print("⚠️  GUI components not available. Falling back to CLI mode.")
                return self._launch_cli_mode()

            print("🚀 Launching GUI mode...")

            # Try to import and launch GUI
            try:
                from gui.main_window import DMSMainWindow

                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)

                # Set application properties
                app.setApplicationName("DMS")
                app.setApplicationVersion("2.0.0")
                app.setOrganizationName("DMS Team")

                # Create main window
                window = DMSMainWindow()

                # Pass authentication info to the window
                if self.authenticated_user:
                    window.authenticated_user = self.authenticated_user
                    window.auth_manager = self.auth_manager

                window.show()

                # Start event loop
                return app.exec_()

            except ImportError:
                # Fall back to GUI launcher
                return self._launch_gui_launcher_fallback()

        except Exception as e:
            print(f"Error launching GUI mode: {e}")
            print("Falling back to CLI mode...")
            return self._launch_cli_mode()

    def _launch_gui_launcher_fallback(self) -> int:
        """Fallback to CLI mode when GUI is not available."""
        try:
            print("GUI components not fully available. Using CLI mode.")
            return self._launch_cli_mode()
        except Exception as e:
            print(f"Fallback error: {e}")
            return 1

    def _launch_cli_mode(self) -> int:
        """Launch the CLI mode."""
        try:
            if not CLI_AVAILABLE:
                print("❌ CLI launcher not available.")
                return 1

            print("🚀 Launching CLI mode...")

            # Create CLI launcher instance
            cli_launcher = CentralLauncher()

            # Pass authentication info
            if self.authenticated_user:
                cli_launcher.authenticated_user = self.authenticated_user
                cli_launcher.auth_manager = self.auth_manager

            # Show main menu
            cli_launcher.show_main_menu()

            return 0

        except Exception as e:
            print(f"Error launching CLI mode: {e}")
            return 1

    def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.auth_manager:
                self.auth_manager.cleanup_expired_sessions()
        except Exception as e:
            print(f"Cleanup error: {e}")


def main():
    """Main entry point."""
    launcher = UnifiedLauncher()
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
