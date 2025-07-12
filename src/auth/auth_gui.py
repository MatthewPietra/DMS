"""
Authentication GUI for DMS

Provides a modern, user-friendly interface for KeyAuth verification
and user registration/login functionality.
"""

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# GUI Framework imports with fallback
try:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *

    QT_VERSION = "PyQt5"
except ImportError:
    try:
        from PyQt6.QtCore import *
        from PyQt6.QtGui import *
        from PyQt6.QtWidgets import *

        QT_VERSION = "PyQt6"
    except ImportError:
        try:
            from PySide6.QtCore import *
            from PySide6.QtGui import *
            from PySide6.QtWidgets import *

            QT_VERSION = "PySide6"
        except ImportError:
            raise ImportError(
                "No Qt framework found. Please install PyQt5, PyQt6, or PySide6"
            )

from .keyauth_api import KeyAuthAPI
from .user_manager import UserManager


class AuthenticationGUI(QMainWindow):
    """Main authentication window for DMS."""

    # Signals
    authentication_successful = (
        pyqtSignal(dict) if QT_VERSION == "PyQt5" else Signal(dict)
    )
    authentication_failed = pyqtSignal(str) if QT_VERSION == "PyQt5" else Signal(str)

    def __init__(self):
        super().__init__()

        # Load configuration
        self.config = self._load_config()

        # Initialize managers
        self.user_manager = UserManager()
        self.keyauth_api = None

        # UI state
        self.current_step = "keyauth"  # keyauth, login, register
        self.keyauth_verified = False
        self.keyauth_data = None

        # Setup UI
        self._setup_ui()
        self._setup_styles()

        # Check for existing session
        self._check_existing_session()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config_path = (
                Path(__file__).parent.parent.parent / "config" / "keyauth_config.json"
            )
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "application": {
                "name": "Kalena's Application",
                "ownerid": "JR8hfS3d4v",
                "secret": "d7c57798279632a99a4429bc07ece9bf6070d5d2229c1d03387c7bc6d0b94c10",
                "version": "1.0",
            },
            "ui": {
                "window_title": "DMS Authentication",
                "window_width": 450,
                "window_height": 600,
                "theme": "dark",
            },
            "settings": {"session_duration_hours": 24, "remember_last_username": True},
        }

    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle(self.config["ui"]["window_title"])
        self.setFixedSize(
            self.config["ui"]["window_width"], self.config["ui"]["window_height"]
        )

        # Center window
        self._center_window()

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Create screens
        self._create_keyauth_screen()
        self._create_login_screen()
        self._create_register_screen()

        # Show initial screen
        self.stacked_widget.setCurrentIndex(0)  # KeyAuth screen

    def _center_window(self):
        """Center the window on screen."""
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()

        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2

        self.move(x, y)

    def _create_keyauth_screen(self):
        """Create the KeyAuth verification screen."""
        keyauth_widget = QWidget()
        layout = QVBoxLayout(keyauth_widget)
        layout.setSpacing(20)

        # Header
        header_layout = QVBoxLayout()

        # Logo/Title
        title_label = QLabel("DMS Authentication")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)

        subtitle_label = QLabel("Detection Model Suite")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle_label)

        layout.addLayout(header_layout)
        layout.addSpacing(20)

        # KeyAuth section
        keyauth_group = QGroupBox("License Verification")
        keyauth_group.setObjectName("group")
        keyauth_layout = QVBoxLayout(keyauth_group)

        # Instructions
        instructions = QLabel(
            "Please enter your KeyAuth license key to continue.\n"
            "This verifies your access to the DMS application."
        )
        instructions.setObjectName("instructions")
        instructions.setWordWrap(True)
        keyauth_layout.addWidget(instructions)

        # KeyAuth input
        self.keyauth_input = QLineEdit()
        self.keyauth_input.setObjectName("keyauth_input")
        self.keyauth_input.setPlaceholderText("Enter your KeyAuth license key...")
        self.keyauth_input.setEchoMode(QLineEdit.Password)
        keyauth_layout.addWidget(self.keyauth_input)

        # Show/Hide key button
        show_key_btn = QPushButton("👁")
        show_key_btn.setObjectName("show_key_btn")
        show_key_btn.setFixedSize(30, 30)
        show_key_btn.clicked.connect(self._toggle_keyauth_visibility)

        key_layout = QHBoxLayout()
        key_layout.addWidget(self.keyauth_input)
        key_layout.addWidget(show_key_btn)
        keyauth_layout.addLayout(key_layout)

        # Verify button
        self.verify_btn = QPushButton("Verify License")
        self.verify_btn.setObjectName("primary_btn")
        self.verify_btn.clicked.connect(self._verify_keyauth)
        keyauth_layout.addWidget(self.verify_btn)

        layout.addWidget(keyauth_group)

        # Status label
        self.keyauth_status = QLabel("")
        self.keyauth_status.setObjectName("status")
        self.keyauth_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.keyauth_status)

        # Progress bar
        self.keyauth_progress = QProgressBar()
        self.keyauth_progress.setObjectName("progress")
        self.keyauth_progress.setVisible(False)
        layout.addWidget(self.keyauth_progress)

        layout.addStretch()

        self.stacked_widget.addWidget(keyauth_widget)

    def _create_login_screen(self):
        """Create the login screen."""
        login_widget = QWidget()
        layout = QVBoxLayout(login_widget)
        layout.setSpacing(20)

        # Header
        header_label = QLabel("Welcome Back!")
        header_label.setObjectName("title")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Login form
        login_group = QGroupBox("Login to DMS")
        login_group.setObjectName("group")
        login_layout = QVBoxLayout(login_group)

        # Username
        self.username_input = QLineEdit()
        self.username_input.setObjectName("input")
        self.username_input.setPlaceholderText("Username")
        login_layout.addWidget(self.username_input)

        # Password
        self.password_input = QLineEdit()
        self.password_input.setObjectName("input")
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        login_layout.addWidget(self.password_input)

        # Login button
        self.login_btn = QPushButton("Login")
        self.login_btn.setObjectName("primary_btn")
        self.login_btn.clicked.connect(self._login_user)
        login_layout.addWidget(self.login_btn)

        layout.addWidget(login_group)

        # Register option
        register_layout = QHBoxLayout()
        register_label = QLabel("Don't have an account?")
        register_btn = QPushButton("Register")
        register_btn.setObjectName("secondary_btn")
        register_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))

        register_layout.addWidget(register_label)
        register_layout.addWidget(register_btn)
        register_layout.addStretch()
        layout.addLayout(register_layout)

        # Status
        self.login_status = QLabel("")
        self.login_status.setObjectName("status")
        self.login_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.login_status)

        layout.addStretch()

        self.stacked_widget.addWidget(login_widget)

    def _create_register_screen(self):
        """Create the registration screen."""
        register_widget = QWidget()
        layout = QVBoxLayout(register_widget)
        layout.setSpacing(20)

        # Header
        header_label = QLabel("Create Account")
        header_label.setObjectName("title")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Registration form
        register_group = QGroupBox("New User Registration")
        register_group.setObjectName("group")
        register_layout = QVBoxLayout(register_group)

        # Username
        self.reg_username_input = QLineEdit()
        self.reg_username_input.setObjectName("input")
        self.reg_username_input.setPlaceholderText("Choose a username")
        register_layout.addWidget(self.reg_username_input)

        # Password
        self.reg_password_input = QLineEdit()
        self.reg_password_input.setObjectName("input")
        self.reg_password_input.setPlaceholderText("Choose a password")
        self.reg_password_input.setEchoMode(QLineEdit.Password)
        register_layout.addWidget(self.reg_password_input)

        # Confirm Password
        self.reg_confirm_input = QLineEdit()
        self.reg_confirm_input.setObjectName("input")
        self.reg_confirm_input.setPlaceholderText("Confirm password")
        self.reg_confirm_input.setEchoMode(QLineEdit.Password)
        register_layout.addWidget(self.reg_confirm_input)

        # Register button
        self.register_btn = QPushButton("Create Account")
        self.register_btn.setObjectName("primary_btn")
        self.register_btn.clicked.connect(self._register_user)
        register_layout.addWidget(self.register_btn)

        layout.addWidget(register_group)

        # Login option
        login_layout = QHBoxLayout()
        login_label = QLabel("Already have an account?")
        login_btn = QPushButton("Login")
        login_btn.setObjectName("secondary_btn")
        login_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

        login_layout.addWidget(login_label)
        login_layout.addWidget(login_btn)
        login_layout.addStretch()
        layout.addLayout(login_layout)

        # Status
        self.register_status = QLabel("")
        self.register_status.setObjectName("status")
        self.register_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.register_status)

        layout.addStretch()

        self.stacked_widget.addWidget(register_widget)

    def _setup_styles(self):
        """Setup the application styles."""
        style = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QLabel#title {
            font-size: 24px;
            font-weight: bold;
            color: #4a9eff;
            margin: 10px 0;
        }
        
        QLabel#subtitle {
            font-size: 14px;
            color: #888888;
            margin-bottom: 20px;
        }
        
        QLabel#instructions {
            font-size: 12px;
            color: #cccccc;
            margin-bottom: 10px;
        }
        
        QGroupBox#group {
            font-size: 14px;
            font-weight: bold;
            color: #4a9eff;
            border: 2px solid #333333;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
        }
        
        QGroupBox#group::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QLineEdit#input, QLineEdit#keyauth_input {
            background-color: #2d2d2d;
            border: 2px solid #404040;
            border-radius: 6px;
            padding: 10px;
            font-size: 14px;
            color: #ffffff;
        }
        
        QLineEdit#input:focus, QLineEdit#keyauth_input:focus {
            border-color: #4a9eff;
        }
        
        QPushButton#primary_btn {
            background-color: #4a9eff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
        }
        
        QPushButton#primary_btn:hover {
            background-color: #357abd;
        }
        
        QPushButton#primary_btn:pressed {
            background-color: #2d5a87;
        }
        
        QPushButton#secondary_btn {
            background-color: transparent;
            color: #4a9eff;
            border: 2px solid #4a9eff;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 12px;
        }
        
        QPushButton#secondary_btn:hover {
            background-color: #4a9eff;
            color: white;
        }
        
        QPushButton#show_key_btn {
            background-color: #404040;
            color: #ffffff;
            border: none;
            border-radius: 4px;
        }
        
        QPushButton#show_key_btn:hover {
            background-color: #505050;
        }
        
        QLabel#status {
            font-size: 12px;
            margin: 10px 0;
        }
        
        QProgressBar#progress {
            border: 2px solid #333333;
            border-radius: 6px;
            text-align: center;
            background-color: #2d2d2d;
        }
        
        QProgressBar#progress::chunk {
            background-color: #4a9eff;
            border-radius: 4px;
        }
        """

        self.setStyleSheet(style)

    def _toggle_keyauth_visibility(self):
        """Toggle KeyAuth key visibility."""
        if self.keyauth_input.echoMode() == QLineEdit.Password:
            self.keyauth_input.setEchoMode(QLineEdit.Normal)
        else:
            self.keyauth_input.setEchoMode(QLineEdit.Password)

    def _verify_keyauth(self):
        """Verify KeyAuth license key."""
        key = self.keyauth_input.text().strip()

        if not key:
            self._show_keyauth_status("Please enter a license key", "error")
            return

        # Disable UI during verification
        self.verify_btn.setEnabled(False)
        self.keyauth_progress.setVisible(True)
        self.keyauth_progress.setRange(0, 0)  # Indeterminate
        self._show_keyauth_status("Verifying license key...", "info")

        # Run verification in separate thread
        self.verify_thread = KeyAuthVerificationThread(key, self.config)
        self.verify_thread.verification_complete.connect(self._on_keyauth_verified)
        self.verify_thread.start()

    def _on_keyauth_verified(self, success: bool, data: Dict[str, Any]):
        """Handle KeyAuth verification result."""
        self.verify_btn.setEnabled(True)
        self.keyauth_progress.setVisible(False)

        if success:
            self.keyauth_verified = True
            self.keyauth_data = data
            self._show_keyauth_status("License verified successfully!", "success")

            # Move to login screen after short delay
            QTimer.singleShot(1000, lambda: self.stacked_widget.setCurrentIndex(1))
        else:
            error_msg = data.get("error", "Unknown error occurred")
            self._show_keyauth_status(f"Verification failed: {error_msg}", "error")

    def _show_keyauth_status(self, message: str, status_type: str):
        """Show KeyAuth status message."""
        self.keyauth_status.setText(message)

        if status_type == "error":
            self.keyauth_status.setStyleSheet("color: #ff4444;")
        elif status_type == "success":
            self.keyauth_status.setStyleSheet("color: #44ff44;")
        else:
            self.keyauth_status.setStyleSheet("color: #4a9eff;")

    def _login_user(self):
        """Login existing user."""
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            self._show_login_status("Please enter both username and password", "error")
            return

        # Authenticate user
        user_data = self.user_manager.authenticate_user(username, password)

        if user_data:
            # Create session
            session_token = self.user_manager.create_session(user_data["id"])

            if session_token:
                user_data["session_token"] = session_token
                self._show_login_status("Login successful!", "success")

                # Emit success signal
                self.authentication_successful.emit(user_data)

                # Close window after short delay
                QTimer.singleShot(1000, self.close)
            else:
                self._show_login_status("Failed to create session", "error")
        else:
            self._show_login_status("Invalid username or password", "error")

    def _register_user(self):
        """Register new user."""
        if not self.keyauth_verified:
            self._show_register_status("Please verify your license key first", "error")
            return

        username = self.reg_username_input.text().strip()
        password = self.reg_password_input.text().strip()
        confirm_password = self.reg_confirm_input.text().strip()

        if not username or not password or not confirm_password:
            self._show_register_status("Please fill in all fields", "error")
            return

        if password != confirm_password:
            self._show_register_status("Passwords do not match", "error")
            return

        if len(password) < 8:
            self._show_register_status(
                "Password must be at least 8 characters", "error"
            )
            return

        # Create user
        keyauth_key = self.keyauth_input.text().strip()
        success = self.user_manager.create_user(
            username, password, keyauth_key, self.keyauth_data
        )

        if success:
            self._show_register_status("Account created successfully!", "success")

            # Auto-login the new user
            QTimer.singleShot(1000, lambda: self._auto_login(username, password))
        else:
            self._show_register_status("Username already exists", "error")

    def _auto_login(self, username: str, password: str):
        """Auto-login after registration."""
        self.username_input.setText(username)
        self.password_input.setText(password)
        self.stacked_widget.setCurrentIndex(1)
        self._login_user()

    def _show_login_status(self, message: str, status_type: str):
        """Show login status message."""
        self.login_status.setText(message)

        if status_type == "error":
            self.login_status.setStyleSheet("color: #ff4444;")
        elif status_type == "success":
            self.login_status.setStyleSheet("color: #44ff44;")
        else:
            self.login_status.setStyleSheet("color: #4a9eff;")

    def _show_register_status(self, message: str, status_type: str):
        """Show registration status message."""
        self.register_status.setText(message)

        if status_type == "error":
            self.register_status.setStyleSheet("color: #ff4444;")
        elif status_type == "success":
            self.register_status.setStyleSheet("color: #44ff44;")
        else:
            self.register_status.setStyleSheet("color: #4a9eff;")

    def _check_existing_session(self):
        """Check for existing valid session."""
        # This would check for saved session tokens
        # For now, we'll skip this and always show auth screen
        pass


class KeyAuthVerificationThread(QThread):
    """Thread for KeyAuth verification to prevent UI blocking."""

    verification_complete = (
        pyqtSignal(bool, dict) if QT_VERSION == "PyQt5" else Signal(bool, dict)
    )

    def __init__(self, key: str, config: Dict[str, Any]):
        super().__init__()
        self.key = key
        self.config = config

    def run(self):
        """Run KeyAuth verification."""
        try:
            # Initialize KeyAuth API
            api = KeyAuthAPI(
                name=self.config["application"]["name"],
                ownerid=self.config["application"]["ownerid"],
                secret=self.config["application"]["secret"],
                version=self.config["application"]["version"],
            )

            # Verify license
            success = api.license(self.key)

            if success:
                # Return user data
                user_data = {
                    "username": api.user_data.username,
                    "expires": api.user_data.expires,
                    "subscription": api.user_data.subscription,
                    "hwid": api.user_data.hwid,
                }
                self.verification_complete.emit(True, user_data)
            else:
                self.verification_complete.emit(
                    False, {"error": "License verification failed"}
                )

        except Exception as e:
            self.verification_complete.emit(False, {"error": str(e)})


def show_authentication_dialog(parent=None) -> Optional[Dict[str, Any]]:
    """Show authentication dialog and return user data if successful."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = AuthenticationGUI()

    # Store result
    result = {"success": False, "user_data": None}

    def on_success(user_data):
        result["success"] = True
        result["user_data"] = user_data

    def on_failure(error):
        result["success"] = False
        result["error"] = error

    dialog.authentication_successful.connect(on_success)
    dialog.authentication_failed.connect(on_failure)

    dialog.show()
    app.exec_()

    return result if result["success"] else None


if __name__ == "__main__":
    # Test the authentication GUI
    result = show_authentication_dialog()
    if result:
        print(f"Authentication successful: {result['user_data']}")
    else:
        print("Authentication failed or cancelled")
