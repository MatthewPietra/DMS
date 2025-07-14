# auth_gui.py

"""
Clean Authentication GUI for DMS

A simplified, clean interface that:
1. Shows login/register screen first
2. After successful auth, shows interface selection
3. Links KeyAuth key only during registration
4. Existing users login normally
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Qt imports with proper fallback
try:
    from PyQt5.QtCore import (
        QThread,
        QTimer,
        pyqtSignal,
        Qt,
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QStackedWidget,
        QLabel,
        QLineEdit,
        QPushButton,
        QProgressBar,
        QTabWidget,
    )

    QT_VERSION = "PyQt5"
    Signal = pyqtSignal
except ImportError:
    try:
        from PyQt6.QtCore import (
            QThread,
            QTimer,
            Signal,
            Qt,
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QStackedWidget,
            QLabel,
            QLineEdit,
            QPushButton,
            QProgressBar,
            QTabWidget,
        )

        QT_VERSION = "PyQt6"
    except ImportError:
        try:
            from PySide6.QtCore import (
                QThread,
                QTimer,
                Signal,
                Qt,
                QApplication,
                QMainWindow,
                QWidget,
                QVBoxLayout,
                QHBoxLayout,
                QStackedWidget,
                QLabel,
                QLineEdit,
                QPushButton,
                QProgressBar,
                QTabWidget,
            )

            QT_VERSION = "PySide6"
        except ImportError:
            raise ImportError(
                "No Qt framework found. Please install PyQt5, PyQt6, " "or PySide6"
            )

from .keyauth_api import KeyAuthAPI
from .user_manager import UserManager

# Configure logging
logger = logging.getLogger(__name__)


class CleanAuthenticationGUI(QMainWindow):
    """Clean authentication window with separated login and interface selection."""

    # Signals
    authentication_successful = (
        pyqtSignal(dict, str) if QT_VERSION.startswith("PyQt") else Signal(dict, str)
    )
    authentication_failed = (
        pyqtSignal(str) if QT_VERSION.startswith("PyQt") else Signal(str)
    )

    def __init__(self) -> None:
        super().__init__()

        # Load configuration
        self.config = self._load_config()

        # Initialize managers
        self.user_manager = UserManager()

        # UI state
        self.current_user: Optional[Dict[str, Any]] = None
        self.current_screen = "login"  # "login" or "interface"

        # Setup UI
        self._setup_ui()
        self._setup_styles()

        # Show login screen
        self._show_login_screen()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config_path = (
                Path(__file__).parent.parent.parent / "config" / "keyauth_config.json"
            )
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.warning("Error loading config: %s", e)
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "application": {
                "name": "Kalena's Application",
                "ownerid": "JR8hfS3d4v",
                # Use env var
                "secret": os.getenv("KEYAUTH_SECRET", ""),
                "version": "1.0",
            },
            "ui": {
                "window_title": "DMS Authentication",
                "window_width": 400,
                "window_height": 500,
                "theme": "dark",
            },
        }

    def _setup_ui(self) -> None:
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
        self.main_layout.setSpacing(30)
        self.main_layout.setContentsMargins(40, 40, 40, 40)

        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Create screens
        self._create_login_screen()
        self._create_interface_screen()

    def _center_window(self) -> None:
        """Center the window on screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return

        screen_geometry = screen.availableGeometry()

        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2

        self.move(x, y)

    def _create_login_screen(self) -> None:
        """Create the login/register screen."""
        login_widget = QWidget()
        layout = QVBoxLayout(login_widget)
        layout.setSpacing(25)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(10)

        title_label = QLabel("DMS")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)

        subtitle_label = QLabel("Detection Model Suite")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle_label)

        layout.addLayout(header_layout)

        # Tab widget for login/register
        self.auth_tabs = QTabWidget()
        self.auth_tabs.setObjectName("auth_tabs")

        # Login tab
        login_tab = QWidget()
        login_layout = QVBoxLayout(login_tab)
        login_layout.setSpacing(15)

        # Login form
        self.username_input = QLineEdit()
        self.username_input.setObjectName("input")
        self.username_input.setPlaceholderText("Username")
        self.username_input.setFixedHeight(45)
        login_layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setObjectName("input")
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFixedHeight(45)
        self.password_input.returnPressed.connect(self._login_user)
        login_layout.addWidget(self.password_input)

        self.login_btn = QPushButton("Login")
        self.login_btn.setObjectName("primary_btn")
        self.login_btn.setFixedHeight(45)
        self.login_btn.clicked.connect(self._login_user)
        login_layout.addWidget(self.login_btn)

        # Login status
        self.login_status = QLabel("")
        self.login_status.setObjectName("status")
        self.login_status.setAlignment(Qt.AlignCenter)
        self.login_status.setWordWrap(True)
        login_layout.addWidget(self.login_status)

        login_layout.addStretch()
        self.auth_tabs.addTab(login_tab, "Login")

        # Register tab
        register_tab = QWidget()
        register_layout = QVBoxLayout(register_tab)
        register_layout.setSpacing(15)

        # Register form
        self.reg_username_input = QLineEdit()
        self.reg_username_input.setObjectName("input")
        self.reg_username_input.setPlaceholderText("Choose username")
        self.reg_username_input.setFixedHeight(45)
        register_layout.addWidget(self.reg_username_input)

        self.reg_password_input = QLineEdit()
        self.reg_password_input.setObjectName("input")
        self.reg_password_input.setPlaceholderText("Choose password")
        self.reg_password_input.setEchoMode(QLineEdit.Password)
        self.reg_password_input.setFixedHeight(45)
        register_layout.addWidget(self.reg_password_input)

        self.reg_confirm_input = QLineEdit()
        self.reg_confirm_input.setObjectName("input")
        self.reg_confirm_input.setPlaceholderText("Confirm password")
        self.reg_confirm_input.setEchoMode(QLineEdit.Password)
        self.reg_confirm_input.setFixedHeight(45)
        register_layout.addWidget(self.reg_confirm_input)

        # KeyAuth key for registration
        self.reg_keyauth_input = QLineEdit()
        self.reg_keyauth_input.setObjectName("input")
        self.reg_keyauth_input.setPlaceholderText("KeyAuth license key")
        self.reg_keyauth_input.setEchoMode(QLineEdit.Password)
        self.reg_keyauth_input.setFixedHeight(45)
        register_layout.addWidget(self.reg_keyauth_input)

        # Show/hide key button
        key_layout = QHBoxLayout()
        key_layout.setContentsMargins(0, 0, 0, 0)

        show_key_btn = QPushButton("Show Key")
        show_key_btn.setObjectName("show_key_btn")
        show_key_btn.setFixedHeight(30)
        show_key_btn.clicked.connect(self._toggle_register_key_visibility)
        key_layout.addStretch()
        key_layout.addWidget(show_key_btn)
        register_layout.addLayout(key_layout)

        self.register_btn = QPushButton("Create Account")
        self.register_btn.setObjectName("primary_btn")
        self.register_btn.setFixedHeight(45)
        self.register_btn.clicked.connect(self._register_user)
        register_layout.addWidget(self.register_btn)

        # Register status
        self.register_status = QLabel("")
        self.register_status.setObjectName("status")
        self.register_status.setAlignment(Qt.AlignCenter)
        self.register_status.setWordWrap(True)
        register_layout.addWidget(self.register_status)

        register_layout.addStretch()
        self.auth_tabs.addTab(register_tab, "Register")

        layout.addWidget(self.auth_tabs)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progress")
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.stacked_widget.addWidget(login_widget)

    def _create_interface_screen(self) -> None:
        """Create the interface selection screen."""
        interface_widget = QWidget()
        layout = QVBoxLayout(interface_widget)
        layout.setSpacing(30)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(10)

        welcome_label = QLabel("Welcome!")
        welcome_label.setObjectName("title")
        welcome_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(welcome_label)

        self.user_label = QLabel("")
        self.user_label.setObjectName("subtitle")
        self.user_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.user_label)

        layout.addLayout(header_layout)

        # Interface selection
        selection_label = QLabel("How would you like to use DMS?")
        selection_label.setObjectName("instructions")
        selection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(selection_label)

        # Interface buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)

        # GUI button
        self.gui_btn = QPushButton(
            "🖥️  Graphical Interface\nFull-featured visual interface"
        )
        self.gui_btn.setObjectName("interface_btn")
        self.gui_btn.setFixedHeight(80)
        self.gui_btn.clicked.connect(lambda: self._launch_interface("gui"))
        button_layout.addWidget(self.gui_btn)

        # CLI button
        self.cli_btn = QPushButton(
            "💻  Command Line Interface\nAdvanced command-line tools"
        )
        self.cli_btn.setObjectName("interface_btn")
        self.cli_btn.setFixedHeight(80)
        self.cli_btn.clicked.connect(lambda: self._launch_interface("cli"))
        button_layout.addWidget(self.cli_btn)

        layout.addLayout(button_layout)

        # Back button
        back_layout = QHBoxLayout()
        back_btn = QPushButton("← Back to Login")
        back_btn.setObjectName("back_btn")
        back_btn.clicked.connect(self._show_login_screen)
        back_layout.addWidget(back_btn)
        back_layout.addStretch()
        layout.addLayout(back_layout)

        layout.addStretch()

        self.stacked_widget.addWidget(interface_widget)

    def _show_login_screen(self) -> None:
        """Show the login screen."""
        self.current_screen = "login"
        self.stacked_widget.setCurrentIndex(0)
        self.current_user = None

        # Clear form data
        self.username_input.clear()
        self.password_input.clear()
        self.login_status.clear()
        self.register_status.clear()

        # Focus on username input
        self.username_input.setFocus()

    def _show_interface_screen(self, username: str) -> None:
        """Show the interface selection screen."""
        self.current_screen = "interface"
        self.user_label.setText(f"Hello, {username}!")
        self.stacked_widget.setCurrentIndex(1)

    def _toggle_register_key_visibility(self) -> None:
        """Toggle KeyAuth key visibility in registration."""
        if self.reg_keyauth_input.echoMode() == QLineEdit.Password:
            self.reg_keyauth_input.setEchoMode(QLineEdit.Normal)
        else:
            self.reg_keyauth_input.setEchoMode(QLineEdit.Password)

    def _validate_input(self, text: str, field_name: str) -> bool:
        """Validate input text for security."""
        if not text or not text.strip():
            return False

        # Check for potentially dangerous characters
        dangerous_chars = ["<", ">", '"', "'", "&", ";", "|", "`", "$", "(", ")"]
        if any(char in text for char in dangerous_chars):
            logger.warning(
                "Potentially dangerous characters detected in %s", field_name
            )
            return False

        return True

    def _login_user(self) -> None:
        """Login existing user."""
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not self._validate_input(username, "username"):
            self._show_login_status("Please enter a valid username", "error")
            return

        if not self._validate_input(password, "password"):
            self._show_login_status("Please enter a valid password", "error")
            return

        self._show_login_status("Logging in...", "info")
        self.login_btn.setEnabled(False)

        # Authenticate user
        user_data = self.user_manager.authenticate_user(username, password)

        if user_data:
            # Create session
            session_token = self.user_manager.create_session(user_data["id"])

            if session_token:
                user_data["session_token"] = session_token
                self.current_user = user_data
                self._show_login_status("Login successful!", "success")

                # Show interface selection after short delay
                QTimer.singleShot(800, lambda: self._show_interface_screen(username))
            else:
                self._show_login_status("Failed to create session", "error")
                self.login_btn.setEnabled(True)
        else:
            self._show_login_status("Invalid username or password", "error")
            self.login_btn.setEnabled(True)

    def _register_user(self) -> None:
        """Register new user with KeyAuth verification."""
        username = self.reg_username_input.text().strip()
        password = self.reg_password_input.text().strip()
        confirm_password = self.reg_confirm_input.text().strip()
        keyauth_key = self.reg_keyauth_input.text().strip()

        # Validate all inputs
        if not all(
            [
                self._validate_input(username, "username"),
                self._validate_input(password, "password"),
                self._validate_input(confirm_password, "confirm_password"),
                self._validate_input(keyauth_key, "keyauth_key"),
            ]
        ):
            self._show_register_status(
                "Please fill in all fields with valid data", "error"
            )
            return

        if password != confirm_password:
            self._show_register_status("Passwords do not match", "error")
            return

        if len(password) < 8:
            self._show_register_status(
                "Password must be at least 8 characters", "error"
            )
            return

        # Disable UI during registration
        self.register_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self._show_register_status("Verifying KeyAuth license...", "info")

        # Run registration in separate thread
        self.register_thread = RegistrationThread(
            username,
            password,
            keyauth_key,
            self.config,
            self.user_manager,
        )
        self.register_thread.registration_complete.connect(
            self._on_registration_complete,
        )
        self.register_thread.start()

    def _on_registration_complete(
        self, success: bool, message: str, user_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle registration completion."""
        self.register_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success and user_data:
            self._show_register_status("Account created successfully!", "success")
            self.current_user = user_data

            # Auto-switch to interface selection
            QTimer.singleShot(
                1000,
                lambda: self._show_interface_screen(user_data["username"]),
            )
        else:
            self._show_register_status(message, "error")

    def _show_login_status(self, message: str, status_type: str) -> None:
        """Show login status message."""
        self.login_status.setText(message)

        if status_type == "error":
            self.login_status.setStyleSheet("color: #ff4444;")
        elif status_type == "success":
            self.login_status.setStyleSheet("color: #44ff44;")
        else:
            self.login_status.setStyleSheet("color: #4a9eff;")

    def _show_register_status(self, message: str, status_type: str) -> None:
        """Show registration status message."""
        self.register_status.setText(message)

        if status_type == "error":
            self.register_status.setStyleSheet("color: #ff4444;")
        elif status_type == "success":
            self.register_status.setStyleSheet("color: #44ff44;")
        else:
            self.register_status.setStyleSheet("color: #4a9eff;")

    def _launch_interface(self, interface: str) -> None:
        """Launch the selected interface."""
        if not self.current_user:
            return

        # Emit success signal with interface choice
        self.authentication_successful.emit(self.current_user, interface)

        # Close window
        self.close()

    def _setup_styles(self) -> None:
        """Setup the application styles."""
        style = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }

        QLabel#title {
            font-size: 28px;
            font-weight: bold;
            color: #4a9eff;
            margin: 15px 0;
        }

        QLabel#subtitle {
            font-size: 16px;
            color: #888888;
            margin-bottom: 20px;
        }

        QLabel#instructions {
            font-size: 14px;
            color: #cccccc;
            margin: 15px 0;
        }

        QTabWidget#auth_tabs {
            background-color: transparent;
            border: none;
        }

        QTabWidget#auth_tabs::pane {
            border: 1px solid #404040;
            border-radius: 8px;
            background-color: #2a2a2a;
            padding: 20px;
        }

        QTabWidget#auth_tabs::tab-bar {
            alignment: center;
        }

        QTabBar::tab {
            background-color: #404040;
            color: #ffffff;
            padding: 12px 24px;
            margin-right: 4px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-size: 14px;
        }

        QTabBar::tab:selected {
            background-color: #4a9eff;
            color: white;
        }

        QTabBar::tab:hover {
            background-color: #505050;
        }

        QTabBar::tab:selected:hover {
            background-color: #357abd;
        }

        QLineEdit#input {
            background-color: #2d2d2d;
            border: 2px solid #404040;
            border-radius: 8px;
            padding: 12px 15px;
            font-size: 14px;
            color: #ffffff;
        }

        QLineEdit#input:focus {
            border-color: #4a9eff;
            background-color: #333333;
        }

        QPushButton#primary_btn {
            background-color: #4a9eff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
        }

        QPushButton#primary_btn:hover {
            background-color: #357abd;
        }

        QPushButton#primary_btn:pressed {
            background-color: #2d5a87;
        }

        QPushButton#primary_btn:disabled {
            background-color: #333333;
            color: #666666;
        }

        QPushButton#interface_btn {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 2px solid #404040;
            border-radius: 12px;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
        }

        QPushButton#interface_btn:hover {
            background-color: #404040;
            border-color: #4a9eff;
            color: #4a9eff;
        }

        QPushButton#interface_btn:pressed {
            background-color: #4a9eff;
            color: white;
        }

        QPushButton#show_key_btn {
            background-color: #404040;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
        }

        QPushButton#show_key_btn:hover {
            background-color: #505050;
        }

        QPushButton#back_btn {
            background-color: transparent;
            color: #888888;
            border: none;
            padding: 8px 12px;
            font-size: 12px;
        }

        QPushButton#back_btn:hover {
            color: #4a9eff;
        }

        QLabel#status {
            font-size: 12px;
            margin: 10px 0;
            padding: 8px;
            border-radius: 4px;
            background-color: #2a2a2a;
        }

        QProgressBar#progress {
            border: 2px solid #333333;
            border-radius: 8px;
            text-align: center;
            background-color: #2d2d2d;
            height: 20px;
        }

        QProgressBar#progress::chunk {
            background-color: #4a9eff;
            border-radius: 6px;
        }
        """

        self.setStyleSheet(style)


class RegistrationThread(QThread):
    """Thread for user registration with KeyAuth verification."""

    registration_complete = (
        pyqtSignal(bool, str, dict)
        if QT_VERSION.startswith("PyQt")
        else Signal(bool, str, dict)
    )

    def __init__(
        self,
        username: str,
        password: str,
        keyauth_key: str,
        config: Dict[str, Any],
        user_manager: UserManager,
    ) -> None:
        super().__init__()
        self.username = username
        self.password = password
        self.keyauth_key = keyauth_key
        self.config = config
        self.user_manager = user_manager

    def run(self) -> None:
        """Run registration with KeyAuth verification."""
        try:
            # First verify KeyAuth license
            api = KeyAuthAPI(
                name=self.config["application"]["name"],
                ownerid=self.config["application"]["ownerid"],
                secret=self.config["application"]["secret"],
                version=self.config["application"]["version"],
            )

            # Verify license
            success = api.license(self.keyauth_key)

            if not success:
                self.registration_complete.emit(
                    False, "Invalid KeyAuth license key", {}
                )
                return

            # Get KeyAuth user data
            keyauth_data = {
                "username": api.user_data.username,
                "expires": api.user_data.expires,
                "subscription": api.user_data.subscription,
                "hwid": api.user_data.hwid,
            }

            # Create user account
            user_created = self.user_manager.create_user(
                self.username, self.password, self.keyauth_key, keyauth_data
            )

            if user_created:
                # Create session for the new user
                user_data = self.user_manager.authenticate_user(
                    self.username, self.password
                )
                if user_data:
                    session_token = self.user_manager.create_session(user_data["id"])
                    if session_token:
                        user_data["session_token"] = session_token
                        self.registration_complete.emit(
                            True, "Account created successfully!", user_data
                        )
                    else:
                        self.registration_complete.emit(
                            False, "Account created but failed to create session", {}
                        )
                else:
                    self.registration_complete.emit(
                        False, "Account created but login failed", {}
                    )
            else:
                self.registration_complete.emit(False, "Username already exists", {})

        except Exception as e:
            logger.error("Registration failed: %s", e)
            self.registration_complete.emit(
                False, "Registration failed: An error occurred during registration.", {}
            )


def show_clean_authentication_dialog(parent=None) -> Optional[Dict[str, Any]]:
    """Show clean authentication dialog and return user data with interface choice if successful."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = CleanAuthenticationGUI()

    # Store result
    result: Dict[str, Any] = {"success": False, "user_data": None, "interface": None}

    def on_success(user_data: Dict[str, Any], interface: str) -> None:
        result["success"] = True
        result["user_data"] = user_data
        result["interface"] = interface

    def on_failure(error: str) -> None:
        result["success"] = False
        result["error"] = error

    dialog.authentication_successful.connect(on_success)
    dialog.authentication_failed.connect(on_failure)

    dialog.show()
    app.exec_()

    return result if result["success"] else None


if __name__ == "__main__":
    # Test the clean authentication GUI
    result = show_clean_authentication_dialog()
    if result:
        print(
            "Authentication successful: {}".format(
                result["user_data"]
            )
        )
        print("Selected interface: {}".format(result["interface"]))
    else:
        print("Authentication failed or cancelled")
