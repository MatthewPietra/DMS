"""Authentication Manager for DMS.

Coordinates the authentication flow between KeyAuth verification,
user management, and session handling.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .keyauth_api import KeyAuthAPI
from .user_manager import UserManager

# Configure logging
logger = logging.getLogger(__name__)


class AuthenticationManager:
    """Manages the complete authentication flow for DMS.

    Flow:
    1. KeyAuth license verification
    2. User registration/login
    3. Session management
    4. Access control

    Attributes:
        config_path: Path to the configuration file.
        config: Loaded configuration dictionary.
        user_manager: User management instance.
        keyauth_api: KeyAuth API instance.
        current_user: Currently authenticated user data.
        current_session: Current session token.
        is_authenticated: Authentication status flag.
        keyauth_verified: KeyAuth verification status.
        keyauth_data: KeyAuth license data.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the authentication manager.

        Args:
            config_path: Optional path to configuration file. If None,
                uses default path.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

        # Initialize components
        self.user_manager = UserManager()
        self.keyauth_api: Optional[KeyAuthAPI] = None

        # Current session
        self.current_user: Optional[Dict[str, Any]] = None
        self.current_session: Optional[str] = None

        # Authentication state
        self.is_authenticated: bool = False
        self.keyauth_verified: bool = False
        self.keyauth_data: Optional[Dict[str, Any]] = None

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path.

        Returns:
            Default configuration file path.
        """
        return str(
            Path(__file__).parent.parent.parent / "config" / "keyauth_config.json"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.

        Returns:
            Configuration dictionary.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading config: %s", e)
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.

        Returns:
            Default configuration dictionary.
        """
        return {
            "application": {
                "name": "DMS - Detection Model Suite",
                "ownerid": "JR8hfS3d4v",
                "secret": "KEYAUTH_SECRET",
                "version": "1.0.0",
            },
            "settings": {
                "session_duration_hours": 24,
                "auto_cleanup_sessions": True,
                "require_keyauth_for_registration": True,
            },
        }

    def verify_keyauth_license(self, license_key: str) -> Dict[str, Any]:
        """Verify KeyAuth license key.

        Args:
            license_key: The license key to verify.

        Returns:
            Dictionary with 'success' boolean and 'data' or 'error' fields.
        """
        if not license_key or not license_key.strip():
            return {"success": False, "error": "Invalid license key"}

        try:
            # Initialize KeyAuth API
            self.keyauth_api = KeyAuthAPI(
                name=self.config["application"]["name"],
                ownerid=self.config["application"]["ownerid"],
                secret=self.config["application"]["secret"],
                version=self.config["application"]["version"],
            )

            # Verify license
            success = self.keyauth_api.license(license_key)

            if success:
                self.keyauth_verified = True
                self.keyauth_data = {
                    "username": self.keyauth_api.user_data.username,
                    "expires": self.keyauth_api.user_data.expires,
                    "subscription": self.keyauth_api.user_data.subscription,
                    "hwid": self.keyauth_api.user_data.hwid,
                    "license_key": license_key,
                }

                return {"success": True, "data": self.keyauth_data}
            else:
                return {"success": False, "error": "License verification failed"}

        except Exception as e:
            logger.error("KeyAuth verification error: %s", e)
            return {"success": False, "error": str(e)}

    def register_user(self, username: str, password: str) -> Dict[str, Any]:
        """Register a new user after KeyAuth verification.

        Args:
            username: Username for the new account.
            password: Password for the new account.

        Returns:
            Dictionary with 'success' boolean and 'data' or 'error' fields.
        """
        if not self.keyauth_verified:
            return {
                "success": False,
                "error": "KeyAuth license must be verified first",
            }

        if not username or not password:
            return {
                "success": False,
                "error": "Username and password are required",
            }

        try:
            # Create user account
            license_key = ""
            if self.keyauth_data and self.keyauth_data.get("license_key"):
                license_key = str(self.keyauth_data.get("license_key", ""))

            success = self.user_manager.create_user(
                username=username,
                password=password,
                keyauth_key=license_key,
                keyauth_data=self.keyauth_data if self.keyauth_data else {},
            )

            if success:
                # Auto-login the new user
                return self.login_user(username, password)
            else:
                return {
                    "success": False,
                    "error": "Username already exists or registration failed",
                }

        except Exception as e:
            logger.error("Registration error: %s", e)
            return {"success": False, "error": f"Registration error: {str(e)}"}

    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """Login user with username and password.

        Args:
            username: Username for authentication.
            password: Password for authentication.

        Returns:
            Dictionary with 'success' boolean and 'data' or 'error' fields.
        """
        if not username or not password:
            return {
                "success": False,
                "error": "Username and password are required",
            }

        try:
            # Authenticate user
            user_data = self.user_manager.authenticate_user(username, password)

            if user_data:
                # Create session
                session_token = self.user_manager.create_session(
                    user_data["id"],
                    duration_hours=self.config["settings"]["session_duration_hours"],
                )

                if session_token:
                    # Update current session
                    self.current_user = user_data
                    self.current_session = session_token
                    self.is_authenticated = True

                    user_data["session_token"] = session_token

                    return {"success": True, "data": user_data}
                else:
                    return {"success": False, "error": "Failed to create session"}
            else:
                return {"success": False, "error": "Invalid username or password"}

        except Exception as e:
            logger.error("Login error: %s", e)
            return {"success": False, "error": f"Login error: {str(e)}"}

    def validate_session(self, session_token: str) -> Dict[str, Any]:
        """Validate an existing session token.

        Args:
            session_token: Session token to validate.

        Returns:
            Dictionary with 'success' boolean and 'data' or 'error' fields.
        """
        if not session_token:
            return {"success": False, "error": "Session token is required"}

        try:
            user_data = self.user_manager.validate_session(session_token)

            if user_data:
                # Update current session
                self.current_user = user_data
                self.current_session = session_token
                self.is_authenticated = True

                return {"success": True, "data": user_data}
            else:
                return {"success": False, "error": "Invalid or expired session"}

        except Exception as e:
            logger.error("Session validation error: %s", e)
            return {"success": False, "error": f"Session validation error: {str(e)}"}

    def logout_user(self) -> bool:
        """Logout the current user.

        Returns:
            True if logout successful, False otherwise.
        """
        try:
            if self.current_session:
                success = self.user_manager.logout_user(self.current_session)

                # Clear current session
                self.current_user = None
                self.current_session = None
                self.is_authenticated = False

                return success

            return True

        except Exception as e:
            logger.error("Logout error: %s", e)
            return False

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user data.

        Returns:
            Current user data dictionary or None if not authenticated.
        """
        return self.current_user

    def is_user_authenticated(self) -> bool:
        """Check if user is currently authenticated.

        Returns:
            True if user is authenticated, False otherwise.
        """
        return self.is_authenticated

    def check_keyauth_expiry(self) -> Dict[str, Any]:
        """Check if KeyAuth license is still valid.

        Returns:
            Dictionary with validity status and expiry information.
        """
        if not self.current_user:
            return {"valid": False, "error": "No user logged in"}

        try:
            expires_str = self.current_user.get("keyauth_expires", "")
            if not expires_str:
                return {"valid": True, "message": "No expiry date set"}

            # Parse expiry date (assuming Unix timestamp)
            try:
                expires_timestamp = int(expires_str)
                expires_date = datetime.fromtimestamp(expires_timestamp)
                current_date = datetime.now()

                if current_date > expires_date:
                    return {
                        "valid": False,
                        "error": (
                            f"License expired on "
                            f'{expires_date.strftime("%Y-%m-%d %H:%M:%S")}'
                        ),
                    }
                else:
                    days_left = (expires_date - current_date).days
                    return {
                        "valid": True,
                        "expires_date": expires_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "days_left": days_left,
                    }

            except (ValueError, TypeError):
                return {"valid": True, "message": "Could not parse expiry date"}

        except Exception as e:
            logger.error("Error checking expiry: %s", e)
            return {"valid": False, "error": f"Error checking expiry: {str(e)}"}

    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics.

        Returns:
            Dictionary containing user statistics.
        """
        return self.user_manager.get_user_stats()

    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        if self.config["settings"]["auto_cleanup_sessions"]:
            self.user_manager.cleanup_expired_sessions()

    def save_session_to_file(self, session_file: Optional[str] = None) -> bool:
        """Save current session to file for persistence.

        Args:
            session_file: Optional path to session file. If None, uses default path.

        Returns:
            True if session saved successfully, False otherwise.
        """
        if not self.current_session:
            return False

        try:
            if session_file is None:
                session_file = str(
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "current_session.json"
                )

            # Ensure directory exists
            os.makedirs(os.path.dirname(session_file), exist_ok=True)

            session_data = {
                "session_token": self.current_session,
                "user_id": (self.current_user["id"] if self.current_user else None),
                "username": (
                    self.current_user["username"] if self.current_user else None
                ),
                "saved_at": datetime.now().isoformat(),
            }

            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)

            return True

        except Exception as e:
            logger.error("Error saving session: %s", e)
            return False

    def load_session_from_file(
        self, session_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load session from file.

        Args:
            session_file: Optional path to session file. If None, uses default path.

        Returns:
            Dictionary with 'success' boolean and session data or error message.
        """
        try:
            if session_file is None:
                session_file = str(
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "current_session.json"
                )

            if not os.path.exists(session_file):
                return {"success": False, "error": "No saved session found"}

            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            session_token = session_data.get("session_token")
            if not session_token:
                return {"success": False, "error": "Invalid session data"}

            # Validate the session
            result = self.validate_session(session_token)

            if result["success"]:
                return {
                    "success": True,
                    "data": result["data"],
                    "message": "Session restored successfully",
                }
            else:
                # Remove invalid session file
                try:
                    os.remove(session_file)
                except Exception as e:
                    logger.warning("Could not remove invalid session file: %s", e)

                return result

        except Exception as e:
            logger.error("Error loading session: %s", e)
            return {"success": False, "error": f"Error loading session: {str(e)}"}

    def require_authentication(self, func: Callable) -> Callable:
        """Decorator to require authentication for a function.

        Args:
            func: Function to protect with authentication.

        Returns:
            Wrapped function that checks authentication before execution.

        Usage:
            @auth_manager.require_authentication
            def protected_function():
                pass
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.is_authenticated:
                raise Exception("Authentication required")
            return func(*args, **kwargs)

        return wrapper

    def get_authentication_status(self) -> Dict[str, Any]:
        """Get comprehensive authentication status.

        Returns:
            Dictionary containing authentication status information.
        """
        return {
            "is_authenticated": self.is_authenticated,
            "keyauth_verified": self.keyauth_verified,
            "current_user": self.current_user,
            "session_active": self.current_session is not None,
            "keyauth_status": (
                self.check_keyauth_expiry() if self.current_user else None
            ),
        }
