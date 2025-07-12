"""
Authentication Manager for DMS

Coordinates the authentication flow between KeyAuth verification,
user management, and session handling.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from .keyauth_api import KeyAuthAPI
from .user_manager import UserManager


class AuthenticationManager:
    """
    Manages the complete authentication flow for DMS.

    Flow:
    1. KeyAuth license verification
    2. User registration/login
    3. Session management
    4. Access control
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the authentication manager."""
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

        # Initialize components
        self.user_manager = UserManager()
        self.keyauth_api = None

        # Current session
        self.current_user = None
        self.current_session = None

        # Authentication state
        self.is_authenticated = False
        self.keyauth_verified = False
        self.keyauth_data = None

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return str(
            Path(__file__).parent.parent.parent / "config" / "keyauth_config.json"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "application": {
                "name": "DMS - Detection Model Suite",
                "ownerid": "JR8hfS3d4v",
                "secret": "d7c57798279632a99a4429bc07ece9bf6070d5d2229c1d03387c7bc6d0b94c10",
                "version": "1.0.0",
            },
            "settings": {
                "session_duration_hours": 24,
                "auto_cleanup_sessions": True,
                "require_keyauth_for_registration": True,
            },
        }

    def verify_keyauth_license(self, license_key: str) -> Dict[str, Any]:
        """
        Verify KeyAuth license key.

        Returns:
            Dict with 'success' boolean and 'data' or 'error' fields
        """
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
            return {"success": False, "error": str(e)}

    def register_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Register a new user after KeyAuth verification.

        Returns:
            Dict with 'success' boolean and 'data' or 'error' fields
        """
        if not self.keyauth_verified:
            return {"success": False, "error": "KeyAuth license must be verified first"}

        try:
            # Create user account
            success = self.user_manager.create_user(
                username=username,
                password=password,
                keyauth_key=self.keyauth_data.get("license_key"),
                keyauth_data=self.keyauth_data,
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
            return {"success": False, "error": f"Registration error: {str(e)}"}

    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Login user with username and password.

        Returns:
            Dict with 'success' boolean and 'data' or 'error' fields
        """
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
            return {"success": False, "error": f"Login error: {str(e)}"}

    def validate_session(self, session_token: str) -> Dict[str, Any]:
        """
        Validate an existing session token.

        Returns:
            Dict with 'success' boolean and 'data' or 'error' fields
        """
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
            return {"success": False, "error": f"Session validation error: {str(e)}"}

    def logout_user(self) -> bool:
        """Logout the current user."""
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
            print(f"Logout error: {e}")
            return False

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user data."""
        return self.current_user

    def is_user_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        return self.is_authenticated

    def check_keyauth_expiry(self) -> Dict[str, Any]:
        """Check if KeyAuth license is still valid."""
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
                        "error": f'License expired on {expires_date.strftime("%Y-%m-%d %H:%M:%S")}',
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
            return {"valid": False, "error": f"Error checking expiry: {str(e)}"}

    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        return self.user_manager.get_user_stats()

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        if self.config["settings"]["auto_cleanup_sessions"]:
            self.user_manager.cleanup_expired_sessions()

    def save_session_to_file(self, session_file: str = None) -> bool:
        """Save current session to file for persistence."""
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
                "user_id": self.current_user["id"] if self.current_user else None,
                "username": (
                    self.current_user["username"] if self.current_user else None
                ),
                "saved_at": datetime.now().isoformat(),
            }

            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    def load_session_from_file(self, session_file: str = None) -> Dict[str, Any]:
        """Load session from file."""
        try:
            if session_file is None:
                session_file = str(
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "current_session.json"
                )

            if not os.path.exists(session_file):
                return {"success": False, "error": "No saved session found"}

            with open(session_file, "r") as f:
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
                    print(f"Warning: Could not remove invalid session file: {e}")

                return result

        except Exception as e:
            return {"success": False, "error": f"Error loading session: {str(e)}"}

    def require_authentication(self, func: Callable) -> Callable:
        """
        Decorator to require authentication for a function.

        Usage:
            @auth_manager.require_authentication
            def protected_function():
                pass
        """

        def wrapper(*args, **kwargs):
            if not self.is_authenticated:
                raise Exception("Authentication required")
            return func(*args, **kwargs)

        return wrapper

    def get_authentication_status(self) -> Dict[str, Any]:
        """Get comprehensive authentication status."""
        return {
            "is_authenticated": self.is_authenticated,
            "keyauth_verified": self.keyauth_verified,
            "current_user": self.current_user,
            "session_active": self.current_session is not None,
            "keyauth_status": (
                self.check_keyauth_expiry() if self.current_user else None
            ),
        }
