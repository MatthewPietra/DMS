"""User Manager for DMS Authentication.

Handles user credential storage, session management, and database operations
for the DMS authentication system.
"""

import hashlib
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class UserManager:
    """Manages user authentication and database operations."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the user manager.

        Args:
            db_path: Optional path to the database file. If not provided,
                defaults to data/users.db in the project root.
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.db_path = db_path or str(self.project_root / "data" / "users.db")
        self.session_path = str(self.project_root / "data" / "sessions.json")

        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize database
        self._init_database()

        # Load sessions
        self.sessions = self._load_sessions()

    def _init_database(self) -> None:
        """Initialize the user database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create users table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    keyauth_key TEXT,
                    keyauth_username TEXT,
                    keyauth_expires TEXT,
                    keyauth_subscription TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    session_token TEXT
                )
                """
            )

            # Create sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                """
            )

            # Create keyauth_logs table for audit
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS keyauth_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    keyauth_key TEXT,
                    action TEXT,
                    success BOOLEAN,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                """
            )

            conn.commit()

    def _hash_password(
        self, password: str, salt: Optional[str] = None
    ) -> tuple[str, str]:
        """Hash password with salt using PBKDF2.

        Args:
            password: The password to hash.
            salt: Optional salt. If not provided, a new salt will be generated.

        Returns:
            A tuple containing (password_hash, salt).
        """
        if salt is None:
            salt = secrets.token_hex(32)

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100_000,
        )
        return password_hash.hex(), salt

    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash.

        Args:
            password: The password to verify.
            password_hash: The stored password hash.
            salt: The salt used for hashing.

        Returns:
            True if password matches, False otherwise.
        """
        computed_hash, _ = self._hash_password(password, salt)
        return computed_hash == password_hash

    def _generate_session_token(self) -> str:
        """Generate a secure session token.

        Returns:
            A secure random token suitable for session management.
        """
        return secrets.token_urlsafe(32)

    def _load_sessions(self) -> Dict[str, Any]:
        """Load active sessions from file."""
        try:
            if os.path.exists(self.session_path):
                with open(self.session_path, "r") as f:
                    sessions: Dict[str, Any] = json.load(f)
                    return sessions
        except Exception as e:
            print(f"Warning: Could not load sessions: {e}")
        return {}

    def _save_sessions(self) -> None:
        """Save sessions to file."""
        try:
            with open(self.session_path, "w") as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save sessions: {e}")

    def create_user(
        self,
        username: str,
        password: str,
        keyauth_key: Optional[str] = None,
        keyauth_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new user account.

        Args:
            username: The username for the new account.
            password: The password for the new account.
            keyauth_key: Optional KeyAuth key.
            keyauth_data: Optional KeyAuth data dictionary.

        Returns:
            True if user was created successfully, False otherwise.
        """
        try:
            password_hash, salt = self._hash_password(password)
            keyauth_username = keyauth_data.get("username", "") if keyauth_data else ""
            keyauth_expires = keyauth_data.get("expires", "") if keyauth_data else ""
            keyauth_subscription = (
                keyauth_data.get("subscription", "") if keyauth_data else ""
            )
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO users (
                        username, password_hash, salt, keyauth_key,
                        keyauth_username, keyauth_expires, keyauth_subscription
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        password_hash,
                        salt,
                        keyauth_key,
                        keyauth_username,
                        keyauth_expires,
                        keyauth_subscription,
                    ),
                )
                user_id = cursor.lastrowid
                conn.commit()
                if user_id is not None:
                    self._log_keyauth_action(
                        user_id,
                        keyauth_key or "",
                        "user_created",
                        True,
                        f"User {username} created successfully",
                    )
                return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    def authenticate_user(
        self, username: str, password: str
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user with username and password.

        Args:
            username: The username to authenticate.
            password: The password to verify.

        Returns:
            User data dictionary if authentication successful, None otherwise.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, username, password_hash, salt, keyauth_key,
                           keyauth_username, keyauth_expires, keyauth_subscription,
                           is_active
                    FROM users
                    WHERE username = ? AND is_active = 1
                    """,
                    (username,),
                )
                user = cursor.fetchone()
                if user and self._verify_password(password, user[2], user[3]):
                    cursor.execute(
                        """
                        UPDATE users
                        SET last_login = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (user[0],),
                    )
                    conn.commit()
                    return {
                        "id": user[0],
                        "username": user[1],
                        "keyauth_key": user[4],
                        "keyauth_username": user[5],
                        "keyauth_expires": user[6],
                        "keyauth_subscription": user[7],
                        "is_active": user[8],
                    }
                return None
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None

    def create_session(self, user_id: int, duration_hours: int = 24) -> Optional[str]:
        """Create a new session for the user.

        Args:
            user_id: The user ID to create a session for.
            duration_hours: Session duration in hours.

        Returns:
            Session token if successful, None otherwise.
        """
        try:
            session_token = self._generate_session_token()
            expires_at = datetime.now() + timedelta(hours=duration_hours)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE sessions
                    SET is_active = 0
                    WHERE user_id = ? AND is_active = 1
                    """,
                    (user_id,),
                )
                cursor.execute(
                    """
                    INSERT INTO sessions (user_id, session_token, expires_at)
                    VALUES (?, ?, ?)
                    """,
                    (user_id, session_token, expires_at),
                )
                conn.commit()
                self.sessions[session_token] = {
                    "user_id": user_id,
                    "expires_at": expires_at.isoformat(),
                    "created_at": datetime.now().isoformat(),
                }
                self._save_sessions()
                return session_token
        except Exception as e:
            print(f"Error creating session: {e}")
            return None

    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate a session token.

        Args:
            session_token: The session token to validate.

        Returns:
            User data if session is valid, None otherwise.
        """
        try:
            if session_token not in self.sessions:
                return None
            session = self.sessions[session_token]
            expires_at = datetime.fromisoformat(session["expires_at"])
            if datetime.now() > expires_at:
                self._invalidate_session(session_token)
                return None
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, username, keyauth_key, keyauth_username,
                           keyauth_expires, keyauth_subscription, is_active
                    FROM users
                    WHERE id = ? AND is_active = 1
                    """,
                    (session["user_id"],),
                )
                user = cursor.fetchone()
                if user:
                    return {
                        "id": user[0],
                        "username": user[1],
                        "keyauth_key": user[2],
                        "keyauth_username": user[3],
                        "keyauth_expires": user[4],
                        "keyauth_subscription": user[5],
                        "is_active": user[6],
                        "session_token": session_token,
                    }
                return None
        except Exception as e:
            print(f"Error validating session: {e}")
            return None

    def _invalidate_session(self, session_token: str) -> None:
        """Invalidate a session.

        Args:
            session_token: The session token to invalidate.
        """
        try:
            if session_token in self.sessions:
                del self.sessions[session_token]
                self._save_sessions()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE sessions
                    SET is_active = 0
                    WHERE session_token = ?
                    """,
                    (session_token,),
                )
                conn.commit()
        except Exception as e:
            print(f"Error invalidating session: {e}")

    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session.

        Args:
            session_token: The session token to invalidate.

        Returns:
            True if logout successful, False otherwise.
        """
        try:
            self._invalidate_session(session_token)
            return True
        except Exception as e:
            print(f"Error during logout: {e}")
            return False

    def update_keyauth_data(self, user_id: int, keyauth_data: Dict[str, Any]) -> bool:
        """Update KeyAuth data for a user.

        Args:
            user_id: The user ID to update.
            keyauth_data: Dictionary containing KeyAuth data.

        Returns:
            True if update successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE users
                    SET keyauth_username = ?, keyauth_expires = ?,
                        keyauth_subscription = ?
                    WHERE id = ?
                    """,
                    (
                        keyauth_data.get("username", ""),
                        keyauth_data.get("expires", ""),
                        keyauth_data.get("subscription", ""),
                        user_id,
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Error updating KeyAuth data: {e}")
            return False

    def _log_keyauth_action(
        self, user_id: int, keyauth_key: str, action: str, success: bool, message: str
    ) -> None:
        """Log KeyAuth actions for audit purposes.

        Args:
            user_id: The user ID associated with the action.
            keyauth_key: The KeyAuth key used.
            action: The action performed.
            success: Whether the action was successful.
            message: Additional message about the action.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO keyauth_logs
                    (user_id, keyauth_key, action, success, message)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, keyauth_key, action, success, message),
                )
                conn.commit()
        except Exception as e:
            print(f"Error logging KeyAuth action: {e}")

    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics.

        Returns:
            Dictionary containing user statistics.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
                total_users = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM sessions WHERE is_active = 1")
                active_sessions = cursor.fetchone()[0]
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM users
                    WHERE last_login > datetime('now', '-24 hours')
                    """
                )
                recent_logins = cursor.fetchone()[0]
                return {
                    "total_users": total_users,
                    "active_sessions": active_sessions,
                    "recent_logins": recent_logins,
                }
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {"total_users": 0, "active_sessions": 0, "recent_logins": 0}

    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions from both memory and database."""
        try:
            now = datetime.now()
            expired_tokens = []
            for token, session in self.sessions.items():
                expires_at = datetime.fromisoformat(session["expires_at"])
                if now > expires_at:
                    expired_tokens.append(token)
            for token in expired_tokens:
                self._invalidate_session(token)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE sessions
                    SET is_active = 0
                    WHERE expires_at < datetime('now')
                    """
                )
                conn.commit()
        except Exception as e:
            print(f"Error cleaning up sessions: {e}")

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID.

        Args:
            user_id: The user ID to retrieve.

        Returns:
            User data dictionary if found, None otherwise.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, username, keyauth_key, keyauth_username,
                           keyauth_expires, keyauth_subscription, created_at,
                           last_login, is_active
                    FROM users
                    WHERE id = ?
                    """,
                    (user_id,),
                )
                user = cursor.fetchone()
                if user:
                    return {
                        "id": user[0],
                        "username": user[1],
                        "keyauth_key": user[2],
                        "keyauth_username": user[3],
                        "keyauth_expires": user[4],
                        "keyauth_subscription": user[5],
                        "created_at": user[6],
                        "last_login": user[7],
                        "is_active": user[8],
                    }
                return None
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None
