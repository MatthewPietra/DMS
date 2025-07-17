#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KeyAuth API Integration Module.

Provides secure authentication services using the KeyAuth API.
Handles license verification, user authentication, and session management.
"""

import binascii
import hashlib
import json
import logging
import os
import platform
import secrets
import sys
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..utils.secure_subprocess import get_system_info

try:
    import requests
except ImportError:
    raise ImportError(
        "The 'requests' module is required for authentication but is not "
        "installed. Please install it with: pip install requests"
    )

try:
    import wmi
except ImportError:
    wmi = None

try:
    import win32security
except ImportError:
    win32security = None


# Runtime imports
CRYPTOGRAPHY_AVAILABLE = False
try:
    import cryptography.hazmat.primitives.padding as padding
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


logger = logging.getLogger(__name__)


class KeyAuthEncryption:
    """Encryption utilities for KeyAuth API communication using modern cryptography."""

    @staticmethod
    def encrypt_string(plain_text: bytes, key: bytes, iv: bytes) -> bytes:
        """Encrypt string using AES CBC mode with modern cryptography library.

        Args:
            plain_text: The plaintext to encrypt.
            key: The encryption key.
            iv: The initialization vector.

        Returns:
            The encrypted data as hex-encoded bytes.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library is required for encryption")

        # Pad the plaintext to be a multiple of 16 bytes
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plain_text)
        padded_data += padder.finalize()

        # Create cipher and encrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return binascii.hexlify(ciphertext)

    @staticmethod
    def decrypt_string(cipher_text: bytes, key: bytes, iv: bytes) -> bytes:
        """Decrypt string using AES CBC mode with a cryptography library.

        Args:
            cipher_text: The encrypted data as hex-encoded bytes.
            key: The decryption key.
            iv: The initialization vector.

        Returns:
            The decrypted plaintext.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library is required for decryption")

        cipher_text = binascii.unhexlify(cipher_text)

        # Create cipher and decrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(cipher_text) + decryptor.finalize()

        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext)
        plaintext += unpadder.finalize()

        return plaintext

    @staticmethod
    def encrypt(message: str, enc_key: str, iv: str) -> str:
        """Encrypt message with given key and IV using SHA-256.

        Args:
            message: The message to encrypt.
            enc_key: The encryption key.
            iv: The initialization vector.

        Returns:
            The encrypted message as a hex string.

        Raises:
            Exception: If encryption fails.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library is required for encryption")

        try:
            # Generate key and IV using SHA-256
            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(enc_key.encode())
            _key = digest.finalize()[:32]

            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(iv.encode())
            _iv = digest.finalize()[:16]

            result = KeyAuthEncryption.encrypt_string(message.encode(), _key, _iv)
            return result.decode()
        except Exception as e:
            raise Exception(f"Encryption failed: {e}")

    @staticmethod
    def decrypt(message: str, enc_key: str, iv: str) -> str:
        """Decrypt message with given key and IV using SHA-256.

        Args:
            message: The encrypted message as a hex string.
            enc_key: The decryption key.
            iv: The initialization vector.

        Returns:
            The decrypted message.

        Raises:
            Exception: If decryption fails.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library is required for decryption")

        try:
            # Generate key and IV using SHA-256
            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(enc_key.encode())
            _key = digest.finalize()[:32]

            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(iv.encode())
            _iv = digest.finalize()[:16]

            result = KeyAuthEncryption.decrypt_string(message.encode(), _key, _iv)
            return result.decode()
        except Exception as e:
            raise Exception(f"Decryption failed: {e}")


class KeyAuthHWID:
    """Hardware ID utilities for KeyAuth."""

    @staticmethod
    def get_hwid() -> str:
        """Get hardware ID based on platform.

        Returns:
            A unique hardware identifier string.
        """
        if platform.system() == "Linux":
            try:
                with open("/etc/machine-id") as f:
                    return f.read().strip()
            except Exception:
                return KeyAuthHWID._fallback_hwid()

        elif platform.system() == "Windows":
            try:
                if wmi is not None:
                    c = wmi.WMI()
                    for disk in c.Win32_DiskDrive():
                        if "PHYSICALDRIVE" in disk.DeviceID:
                            return disk.PNPDeviceID
                else:
                    # Fallback for Windows without wmi
                    try:
                        if win32security is not None:
                            winuser = os.getlogin()
                            sid = win32security.LookupAccountName(None, winuser)[0]
                            return win32security.ConvertSidToStringSid(sid)
                        else:
                            return KeyAuthHWID._fallback_hwid()
                    except Exception:
                        return KeyAuthHWID._fallback_hwid()
            except Exception:
                try:
                    if win32security is not None:
                        winuser = os.getlogin()
                        sid = win32security.LookupAccountName(None, winuser)[0]
                        return win32security.ConvertSidToStringSid(sid)
                    else:
                        return KeyAuthHWID._fallback_hwid()
                except Exception:
                    return KeyAuthHWID._fallback_hwid()

        elif platform.system() == "Darwin":
            try:
                # Use secure subprocess utility
                success, stdout, stderr = get_system_info()
                if success:
                    for line in stdout.split("\n"):
                        if "IOPlatformSerialNumber" in line:
                            serial = line.split("=", 1)[1].replace(" ", "").strip()
                            return serial.strip('"')
                return KeyAuthHWID._fallback_hwid()
            except Exception:
                return KeyAuthHWID._fallback_hwid()

        return KeyAuthHWID._fallback_hwid()

    @staticmethod
    def _fallback_hwid() -> str:
        """Generate fallback HWID based on system information using secure SHA-256.

        Returns:
            A fallback hardware identifier string.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback without cryptography
            system_info = (
                f"{platform.system()}-{platform.node()}-{platform.processor()}"
            )
            return hashlib.sha256(system_info.encode()).hexdigest()

        system_info = f"{platform.system()}-{platform.node()}-{platform.processor()}"
        # Use SHA-256 instead of insecure MD5
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback without cryptography
            return hashlib.sha256(system_info.encode()).hexdigest()
        else:
            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(system_info.encode())
            return digest.finalize().hex()


class KeyAuthAPI:
    """KeyAuth API client for DMS authentication."""

    def __init__(
        self,
        name: str,
        ownerid: str,
        secret: str,
        version: str,
        hash_to_check: str = "",
    ) -> None:
        """Initialize KeyAuth API client.

        Args:
            name: The application name.
            ownerid: The owner ID.
            secret: The application secret.
            version: The application version.
            hash_to_check: Optional file hash for verification.
        """
        self.name = name
        self.ownerid = ownerid
        self.secret = secret
        self.version = version
        self.hash_to_check = hash_to_check or self._get_file_hash()

        self.sessionid = ""
        self.enckey = ""
        self.initialized = False

        # User and app data
        self.user_data = self.UserData()
        self.app_data = self.AppData()

        # Initialize the API
        self.init()

    def _get_file_hash(self) -> str:
        """Get hash of the current executable for verification using secure SHA-256.

        Returns:
            The SHA-256 hash of the current executable file.
        """
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                # Fallback without cryptography
                with open(sys.argv[0], "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()

            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            with open(sys.argv[0], "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    digest.update(chunk)
            return digest.finalize().hex()
        except Exception:
            return ""

    def init(self) -> bool:
        """Initialize the KeyAuth session.

        Returns:
            True if initialization was successful.

        Raises:
            Exception: If initialization fails.
        """
        try:
            if self.sessionid:
                return True

            # Use SHA-256 and key generation
            if not CRYPTOGRAPHY_AVAILABLE:
                # Fallback without cryptography
                init_iv = hashlib.sha256(secrets.token_bytes(8)).hexdigest()
                self.enckey = hashlib.sha256(secrets.token_bytes(8)).hexdigest()
            else:
                digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
                digest.update(str(uuid4())[:8].encode())
                init_iv = digest.finalize().hex()

                digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
                digest.update(str(uuid4())[:8].encode())
                self.enckey = digest.finalize().hex()

            post_data = {
                "type": binascii.hexlify("init".encode()),
                "ver": KeyAuthEncryption.encrypt(self.version, self.secret, init_iv),
                "hash": self.hash_to_check,
                "enckey": KeyAuthEncryption.encrypt(self.enckey, self.secret, init_iv),
                "name": binascii.hexlify(self.name.encode()),
                "ownerid": binascii.hexlify(self.ownerid.encode()),
                "init_iv": init_iv,
            }

            response = self._do_request(post_data)

            if response == "KeyAuth_Invalid":
                raise Exception("The application doesn't exist")

            response = KeyAuthEncryption.decrypt(response, self.secret, init_iv)
            json_response = json.loads(response)

            if json_response["message"] == "invalidver":
                raise Exception("Invalid version - please update the application")

            if not json_response["success"]:
                raise Exception(json_response["message"])

            self.sessionid = json_response["sessionid"]
            self.initialized = True
            self._load_app_data(json_response["appinfo"])

            return True

        except Exception as e:
            raise Exception(f"KeyAuth initialization failed: {e}")

    def license(self, key: str, hwid: Optional[str] = None) -> bool:
        """Verify license key.

        Args:
            key: The license key to verify.
            hwid: Optional hardware ID. If None, uses current system HWID.

        Returns:
            True if license verification was successful.

        Raises:
            Exception: If license verification fails.
        """
        self._check_init()

        if hwid is None:
            hwid = KeyAuthHWID.get_hwid()

        # Use SHA-256 for IV generation
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback without cryptography
            init_iv = hashlib.sha256(str(uuid4())[:8].encode()).hexdigest()
        else:
            digest = hashes.Hash(
                hashes.SHA256(),
                backend=default_backend(),
            )
            digest.update(str(uuid4())[:8].encode())
            init_iv = digest.finalize().hex()

        post_data = {
            "type": binascii.hexlify("license".encode()),
            "key": KeyAuthEncryption.encrypt(key, self.enckey, init_iv),
            "hwid": KeyAuthEncryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv,
        }

        response = self._do_request(post_data)
        response = KeyAuthEncryption.decrypt(response, self.enckey, init_iv)
        json_response = json.loads(response)

        if json_response["success"]:
            self._load_user_data(json_response["info"])
            return True
        else:
            raise Exception(json_response["message"])

    def register(
        self, username: str, password: str, license_key: str, hwid: Optional[str] = None
    ) -> bool:
        """Register a new user with KeyAuth.

        Args:
            username: The username for registration.
            password: The password for registration.
            license_key: The license key for registration.
            hwid: Optional hardware ID. If None, uses current system HWID.

        Returns:
            True if registration was successful.

        Raises:
            Exception: If registration fails.
        """
        self._check_init()

        if hwid is None:
            hwid = KeyAuthHWID.get_hwid()

        # Use SHA-256 for IV generation
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback without cryptography
            init_iv = hashlib.sha256(str(uuid4())[:8].encode()).hexdigest()
        else:
            digest = hashes.Hash(
                hashes.SHA256(),
                backend=default_backend(),
            )
            digest.update(str(uuid4())[:8].encode())
            init_iv = digest.finalize().hex()

        post_data = {
            "type": binascii.hexlify("register".encode()),
            "username": KeyAuthEncryption.encrypt(username, self.enckey, init_iv),
            "pass": KeyAuthEncryption.encrypt(password, self.enckey, init_iv),
            "key": KeyAuthEncryption.encrypt(license_key, self.enckey, init_iv),
            "hwid": KeyAuthEncryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv,
        }

        response = self._do_request(post_data)
        response = KeyAuthEncryption.decrypt(response, self.enckey, init_iv)
        json_response = json.loads(response)

        if json_response["success"]:
            self._load_user_data(json_response["info"])
            return True
        else:
            raise Exception(json_response["message"])

    def login(self, username: str, password: str, hwid: Optional[str] = None) -> bool:
        """Login with username and password.

        Args:
            username: The username for login.
            password: The password for login.
            hwid: Optional hardware ID. If None, uses current system HWID.

        Returns:
            True if login was successful.

        Raises:
            Exception: If login fails.
        """
        self._check_init()

        if hwid is None:
            hwid = KeyAuthHWID.get_hwid()

        # Use SHA-256 for IV generation
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback without cryptography
            init_iv = hashlib.sha256(str(uuid4())[:8].encode()).hexdigest()
        else:
            digest = hashes.Hash(
                hashes.SHA256(),
                backend=default_backend(),
            )
            digest.update(str(uuid4())[:8].encode())
            init_iv = digest.finalize().hex()

        post_data = {
            "type": binascii.hexlify("login".encode()),
            "username": KeyAuthEncryption.encrypt(username, self.enckey, init_iv),
            "pass": KeyAuthEncryption.encrypt(password, self.enckey, init_iv),
            "hwid": KeyAuthEncryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv,
        }

        response = self._do_request(post_data)
        response = KeyAuthEncryption.decrypt(response, self.enckey, init_iv)
        json_response = json.loads(response)

        if json_response["success"]:
            self._load_user_data(json_response["info"])
            return True
        else:
            raise Exception(json_response["message"])

    def _check_init(self) -> None:
        """Check if API is initialized.

        Raises:
            Exception: If API is not initialized.
        """
        if not self.initialized:
            raise Exception("KeyAuth API not initialized")

    def _do_request(self, post_data: Dict[str, Any]) -> str:
        """Execute HTTP request to KeyAuth API.

        Args:
            post_data: The POST data to send.

        Returns:
            The response text from the API.

        Raises:
            Exception: If the network request fails.
        """
        try:
            response = requests.post(
                "https://keyauth.win/api/1.0/",
                data=post_data,
                timeout=30,
            )
            return response.text
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network request failed: {e}")

    def _load_app_data(self, data: Dict[str, Any]) -> None:
        """Load application data from KeyAuth response.

        Args:
            data: The application data dictionary from KeyAuth.
        """
        self.app_data.numUsers = data.get("numUsers", "")
        self.app_data.numKeys = data.get("numKeys", "")
        self.app_data.app_ver = data.get("version", "")
        self.app_data.customer_panel = data.get("customerPanelLink", "")
        self.app_data.onlineUsers = data.get("numOnlineUsers", "")

    def _load_user_data(self, data: Dict[str, Any]) -> None:
        """Load user data from KeyAuth response.

        Args:
            data: The user data dictionary from KeyAuth.
        """
        self.user_data.username = data.get("username", "")
        self.user_data.ip = data.get("ip", "")
        self.user_data.hwid = data.get("hwid", "")
        self.user_data.createdate = data.get("createdate", "")
        self.user_data.lastlogin = data.get("lastlogin", "")

        # Handle subscriptions
        subscriptions = data.get("subscriptions", [])
        if subscriptions:
            self.user_data.expires = subscriptions[0].get("expiry", "")
            self.user_data.subscription = subscriptions[0].get("subscription", "")
        self.user_data.subscriptions = subscriptions

    class UserData:
        """Container for user data."""

        def __init__(self) -> None:
            """Initialize user data container."""
            self.username = ""
            self.ip = ""
            self.hwid = ""
            self.expires = ""
            self.createdate = ""
            self.lastlogin = ""
            self.subscription = ""
            self.subscriptions: List[Dict[str, Any]] = []

    class AppData:
        """Container for application data."""

        def __init__(self) -> None:
            """Initialize application data container."""
            self.numUsers = ""
            self.numKeys = ""
            self.app_ver = ""
            self.customer_panel = ""
            self.onlineUsers = ""
