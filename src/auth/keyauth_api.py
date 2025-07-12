"""
KeyAuth API Implementation for DMS

Adapted from NeuralAim's KeyAuth integration to provide secure authentication
and license verification for the DMS application.
"""

import os
import sys
import json
import time
import hashlib
import platform
import subprocess
import binascii
from uuid import uuid4
from pathlib import Path
from typing import Optional, Dict, Any

# Import dependency manager to ensure dependencies are available
try:
    from .dependency_manager import ensure_auth_dependencies
    # Ensure authentication dependencies are available
    ensure_auth_dependencies()
except ImportError:
    # If dependency manager is not available, try to import dependencies directly
    pass

# Import Crypto modules after ensuring dependencies
try:
    from Crypto.Cipher import AES
    from Crypto.Hash import SHA256
    from Crypto.Util.Padding import pad, unpad
except ImportError:
    # If Crypto modules are not available, provide a helpful error
    raise ImportError(
        "The 'pycryptodome' module is required for authentication but is not installed. "
        "Please run the authentication dependency installer or contact support."
    )

# Import requests after ensuring dependencies
try:
    import requests
except ImportError:
    # If requests is still not available, provide a helpful error
    raise ImportError(
        "The 'requests' module is required for authentication but is not installed. "
        "Please run the authentication dependency installer or contact support."
    )

# Import wmi only on Windows
if platform.system() == "Windows":
    try:
        import wmi
    except ImportError:
        wmi = None
else:
    wmi = None

# Import win32security only on Windows
if platform.system() == "Windows":
    try:
        import win32security
    except ImportError:
        win32security = None
else:
    win32security = None


class KeyAuthEncryption:
    """Encryption utilities for KeyAuth API communication."""
    
    @staticmethod
    def encrypt_string(plain_text: bytes, key: bytes, iv: bytes) -> bytes:
        """Encrypt string using AES CBC mode."""
        plain_text = pad(plain_text, 16)
        aes_instance = AES.new(key, AES.MODE_CBC, iv)
        raw_out = aes_instance.encrypt(plain_text)
        return binascii.hexlify(raw_out)

    @staticmethod
    def decrypt_string(cipher_text: bytes, key: bytes, iv: bytes) -> bytes:
        """Decrypt string using AES CBC mode."""
        cipher_text = binascii.unhexlify(cipher_text)
        aes_instance = AES.new(key, AES.MODE_CBC, iv)
        cipher_text = aes_instance.decrypt(cipher_text)
        return unpad(cipher_text, 16)

    @staticmethod
    def encrypt(message: str, enc_key: str, iv: str) -> str:
        """Encrypt message with given key and IV."""
        try:
            _key = SHA256.new(enc_key.encode()).hexdigest()[:32]
            _iv = SHA256.new(iv.encode()).hexdigest()[:16]
            return KeyAuthEncryption.encrypt_string(
                message.encode(), _key.encode(), _iv.encode()
            ).decode()
        except Exception as e:
            raise Exception(f"Encryption failed: {e}")

    @staticmethod
    def decrypt(message: str, enc_key: str, iv: str) -> str:
        """Decrypt message with given key and IV."""
        try:
            _key = SHA256.new(enc_key.encode()).hexdigest()[:32]
            _iv = SHA256.new(iv.encode()).hexdigest()[:16]
            return KeyAuthEncryption.decrypt_string(
                message.encode(), _key.encode(), _iv.encode()
            ).decode()
        except Exception as e:
            raise Exception(f"Decryption failed: {e}")


class KeyAuthHWID:
    """Hardware ID utilities for KeyAuth."""
    
    @staticmethod
    def get_hwid() -> str:
        """Get hardware ID based on platform."""
        if platform.system() == "Linux":
            try:
                with open("/etc/machine-id") as f:
                    return f.read().strip()
            except:
                return KeyAuthHWID._fallback_hwid()
        
        elif platform.system() == 'Windows':
            try:
                if wmi is not None:
                    c = wmi.WMI()
                    for disk in c.Win32_DiskDrive():
                        if 'PHYSICALDRIVE' in disk.DeviceID:
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
                    except:
                        return KeyAuthHWID._fallback_hwid()
            except:
                try:
                    if win32security is not None:
                        winuser = os.getlogin()
                        sid = win32security.LookupAccountName(None, winuser)[0]
                        return win32security.ConvertSidToStringSid(sid)
                    else:
                        return KeyAuthHWID._fallback_hwid()
                except:
                    return KeyAuthHWID._fallback_hwid()
        
        elif platform.system() == 'Darwin':
            try:
                output = subprocess.Popen(
                    "ioreg -l | grep IOPlatformSerialNumber", 
                    stdout=subprocess.PIPE, 
                    shell=True
                ).communicate()[0]
                serial = output.decode().split('=', 1)[1].replace(' ', '')
                return serial[1:-2]
            except:
                return KeyAuthHWID._fallback_hwid()
        
        return KeyAuthHWID._fallback_hwid()
    
    @staticmethod
    def _fallback_hwid() -> str:
        """Generate fallback HWID based on system information."""
        system_info = f"{platform.system()}-{platform.node()}-{platform.processor()}"
        return hashlib.md5(system_info.encode()).hexdigest()


class KeyAuthAPI:
    """KeyAuth API client for DMS authentication."""
    
    def __init__(self, name: str, ownerid: str, secret: str, version: str, hash_to_check: str = ""):
        """Initialize KeyAuth API client."""
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
        """Get hash of the current executable for verification."""
        try:
            md5_hash = hashlib.md5()
            with open(sys.argv[0], "rb") as f:
                md5_hash.update(f.read())
            return md5_hash.hexdigest()
        except:
            return ""
    
    def init(self) -> bool:
        """Initialize the KeyAuth session."""
        try:
            if self.sessionid:
                return True
            
            init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()
            self.enckey = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

            post_data = {
                "type": binascii.hexlify("init".encode()),
                "ver": KeyAuthEncryption.encrypt(self.version, self.secret, init_iv),
                "hash": self.hash_to_check,
                "enckey": KeyAuthEncryption.encrypt(self.enckey, self.secret, init_iv),
                "name": binascii.hexlify(self.name.encode()),
                "ownerid": binascii.hexlify(self.ownerid.encode()),
                "init_iv": init_iv
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
        """Verify license key."""
        self._check_init()
        
        if hwid is None:
            hwid = KeyAuthHWID.get_hwid()

        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("license".encode()),
            "key": KeyAuthEncryption.encrypt(key, self.enckey, init_iv),
            "hwid": KeyAuthEncryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self._do_request(post_data)
        response = KeyAuthEncryption.decrypt(response, self.enckey, init_iv)
        json_response = json.loads(response)

        if json_response["success"]:
            self._load_user_data(json_response["info"])
            return True
        else:
            raise Exception(json_response["message"])
    
    def register(self, username: str, password: str, license_key: str, hwid: Optional[str] = None) -> bool:
        """Register a new user with KeyAuth."""
        self._check_init()
        
        if hwid is None:
            hwid = KeyAuthHWID.get_hwid()

        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("register".encode()),
            "username": KeyAuthEncryption.encrypt(username, self.enckey, init_iv),
            "pass": KeyAuthEncryption.encrypt(password, self.enckey, init_iv),
            "key": KeyAuthEncryption.encrypt(license_key, self.enckey, init_iv),
            "hwid": KeyAuthEncryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
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
        """Login with username and password."""
        self._check_init()
        
        if hwid is None:
            hwid = KeyAuthHWID.get_hwid()

        init_iv = SHA256.new(str(uuid4())[:8].encode()).hexdigest()

        post_data = {
            "type": binascii.hexlify("login".encode()),
            "username": KeyAuthEncryption.encrypt(username, self.enckey, init_iv),
            "pass": KeyAuthEncryption.encrypt(password, self.enckey, init_iv),
            "hwid": KeyAuthEncryption.encrypt(hwid, self.enckey, init_iv),
            "sessionid": binascii.hexlify(self.sessionid.encode()),
            "name": binascii.hexlify(self.name.encode()),
            "ownerid": binascii.hexlify(self.ownerid.encode()),
            "init_iv": init_iv
        }

        response = self._do_request(post_data)
        response = KeyAuthEncryption.decrypt(response, self.enckey, init_iv)
        json_response = json.loads(response)

        if json_response["success"]:
            self._load_user_data(json_response["info"])
            return True
        else:
            raise Exception(json_response["message"])
    
    def _check_init(self):
        """Check if API is initialized."""
        if not self.initialized:
            raise Exception("KeyAuth API not initialized")
    
    def _do_request(self, post_data: Dict[str, Any]) -> str:
        """Execute HTTP request to KeyAuth API."""
        try:
            response = requests.post(
                "https://keyauth.win/api/1.0/", 
                data=post_data, 
                timeout=30
            )
            return response.text
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network request failed: {e}")
    
    def _load_app_data(self, data: Dict[str, Any]):
        """Load application data from KeyAuth response."""
        self.app_data.numUsers = data.get("numUsers", "")
        self.app_data.numKeys = data.get("numKeys", "")
        self.app_data.app_ver = data.get("version", "")
        self.app_data.customer_panel = data.get("customerPanelLink", "")
        self.app_data.onlineUsers = data.get("numOnlineUsers", "")
    
    def _load_user_data(self, data: Dict[str, Any]):
        """Load user data from KeyAuth response."""
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
        def __init__(self):
            self.username = ""
            self.ip = ""
            self.hwid = ""
            self.expires = ""
            self.createdate = ""
            self.lastlogin = ""
            self.subscription = ""
            self.subscriptions = []
    
    class AppData:
        """Container for application data."""
        def __init__(self):
            self.numUsers = ""
            self.numKeys = ""
            self.app_ver = ""
            self.customer_panel = ""
            self.onlineUsers = "" 