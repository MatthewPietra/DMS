#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for DMS Authentication System

This script tests the authentication components without requiring GUI interaction.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def test_imports() -> None:
    """Test that all authentication modules can be imported."""
    print("Testing imports...")

    try:
        from auth.keyauth_api import (  # noqa: F401
            KeyAuthAPI,
            KeyAuthEncryption,
            KeyAuthHWID,
        )

        print("✅ KeyAuth API imported successfully")
    except ImportError as e:
        print(f"❌ KeyAuth API import failed: {e}")
        assert False, f"KeyAuth API import failed: {e}"

    try:
        from auth.user_manager import UserManager  # noqa: F401

        print("✅ User Manager imported successfully")
    except ImportError as e:
        print(f"❌ User Manager import failed: {e}")
        assert False, f"User Manager import failed: {e}"

    try:
        from auth.auth_manager import AuthenticationManager  # noqa: F401

        print("✅ Authentication Manager imported successfully")
    except ImportError as e:
        print(f"❌ Authentication Manager import failed: {e}")
        assert False, f"Authentication Manager import failed: {e}"

    # If all imports succeed
    assert True


def test_encryption() -> None:
    """Test the encryption utilities."""
    print("\nTesting encryption...")

    try:
        from auth.keyauth_api import KeyAuthEncryption

        # Test encryption/decryption
        message = "test message"
        key = "test_key"
        iv = "test_iv"

        encrypted = KeyAuthEncryption.encrypt(message, key, iv)
        decrypted = KeyAuthEncryption.decrypt(encrypted, key, iv)

        assert decrypted == message, "Decrypted message does not match original"
        print("✅ Encryption/decryption test passed")

    except Exception as e:
        print(f"❌ Encryption test failed: {e}")
        assert False, f"Encryption test failed: {e}"


def test_hwid() -> None:
    """Test hardware ID generation."""
    print("\nTesting hardware ID generation...")

    try:
        from auth.keyauth_api import KeyAuthHWID

        hwid = KeyAuthHWID.get_hwid()

        assert hwid and len(hwid) > 0, "HWID was not generated"
        print(f"✅ Hardware ID generated: {hwid[:20]}...")

    except Exception as e:
        print(f"❌ Hardware ID test failed: {e}")
        assert False, f"Hardware ID test failed: {e}"


def test_user_manager() -> None:
    """Test user manager database operations."""
    print("\nTesting user manager...")

    try:
        from auth.user_manager import UserManager

        # Use test database
        test_db = project_root / "data" / "test_users.db"
        user_manager = UserManager(str(test_db))

        # Test user creation
        success = user_manager.create_user(
            username="testuser", password="testpass123", keyauth_key="test_key"
        )

        assert success, "User creation failed"
        print("✅ User creation test passed")

        # Test authentication
        user_data = user_manager.authenticate_user("testuser", "testpass123")

        assert user_data, "User authentication failed"
        print("✅ User authentication test passed")

        # Test session creation
        session_token = user_manager.create_session(user_data["id"])

        assert session_token, "Session creation failed"
        print("✅ Session creation test passed")

        # Cleanup test database
        try:
            os.remove(test_db)
        except OSError:
            pass

    except Exception as e:
        print(f"❌ User manager test failed: {e}")
        assert False, f"User manager test failed: {e}"


def test_config_loading() -> None:
    """Test configuration loading."""
    print("\nTesting configuration loading...")

    try:
        from auth.auth_manager import AuthenticationManager

        auth_manager = AuthenticationManager()

        assert (
            auth_manager.config and "application" in auth_manager.config
        ), "Config missing 'application' key"
        print("✅ Configuration loading test passed")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        assert False, f"Configuration test failed: {e}"


def main() -> int:
    """Run all tests."""
    print("=" * 50)
    print("DMS Authentication System Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_encryption,
        test_hwid,
        test_user_manager,
        test_config_loading,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Authentication system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
