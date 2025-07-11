#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for DMS Authentication System

This script tests the authentication components without requiring GUI interaction.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all authentication modules can be imported."""
    print("Testing imports...")
    
    try:
        from auth.keyauth_api import KeyAuthAPI, KeyAuthEncryption, KeyAuthHWID
        print("✅ KeyAuth API imported successfully")
    except ImportError as e:
        print(f"❌ KeyAuth API import failed: {e}")
        return False
    
    try:
        from auth.user_manager import UserManager
        print("✅ User Manager imported successfully")
    except ImportError as e:
        print(f"❌ User Manager import failed: {e}")
        return False
    
    try:
        from auth.auth_manager import AuthenticationManager
        print("✅ Authentication Manager imported successfully")
    except ImportError as e:
        print(f"❌ Authentication Manager import failed: {e}")
        return False
    
    return True

def test_encryption():
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
        
        if decrypted == message:
            print("✅ Encryption/decryption test passed")
            return True
        else:
            print("❌ Encryption/decryption test failed")
            return False
    
    except Exception as e:
        print(f"❌ Encryption test failed: {e}")
        return False

def test_hwid():
    """Test hardware ID generation."""
    print("\nTesting hardware ID generation...")
    
    try:
        from auth.keyauth_api import KeyAuthHWID
        
        hwid = KeyAuthHWID.get_hwid()
        
        if hwid and len(hwid) > 0:
            print(f"✅ Hardware ID generated: {hwid[:20]}...")
            return True
        else:
            print("❌ Hardware ID generation failed")
            return False
    
    except Exception as e:
        print(f"❌ Hardware ID test failed: {e}")
        return False

def test_user_manager():
    """Test user manager database operations."""
    print("\nTesting user manager...")
    
    try:
        from auth.user_manager import UserManager
        
        # Use test database
        test_db = project_root / "data" / "test_users.db"
        user_manager = UserManager(str(test_db))
        
        # Test user creation
        success = user_manager.create_user(
            username="testuser",
            password="testpass123",
            keyauth_key="test_key"
        )
        
        if success:
            print("✅ User creation test passed")
        else:
            print("❌ User creation test failed")
            return False
        
        # Test authentication
        user_data = user_manager.authenticate_user("testuser", "testpass123")
        
        if user_data:
            print("✅ User authentication test passed")
        else:
            print("❌ User authentication test failed")
            return False
        
        # Test session creation
        session_token = user_manager.create_session(user_data['id'])
        
        if session_token:
            print("✅ Session creation test passed")
        else:
            print("❌ Session creation test failed")
            return False
        
        # Cleanup test database
        try:
            os.remove(test_db)
        except:
            pass
        
        return True
    
    except Exception as e:
        print(f"❌ User manager test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from auth.auth_manager import AuthenticationManager
        
        auth_manager = AuthenticationManager()
        
        if auth_manager.config and 'application' in auth_manager.config:
            print("✅ Configuration loading test passed")
            return True
        else:
            print("❌ Configuration loading test failed")
            return False
    
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("DMS Authentication System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_encryption,
        test_hwid,
        test_user_manager,
        test_config_loading
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