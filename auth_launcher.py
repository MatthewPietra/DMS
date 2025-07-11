#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS Authenticated Launcher

Integrates KeyAuth authentication into the DMS launcher system.
This serves as the main entry point that requires authentication
before accessing the DMS application features.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import authentication system
try:
    from auth.auth_manager import AuthenticationManager
    from auth.auth_gui import show_authentication_dialog
    AUTH_AVAILABLE = True
except ImportError as e:
    print(f"Authentication system not available: {e}")
    AUTH_AVAILABLE = False

# Import main launcher
try:
    from main import CentralLauncher
    LAUNCHER_AVAILABLE = True
except ImportError as e:
    print(f"Main launcher not available: {e}")
    LAUNCHER_AVAILABLE = False


class AuthenticatedLauncher:
    """
    Main launcher with integrated KeyAuth authentication.
    
    Authentication Flow:
    1. Check for existing valid session
    2. If no session, show authentication dialog
    3. KeyAuth license verification
    4. User registration/login
    5. Launch main DMS application
    """
    
    def __init__(self):
        """Initialize the authenticated launcher."""
        self.auth_manager = AuthenticationManager() if AUTH_AVAILABLE else None
        self.main_launcher = None
        self.authenticated_user = None
        
        # Setup paths
        self.project_root = project_root
        self.session_file = self.project_root / "data" / "current_session.json"
    
    def run(self):
        """Main entry point for the authenticated launcher."""
        print("=" * 60)
        print("DMS - Detection Model Suite")
        print("Authenticated Launcher v1.0.0")
        print("=" * 60)
        print()
        
        if not AUTH_AVAILABLE:
            print("⚠️  Authentication system not available.")
            print("Please install required dependencies:")
            print("  pip install PyQt5 pycryptodome requests wmi")
            print()
            return self._fallback_launch()
        
        # Check for existing session
        if self._check_existing_session():
            print("✅ Valid session found, launching DMS...")
            return self._launch_main_application()
        
        # Show authentication dialog
        print("🔐 Authentication required...")
        if self._show_authentication():
            print("✅ Authentication successful!")
            return self._launch_main_application()
        else:
            print("❌ Authentication failed or cancelled.")
            return 1
    
    def _check_existing_session(self) -> bool:
        """Check for existing valid session."""
        try:
            if not self.auth_manager:
                return False
            
            result = self.auth_manager.load_session_from_file()
            
            if result['success']:
                self.authenticated_user = result['data']
                print(f"Welcome back, {self.authenticated_user['username']}!")
                
                # Check KeyAuth expiry
                expiry_check = self.auth_manager.check_keyauth_expiry()
                if expiry_check['valid']:
                    if 'days_left' in expiry_check:
                        print(f"License expires in {expiry_check['days_left']} days")
                    return True
                else:
                    print(f"⚠️  License issue: {expiry_check['error']}")
                    return False
            
            return False
        
        except Exception as e:
            print(f"Error checking session: {e}")
            return False
    
    def _show_authentication(self) -> bool:
        """Show authentication dialog."""
        try:
            # Import Qt here to avoid import errors if not available
            from PyQt5.QtWidgets import QApplication
            
            # Create Qt application if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            # Show authentication dialog
            result = show_authentication_dialog()
            
            if result and result.get('user_data'):
                self.authenticated_user = result['user_data']
                
                # Save session for future use
                if self.auth_manager:
                    self.auth_manager.current_user = self.authenticated_user
                    self.auth_manager.current_session = self.authenticated_user.get('session_token')
                    self.auth_manager.is_authenticated = True
                    self.auth_manager.save_session_to_file()
                
                return True
            
            return False
        
        except Exception as e:
            print(f"Error showing authentication dialog: {e}")
            return False
    
    def _launch_main_application(self) -> int:
        """Launch the main DMS application."""
        try:
            if not LAUNCHER_AVAILABLE:
                print("⚠️  Main launcher not available.")
                return self._fallback_launch()
            
            # Initialize main launcher
            self.main_launcher = CentralLauncher()
            
            # Add authentication info to launcher
            if self.authenticated_user:
                self.main_launcher.authenticated_user = self.authenticated_user
                self.main_launcher.auth_manager = self.auth_manager
            
            # Show main menu
            self.main_launcher.show_main_menu()
            
            return 0
        
        except Exception as e:
            print(f"Error launching main application: {e}")
            return 1
    
    def _fallback_launch(self) -> int:
        """Fallback launch without authentication."""
        print("Launching DMS without authentication...")
        print()
        
        try:
            if LAUNCHER_AVAILABLE:
                launcher = CentralLauncher()
                launcher.show_main_menu()
                return 0
            else:
                print("Main launcher not available. Please check installation.")
                return 1
        
        except Exception as e:
            print(f"Error in fallback launch: {e}")
            return 1
    
    def cleanup(self):
        """Cleanup resources."""
        if self.auth_manager:
            self.auth_manager.cleanup_expired_sessions()


def main():
    """Main entry point."""
    try:
        launcher = AuthenticatedLauncher()
        return launcher.run()
    
    except KeyboardInterrupt:
        print("\n⚠️  Launcher interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        return 1
    
    finally:
        # Cleanup
        try:
            if 'launcher' in locals():
                launcher.cleanup()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main()) 