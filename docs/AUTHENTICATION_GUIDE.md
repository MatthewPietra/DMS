# DMS Authentication System Guide

## Overview

The DMS (Detection Model Suite) now includes a comprehensive authentication system based on KeyAuth integration. This system provides secure user access control and licensing verification before allowing access to the DMS application features.

## Authentication Flow

### 1. KeyAuth License Verification
- Users must first enter a valid KeyAuth license key
- The system verifies the license with the KeyAuth server
- License validation includes expiry date and subscription status

### 2. User Account Creation
- After successful KeyAuth verification, users can create a local account
- Username and password are stored securely in a local SQLite database
- Passwords are hashed using PBKDF2 with salt for maximum security

### 3. Session Management
- Successful authentication creates a secure session token
- Sessions are automatically saved for future use
- Session expiry is configurable (default: 24 hours)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Valid KeyAuth license key
- Required dependencies (automatically installed)

### Installation

1. **Using the Unified Launcher (Recommended)**
   ```bash
   # Windows
   launch.bat
   
   # Linux/Mac
   ./launch.sh
   ```

2. **Manual Installation**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   
   # Install authentication dependencies
   pip install -r requirements/requirements_auth.txt
   
   # Launch unified DMS launcher
   python unified_launcher.py
   ```

### First Time Setup

1. **Launch the Application**
   - Run `launch.bat` (Windows) or `./launch.sh` (Linux/Mac)
   - Choose GUI or CLI mode on first launch
   - The authentication window will appear

2. **Enter KeyAuth License**
   - Enter your valid KeyAuth license key
   - Click "Verify License"
   - Wait for verification to complete

3. **Create User Account**
   - After successful verification, you'll be prompted to create an account
   - Choose a username and secure password
   - Click "Create Account"

4. **Access DMS**
   - After registration, you'll be automatically logged in
   - The main DMS application will launch

## Configuration

### KeyAuth Configuration
The KeyAuth settings are stored in `config/keyauth_config.json`:

```json
{
    "application": {
        "name": "DMS - Detection Model Suite",
        "ownerid": "your_owner_id",
        "secret": "your_secret_key",
        "version": "1.0.0"
    },
    "settings": {
        "session_duration_hours": 24,
        "auto_cleanup_sessions": true,
        "require_keyauth_for_registration": true
    },
    "security": {
        "password_min_length": 8,
        "password_require_uppercase": true,
        "password_require_lowercase": true,
        "password_require_numbers": true
    }
}
```

### Database Configuration
User data is stored in `data/users.db` (SQLite database):
- User accounts (username, hashed password, KeyAuth data)
- Session tokens and expiry dates
- Audit logs for KeyAuth actions

## Security Features

### Password Security
- PBKDF2 hashing with 100,000 iterations
- Unique salt for each password
- Secure password requirements (configurable)

### Session Security
- Cryptographically secure session tokens
- Automatic session expiry
- Session invalidation on logout

### KeyAuth Integration
- Encrypted communication with KeyAuth servers
- Hardware ID verification
- License expiry checking
- Subscription status validation

## Usage

### Daily Usage
1. **Launch Application**
   - Use the launch scripts or run `python unified_launcher.py`
   - If you have a valid saved session, you'll be logged in automatically

2. **Manual Login**
   - If no saved session exists, enter your username and password
   - Your session will be saved for future use

3. **Access DMS Features**
   - Once authenticated, you have full access to all DMS features
   - The authentication system runs transparently in the background

### Account Management
- **Change Password**: Currently requires database access (future GUI feature)
- **Session Management**: Sessions are automatically managed
- **License Renewal**: Re-verify your KeyAuth license when it expires

## Troubleshooting

### Common Issues

1. **"KeyAuth API not initialized" Error**
   - Check your internet connection
   - Verify your KeyAuth credentials in the config file
   - Ensure the KeyAuth service is accessible

2. **"License verification failed" Error**
   - Verify your license key is correct
   - Check if your license has expired
   - Ensure your hardware ID is authorized

3. **"Username already exists" Error**
   - Choose a different username
   - Or login with your existing credentials

4. **GUI Not Loading**
   - Ensure PyQt5 is installed: `pip install PyQt5`
   - Try alternative GUI frameworks (PyQt6, PySide6)
   - Check the console for specific error messages

5. **Session Issues**
   - Delete `data/current_session.json` to force re-authentication
   - Check system clock for time synchronization issues

### Debug Mode
Enable debug logging by setting the environment variable:
```bash
# Windows
set DMS_DEBUG=1

# Linux/Mac
export DMS_DEBUG=1
```

## API Reference

### AuthenticationManager
Main class for handling authentication flow:

```python
from auth.auth_manager import AuthenticationManager

auth_manager = AuthenticationManager()

# Verify KeyAuth license
result = auth_manager.verify_keyauth_license("your_license_key")

# Register new user
result = auth_manager.register_user("username", "password")

# Login user
result = auth_manager.login_user("username", "password")

# Check authentication status
status = auth_manager.get_authentication_status()
```

### UserManager
Database operations for user management:

```python
from auth.user_manager import UserManager

user_manager = UserManager()

# Create user
success = user_manager.create_user("username", "password", "keyauth_key")

# Authenticate user
user_data = user_manager.authenticate_user("username", "password")

# Create session
session_token = user_manager.create_session(user_id)
```

## File Structure

```
DMS/
├── unified_launcher.py       # Unified launcher with KeyAuth
├── launch.bat                # Windows launch script
├── launch.sh                 # Linux/Mac launch script
├── config/
│   └── keyauth_config.json   # KeyAuth configuration
├── data/
│   ├── users.db             # User database
│   ├── sessions.json        # Active sessions
│   └── current_session.json # Current user session
├── src/auth/
│   ├── __init__.py
│   ├── keyauth_api.py       # KeyAuth API client
│   ├── user_manager.py      # User database manager
│   ├── auth_manager.py      # Authentication flow manager
│   └── auth_gui.py          # Authentication GUI
├── requirements/
│   └── requirements_auth.txt # Authentication dependencies
└── docs/
    └── AUTHENTICATION_GUIDE.md # This guide
```

## License and Support

This authentication system is integrated into the DMS project and follows the same licensing terms. For support:

1. Check this documentation
2. Review the troubleshooting section
3. Check the project's issue tracker
4. Contact the development team

## Version History

- **v1.0.0**: Initial KeyAuth integration
  - KeyAuth license verification
  - User registration and login
  - Session management
  - SQLite database storage
  - Modern GUI interface

## Future Enhancements

- Web-based authentication interface
- Multi-factor authentication
- Advanced user role management
- License usage analytics
- Automatic license renewal
- Password reset functionality 