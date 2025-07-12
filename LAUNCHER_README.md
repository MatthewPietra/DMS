# DMS Clean Launcher

A streamlined authentication and launcher system for the Detection Model Suite (DMS).

## Features

✅ **Clean, Simple Design**: Spacious layout with clear sections  
✅ **Two-Step Process**: Login first, then choose interface  
✅ **Smart Registration**: Link KeyAuth key only when creating new accounts  
✅ **Normal Login**: Existing users login with just username/password  
✅ **Modern UI**: Dark theme with intuitive design  
✅ **Session Management**: Secure user sessions with token-based authentication  

## Quick Start

### Option 1: Double-click (Windows)
```
Double-click: launch.bat
```

### Option 2: Command Line
```bash
python launch.py
```

## How It Works

### Step 1: Login or Register
**For Existing Users:**
- Enter username and password
- Click "Login"

**For New Users:**
- Switch to "Register" tab
- Enter username, password, confirm password
- Enter your KeyAuth license key
- Click "Create Account"

### Step 2: Choose Interface
After successful authentication:
- **🖥️ Graphical Interface**: Full-featured visual interface
- **💻 Command Line Interface**: Advanced command-line tools

## Authentication Flow

```
Login/Register → Interface Selection → Launch
```

## Key Improvements

### Clean Design
- **Larger window**: 400x500 pixels for better spacing
- **Bigger inputs**: 45px height for easier interaction
- **Clear sections**: Separated login and interface selection
- **Better typography**: Larger, more readable fonts

### Simplified Process
- **No cramped UI**: Everything has proper spacing
- **Two clear steps**: Authentication, then interface choice
- **KeyAuth only for new accounts**: Existing users don't need to re-enter keys
- **Visual feedback**: Clear status messages and progress indicators

### User Experience
- **Tab-based login/register**: Easy switching between modes
- **Enter key support**: Press Enter to login
- **Auto-focus**: Cursor starts in username field
- **Secure key entry**: KeyAuth keys are hidden by default with show/hide option

## File Structure

```
DMS/
├── launch.py              # Main launcher script
├── launch.bat             # Windows batch launcher
├── src/auth/
│   ├── clean_auth_gui.py      # New clean authentication GUI
│   ├── keyauth_api.py         # KeyAuth API integration
│   ├── user_manager.py        # User management
│   └── auth_manager.py        # Authentication manager
└── src/gui/
    └── main_window.py         # Main GUI application
```

## Requirements

- Python 3.8+
- PySide6 (for GUI)
- KeyAuth license key (for new accounts only)
- Internet connection (for license verification during registration)

## Account Management

### New Account Creation
1. **Username & Password**: Choose your credentials
2. **KeyAuth License**: Enter your license key to verify access
3. **Account Created**: Your KeyAuth key is linked to your account

### Existing Account Login
1. **Username & Password**: Use your existing credentials
2. **No KeyAuth Required**: Your key is already linked to your account
3. **Automatic Login**: Quick access to interface selection

## Troubleshooting

### "Authentication failed"
- Check your username/password combination
- For new accounts, verify your KeyAuth license key
- Ensure internet connection for license verification

### "GUI not available"
- Install GUI dependencies: `pip install -r requirements/requirements_gui.txt`
- Ensure PySide6 is installed: `pip install PySide6`

### "Interface selection not showing"
- Ensure you completed the login process
- Check that authentication was successful
- Try logging in again

## Support

For issues or questions:
1. Check the logs in `DMS/logs/`
2. Verify all dependencies are installed
3. For new accounts, ensure your KeyAuth license is valid

---

**Made with ❤️ for the DMS community** 