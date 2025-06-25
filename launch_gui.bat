@echo off
REM DMS GUI Launcher - Windows Batch Script
REM This script creates a dedicated Python 3.10 virtual environment and launches DMS GUI
REM This environment is separate from any existing venv/venv310 directories

title DMS GUI Launcher

echo.
echo ===============================
echo        DMS GUI Launcher
echo   Complete Object Detection Pipeline
echo   Modern Graphical User Interface
echo ===============================
echo.
echo Starting DMS GUI with dedicated virtual environment...
echo.

REM Set dedicated DMS virtual environment path
set VENV_PATH=dms_venv310
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe
set PIP_EXE=%VENV_PATH%\Scripts\pip.exe

REM Check if Python 3.10 is available via different methods
echo Checking Python installations...

REM Method 1: Check if 'python' is 3.10
python --version 2>nul | findstr /C:"3.10" >nul
set PYTHON_IS_310=%errorlevel%

REM Method 2: Check if 'py -3.10' works
py -3.10 --version >nul 2>&1
set PY_310_AVAILABLE=%errorlevel%

REM Method 3: Check if 'python3.10' works
python3.10 --version >nul 2>&1
set PYTHON310_AVAILABLE=%errorlevel%

REM Determine which Python to use for creating venv
if %PYTHON_IS_310% == 0 (
    set PYTHON_CMD=python
    echo Found Python 3.10 via 'python' command
    goto continue_script
) else if %PY_310_AVAILABLE% == 0 (
    set PYTHON_CMD=py -3.10
    echo Found Python 3.10 via 'py -3.10' command
    goto continue_script
) else if %PYTHON310_AVAILABLE% == 0 (
    set PYTHON_CMD=python3.10
    echo Found Python 3.10 via 'python3.10' command
    goto continue_script
) else (
    echo.
    echo ===============================
    echo      PYTHON 3.10 REQUIRED
    echo ===============================
    echo.
    echo WARNING: Python 3.10 not found, but is required for DMS compatibility
    echo.
    echo Available Python versions:
    echo - python: 
    python --version 2>nul
    echo - py: 
    py --version 2>nul
    echo.
    echo ===============================
    echo      INSTALLATION GUIDE
    echo ===============================
    echo.
    echo To install Python 3.10:
    echo 1. Go to: https://www.python.org/downloads/
    echo 2. Download Python 3.10.x (latest 3.10 version)
    echo 3. Run the installer
    echo 4. IMPORTANT: Check "Add Python 3.10 to PATH"
    echo 5. Install for all users (recommended)
    echo.
    echo After installation:
    echo - Close this window
    echo - Open a new command prompt
    echo - Run: python3.10 --version
    echo - Then run this launcher again
    echo.
    echo ===============================
    echo      WHY PYTHON 3.10?
    echo ===============================
    echo.
    echo DMS requires Python 3.10 for optimal compatibility with:
    echo - PyTorch and CUDA libraries
    echo - DirectML support for AMD GPUs
    echo - Specific package versions and dependencies
    echo - Training pipeline stability
    echo.
    pause
    exit /b 1
)

:continue_script
REM Display Python version
echo Detected Python version:
echo Running: %PYTHON_CMD% --version
"%PYTHON_CMD%" --version
echo.

REM Check if dedicated DMS virtual environment exists and check its Python version
if exist "%VENV_PATH%\Scripts\python.exe" (
    echo Checking existing DMS virtual environment...
    echo Running: %PYTHON_EXE% --version
    "%PYTHON_EXE%" --version
    echo.
    
    REM Check if the venv Python is 3.10
    "%PYTHON_EXE%" --version 2>nul | findstr /C:"3.10" >nul
    if errorlevel 1 (
        echo Existing virtual environment was created with wrong Python version
        echo Recreating virtual environment with Python 3.10...
        rmdir /s /q "%VENV_PATH%" 2>nul
        goto create_venv
    ) else (
        echo Using existing DMS virtual environment: %VENV_PATH% (Python 3.10)
        echo This is separate from any existing venv/venv310 environments
        goto install_deps
    )
) else (
    :create_venv
    echo Creating dedicated DMS virtual environment: %VENV_PATH%
    echo This environment is separate from existing venv/venv310 directories
    echo This may take a few moments...
    echo Command: %PYTHON_CMD% -m venv "%VENV_PATH%"
    "%PYTHON_CMD%" -m venv "%VENV_PATH%"
    
    if errorlevel 1 (
        echo ERROR: Failed to create DMS virtual environment
        echo Make sure you have the 'venv' module available
        echo.
        pause
        exit /b 1
    )
    
    echo DMS virtual environment created successfully!
    echo.
    
    REM Upgrade pip in the virtual environment
    echo Upgrading pip in DMS virtual environment...
    echo Running: %PIP_EXE% install --upgrade pip
    "%PIP_EXE%" install --upgrade pip
    echo.
)

:install_deps
REM Verify DMS virtual environment Python version
echo DMS virtual environment Python version:
echo Running: %PYTHON_EXE% --version
"%PYTHON_EXE%" --version
echo.

REM Check if GUI launcher exists
if not exist "gui_launcher.py" (
    echo ERROR: gui_launcher.py not found in current directory
    echo Please make sure you're in the correct directory
    echo.
    pause
    exit /b 1
)

REM Install GUI dependencies
echo Installing GUI dependencies...
echo Running: %PIP_EXE% install PySide6 PyYAML psutil
"%PIP_EXE%" install PySide6 PyYAML psutil

REM Launch the GUI studio using dedicated virtual environment
echo.
echo Launching DMS GUI with dedicated virtual environment...
echo Dedicated Virtual Environment: %VENV_PATH%
echo This is separate from any existing venv/venv310 environments
echo Running: %PYTHON_EXE% gui_launcher.py
echo.
"%PYTHON_EXE%" gui_launcher.py

REM Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
) else (
    echo.
    echo DMS GUI has been closed. Press any key to exit...
    pause >nul
) 