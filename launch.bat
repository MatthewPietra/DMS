@echo off
REM DMS Unified Launcher - Windows
REM This script launches the unified DMS launcher with KeyAuth authentication
REM Supports both GUI and CLI modes with user preference management

title DMS Unified Launcher

REM Set colors for output
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                          DMS Unified Launcher                               ║
echo ║                     Detection Model Suite v2.0.0                           ║
echo ║                                                                              ║
echo ║  🚀 Integrated KeyAuth Authentication    🔧 GUI/CLI Mode Selection          ║
echo ║  📦 Automatic Dependency Management     ✨ Cross-Platform Support          ║
echo ║  🔍 Hardware Auto-Detection             🎯 One-Click Experience            ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo %NC%

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Python is not installed or not in PATH%NC%
    echo.
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%✅ Python %PYTHON_VERSION% detected%NC%

REM Check if we're in the correct directory
if not exist "unified_launcher.py" (
    echo %RED%❌ unified_launcher.py not found in current directory%NC%
    echo Please run this script from the DMS root directory
    echo.
    pause
    exit /b 1
)

REM Set up virtual environment path
set "VENV_PATH=venv"
set "PYTHON_EXE=python"
set "PIP_EXE=pip"

REM Check if virtual environment exists
if exist "%VENV_PATH%" (
    echo %GREEN%✅ Virtual environment found%NC%
    set "PYTHON_EXE=%VENV_PATH%\Scripts\python.exe"
    set "PIP_EXE=%VENV_PATH%\Scripts\pip.exe"
) else (
    echo %YELLOW%⚠️  No virtual environment found%NC%
    echo Creating virtual environment...
    
    python -m venv %VENV_PATH%
    if %errorlevel% neq 0 (
        echo %RED%❌ Failed to create virtual environment%NC%
        echo Proceeding with system Python...
    ) else (
        echo %GREEN%✅ Virtual environment created%NC%
        set "PYTHON_EXE=%VENV_PATH%\Scripts\python.exe"
        set "PIP_EXE=%VENV_PATH%\Scripts\pip.exe"
        
        REM Upgrade pip in virtual environment
        echo Upgrading pip...
        "%PIP_EXE%" install --upgrade pip >nul 2>&1
    )
)

REM Check for required base dependencies
echo %BLUE%🔍 Checking dependencies...%NC%
"%PYTHON_EXE%" -c "import sys, json, pathlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Missing basic Python dependencies%NC%
    echo Please ensure Python installation is complete
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "config" mkdir config

REM Launch the unified launcher
echo %BLUE%🚀 Starting DMS Unified Launcher...%NC%
echo.

"%PYTHON_EXE%" unified_launcher.py

REM Handle exit codes
if %errorlevel% equ 0 (
    echo.
    echo %GREEN%✅ DMS launcher completed successfully%NC%
) else (
    echo.
    echo %RED%❌ DMS launcher exited with error code %errorlevel%%NC%
    echo.
    echo If you encounter issues:
    echo   1. Check that all dependencies are installed
    echo   2. Verify your KeyAuth license key
    echo   3. Check the logs directory for error details
    echo   4. Contact support if the problem persists
)

echo.
echo %BLUE%Press any key to exit...%NC%
pause >nul 