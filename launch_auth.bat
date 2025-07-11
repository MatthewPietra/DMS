@echo off
echo ===============================================
echo DMS - Detection Model Suite
echo Authenticated Launcher
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install authentication dependencies
echo Installing authentication dependencies...
pip install -q PyQt5 pycryptodome requests wmi

REM Launch authenticated DMS
echo.
echo Starting DMS with authentication...
echo.
python auth_launcher.py

REM Deactivate virtual environment
deactivate

echo.
echo DMS session ended.
pause 