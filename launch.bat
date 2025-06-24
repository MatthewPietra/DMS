@echo off
REM DMS Launcher - Windows Batch Script
REM This script creates a dedicated Python 3.10 virtual environment and launches DMS
REM This environment is separate from any existing venv/venv310 directories

title DMS Launcher

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════════╗
echo ║                              DMS Launcher                                       ║
echo ║                      Complete Object Detection Pipeline                          ║
echo ║                      Dedicated Virtual Environment                               ║
echo ╚══════════════════════════════════════════════════════════════════════════════════╝
echo.
echo Starting DMS with dedicated virtual environment...
echo.

REM Set dedicated DMS virtual environment path
set VENV_PATH=dms_venv310
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe
set PIP_EXE=%VENV_PATH%\Scripts\pip.exe

REM Check if Python 3.10 is available
python3.10 --version >nul 2>&1
set PYTHON310_AVAILABLE=%errorlevel%

python --version 2>nul | findstr "3.10" >nul
set CURRENT_PYTHON310=%errorlevel%

REM Determine which Python to use for creating venv
if %PYTHON310_AVAILABLE% == 0 (
    set PYTHON_CMD=python3.10
    echo Found Python 3.10 specifically
) else if %CURRENT_PYTHON310% == 0 (
    set PYTHON_CMD=python
    echo Found Python 3.10 as default
) else (
    echo WARNING: Python 3.10 not found, using available Python version
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: No Python installation found
        echo Please install Python 3.10 from https://python.org
        echo.
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
)

REM Display Python version
echo Detected Python version:
echo Running: %PYTHON_CMD% --version
"%PYTHON_CMD%" --version
echo.

REM Check if dedicated DMS virtual environment exists and is working
if exist "%VENV_PATH%\Scripts\python.exe" (
    REM Test if the virtual environment Python actually works
    %PYTHON_EXE% --version >nul 2>&1
    if errorlevel 1 (
        echo DMS virtual environment is broken, recreating: %VENV_PATH%
        rmdir /s /q "%VENV_PATH%" 2>nul
        goto create_venv
    ) else (
        echo Using existing DMS virtual environment: %VENV_PATH%
        echo This is separate from any existing venv/venv310 environments
    )
) else (
    :create_venv
    echo Creating dedicated DMS virtual environment: %VENV_PATH%
    echo This environment is separate from existing venv/venv310 directories
    echo This may take a few moments...
    echo Command: %PYTHON_CMD% -m venv "%VENV_PATH%"
    %PYTHON_CMD% -m venv "%VENV_PATH%"
    
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

REM Verify DMS virtual environment Python version
echo DMS virtual environment Python version:
echo Running: %PYTHON_EXE% --version
"%PYTHON_EXE%" --version
echo.

REM Check if launcher exists
if not exist "launcher.py" (
    echo ERROR: launcher.py not found in current directory
    echo Please make sure you're in the correct directory
    echo.
    pause
    exit /b 1
)

REM Launch the studio using dedicated virtual environment
echo.
echo Launching DMS with dedicated virtual environment...
echo Dedicated Virtual Environment: %VENV_PATH%
echo This is separate from any existing venv/venv310 environments
echo Running: %PYTHON_EXE% launcher.py
echo.
"%PYTHON_EXE%" launcher.py

REM Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
) 