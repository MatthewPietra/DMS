@echo off
title DMS Launcher
echo Starting DMS (Detection Model Suite)...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Change to the script directory
cd /d "%~dp0"

REM Run the unified launcher
python scripts/unified_launcher.py

REM Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo Press any key to exit...
    pause >nul
) 