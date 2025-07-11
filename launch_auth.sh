#!/bin/bash

echo "==============================================="
echo "DMS - Detection Model Suite"
echo "Authenticated Launcher"
echo "==============================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Install authentication dependencies
echo "Installing authentication dependencies..."
pip install -q PyQt5 pycryptodome requests wmi

# Launch authenticated DMS
echo
echo "Starting DMS with authentication..."
echo
python3 auth_launcher.py

# Deactivate virtual environment
deactivate

echo
echo "DMS session ended." 