#!/bin/bash
# DMS Launcher - Unix/Linux Shell Script
# This script creates a dedicated Python 3.10 virtual environment and launches DMS
# This environment is separate from any existing venv/venv310 directories

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
echo "║                              DMS Launcher                                       ║"
echo "║                      Complete Object Detection Pipeline                          ║"
echo "║                      Dedicated Virtual Environment                               ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "Starting DMS with dedicated virtual environment..."
echo

# Set dedicated DMS virtual environment path
VENV_PATH="dms_venv310"
PYTHON_EXE="$VENV_PATH/bin/python"
PIP_EXE="$VENV_PATH/bin/pip"

# Check if Python 3.10 is available
PYTHON_CMD=""
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}Found Python 3.10 specifically${NC}"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -o "3\.[0-9]\+")
    if [[ "$PYTHON_VERSION" == "3.10" ]]; then
        PYTHON_CMD="python3"
        echo -e "${GREEN}Found Python 3.10 as default${NC}"
    else
        PYTHON_CMD="python3"
        echo -e "${YELLOW}WARNING: Python 3.10 not found, using Python $PYTHON_VERSION${NC}"
    fi
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo -e "${YELLOW}WARNING: Using default python command${NC}"
else
    echo -e "${RED}ERROR: No Python installation found${NC}"
    echo "Please install Python 3.10+ using your package manager"
    echo "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv python3.10-pip"
    echo "  CentOS/RHEL: sudo dnf install python3.10 python3.10-pip"
    echo "  macOS: brew install python@3.10"
    echo ""
    exit 1
fi

# Display Python version
echo "Detected Python version:"
echo "Running: $PYTHON_CMD --version"
$PYTHON_CMD --version
echo

# Check if dedicated DMS virtual environment exists and is working
if [ -f "$PYTHON_EXE" ]; then
    # Test if the virtual environment Python actually works
    if ! $PYTHON_EXE --version &> /dev/null; then
        echo -e "${YELLOW}DMS virtual environment is broken, recreating: $VENV_PATH${NC}"
        rm -rf "$VENV_PATH" 2>/dev/null || true
    else
        echo -e "${GREEN}Using existing DMS virtual environment: $VENV_PATH${NC}"
        echo "This is separate from any existing venv/venv310 environments"
    fi
fi

if [ ! -f "$PYTHON_EXE" ]; then
    echo -e "${BLUE}Creating dedicated DMS virtual environment: $VENV_PATH${NC}"
    echo "This environment is separate from existing venv/venv310 directories"
    echo "This may take a few moments..."
    echo "Command: $PYTHON_CMD -m venv $VENV_PATH"
    
    if ! $PYTHON_CMD -m venv "$VENV_PATH"; then
        echo -e "${RED}ERROR: Failed to create DMS virtual environment${NC}"
        echo "Make sure you have the 'venv' module available"
        echo "  Ubuntu/Debian: sudo apt install python3.10-venv"
        echo "  CentOS/RHEL: python3.10 -m ensurepip --upgrade"
        echo ""
        exit 1
    fi
    
    echo -e "${GREEN}DMS virtual environment created successfully!${NC}"
    echo ""
    
    # Upgrade pip in the virtual environment
    echo "Upgrading pip in DMS virtual environment..."
    echo "Running: $PIP_EXE install --upgrade pip"
    "$PIP_EXE" install --upgrade pip
    echo ""
fi

# Verify DMS virtual environment Python version
echo "DMS virtual environment Python version:"
echo "Running: $PYTHON_EXE --version"
"$PYTHON_EXE" --version
echo

# Check if launcher exists
if [ ! -f "launcher.py" ]; then
    echo -e "${RED}ERROR: launcher.py not found in current directory${NC}"
    echo "Please make sure you're in the correct directory"
    exit 1
fi

# Make sure we have execute permissions on this script
chmod +x "$0" 2>/dev/null || true

# Launch the studio using dedicated virtual environment
echo
echo -e "${BLUE}Launching DMS with dedicated virtual environment...${NC}"
echo -e "${GREEN}Dedicated Virtual Environment: $VENV_PATH${NC}"
echo "This is separate from any existing venv/venv310 environments"
echo "Running: $PYTHON_EXE launcher.py"
echo

"$PYTHON_EXE" launcher.py

# Check exit status
if [ $? -ne 0 ]; then
    echo
    echo -e "${RED}An error occurred during execution${NC}"
    echo "Check the output above for details"
    exit 1
fi

echo
echo -e "${GREEN}DMS session completed${NC}" 