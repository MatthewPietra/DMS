#!/bin/bash
# DMS Unified Launcher - Linux/Unix
# This script launches the unified DMS launcher with KeyAuth authentication
# Supports both GUI and CLI modes with user preference management

# Set colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                          DMS Unified Launcher                               ║"
echo "║                     Detection Model Suite v2.0.0                           ║"
echo "║                                                                              ║"
echo "║  🚀 Integrated KeyAuth Authentication    🔧 GUI/CLI Mode Selection          ║"
echo "║  📦 Automatic Dependency Management     ✨ Cross-Platform Support          ║"
echo "║  🔍 Hardware Auto-Detection             🎯 One-Click Experience            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Python version
get_python_version() {
    python3 --version 2>/dev/null | cut -d' ' -f2 || echo "unknown"
}

# Check if Python 3 is available
if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed or not in PATH${NC}"
    echo ""
    echo "Please install Python 3.8 or higher:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "  CentOS/RHEL:   sudo yum install python3 python3-pip"
    echo "  Fedora:        sudo dnf install python3 python3-pip"
    echo "  Arch:          sudo pacman -S python python-pip"
    echo ""
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(get_python_version)
echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"

# Check if we're in the correct directory
if [ ! -f "unified_launcher.py" ]; then
    echo -e "${RED}❌ unified_launcher.py not found in current directory${NC}"
    echo "Please run this script from the DMS root directory"
    echo ""
    exit 1
fi

# Set up virtual environment path
VENV_PATH="venv"
PYTHON_EXE="python3"
PIP_EXE="pip3"

# Check if virtual environment exists
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}✅ Virtual environment found${NC}"
    PYTHON_EXE="$VENV_PATH/bin/python"
    PIP_EXE="$VENV_PATH/bin/pip"
else
    echo -e "${YELLOW}⚠️  No virtual environment found${NC}"
    echo "Creating virtual environment..."
    
    # Check if venv module is available
    if ! python3 -m venv --help >/dev/null 2>&1; then
        echo -e "${RED}❌ Python venv module not available${NC}"
        echo "Please install python3-venv:"
        echo "  Ubuntu/Debian: sudo apt-get install python3-venv"
        echo "  CentOS/RHEL:   sudo yum install python3-venv"
        echo "Proceeding with system Python..."
    else
        python3 -m venv "$VENV_PATH"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Virtual environment created${NC}"
            PYTHON_EXE="$VENV_PATH/bin/python"
            PIP_EXE="$VENV_PATH/bin/pip"
            
            # Upgrade pip in virtual environment
            echo "Upgrading pip..."
            "$PIP_EXE" install --upgrade pip >/dev/null 2>&1
        else
            echo -e "${RED}❌ Failed to create virtual environment${NC}"
            echo "Proceeding with system Python..."
        fi
    fi
fi

# Check for required base dependencies
echo -e "${BLUE}🔍 Checking dependencies...${NC}"
"$PYTHON_EXE" -c "import sys, json, pathlib" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Missing basic Python dependencies${NC}"
    echo "Please ensure Python installation is complete"
    exit 1
fi

# Create necessary directories
mkdir -p data logs config

# Set executable permissions for Python files
chmod +x unified_launcher.py 2>/dev/null || true

# Launch the unified launcher
echo -e "${BLUE}🚀 Starting DMS Unified Launcher...${NC}"
echo ""

"$PYTHON_EXE" unified_launcher.py

# Handle exit codes
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ DMS launcher completed successfully${NC}"
else
    echo ""
    echo -e "${RED}❌ DMS launcher exited with error code $EXIT_CODE${NC}"
    echo ""
    echo "If you encounter issues:"
    echo "  1. Check that all dependencies are installed"
    echo "  2. Verify your KeyAuth license key"
    echo "  3. Check the logs directory for error details"
    echo "  4. Contact support if the problem persists"
fi

echo ""
echo -e "${BLUE}Press Enter to exit...${NC}"
read -r 