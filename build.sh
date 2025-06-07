#!/bin/bash

# This script builds a distributable binary for the Super Cuts application.
# It should be run from the root of the repository.

# --- Configuration ---
SCRIPT_NAME="process_video.py"
BINARY_NAME="supercuts"
DIST_DIR="dist"

# --- Colors for Output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Functions ---
function check_command() {
    if ! command -v $1 &> /dev/null
    then
        echo -e "${YELLOW}Command not found: $1. Please install it.${NC}"
        exit 1
    fi
}

function build_binary() {
    echo -e "${GREEN}Starting build process...${NC}"

    # 1. Check for Python and Pip
    check_command python
    check_command pip

    # 2. Install/Update Dependencies
    echo "Installing/updating dependencies from requirements.txt..."
    python -m pip install --upgrade -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Failed to install dependencies.${NC}"
        exit 1
    fi

    # 3. Check for PyInstaller
    if ! python -m pip show pyinstaller &> /dev/null; then
        echo -e "${YELLOW}PyInstaller is not installed. Please add it to requirements.txt.${NC}"
        exit 1
    fi

    # 4. Run PyInstaller
    echo "Running PyInstaller to create the binary..."
    pyinstaller \
        --name $BINARY_NAME \
        --onefile \
        --console \
        --distpath $DIST_DIR \
        --workpath build \
        --specpath . \
        $SCRIPT_NAME

    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}PyInstaller failed to build the binary.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Build successful!${NC}"
    echo -e "The binary is located at: ${GREEN}${DIST_DIR}/${BINARY_NAME}${NC}"
    echo "You can now distribute the '${DIST_DIR}' directory."
    
    # Clean up PyInstaller build artifacts
    rm -rf build
    rm -f ${BINARY_NAME}.spec
}

# --- Main Execution ---
build_binary 