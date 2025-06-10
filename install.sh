#!/bin/bash

# This script installs the 'supercuts' binary to /usr/local/bin.
# It needs to be run with sudo privileges to write to that directory.

# --- Configuration ---
BINARY_NAME="supercuts"
INSTALL_DIR="/usr/local/bin"
REPO_URL="https://github.com/yourbuddyconner/super-cuts"

# --- Platform Detection ---
OS_TYPE=$(uname -s)
ARCH=$(uname -m)

case "$OS_TYPE" in
    Linux*)     
        case "$ARCH" in
            x86_64) ASSET_NAME="supercuts-linux-x86_64";;
            aarch64) ASSET_NAME="supercuts-linux-arm64";;
            *)      echo "Error: Unsupported Linux architecture '$ARCH'. Only x86_64 and aarch64 are supported."; exit 1;;
        esac
        ;;
    Darwin*)    
        case "$ARCH" in
            arm64)  ASSET_NAME="supercuts-macos-arm64";;
            x86_64) ASSET_NAME="supercuts-macos-x86_64";;
            *)      echo "Error: Unsupported macOS architecture '$ARCH'. Only arm64 and x86_64 are supported."; exit 1;;
        esac
        ;;
    *)          
        echo "Error: Unsupported operating system '$OS_TYPE'."; exit 1;;
esac

BINARY_URL="${REPO_URL}/releases/latest/download/${ASSET_NAME}"

# --- Colors for Output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Functions ---
function check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${YELLOW}This script requires sudo privileges to install to ${INSTALL_DIR}.${NC}"
        echo "Please run it with 'sudo'."
        exit 1
    fi
}

function install_binary() {
    echo "Starting Super Cuts installation for ${OS_TYPE}..."

    # 1. Check for required tools
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: 'curl' is not installed. Please install it to continue.${NC}"
        exit 1
    fi

    # 2. Download the binary
    echo "Downloading the '${BINARY_NAME}' binary from ${BINARY_URL}..."
    if ! curl -L -o "/tmp/${BINARY_NAME}" "${BINARY_URL}"; then
        echo -e "${RED}Failed to download the binary. Please check the URL and your connection.${NC}"
        exit 1
    fi

    # 3. Make the binary executable
    echo "Making the binary executable..."
    chmod +x "/tmp/${BINARY_NAME}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to set executable permissions on the binary.${NC}"
        rm -f "/tmp/${BINARY_NAME}"
        exit 1
    fi

    # 4. Move the binary to the installation directory
    echo "Moving '${BINARY_NAME}' to ${INSTALL_DIR}..."
    if ! mv "/tmp/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"; then
        echo -e "${RED}Failed to move the binary to ${INSTALL_DIR}.${NC}"
        echo "Please ensure you have the necessary permissions."
        rm -f "/tmp/${BINARY_NAME}"
        exit 1
    fi

    echo -e "${GREEN}Installation successful!${NC}"
    echo "You can now run 'supercuts' from anywhere in your terminal."
    echo "Make sure you have an .env file with your OPENAI_API_KEY in the directory where you run the command."
}

# --- Main Execution ---
check_sudo
install_binary 