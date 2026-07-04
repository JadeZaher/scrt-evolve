#!/bin/bash
# scrt-evolve install script for Linux (x86_64)
# Usage: curl -fsSL https://raw.githubusercontent.com/nousresearch/scrt-evolve/main/scripts/install.sh | sh
# Or: bash install.sh

set -euo pipefail

# Configuration
REPO_OWNER="${REPO_OWNER:-nousresearch}"
REPO_NAME="${REPO_NAME:-scrt-evolve}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
VENV_DIR="${VENV_DIR:-$HOME/.local/share/scrt-evolve}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS and architecture
detect_platform() {
    if [[ "$(uname)" != "Linux" ]]; then
        log_error "This script is for Linux only. On Windows, use install.ps1"
        exit 1
    fi
    
    local arch
    arch=$(uname -m)
    if [[ "$arch" != "x86_64" ]]; then
        log_error "This script supports x86_64 Linux only. Detected: $arch"
        exit 1
    fi
    
    echo "linux-x86_64"
}

# Detect Python version
detect_python() {
    log_info "Detecting Python..."
    
    # Check for python3 first, then python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python 3.9+ not found. Please install Python first."
        exit 1
    fi
    
    # Check version
    local version
    version=$($PYTHON_CMD -c 'import sys; print(f"{v.major}.{v.minor}")' 2>/dev/null || echo "0.0")
    local major minor
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    
    if [[ "$major" -lt 3 ]] || [[ "$major" -eq 3 && "$minor" -lt 9 ]]; then
        log_error "Python 3.9+ required. Detected: $version"
        exit 1
    fi
    
    log_info "Found Python $version at $(which $PYTHON_CMD)"
    echo "$PYTHON_CMD"
}

# Detect CUDA availability
detect_cuda() {
    log_info "Checking for CUDA..."
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            log_info "CUDA detected via nvidia-smi"
            echo "cuda"
            return
        fi
    fi
    
    log_info "No CUDA detected, using CPU"
    echo "cpu"
}

# Create installation directory
ensure_install_dir() {
    log_info "Ensuring installation directory exists: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    
    # Check if directory is on PATH
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        log_warn "$INSTALL_DIR is not on your PATH"
        echo ""
        echo "Add to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
        echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
        echo ""
    fi
}

# Fetch latest release
fetch_latest_release() {
    log_info "Fetching latest release from GitHub..."
    
    local api_url="https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/releases/latest"
    local tag
    
    tag=$(curl -fsSL "$api_url" | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
    
    if [[ -z "$tag" ]]; then
        log_error "Could not fetch latest release. Using fallback tag."
        echo "v0.1.0"
        return
    fi
    
    log_info "Latest release: $tag"
    echo "$tag"
}

# Download binary
download_binary() {
    local tag="$1"
    local platform="$2"
    
    log_info "Downloading scrt-evolve $tag for $platform..."
    
    local download_url="https://github.com/$REPO_OWNER/$REPO_NAME/releases/download/$tag/evolve-$platform"
    local output_path="$INSTALL_DIR/evolve"
    
    if curl -fsSL "$download_url" -o "$output_path"; then
        chmod +x "$output_path"
        log_info "Downloaded to $output_path"
    else
        log_error "Download failed. The release may not have artifacts for this platform."
        log_error "You may need to build from source: cargo build --release -p scrt-evolve"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    local python_cmd="$1"
    local accelerator="$2"
    
    log_info "Setting up Python virtual environment at $VENV_DIR"
    
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Virtual environment already exists, skipping creation"
        return 0
    fi
    
    mkdir -p "$(dirname "$VENV_DIR")"
    
    log_info "Creating virtual environment..."
    if ! $python_cmd -m venv "$VENV_DIR"; then
        log_error "Failed to create virtual environment"
        exit 1
    fi
    
    log_info "Installing scrt-evolve-ml[$accelerator]..."
    local pip_cmd
    if [[ "$(uname)" == "Linux" ]]; then
        pip_cmd="$VENV_DIR/bin/pip"
    else
        pip_cmd="$VENV_DIR/Scripts/pip.exe"
    fi
    
    if ! $pip_cmd install --upgrade pip 2>/dev/null; then
        log_warn "Could not upgrade pip, continuing with existing version"
    fi
    
    if ! $pip_cmd install "scrt-evolve-ml[$accelerator]"; then
        log_error "Failed to install scrt-evolve-ml[$accelerator]"
        exit 1
    fi
    
    log_info "Installation complete!"
}

# Main installation flow
main() {
    echo "=========================================="
    echo "  scrt-evolve Installer (Linux x86_64)"
    echo "=========================================="
    echo ""
    
    # Detect platform
    local platform
    platform=$(detect_platform)
    
    # Ensure install directory
    ensure_install_dir
    
    # Fetch latest release
    local tag
    tag=$(fetch_latest_release)
    
    # Download binary
    download_binary "$tag" "$platform"
    
    # Detect Python
    local python_cmd
    python_cmd=$(detect_python)
    
    # Detect accelerator
    local accelerator
    accelerator=$(detect_cuda)
    
    # Create virtual environment
    create_venv "$python_cmd" "$accelerator"
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Installation complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Add $INSTALL_DIR to your PATH if not already"
    echo "  2. Configure Python: export SCRT_EVOLVE_PYTHON=$VENV_DIR/bin/python"
    echo "  3. Run: evolve doctor"
    echo ""
    
    # Offer to configure environment
    if [[ -t 0 ]]; then
        echo "Would you like to add these to your shell profile? (y/N)"
        read -r response
        if [[ "$response" == "y" || "$response" == "Y" ]]; then
            local shell_rc
            shell_rc="$HOME/.bashrc"
            if [[ -n "${ZSH_VERSION:-}" ]]; then
                shell_rc="$HOME/.zshrc"
            fi
            
            echo "" >> "$shell_rc"
            echo "# scrt-evolve" >> "$shell_rc"
            echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$shell_rc"
            echo "export SCRT_EVOLVE_PYTHON=$VENV_DIR/bin/python" >> "$shell_rc"
            
            log_info "Added to $shell_rc"
            log_info "Please restart your shell or run: source $shell_rc"
        fi
    fi
}

# Run main
main "$@"