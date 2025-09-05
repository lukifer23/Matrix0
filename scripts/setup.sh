#!/usr/bin/env bash
set -euo pipefail

# Default values
VENV=${1:-.venv}
PYTHON=${PYTHON:-python3}
REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-requirements.txt}
QUIET=${QUIET:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Python version
get_python_version() {
    $PYTHON --version 2>&1 | grep -oP '\d+\.\d+' || echo "unknown"
}

# Function to check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."

    # Check Python
    if ! command_exists "$PYTHON"; then
        log_error "Python ($PYTHON) not found. Please install Python 3.8+ first."
        exit 1
    fi

    PYTHON_VERSION=$(get_python_version)
    log_info "Using Python $PYTHON_VERSION at $(which $PYTHON)"

    # Check Python version (basic check)
    if [[ "$PYTHON_VERSION" =~ ^[0-2]\. ]]; then
        log_warning "Python version $PYTHON_VERSION detected. Matrix0 requires Python 3.8+"
    fi

    # Check if we're in the right directory
    if [[ ! -f "requirements.txt" ]]; then
        log_error "requirements.txt not found. Are you in the Matrix0 root directory?"
        exit 1
    fi

    log_success "System requirements check passed"
}

# Function to create virtual environment
create_venv() {
    log_info "Creating virtual environment: $VENV"

    if [[ -d "$VENV" ]]; then
        log_warning "Virtual environment $VENV already exists"
        read -p "Remove existing virtual environment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV"
            log_info "Removed existing virtual environment"
        fi
    fi

    if ! $PYTHON -m venv "$VENV"; then
        log_error "Failed to create virtual environment"
        exit 1
    fi

    log_success "Virtual environment created successfully"
}

# Function to activate virtual environment
activate_venv() {
    log_info "Activating virtual environment..."

    if [[ ! -f "$VENV/bin/activate" ]]; then
        log_error "Virtual environment activation script not found"
        exit 1
    fi

    source "$VENV/bin/activate"
    log_success "Virtual environment activated"
}

# Function to upgrade pip
upgrade_pip() {
    log_info "Upgrading pip..."

    if ! python -m pip install --upgrade pip; then
        log_warning "Failed to upgrade pip, continuing with existing version"
    else
        log_success "Pip upgraded successfully"
    fi
}

# Function to install requirements
install_requirements() {
    log_info "Installing requirements from $REQUIREMENTS_FILE..."

    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log_error "Requirements file $REQUIREMENTS_FILE not found"
        exit 1
    fi

    if ! pip install -r "$REQUIREMENTS_FILE"; then
        log_error "Failed to install requirements"
        exit 1
    fi

    log_success "Requirements installed successfully"
}

# Function to verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Test basic imports
    if ! python -c "import torch; print(f'PyTorch {torch.__version__} available')"; then
        log_warning "PyTorch import failed. Some features may not work."
    else
        log_success "PyTorch import successful"
    fi

    # Test azchess import
    if ! python -c "import azchess; print('azchess import successful')"; then
        log_warning "azchess import failed. Installation may be incomplete."
    else
        log_success "azchess import successful"
    fi
}

# Function to show usage information
show_usage() {
    cat << EOF
Matrix0 Setup Script

Usage: $0 [VENV_NAME] [OPTIONS]

Arguments:
    VENV_NAME       Name of virtual environment (default: .venv)

Options:
    --python PATH   Python executable to use (default: python3)
    --requirements FILE    Requirements file to use (default: requirements.txt)
    --quiet         Suppress informational messages
    --help          Show this help message

Environment Variables:
    PYTHON          Python executable (same as --python)
    REQUIREMENTS_FILE    Requirements file (same as --requirements)
    QUIET           Suppress messages (same as --quiet)

Examples:
    $0                          # Create .venv with default settings
    $0 myenv                    # Create custom virtual environment
    $0 --python python3.9        # Use specific Python version
    $0 --requirements dev.txt   # Use different requirements file

After setup, activate with: source $VENV/bin/activate
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --requirements)
            REQUIREMENTS_FILE="$2"
            shift 2
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            VENV="$1"
            shift
            ;;
    esac
done

# Main setup process
main() {
    echo "🚀 Matrix0 Development Environment Setup"
    echo "=" * 50

    check_system_requirements
    create_venv
    activate_venv
    upgrade_pip
    install_requirements
    verify_installation

    echo
    log_success "Matrix0 setup complete!"
    echo
    echo "Next steps:"
    echo "  1. Activate the environment: source $VENV/bin/activate"
    echo "  2. Run tests: python -m pytest tests/"
    echo "  3. Start training: python -m azchess.training.train"
    echo "  4. View WebUI: python webui/server.py"
    echo
    echo "For more information, see README.md"
}

# Run main function
main "$@"
