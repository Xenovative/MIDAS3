#!/bin/bash
# MIDAS3 macOS Setup Script with ComfyUI

# Exit on error
set -e

# Set architecture flags for Apple Silicon
if [[ $(uname -m) == 'arm64' ]]; then
    echo "Detected Apple Silicon (ARM64) architecture"
    # Ensure we're using ARM64 Homebrew
    if [[ $(which brew) != "/opt/homebrew"* ]]; then
        echo "Configuring for ARM64 Homebrew..."
        export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
    fi
    export HOMEBREW_PREFIX="/opt/homebrew"
else
    echo "Detected Intel architecture"
    export HOMEBREW_PREFIX="/usr/local"
fi

# Set Python and pip paths for ARM64
PYTHON_BIN="$HOMEBREW_PREFIX/opt/python@3.10/bin/python3.10"
PIP_BIN="$HOMEBREW_PREFIX/opt/python@3.10/bin/pip3.10"

echo "Using Python from: $PYTHON_BIN"

# Function to start services in new terminal windows
start_in_terminal() {
    osascript -e "tell app \"Terminal\" to do script \"cd '$PWD' && $1\""
}

# Install Homebrew if not installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
    source ~/.zshrc
fi

# Install Python 3.10 if not installed
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Installing Python 3.10..."
    # Install Python using Homebrew with architecture flags for Apple Silicon
    arch -arm64 brew install python@3.10
    # Verify installation
    if [ ! -f "$PYTHON_BIN" ]; then
        echo "Failed to install Python 3.10 at $PYTHON_BIN"
        exit 1
    fi
    # Add Python to PATH
    echo "export PATH=\"$HOMEBREW_PREFIX/opt/python@3.10/bin:\$PATH\"" >> ~/.zshrc
    source ~/.zshrc
fi

# Verify Python architecture
echo "Verifying Python installation..."
"$PYTHON_BIN" -c "import platform; print(f'Python {platform.python_version()} ({platform.machine()})')"

# Install required build tools
arch -arm64 brew install cmake protobuf rust

# Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    brew install ollama
    # Start Ollama service
    ollama serve &
    # Pull the default model
    ollama pull llama3.1:8b
fi

# Install ComfyUI dependencies
if ! command -v git-lfs &> /dev/null; then
    echo "Installing Git LFS for ComfyUI models..."
    brew install git-lfs
    git lfs install
fi

# Clone MIDAS3 repository if not already cloned
if [ ! -d "MIDAS3" ]; then
    echo "Cloning MIDAS3 repository..."
    git clone https://github.com/Xenovative/MIDAS3.git
    cd MIDAS3
else
    cd MIDAS3
fi

# Create and activate MIDAS3 virtual environment
echo "Setting up MIDAS3 Python virtual environment..."
python3.10 -m venv midas_venv
source midas_venv/bin/activate

# Install MIDAS3 dependencies
echo "Installing MIDAS3 Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional dependencies
brew install libmagic

# Set environment variables for MIDAS3
export FLASK_APP=app.py
export FLASK_ENV=development
export OLLAMA_HOST=http://localhost:11434

# Initialize the database
echo "Initializing MIDAS3 database..."
flask init-db

# Clone or update ComfyUI
if [ ! -d "ComfyUI" ]; then
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
fi

# Navigate to ComfyUI directory
cd ComfyUI

# Set up ComfyUI virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating ComfyUI virtual environment with ARM Python..."
    # Create virtual environment using the ARM64 Python
    /opt/homebrew/opt/python@3.10/bin/python3.10 -m venv venv
    
    # Activate the virtual environment
    source venv/bin/activate
    
    # Upgrade pip and install dependencies
    echo "Installing ComfyUI dependencies..."
    "$PYTHON_BIN" -m pip install --upgrade pip
    
    # Install PyTorch with Metal Performance Shaders (MPS) for Apple Silicon
    echo "Installing PyTorch with MPS support for Apple Silicon..."
    
    # Install PyTorch with MPS support using the correct pip
    "$PIP_BIN" install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
    
    # Install specific versions known to work well with MPS
    "$PIP_BIN" install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    
    # Verify PyTorch MPS availability
    echo "Verifying PyTorch MPS support..."
    "$PYTHON_BIN" -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); device = 'mps' if torch.backends.mps.is_available() else 'cpu'; print(f'Device: {device}')"
    
    # Install ComfyUI requirements
    echo "Installing ComfyUI requirements..."
    "$PIP_BIN" install -r requirements.txt
    
    # Install additional common nodes
    echo "Installing common custom nodes..."
    mkdir -p custom_nodes
    
    # Example: Install ComfyUI Manager
    if [ ! -d "custom_nodes/ComfyUI-Manager" ]; then
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
    fi
    
    deactivate  # Exit ComfyUI venv
    
    echo "ComfyUI virtual environment setup complete."
else
    echo "ComfyUI virtual environment already exists."
    
    # Update ComfyUI
    echo "Updating ComfyUI..."
    git pull
    
    # Update dependencies if needed
    source venv/bin/activate
    pip install --upgrade -r requirements.txt
    deactivate
fi

# Return to MIDAS3 directory
cd ..

# Start services in separate terminal windows
echo "Starting services in new terminal windows..."

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    start_in_terminal "ollama serve"
    echo "Ollama started in new terminal"
    sleep 5  # Give Ollama time to start
fi

# Start ComfyUI in its own virtual environment
start_in_terminal "cd ComfyUI && source venv/bin/activate && python main.py --cpu"
echo "ComfyUI starting in new terminal at http://127.0.0.1:8188"

# Start MIDAS3
echo "Starting MIDAS3..."
echo "Open http://localhost:5000 in your browser"
source midas_venv/bin/activate  # Reactivate MIDAS3 venv
flask run