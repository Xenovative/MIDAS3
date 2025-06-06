#!/bin/bash
# MIDAS3 macOS Setup Script with ComfyUI

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
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    brew install python@3.10
fi

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

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional dependencies
brew install libmagic

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export OLLAMA_HOST=http://localhost:11434

# Initialize the database
echo "Initializing database..."
flask init-db

# Clone ComfyUI if not exists
if [ ! -d "ComfyUI" ]; then
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    pip install -r requirements.txt
    cd ..
fi

# Start services in separate terminal windows
echo "Starting services in new terminal windows..."

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    start_in_terminal "ollama serve"
    echo "Ollama started in new terminal"
    sleep 5  # Give Ollama time to start
fi

# Start ComfyUI
start_in_terminal "cd ComfyUI && python main.py --cpu"
echo "ComfyUI starting in new terminal at http://127.0.0.1:8188"

# Start MIDAS3
echo "Starting MIDAS3..."
echo "Open http://localhost:5000 in your browser"
flask run