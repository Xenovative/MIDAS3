#!/bin/bash
# MIDAS3 macOS Run Script

# Function to start services in new terminal windows
start_in_terminal() {
    osascript -e "tell app \"Terminal\" to do script \"cd '$PWD' && $1\""
}

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please run setup_mac.sh first."
    exit 1
fi

# Check if virtual environments exist
if [ ! -d "midas_venv" ]; then
    echo "MIDAS3 virtual environment not found. Please run setup_mac.sh first."
    exit 1
fi

if [ ! -d "ComfyUI/venv" ]; then
    echo "ComfyUI virtual environment not found. Please run setup_mac.sh first."
    exit 1
fi

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export OLLAMA_HOST=http://localhost:11434

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    start_in_terminal "ollama serve"
    echo "Ollama started in new terminal"
    sleep 5  # Give Ollama time to start
fi

# Start ComfyUI in its own virtual environment
echo "Starting ComfyUI..."
start_in_terminal "cd '$PWD/ComfyUI' && source venv/bin/activate && python main.py --cpu"
echo "ComfyUI starting in new terminal at http://127.0.0.1:8188"

# Start MIDAS3
echo "Starting MIDAS3..."
echo "Open http://localhost:5000 in your browser"
source midas_venv/bin/activate
python app.py
