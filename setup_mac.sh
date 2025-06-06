#!/bin/bash
# MIDAS3 macOS Setup Script

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

# Install additional dependencies that might be needed
brew install libmagic

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export OLLAMA_HOST=http://localhost:11434

# Initialize the database
echo "Initializing database..."
flask init-db

# Start the application
echo "Starting MIDAS3..."
echo "Open http://localhost:5000 in your browser after the server starts"
flask run