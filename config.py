import os
import json
import ollama

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_preferences.json')

def get_available_models():
    """Fetch available models from Ollama"""
    try:
        print("Attempting to connect to Ollama at http://localhost:11434")
        client = ollama.Client(host='http://localhost:11434', timeout=5.0)
        
        # Get the raw models data
        models_data = client.list()
        print(f"Ollama response: {models_data}")
        
        # Get embedding models to exclude them
        embedding_models = get_available_embedding_models()
        print(f"Found {len(embedding_models)} embedding models to exclude")
        
        # New Ollama API format has models in models_data['models']
        if 'models' in models_data and models_data['models']:
            print(f"Found {len(models_data['models'])} total models")
            filtered_models = []
            
            for model in models_data['models']:
                # Handle both new and old API formats
                model_name = model.get('model', model.get('name', ''))
                if not model_name:
                    continue
                    
                print(f"Processing model: {model_name}")
                
                # Skip if model is in the embedding models list
                if model_name in embedding_models:
                    print(f"Skipping embedding model: {model_name}")
                    continue
                
                filtered_models.append(model_name)
            
            print(f"Found {len(filtered_models)} non-embedding models")
            return filtered_models if filtered_models else default_models()
            
        # Fallback to direct API call if models key not found
        import requests
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            data = response.json()
            if 'models' in data:
                print(f"Found {len(data['models'])} models via direct API call")
                return [m['name'] for m in data['models'] if m.get('name') and m.get('name') not in embedding_models]
        
        print("No models found in Ollama response")
        return default_models()
        
    except Exception as e:
        print(f"Error fetching models from Ollama: {str(e)}")
        import traceback
        traceback.print_exc()
        return default_models()

def get_available_embedding_models():
    """Fetch available embedding models from Ollama"""
    try:
        client = ollama.Client(host='http://localhost:11434')
        models_data = client.list()
        
        # List of known embedding models for fallback detection
        known_embedding_models = [
            'nomic-embed-text',
            'all-minilm',
            'mxbai-embed-large',
            'e5',
            'bge',
            'dmeta-embedding-zh',
            'herald/dmeta-embedding-zh'
            'all-minilm'
            'all-minilm-latest'
        ]
        
        if 'models' in models_data and models_data['models']:
            # Filter for embedding models
            filtered_models = []
            for model in models_data['models']:
                model_name = model.get('model', model.get('name', ''))
                
                # Method 1: Check model details for embedding indicators
                try:
                    model_info = client.show(model_name)
                    # Check if model has embedding length property or contains 'embed' in architecture
                    if ('embedding length' in str(model_info) or 
                        'embed' in str(model_info.get('model', {}).get('architecture', '')).lower() or
                        'bert' in str(model_info.get('model', {}).get('architecture', '')).lower()):
                        filtered_models.append(model_name)
                        continue
                except Exception as e:
                    print(f"Error getting details for model {model_name}: {e}")
                
                # Method 2: Fallback to name-based detection
                if any(embed_model in model_name.lower() for embed_model in known_embedding_models):
                    filtered_models.append(model_name)
            
            return filtered_models if filtered_models else ['nomic-embed-text']
    except Exception as e:
        print(f"Error fetching embedding models from Ollama: {e}")
        
    return ['nomic-embed-text']

def default_models():
    """Return default models if Ollama isn't running or no models found"""
    return [
        'llama2', 
        'llama3:8b', 
        'mistral:7b', 
        'gemma:7b',
        'codellama:latest',
        'mistral-small3.1:latest'
    ]

def load_user_preferences():
    """Load user preferences from config file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading user preferences: {e}")
    
    # Default preferences
    return {
        'default_model': None,
        'default_embedding_model': 'nomic-embed-text',
        'visible_models': [],  # Empty means show all
        'theme': 'system',
        'show_thinking': False  # Default to collapsed thinking process
    }

def save_user_preferences(preferences):
    """Save user preferences to config file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(preferences, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving user preferences: {e}")
        return False

# Get available models
AVAILABLE_MODELS = get_available_models()
AVAILABLE_EMBEDDING_MODELS = get_available_embedding_models()

# ComfyUI Configuration
COMFYUI_HOST = os.environ.get('COMFYUI_HOST', 'localhost')
COMFYUI_PORT = os.environ.get('COMFYUI_PORT', '8188')
COMFYUI_BASE_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Load user preferences
USER_PREFERENCES = load_user_preferences()

# Determine default model - prefer user preference, fall back to first available
DEFAULT_MODEL = USER_PREFERENCES.get('default_model') or (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else 'llama2')
DEFAULT_EMBEDDING_MODEL = USER_PREFERENCES.get('default_embedding_model', 'nomic-embed-text')

# Define visible models - if user has preferences, use those, otherwise show all
VISIBLE_MODELS = USER_PREFERENCES.get('visible_models', []) or AVAILABLE_MODELS

# Model options (can still be customized per model)
MODEL_OPTIONS = {
    'llama2': {'temperature': 0.7, 'top_p': 0.9},
    'llama3:8b': {'temperature': 0.7, 'top_p': 0.9},
    'mistral:7b': {'temperature': 0.5, 'top_p': 0.85},
    'gemma:7b': {'temperature': 0.7, 'top_p': 0.9},
    # Add other models and their default parameters here
}

# Thinking process configuration
THINKING_MODELS = [
    'deepseek-r1:7b',
    'deepseek-r1:32b',
    'qwen3:30b'
]

# API configuration
API_TIMEOUT = 300  # Timeout in seconds for API calls

# Ollama API host
OLLAMA_HOST = 'http://localhost:11434'

# RAG Configuration
RAG_SOURCE_DIR = "docs" # Directory containing documents for RAG indexing
# Ensure this directory exists or is created before running RAG setup.
