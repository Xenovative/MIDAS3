from flask import Flask, request, jsonify, render_template, Response, abort
import ollama
import config # Added import
from config import DEFAULT_MODEL, MODEL_OPTIONS, API_TIMEOUT, THINKING_MODELS, AVAILABLE_MODELS, OLLAMA_HOST, AVAILABLE_EMBEDDING_MODELS, VISIBLE_MODELS, USER_PREFERENCES
import os
import re
import logging
from logging.handlers import RotatingFileHandler
import json
import time
import database as db
import rag # Import the RAG module
from models.bot import Bot
from progress_tracker import create_progress, update_progress, get_progress, get_all_progress, clear_progress
from werkzeug.utils import secure_filename # Import for secure filenames
import requests  # For ComfyUI API calls
import base64     # For encoding images if needed
import re
import math
import random
import uuid
import time
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session, send_from_directory
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from passlib.hash import bcrypt
import sqlite3  # Import sqlite3 here
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredXMLLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.secret_key = 'CHANGE_THIS_TO_A_RANDOM_SECRET_KEY'

# Configure upload limits
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['MAX_CONTENT_PATH'] = 100 * 1024 * 1024  # 100MB

# Configure request buffers
app.config['REQUEST_BUFFER_SIZE'] = 100 * 1024 * 1024  # 100MB

# Global progress tracking
if not hasattr(app, 'indexing_progress'):
    app.indexing_progress = {}
    # Structure: {bot_id: {current: 0, total: 0, status: '', file: ''}}

# Import for SSE
from flask import Response, stream_with_context

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

log_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- User Class for Flask-Login ---
class User(UserMixin):
    def __init__(self, user_dict):
        self.id = user_dict['id']
        self.username = user_dict['username']
        self.password_hash = user_dict['password_hash']
        self.role = user_dict['role']
        self.created_at = user_dict['created_at']
        # Add display_name attribute
        self.display_name = user_dict.get('display_name', self.username)

    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    user_dict = db.get_user_by_id(user_id)
    if user_dict:
        return User(user_dict)
    return None
    
# --- Helper Functions ---
def call_llm(model, messages, max_tokens=500, temperature=0.7, top_p=0.95, timeout=10):
    """Call an LLM model with the given messages and parameters"""
    try:
        app.logger.info(f"Calling LLM model {model} with {len(messages)} messages (timeout: {timeout}s)")
        
        # Handle different model types
        if model.startswith('bot:'):
            # For bot models, use the generate_bot_response function
            bot_id = model.split(':', 1)[1]
            bot = Bot.get_bot_by_id(bot_id)
            if not bot:
                raise ValueError(f"Bot {bot_id} not found")
            
            # Extract just the content from the messages
            prompt = "\n".join([msg.get('content', '') for msg in messages])
            response = bot.generate_response(prompt, max_tokens=max_tokens)
            return response
        else:
            # For Ollama models
            try:
                # Set a custom timeout for the request
                import socket
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(timeout)
                
                try:
                    # Check if Ollama is available with the specified timeout
                    app.logger.info(f"Calling Ollama with model: {model} at host: {OLLAMA_HOST}")
                    client = ollama.Client(host=OLLAMA_HOST)
                    response = client.chat(
                        model=model,
                        messages=messages,
                        options={
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_predict": max_tokens
                        }
                    )
                    return response['message']['content']
                finally:
                    # Restore original timeout
                    socket.setdefaulttimeout(original_timeout)
                    
            except requests.exceptions.ConnectionError:
                raise ConnectionError("Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible.")
            except socket.timeout:
                raise ConnectionError(f"Connection to Ollama timed out after {timeout} seconds.")
            except Exception as e:
                error_msg = str(e).lower()
                app.logger.error(f"Ollama error: {error_msg}")
                
                if "not found" in error_msg:
                    # Try to extract the model name from the error message
                    raise ValueError(f"Model '{model}' not found in Ollama. Please check available models.")
                elif "connection" in error_msg or "timeout" in error_msg:
                    raise ConnectionError("Failed to connect to Ollama. Please check that Ollama is running.")
                else:
                    raise ValueError(f"Error calling model '{model}': {str(e)}")

    except Exception as e:
        app.logger.error(f"Error calling LLM: {str(e)}")
        raise

@login_manager.unauthorized_handler
def unauthorized():
    # Return JSON for API requests, HTML for browser navigation
    if request.accept_mimetypes.accept_json or request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Authentication required'}), 401
    else:
        return render_template('login.html'), 401

# --- Admin Required Decorator ---
def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or getattr(current_user, 'role', None) != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# --- Registration Route ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form if request.form else request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        # Check if this is the first user
        is_first_user = db.get_user_by_id(1) is None
        role = 'admin' if is_first_user else 'user'
        if not username or not email or not password:
            return jsonify({'status': 'error', 'message': 'Username, email, and password required'}), 400
        if db.get_user_by_username(username):
            return jsonify({'status': 'error', 'message': 'Username already exists'}), 400
        # Validate email format
        import re
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            return jsonify({'status': 'error', 'message': 'Invalid email format'}), 400
        password_hash = bcrypt.hash(password)
        user_id = db.create_user(username, password_hash, role, email=email)
        if user_id:
            user_dict = db.get_user_by_id(user_id)
            user = User(user_dict)
            login_user(user)
            return jsonify({'status': 'success', 'user': {'username': user.username, 'role': user.role}})
        else:
            return jsonify({'status': 'error', 'message': 'Registration failed'}), 500
    # For GET, return a simple registration form (for testing)
    return '''<form method="post"><input name="username" placeholder="Username"><input name="email" placeholder="Email"><input name="password" type="password" placeholder="Password"><button type="submit">Register</button></form>'''

# --- Login Route ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form if request.form else request.json
        username = data.get('username')
        password = data.get('password')
        user_dict = db.get_user_by_username(username)
        if user_dict and bcrypt.verify(password, user_dict['password_hash']):
            user = User(user_dict)
            login_user(user)
            return jsonify({'status': 'success', 'user': {'username': user.username, 'role': user.role}})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
    # For GET, return a simple login form (for testing)
    return '''<form method="post"><input name="username" placeholder="Username"><input name="password" type="password" placeholder="Password"><button type="submit">Login</button></form>'''

# --- Logout Route ---
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'status': 'success', 'message': 'Logged out'})

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    """Endpoint to list available models"""
    try:
        app.logger.info("Attempting to get available models...")
        
        # Get user preferences if user is logged in
        user_preferred_models = []
        if current_user.is_authenticated:
            user_id = str(current_user.id)
            try:
                if os.path.exists('data/user_preferences.json'):
                    with open('data/user_preferences.json', 'r') as f:
                        all_prefs = json.load(f)
                        user_prefs = all_prefs.get(user_id, {})
                        if user_prefs and 'visible_models' in user_prefs and user_prefs['visible_models']:
                            user_preferred_models = user_prefs['visible_models']
                            app.logger.info(f"User preferred models: {user_preferred_models}")
            except Exception as e:
                app.logger.error(f"Error loading user preferences: {e}")
        
        # Get available models from Ollama
        models_list = config.AVAILABLE_MODELS.copy()
        app.logger.info(f"Available models from Ollama: {models_list}")
        
        # Filter models based on user preferences if they exist
        if user_preferred_models:
            # If user has specific visible models, use only those that are available
            filtered_models = []
            for model_name in user_preferred_models:
                # Only add models that exist in available models or are explicitly preferred
                if model_name in models_list:
                    filtered_models.append(model_name)
            
            # Replace models_list with filtered list if we have preferences
            if filtered_models:
                models_list = filtered_models
                app.logger.info(f"Filtered models based on user preferences: {models_list}")
        
        # Convert simple string models to objects for consistent frontend handling
        serializable_models = []
        for model_name in models_list:
            model_obj = {
                'name': model_name,
                'details': {
                    'parameter_size': '7B-8B' if '7b' in model_name.lower() or '8b' in model_name.lower() else '',
                }
            }
            serializable_models.append(model_obj)
            
        app.logger.info(f"Serialized models: {serializable_models}")
        
        return jsonify({
            'models': serializable_models,
            'status': 'success'
        })
            
    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        return jsonify({
            'message': 'Server error while processing request',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/conversations', methods=['GET'])
@login_required
def list_conversations():
    """List all conversations for the current user"""
    try:
        conversations = db.get_user_conversations(current_user.id)
        return jsonify({
            'conversations': conversations,
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Error listing conversations: {str(e)}")
        return jsonify({
            'message': 'Failed to list conversations',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/conversations', methods=['POST'])
@login_required
def create_conversation():
    """Create a new conversation for the current user"""
    try:
        data = request.get_json()
        title = data.get('title', f"New Chat {time.strftime('%Y-%m-%d %H:%M')}")
        model = data.get('model', DEFAULT_MODEL)
        secret = data.get('secret', False)
        if secret:
            conversation_id = f"secret-{uuid.uuid4()}"
        else:
            conversation_id = db.create_conversation(current_user.id, title, model)
        return jsonify({
            'conversation_id': conversation_id,
            'title': title,
            'model': model,
            'secret': secret,
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Error creating conversation: {str(e)}")
        return jsonify({
            'message': 'Failed to create conversation',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/conversations/<int:conversation_id>', methods=['GET'])
@login_required
def get_conversation(conversation_id):
    """Get a conversation by ID with all its messages (only if owned by user)"""
    try:
        conversation = db.get_user_conversation(current_user.id, conversation_id)
        if not conversation:
            return jsonify({'status': 'error', 'message': 'Conversation not found'}), 404
        messages_list = conversation['messages']
        del conversation['messages']
        return jsonify({
            'status': 'success',
            'conversation': conversation,
            'messages': messages_list
        })
    except Exception as e:
        app.logger.error(f"Error getting conversation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/conversations/<int:conversation_id>', methods=['PUT'])
@login_required
def update_conversation(conversation_id):
    """Update a conversation's title"""
    try:
        data = request.get_json()
        title = data.get('title')
        model = data.get('model')
        
        if model:
            # If the model starts with 'bot:', fetch the bot's base model
            if model.startswith('bot:'):
                try:
                    bot_id = model[4:]  # Remove 'bot:' prefix
                    bot = Bot.get(bot_id)
                    if bot:
                        model = bot.base_model or DEFAULT_MODEL
                        app.logger.info(f"Using bot's base model {model} for conversation")
                    else:
                        app.logger.warning(f"Bot {bot_id} not found, falling back to default model")
                        model = DEFAULT_MODEL
                except Exception as e:
                    app.logger.error(f"Error fetching bot details: {str(e)}")
                    model = DEFAULT_MODEL
            
            db.update_conversation_model(conversation_id, model)
            return jsonify({
                'status': 'success',
                'message': 'Model updated successfully'
            })
        
        if not title and data.get('generate_title', False):
            # Check if a first_message was provided in the request
            first_user_message = data.get('first_message')
            
            # If no first_message was provided, try to get it from the database
            if not first_user_message:
                # Get the conversation messages
                messages = db.get_conversation_messages(conversation_id)
                app.logger.info(f"Generating title for conversation {conversation_id} with {len(messages)} messages")
                
                # Need at least one user message
                if len(messages) >= 1:
                    # Extract the first user message
                    first_user_message = next((m['content'] for m in messages if m['role'] == 'user'), None)
            
            # If we have a first_user_message (either from request or database), generate a title
            if first_user_message:
                app.logger.info(f"Generating title based on message: {first_user_message[:50]}...")
                
                # Get the model used for this conversation
                conversation = db.get_conversation(conversation_id)
                model_name = conversation.get('model', DEFAULT_MODEL)
                
                # If the model starts with 'bot:', fetch the bot's base model
                if model_name.startswith('bot:'):
                    try:
                        bot_id = model_name[4:]  # Remove 'bot:' prefix
                        bot = Bot.get_bot_by_id(bot_id)
                        if bot:
                            model_name = bot.base_model or DEFAULT_MODEL
                            app.logger.info(f"Using bot's base model {model_name} for title generation")
                        else:
                            app.logger.warning(f"Bot {bot_id} not found, falling back to default model")
                            model_name = DEFAULT_MODEL
                    except Exception as e:
                        app.logger.error(f"Error fetching bot details: {str(e)}")
                        model_name = DEFAULT_MODEL
                
                # Always use a simple title prompt regardless of model
                title_prompt = (
                    "Generate a short, descriptive title for this conversation (3-5 words). "
                    "Respond with ONLY the title text. No explanation, no thinking, no extra text. "
                    "Just the title itself.\n\n"
                    f"User message: {first_user_message}"
                )
                
                try:
                    client = ollama.Client(host=OLLAMA_HOST)
                    response = client.chat(
                        model=model_name,
                        messages=[{"role": "user", "content": title_prompt}]
                    )
                    
                    generated_title = response['message']['content']
                    app.logger.info(f"Raw generated title: {generated_title}")
                    
                    # Simple cleanup approach - take first line, remove tags, clean up formatting
                    
                    # Take first line only if multiple lines
                    if '\n' in generated_title:
                        generated_title = generated_title.split('\n')[0].strip()
                    
                    # Remove any HTML-like tags
                    import re
                    generated_title = re.sub(r'<[^>]+>', '', generated_title)
                    
                    # Remove quotes, punctuation, and extra spaces
                    title = generated_title.strip('"\'\'.,!?').strip()
                    
                    # Capitalize the first letter of each word for consistency
                    title = ' '.join(word.capitalize() for word in title.split())
                    
                    # Limit to 50 characters if somehow still too long
                    if len(title) > 50:
                        title = title[:47] + '...'
                        
                    # If we ended up with nothing useful, use fallback
                    if not title or len(title) < 3:
                        title = fallback_title
                        
                    app.logger.info(f"Final generated title: {title}")
                except Exception as e:
                    app.logger.error(f"Error generating title: {str(e)}")
                    title = f"Chat {time.strftime('%Y-%m-%d %H:%M')}"
            else:
                app.logger.warning("No user message found for title generation")
                title = f"Chat {time.strftime('%Y-%m-%d %H:%M')}"
        
        if not title:
            return jsonify({
                'message': 'Title is required',
                'status': 'error'
            }), 400
            
        # Update the conversation title
        db.update_conversation_title(conversation_id, title)
        
        return jsonify({
            'status': 'success',
            'message': 'Conversation updated successfully',
            'title': title
        })
    except Exception as e:
        app.logger.error(f"Error updating conversation: {str(e)}")
        return jsonify({
            'message': 'Failed to update conversation',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/messages/<int:message_id>', methods=['DELETE'])
@login_required
def delete_message_route(message_id):
    """Delete a specific message"""
    try:
        app.logger.info(f"Attempting to delete message {message_id}")
        success = db.delete_message(message_id)
        
        if not success:
            app.logger.warning(f"Message {message_id} not found for deletion.")
            return jsonify({
                'message': 'Message not found',
                'status': 'error'
            }), 404
            
        app.logger.info(f"Message {message_id} deleted successfully.")
        return jsonify({
            'message': 'Message deleted successfully',
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Error deleting message {message_id}: {str(e)}")
        return jsonify({
            'message': 'Failed to delete message',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/conversations/<int:conversation_id>', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    """Delete a conversation"""
    try:
        success = db.delete_conversation(conversation_id)
        
        if not success:
            return jsonify({
                'message': 'Conversation not found',
                'status': 'error'
            }), 404
            
        return jsonify({
            'message': 'Conversation deleted',
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Error deleting conversation: {str(e)}")
        return jsonify({
            'message': 'Failed to delete conversation',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/conversations/<int:conversation_id>/messages', methods=['POST'])
@login_required
def add_message(conversation_id):
    """Add a message to a conversation"""
    try:
        data = request.get_json()
        role = data.get('role')
        content = data.get('content')
        thinking = data.get('thinking')
        images = data.get('images') # Get images from request
        attachment_filename = data.get('attachment_filename')  # Get attachment filename
        secret = data.get('secret', False)
        
        if not role or not content:
            return jsonify({
                'message': 'Role and content are required',
                'status': 'error'
            }), 400
            
        if secret:
            # Do not save to DB, just return a fake message ID
            message_id = f"secret-{uuid.uuid4()}"
        else:
            # Pass images and attachment_filename to db.add_message
            message_id = db.add_message(
                conversation_id, 
                role, 
                content, 
                thinking, 
                images, 
                attachment_filename
            ) 
        
        return jsonify({
            'message_id': message_id,
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Error adding message: {str(e)}")
        return jsonify({
            'message': 'Failed to add message',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/generate', methods=['POST'])
@login_required
def generate():
    """Generate a response to a user's message"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        model_name = data.get('model', 'llama3:8b')
        original_model = model_name  # Store the original model for reference
        system_prompt = data.get('system_prompt', '')
        conversation_id = data.get('conversation_id')
        parameters = data.get('parameters', {})
        knowledge_files = data.get('knowledge_files', [])
        attachment_filename = data.get('attachment_filename') # Get attachment filename
        secret = data.get('secret', False)
        
        # Check user quota
        quota_check = db.check_user_quota(current_user.id)
        app.logger.info(f"Quota check result: {quota_check}")
        
        if not quota_check['allowed'] and current_user.role != 'admin':  # Admins bypass quota limits
            reason = quota_check['reason']
            limit = quota_check['limit']
            used = quota_check['used']
            
            if reason == 'daily_limit':
                message = f"You've reached your daily message limit ({used}/{limit}). Please try again tomorrow."
            elif reason == 'monthly_limit':
                message = f"You've reached your monthly message limit ({used}/{limit}). Please try again next month."
            else:
                message = "You've exceeded your message quota."
            
            # Get the full quota information for debugging
            full_quota = db.get_user_quota(current_user.id)
            app.logger.info(f"Full quota info: {full_quota}")
            
            # Include both quota check and full quota info
            quota_info = {
                **quota_check,
                'full_quota': full_quota
            }
                
            return jsonify({
                'status': 'error',
                'message': message,
                'quota_exceeded': True,
                'quota_info': quota_info
            }), 429  # Too Many Requests
        
        # Check for web search flag
        web_search = '--web' in user_message
        if web_search:
            # Remove the --web flag from the message
            user_message = re.sub(r'\s*--web\b', '', user_message).strip()
            # Force RAG to be used for web search
            use_rag = True
            rag_enabled = True
            
        # If no message and no attachment, return error
        if not user_message and not attachment_filename:
            return jsonify({'status': 'error', 'message': 'No message provided'}), 400
            
        # If only attachment, create a default message
        if attachment_filename:
            user_message = f"(Attached file: {attachment_filename})" if not user_message else user_message
            
        # Create a new conversation if none exists
        if not conversation_id:
            conversation_id = db.create_conversation(current_user.id, 'New Conversation', model_name)
            
        # --- 1. Add User Message to DB ---
        try:
            if not secret:
                # Pass attachment_filename when adding user message
                user_message_id = db.add_message(conversation_id, 'user', user_message, attachment_filename=attachment_filename)
            else:
                user_message_id = f"secret-{uuid.uuid4()}"
        except Exception as e:
            logging.error(f"Error adding user message to DB: {e}")
            # Continue processing, but log the error

        # --- RAG Integration ---
        # Check for documents in conversation or bot's knowledge base (skip for secret)
        use_rag = False
        # Initialize variables for RAG
        use_rag = False
        use_conversation_docs = False
        use_bot_knowledge = False
        retrieved_context = None
        bot_collection = None
        rag_enabled = False  # New variable to track RAG state
        app.logger.info(f"[RAG] Request data: bot_id={data.get('bot_id')}, conversation_id={conversation_id}")
        
        # Check if bot is selected either via bot_id parameter or model parameter
        bot_id = data.get('bot_id')
        
        # If no explicit bot_id but model starts with 'bot:', extract the bot_id from model
        if not bot_id and model_name and model_name.startswith('bot:'):
            bot_id = model_name.split(':', 1)[1]
            app.logger.info(f"[RAG] Extracted bot_id {bot_id} from model parameter: {model_name}")
        
        # If a bot is selected in the request, update the conversation model
        if bot_id and not secret and conversation_id:
            try:
                # Format as bot:id to indicate this is a bot conversation
                bot_model = f"bot:{bot_id}"
                db.update_conversation_model(conversation_id, bot_model)
                app.logger.info(f"[BOT] Updated conversation {conversation_id} model to {bot_model}")
            except Exception as model_err:
                app.logger.error(f"[BOT] Error updating conversation model: {str(model_err)}")
        
        # Check for knowledge files in the request
        knowledge_files = data.get('knowledge_files', [])
        
        # Check if conversation is associated with a bot (model field starts with 'bot:') 
        # if bot_id wasn't already extracted
        if not bot_id and conversation_id and not secret:
            try:
                conversation = db.get_conversation(conversation_id)
                if conversation and 'model' in conversation and conversation['model'].startswith('bot:'):
                    # Extract bot ID from model string
                    extracted_bot_id = conversation['model'].split(':', 1)[1]
                    bot_id = extracted_bot_id  # Set the bot_id for later use
                    app.logger.info(f"[RAG] Conversation {conversation_id} is associated with bot {bot_id}")
                    
                    # Add the bot_id to the request data for future processing
                    # This ensures the bot's system prompt and base model will be used
                    if 'bot_id' not in data:
                        data['bot_id'] = bot_id
                        app.logger.info(f"[RAG] Added bot_id {bot_id} to request data from conversation model")
            except Exception as e:
                app.logger.error(f"[RAG] Error checking conversation model: {str(e)}")
        
        # Rule 3: Check if conversation has uploaded documents
        has_conversation_docs = False
        if conversation_id and not secret:
            try:
                has_conversation_docs = rag.has_documents(conversation_id=conversation_id)
                if has_conversation_docs:
                    app.logger.info(f"[RAG] Documents found for conversation {conversation_id}")
                    use_conversation_docs = True
                    # Don't set use_rag here - only set it through our rules section below
            except Exception as e:
                app.logger.error(f"[RAG] Error checking conversation documents: {str(e)}")
        
        # Rule 1 & 2: Check if bot has knowledge base
        has_bot_knowledge = False
        if bot_id:
            try:
                bot = Bot.get(bot_id)
                if bot and bot.knowledge_files:
                    has_bot_knowledge = True
                    app.logger.info(f"[RAG] Bot {bot_id} has knowledge base with {len(bot.knowledge_files)} files")
                    use_bot_knowledge = True
                    # Don't set use_rag here - only set it through our rules section below
                else:
                    app.logger.info(f"[RAG] Bot {bot_id} has no knowledge base files")
            except Exception as e:
                app.logger.error(f"[RAG] Error checking bot knowledge base: {str(e)}")
                
        # Apply the rules
        if not secret:
            if web_search:
                # Web search takes priority over other RAG sources
                app.logger.info("[RAG] Web search requested, using web search results")
                use_rag = True
                rag_enabled = True
            elif use_bot_knowledge and use_conversation_docs:
                # Rule 2: Conversation with a bot with knowledge base AND documents uploaded
                app.logger.info(f"[RAG] Using both bot knowledge base AND conversation documents for RAG")
                use_rag = True
            elif use_bot_knowledge and not use_conversation_docs:
                # Rule 1: Conversation with a bot with knowledge base
                app.logger.info(f"[RAG] Using ONLY bot knowledge base for RAG")
                use_rag = True
            elif use_conversation_docs and not use_bot_knowledge:
                # Rule 3: Conversation with files uploaded but no bot or bot without knowledge base
                app.logger.info(f"[RAG] Using ONLY conversation documents for RAG")
                use_rag = True
            elif bot_id and not has_bot_knowledge:
                # Rule 4: Conversation with a bot without knowledge base
                app.logger.info(f"[RAG] Bot has no knowledge base, disabling RAG")
                use_rag = False
            else:
                app.logger.info(f"[RAG] No RAG sources available, disabling RAG")
                use_rag = False
                
            # Update rag_enabled to match use_rag after applying the rules
            rag_enabled = use_rag
            
        if use_rag and not secret:
            # Generate a unique operation ID for tracking
            operation_id = f"rag_{conversation_id}_{int(time.time())}"
            app.logger.info(f"[RAG] Querying with: '{user_message[:100]}...'")
            retrieval_start = time.time()
            
            try:
                import concurrent.futures
                
                # Determine which collections to query based on our rules
                bot_collection = None
                if use_bot_knowledge and bot_id:
                    bot_collection = f"bot_{bot_id}"
                    app.logger.info(f"[RAG] Will query bot collection: {bot_collection}")
                
                # Use ThreadPoolExecutor to run with timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Configure retrieval based on our rules
                    if web_search:
                        # For web search, we want to get web results first
                        app.logger.info("[RAG] Performing web search for query")
                        future = executor.submit(
                            rag.retrieve_context,
                            query=user_message,
                            web_search=True,
                            operation_id=operation_id
                        )
                    elif use_bot_knowledge and use_conversation_docs:
                        # Rule 2: Use both bot knowledge and conversation documents
                        app.logger.info(f"[RAG] Using both bot knowledge and conversation documents")
                        future = executor.submit(rag.retrieve_context, 
                                            query=user_message, 
                                            conversation_id=conversation_id,
                                            bot_collection=bot_collection,
                                            operation_id=operation_id)
                    elif use_bot_knowledge and not use_conversation_docs:
                        # Rule 1: Use ONLY bot knowledge base
                        app.logger.info(f"[RAG] Using ONLY bot knowledge base (collection: {bot_collection})")
                        future = executor.submit(rag.retrieve_context, 
                                            query=user_message, 
                                            bot_collection=bot_collection,
                                            operation_id=operation_id)
                    elif use_conversation_docs and not use_bot_knowledge:
                        # Rule 3: Use ONLY conversation documents
                        app.logger.info(f"[RAG] Using ONLY conversation documents")
                        future = executor.submit(rag.retrieve_context, 
                                            query=user_message, 
                                            conversation_id=conversation_id,
                                            operation_id=operation_id)
                    
                    # Log the operation ID for tracking
                    app.logger.info(f"[RAG] Started retrieval operation: {operation_id}")
                    
                    try:
                        retrieved_context = future.result(timeout=120)  # 2 minute timeout
                        retrieval_time = time.time() - retrieval_start
                        
                        # Check if query is in Chinese for fallback purposes
                        is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in user_message)
                        
                        # Log RAG results
                        if retrieved_context:
                            app.logger.info(f"[RAG] Retrieved {len(retrieved_context)} characters in {retrieval_time:.2f}s")
                            app.logger.info(f"[RAG] Context sample: '{retrieved_context[:200]}...'")
                        else:
                            app.logger.info(f"[RAG] No relevant context found")
                            # Special handling for Chinese queries with no results
                            if is_chinese_query and bot_id:
                                app.logger.info(f"[RAG] Chinese query with no results. Trying direct document lookup.")
                                try:
                                    # Try to get system prompt which might have relevant info
                                    bot = Bot.get(bot_id)
                                    if bot:
                                        # Use description as context
                                        bot_info = f"Bot name: {bot.name}\nDescription: {bot.description}\n"
                                        # Use any knowledge file metadata
                                        if bot.knowledge_files and len(bot.knowledge_files) > 0:
                                            bot_info += "\nKnowledge files:\n"
                                            for file in bot.knowledge_files:
                                                bot_info += f"- {file}\n"
                                        
                                        # Use this as minimal context
                                        retrieved_context = bot_info
                                        app.logger.info(f"[RAG] Using bot metadata as minimal context: {len(retrieved_context)} chars")
                                except Exception as fb_err:
                                    app.logger.error(f"[RAG] Error in Chinese fallback: {str(fb_err)}")
                                    # Continue with empty context
                    except concurrent.futures.TimeoutError:
                        app.logger.warning(f"[RAG] Retrieval timed out after 120 seconds, continuing without context")
                        retrieved_context = ""  # Empty context if timeout
                        future.cancel()
            except Exception as e:
                app.logger.error(f"[RAG] Error during retrieval: {str(e)}")
                retrieved_context = ""  # Empty context on error
                
        # Debug info for bot identification
        if bot_id and not secret:
            app.logger.info(f"[RAG] Using bot_id: {bot_id}")
            bot_collection = f"bot_{bot_id}"
            
            # Log available collections
            try:
                available_collections = rag.list_collections()
                app.logger.info(f"[RAG] Available collections: {available_collections}")
                
                # Check for possible bot collection names if the exact match isn't found
                if bot_collection not in available_collections:
                    possible_names = [
                        f"bot_{bot_id}",
                        f"bot-{bot_id}",
                        f"{bot_id}",
                        f"bot_{bot_id.replace('-', '_')}",
                        f"bot{bot_id}"
                    ]
                    app.logger.info(f"[RAG] Checking possible bot collection names: {possible_names}")
                    
                    for name in possible_names:
                        if name in available_collections:
                            app.logger.info(f"[RAG] Found potential bot collection match: '{name}'")
                            # Check if it has documents
                            try:
                                doc_count = rag.count_documents(name)
                                app.logger.info(f"[RAG] Potential bot collection '{name}' has {doc_count} documents")
                                if doc_count > 0:
                                    app.logger.info(f"[RAG] Trying potential bot collection '{name}' with {doc_count} documents")
                            except Exception as e:
                                app.logger.error(f"[RAG] Error checking potential bot collection '{name}': {str(e)}")
            except Exception as e:
                app.logger.error(f"[RAG] Error listing collections: {str(e)}")
                
            # Get bot information
            try:
                bot = Bot.get(bot_id)
                if bot:
                    app.logger.info(f"[RAG] Found bot: {bot.name} (ID: {bot.id})")
                    if bot.knowledge_files:
                        app.logger.info(f"[RAG] Bot has {len(bot.knowledge_files)} knowledge files: {bot.knowledge_files}")
                    else:
                        app.logger.info(f"[RAG] Bot has no knowledge files")
            except Exception as e:
                app.logger.error(f"[RAG] Error getting bot information: {str(e)}")
                
        # Now check if we need to update the rag.py to support the bot_collection parameter
        # If the rag.retrieve_context function doesn't support the bot_collection parameter yet,
        # it will fail and fall back to the old behavior

            
            # Check bot's knowledge base if we have a bot_id
            if bot_id and bot_collection:
                try:
                    # Get the bot
                    bot = Bot.get(bot_id)
                    if bot and bot.knowledge_files:
                        app.logger.info(f"[RAG] Checking bot {bot_id} knowledge base with {len(bot.knowledge_files)} files")
                        
                        # Update the conversation model to show the bot name instead of base model
                        if not secret and conversation_id:
                            try:
                                # Format as bot:id to indicate this is a bot conversation
                                bot_model = f"bot:{bot.id}"
                                db.update_conversation_model(conversation_id, bot_model)
                                app.logger.info(f"[RAG] Updated conversation {conversation_id} model to {bot_model}")
                            except Exception as model_err:
                                app.logger.error(f"[RAG] Error updating conversation model: {str(model_err)}")
                                # Non-critical error, continue processing
                        
                        # Debug collection name
                        app.logger.info(f"[RAG] Checking bot collection: {bot_collection}")
                        
                        # Function to re-index bot knowledge files if needed
                        def reindex_bot_knowledge(bot):
                            try:
                                app.logger.info(f"[RAG] Re-indexing knowledge files for bot {bot.id}")
                                kb_dir = bot.get_knowledge_base_path()
                                collection_name = f"bot_{bot.id}"
                                
                                # Process each file
                                indexed_files = []
                                for filename in bot.knowledge_files:
                                    try:
                                        file_path = os.path.join(kb_dir, secure_filename(filename))
                                        if not os.path.exists(file_path):
                                            app.logger.warning(f"[RAG] File {filename} not found at {file_path}")
                                            continue
                                            
                                        # Select appropriate loader
                                        if filename.endswith('.pdf'):
                                            loader = PyPDFLoader(file_path)
                                        elif filename.endswith('.xml'):
                                            loader = UnstructuredXMLLoader(file_path)
                                        elif filename.endswith('.md'):
                                            loader = UnstructuredMarkdownLoader(file_path)
                                        else:  # .txt
                                            loader = TextLoader(file_path)
                                        
                                        # Load and process documents
                                        app.logger.info(f"[RAG] Loading {filename} for re-indexing")
                                        documents = loader.load()
                                        
                                        # Add metadata
                                        for doc in documents:
                                            if not hasattr(doc, 'metadata') or not doc.metadata:
                                                doc.metadata = {}
                                            doc.metadata['source'] = file_path
                                            doc.metadata['filename'] = filename
                                            doc.metadata['bot_id'] = bot.id
                                        
                                        # Split documents
                                        text_splitter = rag.RecursiveCharacterTextSplitter(
                                            chunk_size=rag.CHUNK_SIZE, 
                                            chunk_overlap=rag.CHUNK_OVERLAP
                                        )
                                        split_docs = text_splitter.split_documents(documents)
                                        app.logger.info(f"[RAG] Split {filename} into {len(split_docs)} chunks")
                                        
                                        # Add to vector store
                                        vectorstore = rag.Chroma(
                                            persist_directory=rag.CHROMA_PERSIST_DIR, 
                                            embedding_function=rag.ollama_ef, 
                                            collection_name=collection_name
                                        )
                                        vectorstore.add_documents(split_docs)
                                        vectorstore.persist()
                                        
                                        indexed_files.append(filename)
                                        app.logger.info(f"[RAG] Successfully re-indexed {filename}")
                                    except Exception as e:
                                        app.logger.error(f"[RAG] Error re-indexing {filename}: {str(e)}")
                                
                                return indexed_files
                            except Exception as e:
                                app.logger.error(f"[RAG] Error in re-indexing: {str(e)}")
                                return []
                        
                        # Force check for documents in the collection
                        try:
                            # Direct check with Chroma
                            vectorstore = rag.Chroma(
                                persist_directory=rag.CHROMA_PERSIST_DIR, 
                                embedding_function=rag.ollama_ef, 
                                collection_name=bot_collection
                            )
                            collection = vectorstore._collection
                            count = collection.count()
                            app.logger.info(f"[RAG] Bot collection '{bot_collection}' has {count} documents")
                            
                            if count > 0:
                                app.logger.info(f"[RAG] Documents found for bot {bot_id}, using bot RAG")
                                app.logger.info(f"[RAG] Recommend using /api/bots/{bot_id}/knowledge/reindex endpoint to index files")
                        except Exception as e:
                            app.logger.error(f"[RAG] Error checking bot collection: {str(e)}")
                            app.logger.info(f"[RAG] Falling back to has_documents check")
                            
                            # We no longer use fallbacks - RAG can only be enabled through our explicit conditions
                            if rag.has_documents(collection_name=bot_collection):
                                app.logger.info(f"[RAG] Documents found for bot {bot_id} using fallback check, but not enabling RAG")
                                # We don't set use_rag = True here anymore to ensure RAG only runs under our specified conditions
                                
                                # Log RAG query details
                                app.logger.info(f"[RAG] Querying bot knowledge with: '{user_message[:100]}...'")
                                
                                # Retrieve context from bot's knowledge base with timeout handling
                                retrieval_start = time.time()
                                try:
                                    # Generate a unique operation ID for tracking
                                    operation_id = f"rag_bot_{bot_id}_{int(time.time())}"
                                    
                                    # Set a timeout for retrieval (180 seconds - longer for bot knowledge)
                                    import concurrent.futures
                                    
                                    # Use ThreadPoolExecutor to run with timeout
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(rag.retrieve_context, 
                                                              user_message, 
                                                              collection_name=bot_collection,
                                                              operation_id=operation_id)
                                        
                                        # Return the operation_id to the client for progress tracking
                                        app.logger.info(f"[RAG] Started retrieval operation: {operation_id}")
                                        
                                        try:
                                            # Get the result with timeout
                                            bot_context = future.result(timeout=180)  # 3 minute timeout
                                            retrieval_time = time.time() - retrieval_start
                                            
                                            if bot_context:
                                                retrieved_context = bot_context
                                                app.logger.info(f"[RAG] Retrieved {len(retrieved_context)} characters from bot knowledge in {retrieval_time:.2f}s")
                                                app.logger.info(f"[RAG] Bot knowledge sample: '{retrieved_context[:200]}...'")
                                            else:
                                                app.logger.info(f"[RAG] No relevant context found in bot's knowledge base")
                                        except concurrent.futures.TimeoutError:
                                            # Handle timeout - continue without RAG context
                                            app.logger.warning(f"[RAG] Bot knowledge retrieval timed out after 180 seconds, continuing without context")
                                            # Cancel the future if possible
                                            future.cancel()
                                except Exception as e:
                                    # Handle any other exceptions during retrieval
                                    app.logger.error(f"[RAG] Error during bot knowledge retrieval: {str(e)}")
                            else:
                                app.logger.info(f"[RAG] Bot {bot_id} has {len(bot.knowledge_files)} files but no indexed documents in collection {bot_collection}")
                                
                                # Try to list all collections for debugging
                                try:
                                    from chromadb.config import Settings
                                    import chromadb
                                    client = chromadb.PersistentClient(path=rag.CHROMA_PERSIST_DIR)
                                    collections = client.list_collections()
                                    app.logger.info(f"[RAG] Available collections: {[c.name for c in collections]}")
                                except Exception as ce:
                                    app.logger.error(f"[RAG] Error listing collections: {str(ce)}")
                        
                except Exception as e:
                    app.logger.error(f"[RAG] Error checking bot knowledge: {str(e)}")
        
        # If still no RAG context found, do a direct check of all collections
        if not use_rag:
            try:
                # Try to directly list all collections
                from chromadb.config import Settings
                import chromadb
                client = chromadb.PersistentClient(path=rag.CHROMA_PERSIST_DIR)
                app.logger.info(f"[RAG] Looking for information from collections (bot, conversation, etc.)")
                collection_names = rag.list_collections()
                app.logger.info(f"[RAG] Found {len(collection_names)} collections: {collection_names}")
                
                # Set the rag_enabled flag based on use_rag variable
                rag_enabled = use_rag
                
                # First, try to see if we already have an exact collection for this bot ID
                # This helps us avoid checking all collections
                possible_bot_collections = [
                    f"bot_{bot_id}",                # Standard format
                    f"bot-{bot_id}",                # With hyphen
                    bot_id,                         # Just the ID
                    f"bot_{bot_id.replace('-', '_')}",  # Underscores instead of hyphens
                    f"bot{bot_id}",                 # No separator
                ]
                
                app.logger.info(f"[RAG] Checking possible bot collection names: {possible_bot_collections}")
                
                # Check collections that might match our bot ID
                for coll_name in collection_names:
                    if any(possible_name in coll_name for possible_name in possible_bot_collections):
                        app.logger.info(f"[RAG] Found potential bot collection match: '{coll_name}'")
                        try:
                            # Try to get document count
                            vectorstore = rag.Chroma(
                                persist_directory=rag.CHROMA_PERSIST_DIR, 
                                embedding_function=rag.ollama_ef, 
                                collection_name=coll_name
                            )
                            count = vectorstore._collection.count()
                            app.logger.info(f"[RAG] Potential bot collection '{coll_name}' has {count} documents")
                            
                            # If this collection has documents, try using it
                            if count > 0 and not retrieved_context:
                                app.logger.info(f"[RAG] Trying potential bot collection '{coll_name}' with {count} documents")
                                retrieval_start = time.time()
                                alt_context = rag.retrieve_context(user_message, collection_name=coll_name)
                                retrieval_time = time.time() - retrieval_start
                                
                                if alt_context:
                                    # We don't set RAG to True here anymore - only enabled through our explicit rules
                                    app.logger.info(f"[RAG] Found context in potential bot collection, but not enabling RAG")
                                    # Storing the context but NOT enabling RAG outside our rules
                                    retrieved_context = alt_context
                                    app.logger.info(f"[RAG] Retrieved {len(retrieved_context)} characters from potential bot collection '{coll_name}' in {retrieval_time:.2f}s")
                                    app.logger.info(f"[RAG] Content sample: '{retrieved_context[:200]}...'")
                                    break
                        except Exception as e:
                            app.logger.error(f"[RAG] Error checking collection '{coll_name}': {str(e)}")
                
                # If still no context, check if we should try the bot collection directly
                # This happens when the conversation is associated with a bot but the bot_id wasn't explicitly passed
                if not retrieved_context and conversation_id and not bot_id:
                    try:
                        conversation = db.get_conversation(conversation_id)
                        if conversation and 'model' in conversation and conversation['model'].startswith('bot:'):
                            extracted_bot_id = conversation['model'].split(':', 1)[1]
                            bot_collection = f"bot_{extracted_bot_id}"
                            
                            if bot_collection in collection_names:
                                app.logger.info(f"[RAG] Found bot collection {bot_collection} from conversation model")
                                try:
                                    # Try to get document count
                                    vectorstore = rag.Chroma(
                                        persist_directory=rag.CHROMA_PERSIST_DIR, 
                                        embedding_function=rag.ollama_ef, 
                                        collection_name=bot_collection
                                    )
                                    count = vectorstore._collection.count()
                                    app.logger.info(f"[RAG] Bot collection '{bot_collection}' has {count} documents")
                                    
                                    # If this collection has documents, try using it
                                    if count > 0:
                                        app.logger.info(f"[RAG] Trying bot collection '{bot_collection}' with {count} documents")
                                        retrieval_start = time.time()
                                        bot_context = rag.retrieve_context(user_message, collection_name=bot_collection)
                                        retrieval_time = time.time() - retrieval_start
                                        
                                        if bot_context:
                                            retrieved_context = bot_context
                                            app.logger.info(f"[RAG] Retrieved {len(retrieved_context)} characters from collection '{bot_collection}' in {retrieval_time:.2f}s")
                                except Exception as e:
                                    app.logger.error(f"[RAG] Error using bot collection '{bot_collection}': {str(e)}")
                    except Exception as e:
                        app.logger.error(f"[RAG] Error checking conversation model: {str(e)}")
                
                # We no longer check all remaining collections - RAG only runs under our specified conditions
                check_all_collections = data.get('check_all_collections', False)
                if not retrieved_context and check_all_collections and rag_enabled:
                    app.logger.info(f"[RAG] Check all collections feature is disabled to ensure RAG only runs under specified conditions")
                    # We don't check all collections anymore to ensure RAG only runs under our specified conditions
                            
                # We no longer check other collections - RAG only runs under our specified conditions
                # No additional RAG triggering conditions
                if not retrieved_context and rag_enabled:
                    app.logger.info(f"[RAG] All additional RAG features are disabled to ensure RAG only runs under specified conditions")
                    # RAG can only be enabled through our explicit rule conditions
                
                # Log if RAG is enabled for debugging
                if use_rag:
                    app.logger.info(f"[RAG] RAG is enabled according to our specified conditions")
                else:
                    app.logger.info(f"[RAG] RAG is disabled according to our specified conditions")
            except Exception as e:
                app.logger.error(f"[RAG] Error listing collections: {str(e)}")
            
            if not use_rag:
                app.logger.info(f"[RAG] No documents found in conversation or bot knowledge, skipping RAG")
        # --- End RAG Integration ---
        
        # --- Chat History Integration ---
        if not secret:
            previous_messages = db.get_conversation_messages(conversation_id)
        else:
            previous_messages = []
        
        # Format the chat history for the model - Ollama expects an array of messages
        # Each message should have a role (user or assistant) and content
        messages = []
        
        # Check if the current user message is already in the database
        current_user_message_in_history = False
        current_user_message_id = None
        
        # Look for the current message in the history
        for msg in reversed(previous_messages):  # Start from the most recent
            if msg['role'] == 'user' and msg.get('content') == user_message and 'id' in msg:
                current_user_message_in_history = True
                current_user_message_id = msg['id']
                break
        
        # Build the message history
        history_messages = []
        
        # If we found our current message in history, we need to include it and exclude duplicates
        if current_user_message_in_history:
            # Include all messages except duplicates of the current user message
            seen_current_message = False
            for msg in previous_messages:
                # Only include the first instance of the current message
                if current_user_message_id and 'id' in msg and msg['id'] == current_user_message_id:
                    if not seen_current_message:
                        history_messages.append(msg)
                        seen_current_message = True
                    continue  # Skip duplicates
                history_messages.append(msg)
        else:
            # Current message not in history, so include all previous messages
            history_messages = previous_messages
        
        # Limit to the last 20 messages to avoid context length issues
        if len(history_messages) > 20:
            history_messages = history_messages[-20:]
        
        # Add previous messages to the context
        seen_messages = set()
        for msg in history_messages:
            if msg['role'] in ['user', 'assistant']:
                # Create a unique key for each message to detect duplicates
                msg_key = (msg['role'], msg['content'])
                if msg_key not in seen_messages:
                    clean_msg = {
                        "role": msg['role'],
                        "content": msg['content']
                    }
                    messages.append(clean_msg)
                    seen_messages.add(msg_key)
        
        # Add system prompt first if it exists
        system_prompt = data.get('system_prompt', '')
        
        # If a bot is selected, use the bot's system prompt and base model instead
        if bot_id:
            try:
                # Get the bot
                bot = Bot.get(bot_id)
                if bot:
                    # Use bot's system prompt if available
                    if bot.system_prompt:
                        app.logger.info(f"[BOT] Using bot's system prompt for conversation")
                        system_prompt = bot.system_prompt
                    
                    # Use bot's base model if available
                    if bot.base_model:
                        app.logger.info(f"[BOT] Using bot's base model {bot.base_model} for conversation")
                        # Only override the model if the original model was a bot selection
                        if original_model.startswith('bot:'):
                            model_name = bot.base_model
                            app.logger.info(f"[BOT] Overriding model selection from {original_model} to {model_name}")
            except Exception as e:
                app.logger.error(f"[BOT] Error getting bot information: {str(e)}")
                # Continue with the system prompt and model from the request
        
        # If we have RAG context, add it to the system prompt
        if use_rag and retrieved_context:
            if web_search:
                # For web searches, include the raw results directly
                context_prefix = "Here are the latest web search results for your query:\n\n"
                enhanced_system_prompt = f"{system_prompt}\n\n{context_prefix}{retrieved_context}"
            elif system_prompt:
                # For regular RAG, use the standard knowledge base format
                enhanced_system_prompt = f"{system_prompt}\n\nKnowledge Base Information:\n{retrieved_context}"
            else:
                # Create new system prompt with RAG context
                enhanced_system_prompt = f"You have access to the following knowledge base information. Use it to answer the user's question accurately:\n\n{retrieved_context}"
            
            # Log the enhanced system prompt
            app.logger.info(f"[RAG] {'Web search' if web_search else 'Knowledge base'} context added to prompt")
            
            # Add the enhanced system prompt
            system_msg = {
                "role": "system",
                "content": enhanced_system_prompt.strip()
            }
            messages.append(system_msg)
        elif system_prompt:
            # Add original system prompt if no RAG context
            system_msg = {
                "role": "system",
                "content": system_prompt
            }
            messages.append(system_msg)
            
        # Create the current user message
        user_msg = {
            "role": "user",
            "content": user_message
        }
        
        # Check if there are images to include
        images = data.get('images', [])
        if images:
            user_msg["images"] = images
            
        # Only add current message if not already in history
        if ("user", user_message) not in seen_messages:
            messages.append(user_msg)
        
        system_prompt = data.get('system_prompt', '')
        if system_prompt:
            system_msg = {
                "role": "system",
                "content": system_prompt
            }
            messages.insert(0, system_msg)
        messages.append(user_msg)
        # (Generation logic continues unchanged)
        
        # Check if the model supports thinking process
        supports_thinking = model_name in THINKING_MODELS
        
        # Prepare the RAG prompt template
        rag_prompt_template = """Use the following context to answer the question. If the context doesn't contain the answer, state that you don't know based on the provided documents.

Context:
{context}

Question: {question}

Answer:"""
        
        # If RAG context is available, we need to modify the last user message
        if retrieved_context:
            # Prepare the RAG-enhanced prompt
            rag_prompt = rag_prompt_template.format(context=retrieved_context, question=user_message)
            
            # If thinking is supported, add the thinking instruction
            if supports_thinking:
                rag_prompt = (
                    "Please think step-by-step about this question and show your reasoning "
                    "inside <think></think> tags before giving your final answer.\n\n"
                    f"{rag_prompt}"
                )
                
            # Update the last message with the RAG-enhanced prompt
            messages[-1]["content"] = rag_prompt
            
        elif supports_thinking:
            # If no RAG but thinking is supported, add thinking instruction to the last message
            messages[-1]["content"] = (
                "Please think step-by-step about this question and show your reasoning "
                "inside <think></think> tags before giving your final answer.\n\n"
                f"{user_message}"
            )
        
        # Add images if present
        if images and len(messages) > 0:
            messages[-1]["images"] = images  # Add images to the last message (current user message)
        
        def generate_stream():
            try:
                client = ollama.Client(host=OLLAMA_HOST)
                thinking = ""
                full_response = ""
                
                # Stream the response from Ollama
                for chunk in client.chat(
                    model=model_name,
                    messages=messages,  # Pass the complete message history
                    stream=True
                ):
                    if chunk.get('message', {}).get('content'):
                        chunk_text = chunk['message']['content']
                        full_response += chunk_text
                        
                        # Extract thinking content if present and model supports it
                        if supports_thinking and '<think>' in full_response and '</think>' in full_response:
                            think_start = full_response.find('<think>') + len('<think>')
                            think_end = full_response.find('</think>') + len('</think>')
                            thinking = full_response[think_start:think_end].strip()
                        
                        yield f"data: {json.dumps({'chunk': chunk_text, 'done': False})}\n\n"
                
                # After streaming is complete, extract final thinking content if present
                if supports_thinking and '<think>' in full_response and '</think>' in full_response:
                    think_start = full_response.find('<think>') + len('<think>')
                    think_end = full_response.find('</think>') + len('</think>')
                    thinking = full_response[think_start:think_end].strip()
                    # Remove thinking tags from the final response
                    clean_response = full_response.replace(full_response[full_response.find('<think>'):full_response.find('</think>') + len('</think>')], '')
                    full_response = clean_response.strip()
                
                # Save the assistant's response to the database and get its ID
                if not secret:
                    conn = sqlite3.connect(db.DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO messages 
                        (conversation_id, role, content, thinking, images, attachment_filename) 
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        conversation_id,
                        'assistant',
                        full_response,
                        thinking,
                        json.dumps([None]),  # images - stored as JSON list of base64 strings
                        None,  # attachment_filename
                    ))
                    conn.commit()
                    conn.close()
                
                # Check if this is the first message exchange for this conversation
                # We can determine this by counting the messages after our new ones are added
                if not secret:
                    conversation_messages = db.get_conversation_messages(conversation_id)
                    is_first_exchange = len(conversation_messages) <= 2  # Just the user message and this assistant message
                
                # Send the final message with the complete response AND the message IDs
                yield f"data: {json.dumps({'full_response': full_response, 'thinking': thinking, 'done': True, 'is_first_exchange': is_first_exchange if not secret else False})}\n\n"
                
                # Trigger title generation for first exchanges
                if not secret and is_first_exchange:
                    try:
                        # Attempt to generate a title for the conversation
                        app.logger.info(f"Attempting to generate title for first exchange in conversation {conversation_id}")
                        title_result = generate_conversation_title(conversation_id)
                        
                        # If title generation was successful, send a title update event
                        if isinstance(title_result, dict) and title_result.get('status') == 'success':
                            title = title_result.get('title')
                            if title:
                                app.logger.info(f"Generated title: {title}, sending title update event")
                                yield f"data: {json.dumps({'title_update': True, 'title': title, 'done': True})}\n\n"
                    except Exception as e:
                        app.logger.error(f"Error generating title for conversation {conversation_id}: {str(e)}")
                
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        
        return Response(generate_stream(), mimetype='text/event-stream')
    except Exception as e:
        print(f"Error in /api/generate: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/<id>/generate-title', methods=['POST'])
@login_required
def generate_conversation_title(id):
    """Generate a title for the conversation based on its messages using an LLM"""
    try:
        # Get the model from query parameters or use default
        model = request.args.get('model', DEFAULT_MODEL)
        app.logger.info(f"Generating title for conversation {id} using model {model}")
        
        # Get the conversation
        conversation = db.get_user_conversation(current_user.id, id)
        if not conversation:
            app.logger.warning(f"Conversation {id} not found")
            return jsonify({
                'status': 'error',
                'message': 'Conversation not found'
            }), 404
            
        # Get messages
        messages = db.get_conversation_messages(id)
        if not messages or len(messages) == 0:
            app.logger.warning(f"No messages found for conversation {id}")
            return jsonify({
                'status': 'error',
                'message': 'No messages found'
            }), 404
        
        # Check if we have at least one user message and one assistant message
        user_messages = [m for m in messages if m['role'] == 'user']
        assistant_messages = [m for m in messages if m['role'] == 'assistant']
        
        if not user_messages:
            app.logger.warning(f"No user message found in conversation {id}")
            return jsonify({
                'status': 'error',
                'message': 'No user message found'
            }), 404
        
        # Extract first user message for fallback title
        first_message = user_messages[0]['content']
        fallback_title = first_message[:47] + '...' if len(first_message) > 50 else first_message
        
        # Check if the model is valid (don't restrict to AVAILABLE_MODELS)
        valid_model = model and model != 'none'
        
        # Use the model if it's valid, otherwise use fallback
        if not valid_model:
            app.logger.info(f"No valid model specified for title generation, using fallback")
            title = fallback_title
        else:
            try:
                # Prepare messages for LLM
                # Get up to first 3 message pairs for context
                context_messages = []
                for i, msg in enumerate(messages):
                    if i >= 6:  # Limit to first 3 pairs (6 messages)
                        break
                    context_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
                
                # Create a very simple, direct prompt for title generation
                system_prompt = "You will create a short, descriptive title (3-7 words) for a conversation. ONLY respond with the title text."
                
                # Prepare messages for the LLM
                llm_messages = [
                    {"role": "system", "content": system_prompt}
                ]
                
                # Add context messages
                llm_messages.extend(context_messages)
                
                # Add final instruction with explicit formatting - keep it extremely simple
                llm_messages.append({"role": "user", "content": "Create a title for this conversation. Respond with ONLY the title text."})
                
                # For thinking models, use a more direct approach
                if model in THINKING_MODELS:
                    # Replace with a more direct approach for thinking models
                    llm_messages = [
                        {"role": "system", "content": "You will output ONLY a short title (3-7 words). No explanation, no thinking, no tags."}
                    ]
                    llm_messages.extend(context_messages)
                    llm_messages.append({"role": "user", "content": "Title for this conversation (3-7 words):"})



                
                # Call the LLM with timeout
                app.logger.info(f"Calling LLM with {len(llm_messages)} messages")
                
                # No need for a separate clean_title function anymore - we've simplified the approach
                    
                # Try to connect to Ollama with a short timeout and catch ALL exceptions
                title = fallback_title  # Default to fallback
                try:
                    # Use a very short timeout to avoid hanging the request
                    app.logger.info(f"Calling LLM with model {model} for title generation")
                    
                    response = call_llm(model, llm_messages, max_tokens=50, temperature=0.3, timeout=5)
                    app.logger.info(f"Raw LLM response: '{response}'")
                    
                    # Only use the response if it's valid
                    if response and response.strip():
                        # Simple cleanup approach
                        raw_title = response.strip()
                        
                        # Take first line only if multiple lines
                        if '\n' in raw_title:
                            raw_title = raw_title.split('\n')[0].strip()
                        
                        # Remove any HTML-like tags
                        import re
                        raw_title = re.sub(r'<[^>]+>', '', raw_title)
                        
                        # Remove quotes, punctuation, and extra spaces
                        clean_title = raw_title.strip('"\'\'.,!?').strip()
                        
                        # Capitalize the first letter of each word for consistency
                        clean_title = ' '.join(word.capitalize() for word in clean_title.split())
                        
                        # Limit to 50 characters if somehow still too long
                        if len(clean_title) > 50:
                            clean_title = clean_title[:47] + '...'
                            
                        # If we ended up with nothing useful, use fallback
                        if clean_title and len(clean_title) >= 3:
                            title = clean_title
                        
                        app.logger.info(f"Final generated title: {title}")
                    else:
                        app.logger.warning("LLM returned empty title, using fallback")
                        app.logger.info(f"Using fallback title: {title}")
                        
                except ConnectionError as e:
                    # Specific handling for connection errors
                    app.logger.warning(f"Ollama connection error: {str(e)}")
                    app.logger.info(f"Using fallback title: {title}")
                except ValueError as e:
                    # Specific handling for model errors
                    app.logger.warning(f"Model error: {str(e)}")
                    app.logger.info(f"Using fallback title: {title}")
                except Exception as e:
                    # Catch-all for any other errors
                    app.logger.warning(f"Unexpected error in title generation: {str(e)}")
            except Exception as e:
                app.logger.error(f"Error in title generation process: {str(e)}")
                title = fallback_title
        
        # Update conversation title
        db.update_conversation_title(id, title)
        
        # Return the updated conversation data
        updated_conversation = db.get_user_conversation(current_user.id, id)
        return jsonify({
            'status': 'success',
            'title': title,
            'conversation': updated_conversation
        })
    except Exception as e:
        app.logger.error(f"Error generating title for conversation {id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/embedding_models', methods=['GET'])
def list_embedding_models():
    """Endpoint to list available embedding models"""
    try:
        app.logger.info("Attempting to get available embedding models...")
        
        # Get embedding models from config
        embedding_models = []
        for model_name in config.AVAILABLE_EMBEDDING_MODELS:
            embedding_models.append({"name": model_name})
        
        app.logger.info(f"Available embedding models: {embedding_models}")
        
        return jsonify({
            'models': embedding_models,
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Server error listing embedding models: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/preferences', methods=['GET'])
@login_required
def get_preferences():
    """Get user preferences"""
    try:
        # Get user ID
        user_id = str(current_user.id)
        
        # Load preferences from file
        preferences = {}
        try:
            if os.path.exists('data/user_preferences.json'):
                with open('data/user_preferences.json', 'r') as f:
                    all_prefs = json.load(f)
                    preferences = all_prefs.get(user_id, {})
        except Exception as e:
            app.logger.error(f"Error loading preferences: {e}")
            preferences = {}
        
        # If no preferences found, use defaults
        if not preferences:
            preferences = {
                'default_model': None,
                'default_embedding_model': 'nomic-embed-text',
                'visible_models': [],  # Empty means show all
                'theme': 'system',
                'show_thinking': False
            }
        
        # Get available models for the preferences UI
        available_models = config.get_available_models()
        
        # Get available embedding models
        embedding_models = config.get_available_embedding_models()
        
        return jsonify({
            'preferences': preferences,
            'available_models': available_models,
            'embedding_models': embedding_models,
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Error getting preferences: {str(e)}")
        return jsonify({
            'message': 'Failed to get preferences',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/preferences', methods=['PUT'])
@login_required
def update_preferences():
    """Update user preferences"""
    try:
        data = request.json
        user_id = current_user.id
        
        # Get current preferences
        prefs = USER_PREFERENCES.get(user_id, {})
        
        # Update with new values
        if 'default_model' in data:
            # Validate model
            if data['default_model'] in AVAILABLE_MODELS:
                prefs['default_model'] = data['default_model']
        
        if 'default_embedding_model' in data:
            # Validate embedding model
            if data['default_embedding_model'] in AVAILABLE_EMBEDDING_MODELS:
                prefs['default_embedding_model'] = data['default_embedding_model']
        
        if 'visible_models' in data and isinstance(data['visible_models'], list):
            # Filter to only valid models
            visible_models = [m for m in data['visible_models'] if m in AVAILABLE_MODELS]
            prefs['visible_models'] = visible_models
        
        if 'theme' in data:
            if data['theme'] in ['light', 'dark', 'system']:
                prefs['theme'] = data['theme']
        
        if 'show_thinking' in data:
            prefs['show_thinking'] = bool(data['show_thinking'])
        
        # Only admins can update system-wide settings
        if current_user.role == 'admin':
            if 'system_settings' in data and isinstance(data['system_settings'], dict):
                system_settings = data['system_settings']
                
                # Update system settings
                if 'default_models' in system_settings:
                    # TODO: Update system default models
                    pass
        
        # Save preferences
        USER_PREFERENCES[str(user_id)] = prefs
        
        # Save to disk
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Read existing preferences to merge
            existing_prefs = {}
            try:
                if os.path.exists('data/user_preferences.json'):
                    with open('data/user_preferences.json', 'r') as f:
                        existing_prefs = json.load(f)
            except Exception:
                pass  # If file is corrupted, start fresh
            
            # Update with new preferences
            existing_prefs[str(user_id)] = prefs
            
            # Write back to file
            with open('data/user_preferences.json', 'w') as f:
                json.dump(existing_prefs, f, indent=2)
        except Exception as e:
            app.logger.error(f"Error saving preferences to disk: {e}")
        
        return jsonify({
            'status': 'success',
            'preferences': prefs
        })
    except Exception as e:
        app.logger.error(f"Error updating preferences: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to update preferences',
            'details': str(e)
        }), 500

@app.route('/me', methods=['GET'])
def me():
    if current_user.is_authenticated:
        return jsonify({
            'logged_in': True,
            'username': current_user.username,
            'role': current_user.role,
            'display_name': current_user.display_name
        })
    else:
        return jsonify({'logged_in': False})

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    data = request.json
    user_id = current_user.id
    
    # Update display name if provided
    if 'display_name' in data:
        display_name = data['display_name']
        if display_name and display_name.strip():
            success = db.update_user_display_name(user_id, display_name)
            if not success:
                return jsonify({'status': 'error', 'message': 'Failed to update display name'}), 500
    
    # Update password if provided
    if 'current_password' in data and 'new_password' in data:
        current_password = data['current_password']
        new_password = data['new_password']
        
        # Verify current password
        user_dict = db.get_user_by_id(user_id)
        if not bcrypt.verify(current_password, user_dict['password_hash']):
            return jsonify({'status': 'error', 'message': 'Current password is incorrect'}), 400
        
        # Update password
        new_password_hash = bcrypt.hash(new_password)
        success = db.update_user_password(user_id, new_password_hash)
        if not success:
            return jsonify({'status': 'error', 'message': 'Failed to update password'}), 500
    
    return jsonify({'status': 'success', 'message': 'Profile updated successfully'})

# --- Admin User Management Routes ---
@app.route('/api/users', methods=['GET'])
@login_required
@admin_required
def list_users():
    """List all users (admin only)"""
    try:
        users = db.get_all_users()
        return jsonify({
            'status': 'success',
            'users': users
        })
    except Exception as e:
        app.logger.error(f"Error listing users: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve users'
        }), 500

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@login_required
def update_user(user_id):
    """Update a user's information"""
    try:
        data = request.json
        
        # Only admins can change roles, and users can only update their own info
        is_admin = current_user.role == 'admin'
        is_self = user_id == current_user.id
        
        if not is_admin and not is_self:
            return jsonify({
                'status': 'error',
                'message': 'You can only update your own information'
            }), 403
        
        # Update role if provided (admin only)
        if 'role' in data:
            if not is_admin:
                return jsonify({
                    'status': 'error',
                    'message': 'Only admins can change roles'
                }), 403
                
            # Don't allow changing your own role
            if is_self:
                return jsonify({
                    'status': 'error',
                    'message': 'You cannot change your own role'
                }), 400
                
            role = data['role']
            if role not in ['admin', 'user']:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid role'
                }), 400
                
            success = db.update_user_role(user_id, role)
            if not success:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to update user role'
                }), 500
        
        # Update display name if provided
        if 'display_name' in data:
            display_name = data['display_name']
            if not display_name:
                return jsonify({
                    'status': 'error',
                    'message': 'Display name cannot be empty'
                }), 400
                
            success = db.update_user_display_name(user_id, display_name)
            if not success:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to update display name'
                }), 500
                
        # Update email if provided
        if 'email' in data:
            email = data['email']
            if email and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid email format'
                }), 400
                
            success = db.update_user_email(user_id, email)
            if not success:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to update email'
                }), 500
        
        return jsonify({
            'status': 'success',
            'message': 'User updated successfully'
        })
    except Exception as e:
        app.logger.error(f"Error updating user: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to update user'
        }), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user (admin only)"""
    try:
        # Don't allow deleting yourself
        if user_id == current_user.id:
            return jsonify({
                'status': 'error',
                'message': 'You cannot delete your own account'
            }), 400
            
        success = db.delete_user(user_id)
        if not success:
            return jsonify({
                'status': 'error',
                'message': 'Failed to delete user'
            }), 500
            
        return jsonify({
            'status': 'success',
            'message': 'User deleted successfully'
        })
    except Exception as e:
        app.logger.error(f"Error deleting user: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to delete user'
        }), 500

@app.route('/api/users/<int:user_id>/password', methods=['PUT'])
@login_required
@admin_required
def admin_change_user_password(user_id):
    """Allow admin to change another user's password."""
    try:
        data = request.json
        new_password = data.get('new_password')
        if not new_password or len(new_password) < 6:
            return jsonify({'status': 'error', 'message': 'Password must be at least 6 characters'}), 400
        # Don't allow changing your own password here (use profile update)
        if user_id == current_user.id:
            return jsonify({'status': 'error', 'message': 'You cannot change your own password here'}), 400
        from passlib.hash import bcrypt
        new_password_hash = bcrypt.hash(new_password)
        success = db.update_user_password(user_id, new_password_hash)
        if not success:
            return jsonify({'status': 'error', 'message': 'Failed to update password'}), 500
        return jsonify({'status': 'success', 'message': 'Password updated successfully'})
    except Exception as e:
        app.logger.error(f"Error changing user password: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to update password'}), 500

@app.route('/api/users/<int:user_id>/quota', methods=['GET'])
@login_required
@admin_required
def get_user_quota_api(user_id):
    """Get quota information for a user (admin only)"""
    try:
        # Check if user exists
        user = db.get_user_by_id(user_id)
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
            
        quota = db.get_user_quota(user_id)
        if not quota:
            return jsonify({
                'status': 'error',
                'message': 'Failed to retrieve quota information'
            }), 500
            
        return jsonify({
            'status': 'success',
            'quota': quota
        })
    except Exception as e:
        app.logger.error(f"Error getting user quota: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve quota information',
            'details': str(e)
        }), 500

@app.route('/api/users/<int:user_id>/quota', methods=['PUT'])
@login_required
@admin_required
def update_user_quota_api(user_id):
    """Update quota settings for a user (admin only)"""
    try:
        data = request.json
        app.logger.info(f"Received quota update request: {data}")
        
        # Check if user exists
        user = db.get_user_by_id(user_id)
        if not user:
            app.logger.error(f"User not found: {user_id}")
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
            
        # Get quota parameters
        daily_limit = data.get('daily_message_limit')
        monthly_limit = data.get('monthly_message_limit')
        max_attachment_size = data.get('max_attachment_size_kb')
        
        app.logger.info(f"Quota parameters: daily={daily_limit}, monthly={monthly_limit}, attachment={max_attachment_size}")
        
        # Validate input
        if daily_limit is not None and (not isinstance(daily_limit, int) or daily_limit < 0):
            return jsonify({
                'status': 'error',
                'message': 'Daily message limit must be a positive integer'
            }), 400
            
        if monthly_limit is not None and (not isinstance(monthly_limit, int) or monthly_limit < 0):
            return jsonify({
                'status': 'error',
                'message': 'Monthly message limit must be a positive integer'
            }), 400
            
        if max_attachment_size is not None and (not isinstance(max_attachment_size, int) or max_attachment_size < 0):
            return jsonify({
                'status': 'error',
                'message': 'Max attachment size must be a positive integer'
            }), 400
            
        # Update quota
        updated_quota = db.update_user_quota(user_id, daily_limit, monthly_limit, max_attachment_size)
        if not updated_quota:
            app.logger.error(f"Failed to update quota for user {user_id}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to update quota settings'
            }), 500
            
        app.logger.info(f"Updated quota for user {user_id}: {updated_quota}")
        
        return jsonify({
            'status': 'success',
            'message': 'Quota settings updated successfully',
            'quota': updated_quota
        })
    except Exception as e:
        app.logger.error(f"Error updating user quota: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to update quota settings',
            'details': str(e)
        }), 500

@app.route('/api/quota', methods=['GET'])
@login_required
def get_my_quota():
    """Get quota information for the current user"""
    try:
        user_id = current_user.id
        quota_check = db.check_user_quota(user_id)
        
        return jsonify({
            'status': 'success',
            'quota': quota_check
        })
    except Exception as e:
        app.logger.error(f"Error getting quota: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve quota information',
            'details': str(e)
        }), 500

@app.route('/api/users/<int:user_id>/quota/reset', methods=['POST'])
@login_required
@admin_required
def reset_user_quota_counters(user_id):
    """Reset a user's quota usage counters (admin only)"""
    try:
        # Check if user exists
        user = db.get_user_by_id(user_id)
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
            
        # Reset counters
        conn = sqlite3.connect(db.DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE user_quotas 
                SET messages_used_today = 0,
                    messages_used_month = 0,
                    last_reset_date = CURRENT_DATE
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
            
            return jsonify({
                'status': 'success',
                'message': 'Quota counters reset successfully'
            })
        except Exception as e:
            conn.rollback()
            app.logger.error(f"Database error resetting quota counters: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to reset quota counters',
                'details': str(e)
            }), 500
        finally:
            conn.close()
    except Exception as e:
        app.logger.error(f"Error resetting quota counters: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to reset quota counters',
            'details': str(e)
        }), 500

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'docs')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md', 'xml', 'json', 'csv', 'xls', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload-doc', methods=['POST'])
@login_required
def upload_document():
    """Handles file uploads for the RAG system."""
    if 'document' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request'}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    # Get conversation_id from the request
    conversation_id = request.form.get('conversation_id')
    if not conversation_id:
        return jsonify({'status': 'error', 'message': 'No conversation ID provided'}), 400
    
    # Validate conversation_id exists
    conversation = db.get_user_conversation(current_user.id, conversation_id)
    if not conversation:
        return jsonify({'status': 'error', 'message': 'Conversation not found'}), 404

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Create a conversation-specific directory
        conversation_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"conversation_{conversation_id}")
        os.makedirs(conversation_dir, exist_ok=True)
        
        file_path = os.path.join(conversation_dir, filename)
        try:
            file.save(file_path)
            app.logger.info(f"File '{filename}' saved successfully to '{file_path}'")
            
            # Process and index the file
            try:
                # For all file types, use the appropriate loader via add_single_document_to_store
                app.logger.info(f"Starting document indexing for {filename}...")
                
                # For CSV/Excel files, add more detailed logging
                if filename.lower().endswith(('.csv', '.xls', '.xlsx')):
                    app.logger.info(f"Processing {filename} with custom loader. This may take a moment for large files...")
                
                # Process the file with the appropriate loader
                success = rag.add_single_document_to_store(file_path, conversation_id=conversation_id)
                
                if success:
                    app.logger.info(f'Successfully processed and indexed document: {filename}')
                else:
                    raise Exception('Failed to process and index document')
                    
            except Exception as e:
                app.logger.error(f'Error processing file {filename}: {str(e)}')
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as oe:
                        app.logger.error(f"Error removing file {file_path} after processing error: {oe}")
                return jsonify({'status': 'error', 'message': f'Error processing file: {str(e)}'}), 500
            
            if success:
                if filename.lower().endswith(('.csv', '.xls', '.xlsx')):
                    app.logger.info(f"Successfully processed {filename} with custom loader")
                
                doc_id = db.add_conversation_document(conversation_id, filename, file_path)
                app.logger.info(f"Document {filename} indexed successfully with ID: {doc_id}")
                
                return jsonify({
                    'status': 'success', 
                    'message': f'Document \'{filename}\' uploaded and indexed successfully.',
                    'document_id': doc_id
                })
            else:
                # Remove the document from the database if indexing failed
                db.delete_conversation_document(db.add_conversation_document(conversation_id, filename, file_path))
                app.logger.error(f"Failed to index document: {filename}")
                return jsonify({'status': 'error', 'message': f'File uploaded but failed to index: {filename}'}), 500
                
        except Exception as e:
            app.logger.error(f"Error uploading or indexing file {filename}: {e}")
            # Clean up partially saved file if error occurs during save or indexing
            if os.path.exists(file_path):
                 try:
                     os.remove(file_path)
                 except OSError as oe:
                     app.logger.error(f"Error removing file {file_path} after upload error: {oe}")
            return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    """Upload a file to be used as an attachment"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400
            
        # Get user's attachment size limit
        quota = db.get_user_quota(current_user.id)
        max_size_kb = quota.get('max_attachment_size_kb', 5120)  # Default 5MB if not set
        
        # Check file size (convert bytes to KB)
        file_size_kb = len(file.read()) / 1024
        file.seek(0)  # Reset file pointer after reading
        
        # Skip size check for admins
        if file_size_kb > max_size_kb and current_user.role != 'admin':
            return jsonify({
                'status': 'error', 
                'message': f'File too large. Maximum size is {max_size_kb} KB ({max_size_kb/1024:.1f} MB)',
                'size_limit_exceeded': True,
                'max_size_kb': max_size_kb,
                'file_size_kb': file_size_kb
            }), 413  # Payload Too Large
            
        if file and allowed_file(file.filename):
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Generate a unique filename to avoid collisions
            unique_filename = f"{int(time.time())}_{filename}"
            
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            return jsonify({
                'status': 'success',
                'message': 'File uploaded successfully',
                'filename': unique_filename,
                'original_filename': filename
            })
        else:
            return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400
    except Exception as e:
        app.logger.error(f"Error uploading file: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Legacy chat endpoint"""
    return generate()

# ============================================================
# Image Generation API (ComfyUI Integration)
# ============================================================

# --- API: List available workflows (for image generation) ---
@app.route('/api/workflows', methods=['GET'])
def list_workflows():
    import os
    workflows_dir = os.path.join(os.path.dirname(__file__), 'workflows')
    if not os.path.isdir(workflows_dir):
        return jsonify({'status': 'success', 'workflows': []})
    files = [f for f in os.listdir(workflows_dir) if f.endswith('.json')]
    # Optionally strip .json and sort
    workflows = sorted([os.path.splitext(f)[0] for f in files])
    return jsonify({'status': 'success', 'workflows': workflows})

# Track recent image generation requests to prevent duplicates
recent_image_requests = {}
RECENT_REQUEST_TIMEOUT = 10  # seconds
MAX_STORED_REQUESTS = 100  # Maximum number of request hashes to store

@app.route('/api/generate_image', methods=['POST'])
@login_required
def generate_image():
    """Generate an image using ComfyUI based on a prompt or workflow"""
    import os
    import json
    import time
    import traceback
    import requests
    import base64
    import hashlib
    import re
    import glob
    from flask import request, jsonify
    from datetime import datetime
    
    # Initialize logger at the start
    app.logger.info("=== Starting image generation request ===")

    # Get request data
    app.logger.info("Parsing request data...")
    try:
        data = request.get_json()
        if not data:
            app.logger.error("No JSON data in request")
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
            
        prompt = data.get('prompt', '')
        model = data.get('model', '')  # Optional: workflow name if using a specific workflow
        conversation_id = data.get('conversation_id')  # Add conversation_id
        
        app.logger.info(f"Request data - Prompt: {prompt[:50]}..., Model: {model}, Conversation ID: {conversation_id}")
        
    except Exception as e:
        error_trace = traceback.format_exc()
        app.logger.error(f"Error parsing request data: {str(e)}\n{error_trace}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid request data',
            'details': str(e),
            'traceback': error_trace if app.debug else None
        }), 400
    
    # Get ComfyUI URL from config
    try:
        from config import COMFYUI_BASE_URL
        comfyui_url = f"{COMFYUI_BASE_URL}/prompt"
        app.logger.info(f"Using ComfyUI URL: {comfyui_url}")
        
        # Verify ComfyUI is reachable
        try:
            health_check = requests.get(f"{COMFYUI_BASE_URL}/system_stats", timeout=5)
            if health_check.status_code != 200:
                app.logger.error(f"ComfyUI health check failed with status {health_check.status_code}")
                return jsonify({
                    'status': 'error',
                    'message': f'ComfyUI service is not responding (HTTP {health_check.status_code})',
                    'details': health_check.text[:500] if health_check.text else 'No response body'
                }), 502
        except requests.exceptions.RequestException as e:
            app.logger.error(f"ComfyUI connection error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to connect to ComfyUI service',
                'details': str(e)
            }), 503
            
        # Create a hash of the request to deduplicate
        request_key = hashlib.md5(f"{prompt}:{model}:{conversation_id}".encode()).hexdigest()
        
    except Exception as e:
        error_trace = traceback.format_exc()
        app.logger.error(f"Error initializing ComfyUI configuration: {str(e)}\n{error_trace}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to initialize image generation service',
            'details': str(e),
            'traceback': error_trace if app.debug else None
        }), 500
    
    # Check if we've seen this exact request recently (within the last 10 seconds)
    app.logger.info("Checking for duplicate requests...")
    try:
        current_time = time.time()
        if request_key in recent_image_requests:
            last_request_time, processing = recent_image_requests[request_key]
            # If request is very recent and still processing, return 429 Too Many Requests
            if current_time - last_request_time < RECENT_REQUEST_TIMEOUT and processing:
                app.logger.warning(f"Duplicate image generation request detected within {RECENT_REQUEST_TIMEOUT} seconds: {prompt[:50]}")
                return jsonify({
                    'status': 'error',
                    'message': 'Too many similar requests. Please wait before trying again.'
                }), 429
    except Exception as e:
        error_trace = traceback.format_exc()
        app.logger.error(f"Error checking for duplicate requests: {str(e)}\n{error_trace}")
        # Continue execution even if this check fails
    
    # Mark this request as being processed
    app.logger.info("Marking request as processing...")
    try:
        recent_image_requests[request_key] = (current_time, True)
        
        # Clean up old requests to prevent memory leaks
        cleanup_keys = [k for k, (t, _) in recent_image_requests.items() 
                      if current_time - t > RECENT_REQUEST_TIMEOUT * 2]
        for k in cleanup_keys:
            recent_image_requests.pop(k, None)
            
    except Exception as e:
        app.logger.error(f"Error managing request tracking: {str(e)}")
        # Continue anyway, this isn't critical
    
    # Store the last used seed in a global variable
    app.logger.info("Initializing seed...")
    if not hasattr(app, 'last_used_seed'):
        app.last_used_seed = random.randint(0, 2147483647)
        app.logger.info(f"Initialized new random seed: {app.last_used_seed}")
        
    # Generate a unique identifier for this request to prevent caching
    request_id = str(uuid.uuid4())
    timestamp = int(time.time())
        
    # Extract augment parameters from the prompt
    app.logger.info("Extracting parameters from prompt...")
    aspect_ratio = None
    steps = 20  # Default steps
    seed = None
    
    # Log initial parameters
    app.logger.info(f"Initial parameters - Steps: {steps}, Seed: {seed}, Aspect Ratio: {aspect_ratio}")
    
    # Check for aspect ratio command
    app.logger.info("Processing aspect ratio...")
    try:
        aspect_ratio_match = re.search(r'--ar\s+(\d+:\d+)', prompt)
        if aspect_ratio_match:
            aspect_ratio = aspect_ratio_match.group(1)
            # Remove the command from the prompt
            prompt = re.sub(r'--ar\s+\d+:\d+', '', prompt).strip()
            app.logger.info(f"Found aspect ratio: {aspect_ratio}")
    except Exception as e:
        app.logger.error(f"Error processing aspect ratio: {str(e)}")
        aspect_ratio = None
        
    # Check for steps command
    app.logger.info("Processing steps...")
    try:
        steps_match = re.search(r'--steps\s+(\d+)', prompt)
        if steps_match:
            steps = int(steps_match.group(1))
            # Remove the command from the prompt
            prompt = re.sub(r'--steps\s+\d+', '', prompt).strip()
            app.logger.info(f"Found steps: {steps}")
    except Exception as e:
        app.logger.error(f"Error processing steps: {str(e)}")
        steps = 20  # Default value
        
    # Check for seed command
    app.logger.info("Processing seed...")
    try:
        seed_match = re.search(r'--seed\s+(\w+)', prompt)
        if seed_match:
            seed_value = seed_match.group(1)
            # Handle 'last' keyword for seed
            if seed_value.lower() == 'last':
                seed = getattr(app, 'last_used_seed', random.randint(0, 2147483647))
                app.logger.info(f"Using last seed: {seed}")
            else:
                try:
                    seed = int(seed_value)
                    app.logger.info(f"Using specified seed: {seed}")
                except ValueError:
                    # If not a valid integer, use a random seed
                    seed = random.randint(0, 2147483647)
                    app.logger.info(f"Using random seed (invalid input): {seed}")
            # Remove the command from the prompt
            prompt = re.sub(r'--seed\s+\w+', '', prompt).strip()
        else:
            # If no seed specified, use a random seed
            seed = random.randint(0, 2147483647)
            app.logger.info(f"Using new random seed: {seed}")
    except Exception as e:
        app.logger.error(f"Error processing seed: {str(e)}")
        seed = random.randint(0, 2147483647)
        
    # Parse aspect ratio
    width, height = 1024, 1024  # Default size
    if aspect_ratio:
        try:
            w_ratio, h_ratio = map(int, aspect_ratio.split(':'))
            # Scale to maintain approximately the same pixel count
            scale = math.sqrt((1024 * 1024) / (w_ratio * h_ratio))
            width = int(w_ratio * scale)
            height = int(h_ratio * scale)
            # Ensure dimensions are multiples of 8 (required by many models)
            width = (width // 8) * 8
            height = (height // 8) * 8
        except Exception as e:
            print(f"Error parsing aspect ratio: {e}")
            # Fall back to default size
            width, height = 1024, 1024
        
    # Import ComfyUI config
    from config import COMFYUI_BASE_URL
    
    # ComfyUI paths
    comfy_api_url = COMFYUI_BASE_URL
    comfy_output_dir = os.environ.get('COMFYUI_OUTPUT_DIR', 'MIDAS_standalone/ComfyUI/Output')
        
    # Determine which workflow to use
    app.logger.info("Determining workflow to use...")
    workflow = None
    workflow_file = None
    
    try:
        if model and model.startswith('workflow:'):
            # Extract workflow name from model string
            workflow = model.replace('workflow:', '')
            workflow_file = os.path.join('workflows', f'{workflow}.json')
            app.logger.info(f"Using specified workflow: {workflow} from {workflow_file}")
        else:
            # Default workflow
            workflow = "Flux Quick"
            workflow_file = os.path.join('workflows', 'Flux Quick.json')
            app.logger.info(f"Using default workflow: {workflow} from {workflow_file}")
            
        # Verify workflow file exists
        if not os.path.exists(workflow_file):
            error_msg = f"Workflow file not found: {workflow_file}"
            app.logger.error(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'available_workflows': [f for f in os.listdir('workflows') if f.endswith('.json')]
            }), 404
            
    except Exception as e:
        error_trace = traceback.format_exc()
        app.logger.error(f"Error determining workflow: {str(e)}\n{error_trace}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to determine workflow',
            'details': str(e),
            'traceback': error_trace if app.debug else None
        }), 500
        
    if workflow:
        # Load the workflow JSON
        app.logger.info(f"Loading workflow from file: {workflow_file}")
        try:
            # Get absolute path for better error reporting
            abs_workflow_path = os.path.abspath(workflow_file)
            app.logger.info(f"Absolute workflow path: {abs_workflow_path}")
            
            # Check file exists again with absolute path
            if not os.path.exists(abs_workflow_path):
                raise FileNotFoundError(f"Workflow file not found at: {abs_workflow_path}")
                
            # Check file permissions
            if not os.access(abs_workflow_path, os.R_OK):
                raise PermissionError(f"No read permissions for workflow file: {abs_workflow_path}")
                
            # Read file content
            with open(abs_workflow_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                app.logger.info(f"Successfully read {len(file_content)} bytes from workflow file")
                
                # Parse JSON
                comfyui_payload = json.loads(file_content)
                app.logger.info(f"Successfully parsed workflow JSON with {len(comfyui_payload)} top-level keys")
            
            # Process workflow nodes
            app.logger.info("Processing workflow nodes...")
            try:
                prompt_nodes_updated = 0
                resolution_nodes_updated = 0
                steps_nodes_updated = 0
                seed_nodes_updated = 0
                
                for node_id, node_data in comfyui_payload.items():
                    if not isinstance(node_data, dict):
                        app.logger.warning(f"Skipping non-dict node {node_id}")
                        continue
                        
                    node_type = node_data.get('class_type', 'unknown')
                    inputs = node_data.get('inputs', {})
                    
                    # Update prompt in text encoder nodes
                    if node_type == 'CLIPTextEncode' and 'text' in inputs:
                        app.logger.debug(f"Updating prompt in node {node_id} ({node_type})")
                        node_data['inputs']['text'] = prompt
                        prompt_nodes_updated += 1
                    
                    # Update resolution parameters
                    if 'width' in inputs and 'height' in inputs:
                        app.logger.debug(f"Updating resolution in node {node_id} ({node_type}): {width}x{height}")
                        node_data['inputs']['width'] = width
                        node_data['inputs']['height'] = height
                        resolution_nodes_updated += 1
                        
                    # Update steps parameter
                    if 'steps' in inputs:
                        app.logger.debug(f"Updating steps in node {node_id} ({node_type}): {steps}")
                        node_data['inputs']['steps'] = steps
                        steps_nodes_updated += 1
                        
                    # Update seed parameters
                    if 'seed' in inputs:
                        app.logger.debug(f"Updating seed in node {node_id} ({node_type}): {seed}")
                        node_data['inputs']['seed'] = seed
                        seed_nodes_updated += 1
                    elif 'noise_seed' in inputs:
                        app.logger.debug(f"Updating noise_seed in node {node_id} ({node_type}): {seed}")
                        node_data['inputs']['noise_seed'] = seed
                        seed_nodes_updated += 1
                
                app.logger.info(f"Workflow processing complete - "
                              f"Prompt nodes: {prompt_nodes_updated}, "
                              f"Resolution nodes: {resolution_nodes_updated}, "
                              f"Steps nodes: {steps_nodes_updated}, "
                              f"Seed nodes: {seed_nodes_updated}")
                
            except Exception as e:
                error_trace = traceback.format_exc()
                app.logger.error(f"Error processing workflow nodes: {str(e)}\n{error_trace}")
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to process workflow nodes',
                    'details': str(e),
                    'traceback': error_trace if app.debug else None
                }), 500
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error loading workflow: {str(e)}'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'No workflow specified'}), 400
        
    # Submit to ComfyUI
    comfyui_url = f"{comfy_api_url}/prompt"
    app.logger.info(f"Submitting to ComfyUI at {comfyui_url}")
    
    try:
        # Log the payload (without the full prompt to avoid log spam)
        payload_for_log = json.dumps(comfyui_payload, indent=2)[:1000] + '...'  # Truncate long payloads
        app.logger.info(f"Sending request to ComfyUI with payload: {payload_for_log}")
        
        # Increased timeouts for slower systems
        CONNECT_TIMEOUT = 60  # seconds
        READ_TIMEOUT = 300    # 5 minutes for initial response
        
        app.logger.info(f"Using timeouts - Connect: {CONNECT_TIMEOUT}s, Read: {READ_TIMEOUT}s")
        
        start_time = time.time()
        resp = requests.post(
            comfyui_url, 
            json={'prompt': comfyui_payload}, 
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        )
        request_duration = time.time() - start_time
        
        app.logger.info(f"ComfyUI response received in {request_duration:.2f}s - Status: {resp.status_code}")
        
        # Log response details
        try:
            response_data = resp.json()
            app.logger.debug(f"ComfyUI response data: {json.dumps(response_data, indent=2)[:1000]}...")
            
            if 'error' in response_data:
                app.logger.error(f"ComfyUI returned an error: {response_data.get('error', 'Unknown error')}")
                return jsonify({
                    'status': 'error',
                    'message': 'Image generation service returned an error',
                    'details': response_data.get('error')
                }), 502
                
            result = response_data
            prompt_id = result.get('prompt_id')
            if not prompt_id:
                raise ValueError("No prompt_id in ComfyUI response")
                
            app.logger.info(f"Successfully submitted to ComfyUI with prompt_id: {prompt_id}")
            
        except json.JSONDecodeError:
            app.logger.error(f"Failed to parse ComfyUI response as JSON. Status: {resp.status_code}, Content: {resp.text[:500]}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid response from image generation service',
                'details': f'Expected JSON but got: {resp.text[:200]}...'
            }), 502
            
        resp.raise_for_status()
        
    except requests.exceptions.Timeout as e:
        app.logger.error(f"ComfyUI API request timed out after 30s: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Image generation service timed out. The request took too long to process.',
            'details': str(e)
        }), 504
        
    except requests.exceptions.RequestException as e:
        error_type = type(e).__name__
        app.logger.error(f"Error calling ComfyUI API ({error_type}): {str(e)}")
        
        # Provide more specific error messages for common issues
        if 'ConnectionError' in error_type:
            error_msg = 'Failed to connect to the image generation service. Please check if the service is running.'
        elif 'Timeout' in error_type:
            error_msg = 'The image generation service took too long to respond.'
        else:
            error_msg = f'Failed to communicate with the image generation service: {str(e)}'
            
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'details': str(e),
            'error_type': error_type
        }), 500 if 'ConnectionError' in error_type else 502
        
    # Wait for the generation to complete with more frequent checks
    initial_wait = 10  # Initial wait before starting to poll
    check_interval = 5  # Check every 5 seconds
    max_checks = 180   # Maximum number of checks (15 min total with 5 sec intervals)
    
    app.logger.info(f"Starting image generation polling for prompt_id {result.get('prompt_id')}")
    app.logger.info(f"Polling parameters - Initial wait: {initial_wait}s, Interval: {check_interval}s, Max checks: {max_checks}")
    
    # Initial wait before starting to poll
    time.sleep(initial_wait)
    
    app.logger.info(f"Starting image generation polling for prompt_id {result.get('prompt_id')}")
    
    # Poll until we get a result or hit max attempts
    last_progress = 0
    last_status = None
    
    for attempt in range(max_checks):
        try:
            # Check for results using the ComfyUI API
            history_url = f'{COMFYUI_BASE_URL}/history/{result.get("prompt_id")}'
            queue_url = f'{COMFYUI_BASE_URL}/queue'
            
            # Check queue status first
            try:
                queue_resp = requests.get(queue_url, timeout=5)
                if queue_resp.status_code == 200:
                    queue_data = queue_resp.json()
                    app.logger.debug(f"Queue status: {json.dumps(queue_data, indent=2)}")
            except Exception as e:
                app.logger.warning(f"Error checking queue status: {str(e)}")
            
            # Get generation status
            history_resp = requests.get(history_url, timeout=10)
            
            # Log progress periodically
            if attempt % 5 == 0:  # Log every 5 attempts (every ~25 seconds with 5s interval)
                app.logger.info(f"Polling attempt {attempt + 1}/{max_checks} for prompt_id {result.get('prompt_id')}")
                
                # Get system stats to check load
                try:
                    stats_url = f'{COMFYUI_BASE_URL}/system_stats'
                    stats_resp = requests.get(stats_url, timeout=5)
                    if stats_resp.status_code == 200:
                        stats = stats_resp.json()
                        app.logger.info(f"System stats - CPU: {stats.get('cpu_usage')}%, GPU: {stats.get('gpu_usage')}%, Memory: {stats.get('memory_usage')}%")
                except Exception as e:
                    app.logger.warning(f"Error getting system stats: {str(e)}")
            
            if history_resp.status_code == 200:
                history_data = history_resp.json()
                outputs = history_data.get(result.get('prompt_id'), {}).get('outputs', {})
                
                # Look for image data in the outputs
                for node_id, node_output in outputs.items():
                    if node_output.get('images'):
                        # Get the first image
                        image_data = node_output['images'][0]
                        filename = image_data['filename']
                        
                        # Get the image directly from ComfyUI
                        image_url = f"{COMFYUI_BASE_URL}/view?filename={filename}"
                        if 'subfolder' in image_data and image_data['subfolder']:
                            image_url += f"&subfolder={image_data['subfolder']}"
                        
                        img_resp = requests.get(image_url, timeout=30)
                        if img_resp.status_code == 200 and img_resp.content:
                            img_b64 = base64.b64encode(img_resp.content).decode('utf-8')
                            
                            # Save to conversation history if needed
                            if conversation_id:
                                try:
                                    db.add_message(
                                        conversation_id=conversation_id,
                                        role='assistant',
                                        content='',
                                        thinking=None,
                                        images=[img_b64],
                                        attachment_filename=filename
                                    )
                                except Exception as db_error:
                                    app.logger.error(f"Error saving image to database: {str(db_error)}")
                            
                            # Mark request as completed
                            if request_key in recent_image_requests:
                                recent_image_requests[request_key] = (time.time(), False)
                            
                            return jsonify({
                                'status': 'success',
                                'image_base64': img_b64,
                                'filename': filename,
                                'workflow': workflow,
                                'seed': seed
                            })
            
            # If we get here, image isn't ready yet
            if attempt < max_checks - 1:
                time.sleep(check_interval)
                
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error checking image status: {str(e)}")
            if attempt < max_checks - 1:
                time.sleep(check_interval)
    
    # If we get here, we've timed out
    app.logger.error(f"Timed out waiting for image generation (prompt_id: {result.get('prompt_id')})")
    return jsonify({
        'status': 'error',
        'message': 'Image generation timed out. Please try again.'
    }), 504

    # If we've exhausted all attempts and still haven't found the image
    app.logger.error(f"Timed out waiting for image generation to complete for prompt_id {result.get('prompt_id')}")
    # Mark this request as completed (with error)
    if request_key in recent_image_requests:
        recent_image_requests[request_key] = (current_time, False)
        
    return jsonify({
        'status': 'error',
        'message': 'Timed out waiting for image generation to complete'
    }), 504

# ============================================================
# Bot Management API
# ============================================================

@app.route('/api/bots', methods=['GET'])
@login_required
def list_bots():
    """List all bots"""
    try:
        bots = Bot.get_all()
        return jsonify({
            'status': 'success',
            'bots': [bot.to_dict() for bot in bots]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots', methods=['POST'])
@login_required
@admin_required
def create_bot():
    """Create a new bot"""
    try:
        data = request.json

        # Create new bot
        bot = Bot(
            name=data.get('name', 'New Bot'),
            description=data.get('description', ''),
            greeting=data.get('greeting', 'Hello! How can I help you today?'),
            base_model=data.get('base_model'),
            embedding_model=data.get('embedding_model', 'nomic-embed-text'),
            system_prompt=data.get('system_prompt', ''),
            parameters=data.get('parameters', {
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 2048
            })
        )

        # Save bot
        bot.save()

        return jsonify({
            'status': 'success',
            'bot': bot.to_dict()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>', methods=['GET'])
@login_required
def get_bot(bot_id):
    """Get a bot by ID"""
    try:
        bot = Bot.get(bot_id)
        if not bot:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404

        return jsonify({
            'status': 'success',
            'bot': bot.to_dict()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>', methods=['PUT'])
@login_required
@admin_required
def update_bot(bot_id):
    """Update a bot"""
    try:
        bot = Bot.get(bot_id)
        if not bot:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404

        data = request.json

        # Update bot properties
        bot.name = data.get('name', bot.name)
        bot.description = data.get('description', bot.description)
        bot.greeting = data.get('greeting', bot.greeting)
        bot.base_model = data.get('base_model', bot.base_model)
        bot.embedding_model = data.get('embedding_model', bot.embedding_model)
        bot.system_prompt = data.get('system_prompt', bot.system_prompt)
        bot.parameters = data.get('parameters', bot.parameters)

        # Save bot
        bot.save()

        return jsonify({
            'status': 'success',
            'bot': bot.to_dict()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_bot(bot_id):
    """Delete a bot"""
    try:
        success = Bot.delete(bot_id)
        if not success:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404

        return jsonify({
            'status': 'success',
            'message': 'Bot deleted successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>/knowledge', methods=['POST'])
@login_required
@admin_required
def upload_knowledge_files(bot_id):
    """Upload knowledge files for a bot"""
    try:
        app.logger.info('='*50)
        app.logger.info(f'Upload request received for bot_id: {bot_id}')
        app.logger.info(f'Request headers: {dict(request.headers)}')
        app.logger.info(f'Request form data: {request.form}')
        app.logger.info(f'Request files: {request.files}')
        app.logger.info(f'Content length: {request.content_length}')
        
        # Log all files in the BOTS_DIR for debugging
        if os.path.exists(BOTS_DIR):
            bot_files = [f for f in os.listdir(BOTS_DIR) if f.endswith('.json')]
            app.logger.info(f'Found {len(bot_files)} bot files in {BOTS_DIR}: {bot_files}')
        else:
            app.logger.error(f'BOTS_DIR does not exist: {BOTS_DIR}')
            return jsonify({
                'status': 'error',
                'message': 'Bots directory not found',
                'details': f'Directory not found: {BOTS_DIR}'
            }), 500
        
        # Debug: Check if BOTS_DIR exists and is writable
        app.logger.info(f'BOTS_DIR: {BOTS_DIR}')
        app.logger.info(f'BOTS_DIR exists: {os.path.exists(BOTS_DIR)}')
        app.logger.info(f'BOTS_DIR writable: {os.access(BOTS_DIR, os.W_OK)}')
        
        # Debug: List all bot files
        if os.path.exists(BOTS_DIR):
            bot_files = [f for f in os.listdir(BOTS_DIR) if f.endswith('.json')]
            app.logger.info(f'Found {len(bot_files)} bot files: {bot_files}')
        else:
            app.logger.error(f'BOTS_DIR does not exist: {BOTS_DIR}')
            return jsonify({
                'status': 'error',
                'message': f'Bots directory not found: {BOTS_DIR}'
            }), 500

        bot = Bot.get(bot_id)
        if not bot:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404

        # Check if files were provided
        if 'files' not in request.files:
            app.logger.error('No files part in the request')
            return jsonify({
                'status': 'error',
                'message': 'No files provided',
                'details': 'The request did not contain any files'
            }), 400
            
        files = request.files.getlist('files')
        if not files or all(not file.filename for file in files):
            app.logger.error('No files selected or empty filenames')
            return jsonify({
                'status': 'error',
                'message': 'No files selected',
                'details': 'Please select one or more files to upload'
            }), 400

        saved_files = []

        # Create knowledge base directory if it doesn't exist
        try:
            kb_dir = bot.get_knowledge_base_path()
            app.logger.info(f'Creating/verifying knowledge base directory: {kb_dir}')
            os.makedirs(kb_dir, exist_ok=True)
            
            # Verify directory is writable
            test_file = os.path.join(kb_dir, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
                
        except Exception as e:
            app.logger.error(f'Error creating/accessing knowledge base directory: {str(e)}', exc_info=True)
            app.logger.error(f'Error creating knowledge base directory: {str(e)}')
            return jsonify({
                'status': 'error',
                'message': f'Failed to create knowledge base directory: {str(e)}'
            }), 500

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                if not (filename.endswith('.txt') or filename.endswith('.pdf') or 
                        filename.endswith('.md') or filename.endswith('.xml')):
                    continue
                    
                file_path = os.path.join(kb_dir, filename)
                file.save(file_path)
                saved_files.append(filename)
                
                # Process and index the file
                try:
                    # Initialize processing stats if not already done
                    if not hasattr(app, 'processing_stats'):
                        app.processing_stats = {}
                    if bot_id not in app.processing_stats:
                        app.processing_stats[bot_id] = {
                            'total_chunks': 0,
                            'processing_time': 0,
                            'file_stats': []
                        }
                    
                    # Start timing the processing
                    start_time = time.time()
                    
                    # Select appropriate loader based on file extension
                    if filename.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif filename.endswith('.xml'):
                        loader = UnstructuredXMLLoader(file_path)
                    elif filename.endswith('.md'):
                        loader = UnstructuredMarkdownLoader(file_path)
                    else:  # .txt
                        loader = TextLoader(file_path)
                    
                    # Load documents
                    app.logger.info(f'Loading {filename}...')
                    documents = loader.load()
                    
                    # Index documents in bot's knowledge base collection
                    collection_name = f"bot_{bot_id}"
                    app.logger.info(f'Indexing {len(documents)} documents from {filename} in collection {collection_name}')
                    
                    # Ensure documents have metadata
                    for doc in documents:
                        if not hasattr(doc, 'metadata') or not doc.metadata:
                            doc.metadata = {}
                        doc.metadata['source'] = file_path
                        doc.metadata['filename'] = filename
                        doc.metadata['bot_id'] = bot_id
                    
                    # Split documents into chunks
                    app.logger.info(f'Splitting {filename} into chunks...')
                    text_splitter = rag.RecursiveCharacterTextSplitter(
                        chunk_size=rag.CHUNK_SIZE, 
                        chunk_overlap=rag.CHUNK_OVERLAP
                    )
                    split_docs = text_splitter.split_documents(documents)
                    app.logger.info(f'Split into {len(split_docs)} chunks for indexing')
                    
                    # Ensure all chunks have metadata
                    for chunk in split_docs:
                        if not hasattr(chunk, 'metadata') or not chunk.metadata:
                            chunk.metadata = {}
                        chunk.metadata['source'] = file_path
                        chunk.metadata['filename'] = filename
                        chunk.metadata['bot_id'] = bot_id
                    
                    # Add to vector store
                    try:
                        # Load existing or create new vector store
                        app.logger.info(f'Loading vector store for {filename}...')
                        vectorstore = rag.Chroma(
                            persist_directory=rag.CHROMA_PERSIST_DIR, 
                            embedding_function=rag.ollama_ef, 
                            collection_name=collection_name
                        )
                        
                        # Add documents to vector store in batches with progress updates
                        total_chunks = len(split_docs)
                        app.logger.info(f'Adding {total_chunks} chunks to vector store in batches...')
                        
                        # Process in batches of 500 chunks
                        batch_size = 500
                        for i in range(0, total_chunks, batch_size):
                            batch_end = min(i + batch_size, total_chunks)
                            current_batch = split_docs[i:batch_end]
                            
                            # Add this batch
                            vectorstore.add_documents(current_batch)
                            
                            # Log progress
                            progress_pct = min(100, int((batch_end / total_chunks) * 100))
                            app.logger.info(f'Indexing progress: {batch_end}/{total_chunks} chunks ({progress_pct}%)')
                            
                            # Update global progress tracking
                            if not hasattr(app, 'indexing_progress'):
                                app.indexing_progress = {}
                            app.indexing_progress[bot_id] = {
                                'current': batch_end,
                                'total': total_chunks,
                                'percent': progress_pct,
                                'file': filename,
                                'status': 'indexing'
                            }
                            
                            # Persist after each batch to avoid memory issues
                            if batch_end % 2000 == 0 or batch_end == total_chunks:
                                app.logger.info(f'Persisting vector store at {batch_end}/{total_chunks} chunks...')
                                vectorstore.persist()
                        
                        # Final persist
                        app.logger.info(f'Persisting vector store...')
                        vectorstore.persist()
                        app.logger.info(f'Successfully indexed {total_chunks} chunks in bot knowledge base')
                    except Exception as ve:
                        app.logger.error(f'Error indexing in vector store: {str(ve)}')
                        raise
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Update processing stats
                    app.processing_stats[bot_id]['total_chunks'] += len(split_docs)
                    app.processing_stats[bot_id]['processing_time'] += processing_time
                    app.processing_stats[bot_id]['file_stats'].append({
                        'filename': filename,
                        'chunks': len(split_docs),
                        'processing_time': processing_time,
                        'size_kb': os.path.getsize(file_path) / 1024
                    })
                    
                    app.logger.info(f'Processed {len(documents)} documents from {filename} in {processing_time:.2f} seconds')
                except Exception as e:
                    app.logger.error(f'Error processing file {filename}: {str(e)}')
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    saved_files.remove(filename)
                    
        if not saved_files:
            return jsonify({
                'status': 'error', 
                'message': 'No valid files uploaded'
            }), 400

        # Update bot's knowledge files
        bot.knowledge_files = list(set(bot.knowledge_files + saved_files))
        bot.save()

        # Calculate processing stats
        total_chunks = 0
        processing_time = 0
        file_stats = []
        
        # Get stats from the app context if available
        if hasattr(app, 'processing_stats') and bot_id in app.processing_stats:
            stats = app.processing_stats[bot_id]
            total_chunks = stats.get('total_chunks', 0)
            processing_time = stats.get('processing_time', 0)
            file_stats = stats.get('file_stats', [])
            
            # Clear the stats after retrieving
            del app.processing_stats[bot_id]
        
        return jsonify({
            'status': 'success',
            'message': f'{len(saved_files)} files uploaded',
            'bot': bot.to_dict(),
            'saved_files': saved_files,
            'processing_stats': {
                'total_chunks': total_chunks,
                'processing_time': processing_time,
                'file_stats': file_stats
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>/knowledge/<filename>', methods=['DELETE'])
@login_required
@admin_required
def delete_knowledge_file(bot_id, filename):
    """Delete a knowledge file from a bot"""
    try:
        bot = Bot.get(bot_id)
        if not bot:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404

        # Get bot's knowledge base directory
        kb_dir = bot.get_knowledge_base_path()

        # Check if file exists
        file_path = os.path.join(kb_dir, secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404

        # Delete file
        os.remove(file_path)

        # Update bot's knowledge files list
        if filename in bot.knowledge_files:
            bot.knowledge_files.remove(filename)
            bot.save()

        return jsonify({
            'status': 'success',
            'message': 'File deleted successfully',
            'bot': bot.to_dict()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
def reindex_bot_knowledge(bot):
    """Re-index all knowledge files for a bot"""
    try:
        app.logger.info(f"[RAG] Re-indexing knowledge files for bot {bot.id}")
        kb_dir = bot.get_knowledge_base_path()
        collection_name = f"bot_{bot.id}"
        
        # Initialize progress tracking
        total_files = len(bot.knowledge_files)
        app.indexing_progress[bot.id] = {
            'status': 'processing',
            'progress': 0.0,
            'current_file': '',
            'files_processed': 0,
            'total_files': total_files,
            'chunks_processed': 0,
            'total_chunks': 0
        }
        
        # Process each file
        indexed_files = []
        total_chunks = 0
        start_time = time.time()
        
        for i, filename in enumerate(bot.knowledge_files):
            try:
                # Update progress
                app.indexing_progress[bot.id]['current_file'] = filename
                app.indexing_progress[bot.id]['files_processed'] = i
                app.indexing_progress[bot.id]['progress'] = i / total_files if total_files > 0 else 0
                
                file_path = os.path.join(kb_dir, secure_filename(filename))
                if not os.path.exists(file_path):
                    app.logger.warning(f"[RAG] File {filename} not found at {file_path}")
                    continue
                    
                # Select appropriate loader
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.xml'):
                    loader = UnstructuredXMLLoader(file_path)
                elif filename.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(file_path)
                else:  # .txt
                    loader = TextLoader(file_path)
                
                # Load and process documents
                app.logger.info(f"[RAG] Loading {filename} for re-indexing")
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    if not hasattr(doc, 'metadata') or not doc.metadata:
                        doc.metadata = {}
                    doc.metadata['source'] = file_path
                    doc.metadata['filename'] = filename
                    doc.metadata['bot_id'] = bot.id
                
                # Split documents
                text_splitter = rag.RecursiveCharacterTextSplitter(
                    chunk_size=rag.CHUNK_SIZE, 
                    chunk_overlap=rag.CHUNK_OVERLAP
                )
                split_docs = text_splitter.split_documents(documents)
                chunk_count = len(split_docs)
                app.logger.info(f"[RAG] Split {filename} into {chunk_count} chunks")
                
                # Update progress with chunk information
                total_chunks += chunk_count
                app.indexing_progress[bot.id]['total_chunks'] = total_chunks
                
                # Add to vector store in batches to show progress
                vectorstore = rag.Chroma(
                    persist_directory=rag.CHROMA_PERSIST_DIR, 
                    embedding_function=rag.ollama_ef, 
                    collection_name=collection_name
                )
                
                # Process in batches of 100 chunks
                batch_size = 100
                for j in range(0, len(split_docs), batch_size):
                    batch = split_docs[j:j+batch_size]
                    vectorstore.add_documents(batch)
                    
                    # Update progress after each batch
                    chunks_processed = j + len(batch)
                    app.indexing_progress[bot.id]['chunks_processed'] = app.indexing_progress[bot.id].get('chunks_processed', 0) + len(batch)
                    
                    # Calculate overall progress (files + current chunks)
                    if total_chunks > 0:
                        chunk_progress = chunks_processed / chunk_count
                        file_progress = i / total_files
                        # Weight: 20% for file progress, 80% for chunk progress within current file
                        combined_progress = (file_progress * 0.2) + (chunk_progress * 0.8 / total_files)
                        app.indexing_progress[bot.id]['progress'] = combined_progress
                
                vectorstore.persist()
                
                indexed_files.append(filename)
                app.logger.info(f"[RAG] Successfully re-indexed {filename}")
            except Exception as e:
                app.logger.error(f"[RAG] Error re-indexing {filename}: {str(e)}")
        
        # Mark processing as complete
        processing_time = time.time() - start_time
        app.indexing_progress[bot.id] = {
            'status': 'complete',
            'progress': 1.0,
            'files_processed': len(indexed_files),
            'total_files': total_files,
            'chunks_processed': total_chunks,
            'total_chunks': total_chunks,
            'processing_time': processing_time
        }
        
        app.logger.info(f"[RAG] Re-indexing complete for bot {bot.id}. Processed {total_chunks} chunks in {processing_time:.2f}s")
        return indexed_files
    except Exception as e:
        app.logger.error(f"[RAG] Error in re-indexing: {str(e)}")
        
        # Update progress with error
        if hasattr(app, 'indexing_progress') and bot.id in app.indexing_progress:
            app.indexing_progress[bot.id]['status'] = 'error'
            app.indexing_progress[bot.id]['error'] = str(e)
        
        return []

@app.route('/api/bots/<bot_id>/knowledge/reindex', methods=['POST'])
@login_required
def reindex_bot_knowledge_files(bot_id):
    """Re-index all knowledge files for a bot"""
    try:
        # Get the bot
        bot = Bot.get(bot_id)
        if not bot:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404
            
        # Check if bot has knowledge files
        if not bot.knowledge_files:
            return jsonify({
                'status': 'error',
                'message': 'Bot has no knowledge files to index'
            }), 400
            
        # Start background task for indexing
        # This prevents timeouts during indexing of large files
        def background_indexing():
            try:
                app.logger.info(f"Starting background re-indexing for bot {bot_id}")
                indexed_files = reindex_bot_knowledge(bot)
                app.logger.info(f"Background re-indexing complete for bot {bot_id}: {indexed_files}")
            except Exception as e:
                app.logger.error(f"Error in background re-indexing for bot {bot_id}: {e}")
                
        # Start the background thread
        import threading
        indexing_thread = threading.Thread(target=background_indexing)
        indexing_thread.daemon = True
        indexing_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Re-indexing started for {len(bot.knowledge_files)} files',
            'files': bot.knowledge_files
        })
    except Exception as e:
        app.logger.error(f"Error starting re-indexing: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>/knowledge/progress', methods=['GET'])
@login_required
def get_indexing_progress(bot_id):
    """Stream indexing progress as server-sent events"""
    try:
        def generate():
            # Initial response
            if not hasattr(app, 'indexing_progress') or bot_id not in app.indexing_progress:
                yield f"data: {json.dumps({'status': 'idle', 'percent': 0, 'current': 0, 'total': 0})}\n\n"
                return
                
            # Send current progress immediately
            yield f"data: {json.dumps(app.indexing_progress.get(bot_id, {'status': 'idle', 'percent': 0}))}\n\n"
            
            # Then keep checking for updates
            last_progress = app.indexing_progress.get(bot_id, {}).get('current', 0)
            check_count = 0
            
            while check_count < 60:  # Limit to 60 checks (30 seconds)
                time.sleep(0.5)
                check_count += 1
                
                # If progress exists and has changed
                if hasattr(app, 'indexing_progress') and bot_id in app.indexing_progress:
                    current_progress = app.indexing_progress[bot_id]
                    
                    # Only send if progress has changed
                    if current_progress.get('current', 0) != last_progress:
                        yield f"data: {json.dumps(current_progress)}\n\n"
                        last_progress = current_progress.get('current', 0)
                        
                    # If processing is complete, send final update and exit
                    if current_progress.get('status') == 'complete' or \
                       (current_progress.get('current', 0) >= current_progress.get('total', 0) and \
                        current_progress.get('total', 0) > 0):
                        yield f"data: {json.dumps({'status': 'complete', 'percent': 100})}\n\n"
                        break
                else:
                    # If progress tracking was removed, send idle status
                    yield f"data: {json.dumps({'status': 'idle', 'percent': 0})}\n\n"
                    break
                    
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>/chat', methods=['POST'])
@login_required
def chat_with_bot(bot_id):
    """Chat with a bot"""
    try:
        bot = Bot.get(bot_id)
        if not bot:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404

        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'No message provided'
            }), 400

        # Get bot's knowledge base directory
        kb_dir = bot.get_knowledge_base_path()

        # Check if bot has knowledge files
        has_knowledge = len(bot.knowledge_files) > 0

        # Get context from knowledge base if available
        context = ""
        if has_knowledge:
            try:
                # Use the existing RAG system to get context
                context = rag.get_context(user_message, kb_dir, bot.embedding_model)
            except Exception as e:
                print(f"Error getting context: {e}")

        # Prepare system prompt
        system_prompt = bot.system_prompt
        if context:
            system_prompt += f"\n\nRelevant context:\n{context}"

        # Call Ollama API with the bot's parameters
        response = generate_response_with_ollama(
            user_message, 
            bot.base_model, 
            system_prompt=system_prompt,
            temperature=bot.parameters.get('temperature', 0.7),
            top_p=bot.parameters.get('top_p', 0.9),
            max_tokens=bot.parameters.get('max_tokens', 2048)
        )

        return jsonify({
            'status': 'success',
            'message': response
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# --- Progress Tracking API Routes ---

@app.route('/api/progress/<operation_id>', methods=['GET'])
@login_required
def get_operation_progress(operation_id):
    """Get the progress of a long-running operation"""
    progress = get_progress(operation_id)
    if not progress:
        return jsonify({
            'status': 'error',
            'message': f'No progress found for operation ID: {operation_id}'
        }), 404
    
    return jsonify({
        'status': 'success',
        'progress': progress
    })


@app.route('/api/progress', methods=['GET'])
@login_required
def get_all_operations_progress():
    """Get the progress of all operations"""
    all_progress = get_all_progress()
    return jsonify({
        'status': 'success',
        'operations': list(all_progress.values())
    })


# --- End of API Routes ---

if __name__ == '__main__':
    app.run(debug=True)
