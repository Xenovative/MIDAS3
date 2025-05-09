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
        retrieved_context = ""
        bot_id = data.get('bot_id')
        
        # First check conversation-specific documents
        if not secret and rag.has_documents(conversation_id=conversation_id):
            app.logger.info(f"[RAG] Documents found for conversation {conversation_id}, using conversation RAG")
            use_rag = True
            
            # Log RAG query details before retrieval
            app.logger.info(f"[RAG] Querying conversation knowledge with: '{user_message[:100]}...'")
            
            # Retrieve context based on the user message from this conversation's collection
            retrieval_start = time.time()
            retrieved_context = rag.retrieve_context(user_message, conversation_id=conversation_id)
            retrieval_time = time.time() - retrieval_start
            
            # Log RAG results
            if retrieved_context:
                app.logger.info(f"[RAG] Retrieved {len(retrieved_context)} characters from conversation knowledge in {retrieval_time:.2f}s")
                app.logger.info(f"[RAG] Conversation knowledge sample: '{retrieved_context[:200]}...'")
            else:
                app.logger.info(f"[RAG] No relevant context found in conversation documents")
        
        # If no conversation documents or no results, check bot's knowledge base
        if not secret and not retrieved_context:
            # First check if bot_id is provided in the request
            if bot_id:
                bot_collection = f"bot_{bot_id}"
            else:
                # If no bot_id in request, try to get the bot associated with this conversation
                try:
                    conversation = db.get_conversation(conversation_id)
                    if conversation and 'bot_id' in conversation:
                        bot_id = conversation['bot_id']
                        bot_collection = f"bot_{bot_id}"
                        app.logger.info(f"[RAG] Found bot {bot_id} associated with conversation {conversation_id}")
                    else:
                        bot_collection = None
                except Exception as e:
                    app.logger.error(f"[RAG] Error getting conversation bot: {str(e)}")
                    bot_collection = None
            
            # Check bot's knowledge base if we have a bot_id
            if bot_id and bot_collection:
                try:
                    # Get the bot
                    bot = Bot.get(bot_id)
                    if bot and bot.knowledge_files:
                        app.logger.info(f"[RAG] Checking bot {bot_id} knowledge base with {len(bot.knowledge_files)} files")
                        
                        # Debug collection name
                        app.logger.info(f"[RAG] Checking bot collection: {bot_collection}")
                        
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
                                use_rag = True
                                
                                # Log RAG query details
                                app.logger.info(f"[RAG] Querying bot knowledge with: '{user_message[:100]}...'")
                                
                                # Retrieve context from bot's knowledge base
                                retrieval_start = time.time()
                                bot_context = rag.retrieve_context(user_message, collection_name=bot_collection)
                                retrieval_time = time.time() - retrieval_start
                                
                                if bot_context:
                                    retrieved_context = bot_context
                                    app.logger.info(f"[RAG] Retrieved {len(retrieved_context)} characters from bot knowledge in {retrieval_time:.2f}s")
                                    app.logger.info(f"[RAG] Bot knowledge sample: '{retrieved_context[:200]}...'")
                                else:
                                    app.logger.info(f"[RAG] No relevant context found in bot's knowledge base despite {count} documents")
                            else:
                                app.logger.info(f"[RAG] Bot {bot_id} has {len(bot.knowledge_files)} files but no indexed documents in collection {bot_collection}")
                        except Exception as e:
                            app.logger.error(f"[RAG] Error checking bot collection: {str(e)}")
                            app.logger.info(f"[RAG] Falling back to has_documents check")
                            
                            # Fallback to original check
                            if rag.has_documents(collection_name=bot_collection):
                                app.logger.info(f"[RAG] Documents found for bot {bot_id} using fallback check")
                                use_rag = True
                                
                                # Log RAG query details
                                app.logger.info(f"[RAG] Querying bot knowledge with: '{user_message[:100]}...'")
                                
                                # Retrieve context from bot's knowledge base
                                retrieval_start = time.time()
                                bot_context = rag.retrieve_context(user_message, collection_name=bot_collection)
                                retrieval_time = time.time() - retrieval_start
                                
                                if bot_context:
                                    retrieved_context = bot_context
                                    app.logger.info(f"[RAG] Retrieved {len(retrieved_context)} characters from bot knowledge in {retrieval_time:.2f}s")
                                    app.logger.info(f"[RAG] Bot knowledge sample: '{retrieved_context[:200]}...'")
                                else:
                                    app.logger.info(f"[RAG] No relevant context found in bot's knowledge base")
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
        if system_prompt:
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
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md', 'xml'}

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
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.xml'):
                    loader = UnstructuredXMLLoader(file_path)
                elif filename.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(file_path)
                else:  # .txt
                    loader = TextLoader(file_path)
                    
                documents = loader.load()
                # Add document to vector store
                rag.add_documents(documents)
                app.logger.info(f'Processed {len(documents)} documents from {filename}')
            except Exception as e:
                app.logger.error(f'Error processing file {filename}: {str(e)}')
                if os.path.exists(file_path):
                    os.remove(file_path)
                saved_files.remove(filename)
            
            # Call RAG to index the new file with conversation-specific collection
            success = rag.add_single_document_to_store(file_path, conversation_id=conversation_id)
            
            if success:
                return jsonify({
                    'status': 'success', 
                    'message': f'Document \'{filename}\' uploaded and indexed successfully.',
                    'document_id': db.add_conversation_document(conversation_id, filename, file_path)
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

@app.route('/api/generate_image', methods=['POST'])
@login_required
def generate_image():
    """Generate an image using ComfyUI based on a prompt or workflow"""
    import os
    import json
    import time
    import requests
    import base64
    from flask import request, jsonify
    from datetime import datetime

    data = request.get_json()
    prompt = data.get('prompt', '')
    model = data.get('model', '')  # Optional: workflow name if using a specific workflow
    conversation_id = data.get('conversation_id')  # Add conversation_id
        
    # Store the last used seed in a global variable
    global last_used_seed
    if not hasattr(app, 'last_used_seed'):
        app.last_used_seed = random.randint(0, 2147483647)
        
    # Generate a unique identifier for this request to prevent caching
    request_id = str(uuid.uuid4())
    timestamp = int(time.time())
        
    # Extract augment parameters from the prompt
    aspect_ratio = "1:1"  # Default square
    steps = 20  # Default number of steps
    seed = None  # Default to None, will be randomized later if not specified
        
    # Check for aspect ratio command
    aspect_ratio_match = re.search(r'--ar\s+(\d+:\d+)', prompt)
    if aspect_ratio_match:
        aspect_ratio = aspect_ratio_match.group(1)
        # Remove the command from the prompt
        prompt = re.sub(r'--ar\s+\d+:\d+', '', prompt).strip()
        
    # Check for steps command
    steps_match = re.search(r'--steps\s+(\d+)', prompt)
    if steps_match:
        steps = int(steps_match.group(1))
        # Remove the command from the prompt
        prompt = re.sub(r'--steps\s+\d+', '', prompt).strip()
        
    # Check for seed command
    seed_match = re.search(r'--seed\s+(\w+)', prompt)
    if seed_match:
        seed_value = seed_match.group(1)
        # Handle 'last' keyword for seed
        if seed_value.lower() == 'last':
            seed = app.last_used_seed
        else:
            try:
                seed = int(seed_value)
            except ValueError:
                # If not a valid integer, use a random seed
                seed = random.randint(0, 2147483647)
        # Remove the command from the prompt
        prompt = re.sub(r'--seed\s+\w+', '', prompt).strip()
    else:
        # If no seed specified, use a random seed
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
        
    # ComfyUI API endpoint
    comfy_api_url = os.environ.get('COMFYUI_API_URL', 'http://localhost:8188')
    comfy_output_dir = os.environ.get('COMFYUI_OUTPUT_DIR', 'MIDAS_standalone/ComfyUI/output')
        
    # Determine which workflow to use
    workflow = None
    workflow_file = None
        
    if model and model.startswith('workflow:'):
        # Extract workflow name from model string
        workflow = model.replace('workflow:', '')
        workflow_file = f'workflows/{workflow}.json'
    else:
        # Default workflow
        workflow = "Flux 1.1"
        workflow_file = 'workflows/flux1.1.json'
        
    if workflow:
        # Load the workflow JSON
        try:
            with open(workflow_file, 'r') as f:
                comfyui_payload = json.load(f)
            
            # Find the text prompt node and update it
            for node_id, node_data in comfyui_payload.items():
                if node_data.get('class_type') == 'CLIPTextEncode':
                    # Update the text prompt with a unique identifier to prevent caching
                    # The identifier is added as a comment that won't affect generation
                    node_data['inputs']['text'] = f"{prompt} \\n<!-- request_id: {request_id} timestamp: {timestamp} -->"
                
                # Update resolution if there's a resolution node
                if 'width' in node_data.get('inputs', {}) and 'height' in node_data.get('inputs', {}):
                    node_data['inputs']['width'] = width
                    node_data['inputs']['height'] = height
                    
                # Update steps if there's a sampler node
                if 'steps' in node_data.get('inputs', {}):
                    node_data['inputs']['steps'] = steps
                    
                # Update seed if there's a seed node
                if 'seed' in node_data.get('inputs', {}):
                    node_data['inputs']['seed'] = seed
                # Check for noise_seed parameter
                elif 'noise_seed' in node_data.get('inputs', {}):
                    node_data['inputs']['noise_seed'] = seed
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error loading workflow: {str(e)}'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'No workflow specified'}), 400
        
    # Submit to ComfyUI
    comfyui_url = f"{comfy_api_url}/prompt"
    print(f"Submitting to ComfyUI at {comfyui_url} with payload: {json.dumps(comfyui_payload, indent=2)}")
    resp = requests.post(comfyui_url, json={'prompt': comfyui_payload}, timeout=900)
    print(f"ComfyUI response status: {resp.status_code}, content: {resp.text}")
    resp.raise_for_status()
    result = resp.json()
    print(f"ComfyUI prompt_id: {result.get('prompt_id')}")
        
    # Wait for the generation to complete
    max_attempts = 300  # Maximum wait time = max_attempts * sleep_time (5 minutes)
    sleep_time = 1  # seconds

    # Wait at least 120 seconds before first check to give time for image generation
    print(f"Waiting 120 seconds for image generation to complete for prompt_id {result.get('prompt_id')}")
    time.sleep(270)
    
    # First check after initial wait
    print(f"Checking history for prompt_id {result.get('prompt_id')} (first check)")
    history_resp = requests.get(f'{comfy_api_url}/history/{result.get("prompt_id")}', timeout=10)
    print(f"History response: {history_resp.status_code}, content: {history_resp.text}")
    
    # If not ready, wait another 150 seconds
    if history_resp.status_code != 200 or not history_resp.json().get(result.get('prompt_id'), {}).get('outputs'):
        print(f"Image not ready yet. Waiting additional 150 seconds for prompt_id {result.get('prompt_id')}")
        time.sleep(150)
        
        # Second check after additional wait
        print(f"Checking history again for prompt_id {result.get('prompt_id')} (second check)")
        history_resp = requests.get(f'{comfy_api_url}/history/{result.get("prompt_id")}', timeout=10)
        print(f"History response: {history_resp.status_code}, content: {history_resp.text}")
    
    if history_resp.status_code == 200:
        history_data = history_resp.json()
        # Check if the prompt has outputs (meaning it's complete)
        if history_data.get(result.get('prompt_id'), {}).get('outputs'):
            # Get the outputs for the SaveImage node
            outputs = history_data[result.get('prompt_id')]['outputs']
            image_data = None
            image_filename = None

            # Look through all outputs to find image data
            for node_id, node_output in outputs.items():
                # Check if this is an image output
                if node_output.get('images'):
                    # Get the first image
                    image_filename = node_output['images'][0]['filename']
                    image_data = node_output['images'][0]
                    break

                if image_filename:
                    # Construct path to the image in ComfyUI's output directory
                    image_path = os.path.join(comfy_output_dir, image_filename)
                    print(f"Looking for image at: {image_path}")  # Debug output
                    # Make sure the file exists
                    if os.path.exists(image_path):
                        print(f"Found image at: {image_path}")  # Debug output
                        # Read and encode the image
                        with open(image_path, 'rb') as img_file:
                            img_b64 = base64.b64encode(img_file.read()).decode('utf-8')

                            # Save message to conversation history if conversation_id is provided
                            if conversation_id:
                                try:
                                    # Use the db.add_message function to properly save the message
                                    # This ensures proper persistence and updates conversation timestamps
                                    message_id = db.add_message(
                                        conversation_id=conversation_id,
                                        role='assistant',
                                        content=f'',
                                        thinking=None,
                                        images=[img_b64],  
                                        attachment_filename=image_filename
                                    )

                                    if not message_id:
                                        print("Failed to save image message to database")
                                        # Log the error but don't fail the request
                                        import traceback
                                        traceback.print_exc()
                                except Exception as db_error:
                                    print(f"Error saving image message to database: {db_error}")
                                    # Log the full error details for debugging
                                    import traceback
                                    traceback.print_exc()
                                    # Return a more informative error response
                                    return jsonify({
                                        'status': 'error',
                                        'message': f'Failed to save image message: {str(db_error)}'
                                    }), 500

                            return jsonify({
                                'status': 'success', 
                                'image_base64': img_b64, 
                                'filename': image_filename,
                                'workflow': workflow,
                                'seed': seed  # Include the seed in the response
                            })

                # If we couldn't find the image or it wasn't saved yet
                return jsonify({
                    'status': 'error',
                    'message': 'Image generation completed but image file not found'
                }), 404

    # If we've exhausted all attempts and still haven't found the image
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
        app.logger.info(f'Upload request received for bot {bot_id}')
        app.logger.info(f'Request files: {request.files}')
        app.logger.info(f'Content length: {request.content_length}')

        bot = Bot.get(bot_id)
        if not bot:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404

        if 'files' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No files provided'
            }), 400

        files = request.files.getlist('files')
        saved_files = []

        # Create knowledge base directory if it doesn't exist
        kb_dir = bot.get_knowledge_base_path()
        os.makedirs(kb_dir, exist_ok=True)

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
                    
                    # Split documents into chunks
                    app.logger.info(f'Splitting {filename} into chunks...')
                    text_splitter = rag.RecursiveCharacterTextSplitter(
                        chunk_size=rag.CHUNK_SIZE, 
                        chunk_overlap=rag.CHUNK_OVERLAP
                    )
                    split_docs = text_splitter.split_documents(documents)
                    app.logger.info(f'Split into {len(split_docs)} chunks for indexing')
                    
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

if __name__ == '__main__':
    app.run(debug=True)
