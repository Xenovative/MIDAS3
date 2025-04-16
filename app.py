from flask import Flask, request, jsonify, render_template, Response
import ollama
import config # Added import
from config import DEFAULT_MODEL, MODEL_OPTIONS, API_TIMEOUT, THINKING_MODELS, AVAILABLE_MODELS, OLLAMA_HOST, AVAILABLE_EMBEDDING_MODELS, VISIBLE_MODELS, USER_PREFERENCES
import os
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

app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

log_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    """Endpoint to list available models"""
    try:
        app.logger.info("Attempting to get available models...")
        
        # Use the filtered models from config based on user preferences
        models_list = VISIBLE_MODELS
        app.logger.info(f"Visible models: {models_list}")
        
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
def list_conversations():
    """List all conversations"""
    try:
        conversations = db.get_all_conversations()
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
def create_conversation():
    """Create a new conversation"""
    try:
        data = request.get_json()
        title = data.get('title', f"New Chat {time.strftime('%Y-%m-%d %H:%M')}")
        model = data.get('model', DEFAULT_MODEL)
        
        conversation_id = db.create_conversation(title, model)
        
        return jsonify({
            'conversation_id': conversation_id,
            'title': title,
            'model': model,
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
def get_conversation(conversation_id):
    """Get a conversation by ID with all its messages"""
    try:
        conversation = db.get_conversation(conversation_id)
        
        if not conversation:
            return jsonify({'status': 'error', 'message': 'Conversation not found'}), 404
        
        # db.get_conversation already includes the messages list
        # No need to fetch messages separately here
        
        # The 'conversation' dict now contains a 'messages' list
        # We need to ensure the frontend expects this structure
        # The frontend currently expects 'conversation' and 'messages' as separate top-level keys.
        # Let's adjust the response to match the frontend expectation for now.
        
        messages_list = conversation.pop('messages', []) # Extract messages and remove from conversation dict
        
        # Debug: Log image data in messages
        for msg in messages_list:
            if msg.get('images'):
                print(f"Message {msg['id']} has images: {type(msg['images'])}, length: {len(msg['images']) if isinstance(msg['images'], list) else 'not list'}")
                if isinstance(msg['images'], list) and len(msg['images']) > 0:
                    print(f"First image data type: {type(msg['images'][0])}, length: {len(str(msg['images'][0])) if msg['images'][0] else 'None'}")

        return jsonify({
            'status': 'success',
            'conversation': conversation, # Conversation details without messages
            'messages': messages_list      # Messages as a separate list
        })
    except Exception as e:
        app.logger.error(f"Error getting conversation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/conversations/<int:conversation_id>', methods=['PUT'])
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
                        bot = Bot.get(bot_id)
                        if bot:
                            model_name = bot.base_model or DEFAULT_MODEL
                            app.logger.info(f"Using bot's base model {model_name} for title generation")
                        else:
                            app.logger.warning(f"Bot {bot_id} not found, falling back to default model")
                            model_name = DEFAULT_MODEL
                    except Exception as e:
                        app.logger.error(f"Error fetching bot details: {str(e)}")
                        model_name = DEFAULT_MODEL
                
                # Check if model supports thinking process
                supports_thinking = model_name in THINKING_MODELS
                
                # Create prompt for title generation
                if supports_thinking:
                    title_prompt = (
                        "Based on this user message, generate a concise, descriptive title "
                        "for this conversation (3-5 words). The title should clearly reflect "
                        "the main topic or question. First think about it inside "
                        "<think></think> tags, then provide ONLY the title with no extra text, "
                        "quotes, or punctuation.\n\n"
                        f"User message: {first_user_message}"
                    )
                else:
                    title_prompt = (
                        "Based on this user message, generate a concise, descriptive title "
                        "for this conversation (3-5 words). The title should clearly reflect "
                        "the main topic or question. Provide ONLY the title with no extra text, "
                        "quotes, or punctuation.\n\n"
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
                    
                    # Clean up the title if using a thinking model
                    if supports_thinking and '<think>' in generated_title and '</think>' in generated_title:
                        # Remove the thinking section
                        think_start = generated_title.find('<think>')
                        think_end = generated_title.find('</think>') + len('</think>')
                        generated_title = generated_title.replace(generated_title[think_start:think_end], '').strip()
                    
                    # Clean up the title - remove quotes, periods, and extra spaces
                    title = generated_title.strip('"\'.,!?').strip()
                    
                    # Capitalize the first letter of each word for consistency
                    title = ' '.join(word.capitalize() for word in title.split())
                    
                    # Limit to 50 characters if somehow still too long
                    if len(title) > 50:
                        title = title[:47] + '...'
                        
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
def add_message(conversation_id):
    """Add a message to a conversation"""
    try:
        data = request.get_json()
        role = data.get('role')
        content = data.get('content')
        thinking = data.get('thinking')
        images = data.get('images') # Get images from request
        
        if not role or not content:
            return jsonify({
                'message': 'Role and content are required',
                'status': 'error'
            }), 400
            
        # Pass images to db.add_message
        message_id = db.add_message(conversation_id, role, content, thinking, images) 
        
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
def generate():
    """Generate a response to a user's message"""
    try:
        data = request.json
        user_message = data.get('message')
        conversation_id = data.get('conversation_id')
        model_name = data.get('model', config.DEFAULT_MODEL) # Get model from request or config
        attachment_filename = data.get('attachment_filename') # Get attachment filename
        images = data.get('images') # Get optional images list (base64 strings)
        
        if not conversation_id:
            return jsonify({'error': 'Conversation ID is required'}), 400
        
        if not user_message and not attachment_filename:
            # Allow sending request if only an attachment is present
            # If user message is empty but there's an attachment, create a placeholder message
            if attachment_filename:
                user_message = f"(Attached file: {attachment_filename})"
            else:
                return jsonify({'error': 'Message content is required if no file is attached'}), 400

        # --- 1. Add User Message to DB ---
        try:
            # Pass attachment_filename when adding user message
            user_message_id = db.add_message(conversation_id, 'user', user_message, attachment_filename=attachment_filename)
        except Exception as e:
            logging.error(f"Error adding user message to DB: {e}")
            # Continue processing, but log the error

        # --- RAG Integration ---
        # Only use RAG if documents have been uploaded for this conversation
        use_rag = False
        retrieved_context = ""
        
        # Check if any documents exist in the vector store for this conversation
        if rag.has_documents(conversation_id=conversation_id):
            app.logger.info(f"Documents found for conversation {conversation_id}, using RAG")
            use_rag = True
            # Retrieve context based on the user message from this conversation's collection
            retrieved_context = rag.retrieve_context(user_message, conversation_id=conversation_id)
        else:
            app.logger.info(f"No documents found for conversation {conversation_id}, skipping RAG")
        # --- End RAG Integration ---
        
        # --- Chat History Integration ---
        # Get previous messages for this conversation to use as context
        previous_messages = db.get_conversation_messages(conversation_id)
        
        # Format the chat history for the model - Ollama expects an array of messages
        # Each message should have a role (user or assistant) and content
        messages = []
        
        # Process previous messages and add them to the messages array
        # We need to exclude the current user message which might already be in the database
        # This happens because we add the user message to the database before generating a response
        
        # First, find the ID of the current user message in the database (if it exists)
        current_user_message_id = None
        for msg in reversed(previous_messages):  # Start from the most recent
            if msg['role'] == 'user' and msg.get('content') == user_message and 'id' in msg:
                current_user_message_id = msg['id']
                break
        
        # Now build the history excluding the current message
        history_messages = []
        for msg in previous_messages:
            # Skip the current user message that we just added to the database
            if current_user_message_id and 'id' in msg and msg['id'] == current_user_message_id:
                continue
            history_messages.append(msg)
        
        # Limit to the last 20 messages to avoid context length issues
        if len(history_messages) > 20:
            history_messages = history_messages[-20:]
        
        # Add previous messages to the context
        for msg in history_messages:
            if msg['role'] in ['user', 'assistant']:  # Only include user and assistant messages
                # Create a clean message object with just role and content
                clean_msg = {
                    "role": msg['role'],
                    "content": msg['content']
                }
                messages.append(clean_msg)
        
        # Add the current user message
        user_msg = {
            "role": "user",
            "content": user_message
        }
        
        # If images are present, add them to the user message
        if images:
            user_msg["images"] = images
        
        # Check for system prompt in the request
        system_prompt = data.get('system_prompt', '')
        
        # If a system prompt is provided, add it to the messages
        if system_prompt:
            system_msg = {
                "role": "system",
                "content": system_prompt
            }
            messages.insert(0, system_msg)
        
        # Add the current user message to the messages array
        messages.append(user_msg)
        
        # Improved logging to better understand what's happening with chat history
        previous_msg_count = len(messages) - 1  # Subtract 1 to exclude the current message
        app.logger.info(f"Conversation {conversation_id}: Found {len(previous_messages)} total messages in DB")
        app.logger.info(f"Conversation {conversation_id}: Using {previous_msg_count} previous messages as context")
        
        # Log the first few messages for debugging
        if previous_msg_count > 0:
            app.logger.info(f"First few context messages: {[m.get('content', '')[:30] + '...' for m in messages[:-1][:3]]}")
        # --- End Chat History Integration ---
        
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
                import sqlite3
                DB_PATH = os.path.join('data', 'conversations.db')
                conn = sqlite3.connect(DB_PATH)
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
                conversation_messages = db.get_conversation_messages(conversation_id)
                is_first_exchange = len(conversation_messages) <= 2  # Just the user message and this assistant message
                
                # Send the final message with the complete response AND the message IDs
                yield f"data: {json.dumps({'full_response': full_response, 'thinking': thinking, 'done': True, 'is_first_exchange': is_first_exchange})}\n\n"
                
                # Trigger title generation for first exchanges
                if is_first_exchange:
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
def generate_conversation_title(id):
    """Generate a title for the conversation based on its first message"""
    try:
        # Get the conversation
        conversation = db.get_conversation(id)
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
        
        # Find first user message
        first_user_message = next((m for m in messages if m['role'] == 'user'), None)
        if not first_user_message:
            app.logger.warning(f"No user message found in conversation {id}")
            return jsonify({
                'status': 'error',
                'message': 'No user message found'
            }), 404
        
        # Generate title from first message
        title = first_user_message['content']
        if len(title) > 50:
            title = title[:47] + '...'
            
        # Update conversation title
        db.update_conversation_title(id, title)
        
        # Return the updated conversation data
        updated_conversation = db.get_conversation(id)
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
        
        # List of supported embedding models
        embedding_models = [
            {"name": "nomic-embed-text"},
            {"name": "all-MiniLM-L6-v2"},
            {"name": "e5-large-v2"}
        ]
        
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
def get_preferences():
    """Get user preferences"""
    try:
        # Get user preferences from config
        preferences = config.load_user_preferences()
        
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
def update_preferences():
    """Update user preferences"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Create a copy of current preferences
        new_preferences = USER_PREFERENCES.copy()
        
        # Update only the fields that are provided in the request
        if 'default_model' in data:
            new_preferences['default_model'] = data['default_model']
        
        if 'default_embedding_model' in data:
            new_preferences['default_embedding_model'] = data['default_embedding_model']
        
        if 'visible_models' in data:
            new_preferences['visible_models'] = data['visible_models']
        
        if 'theme' in data:
            new_preferences['theme'] = data['theme']
        
        # Save preferences to file
        success = config.save_user_preferences(new_preferences)
        
        if success:
            # Update the global USER_PREFERENCES variable
            for key, value in new_preferences.items():
                USER_PREFERENCES[key] = value
            
            # Update other related global variables
            config.DEFAULT_MODEL = USER_PREFERENCES.get('default_model') or (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else 'llama2')
            config.DEFAULT_EMBEDDING_MODEL = USER_PREFERENCES.get('default_embedding_model', 'nomic-embed-text')
            config.VISIBLE_MODELS = USER_PREFERENCES.get('visible_models', []) or AVAILABLE_MODELS
            
            # Update imported globals in this module
            global DEFAULT_MODEL, VISIBLE_MODELS
            DEFAULT_MODEL = config.DEFAULT_MODEL
            VISIBLE_MODELS = config.VISIBLE_MODELS
            
            return jsonify({
                'status': 'success',
                'message': 'Preferences updated successfully',
                'preferences': USER_PREFERENCES
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to save preferences to file'
            }), 500
            
    except Exception as e:
        app.logger.error(f"Error updating preferences: {str(e)}")
        return jsonify({
            'message': 'Failed to update preferences',
            'status': 'error',
            'details': str(e)
        }), 500

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'docs')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload-doc', methods=['POST'])
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
    conversation = db.get_conversation(conversation_id)
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
            
            # Add document to the conversation in the database
            document_id = db.add_conversation_document(conversation_id, filename, file_path)
            if not document_id:
                app.logger.error(f"Failed to add document to conversation in database: {filename}")
                return jsonify({'status': 'error', 'message': 'Failed to record document in database'}), 500
            
            # Call RAG to index the new file with conversation-specific collection
            success = rag.add_single_document_to_store(file_path, conversation_id=conversation_id)
            
            if success:
                return jsonify({
                    'status': 'success', 
                    'message': f'Document \'{filename}\' uploaded and indexed successfully.',
                    'document_id': document_id
                })
            else:
                # Remove the document from the database if indexing failed
                db.delete_conversation_document(document_id)
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

@app.route('/api/chat', methods=['POST'])
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
        
    # Store the current seed as the last used seed
    app.last_used_seed = seed
        
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
    comfy_api_url = os.environ.get('COMFYUI_API_URL', 'http://127.0.0.1:8188')
    comfy_ws_url = os.environ.get('COMFYUI_WS_URL', 'ws://127.0.0.1:8188')
    comfy_output_dir = os.environ.get('COMFYUI_OUTPUT_DIR', 'ComfyUI/output')
        
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
        
    # Send to ComfyUI
    try:
        # ComfyUI API endpoint for prompt submission
        comfyui_url = 'http://localhost:8188/prompt'

        # Submit the workflow or prompt to ComfyUI
        resp = requests.post(comfyui_url, json={'prompt': comfyui_payload}, timeout=120)
        resp.raise_for_status()
        result = resp.json()

        # Extract prompt_id from the response
        prompt_id = result.get('prompt_id')
        if not prompt_id:
            return jsonify({'status': 'error', 'message': 'No prompt_id in ComfyUI response'}), 500

        # Wait for the generation to complete
        max_attempts = 300  # Maximum wait time = max_attempts * sleep_time (5 minutes)
        sleep_time = 1  # seconds

        for _ in range(max_attempts):
            # Check the history endpoint to see if our prompt has completed
            history_resp = requests.get(f'http://localhost:8188/history/{prompt_id}', timeout=10)

            if history_resp.status_code == 200:
                history_data = history_resp.json()
                # Check if the prompt has outputs (meaning it's complete)
                if history_data.get(prompt_id, {}).get('outputs'):
                    # Get the outputs for the SaveImage node
                    outputs = history_data[prompt_id]['outputs']
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
                        comfy_output_dir = r'C:\AIapps\MIDAS_standaloneapp\ComfyUI\output'
                        image_path = os.path.join(comfy_output_dir, image_filename)

                        # Make sure the file exists
                        if os.path.exists(image_path):
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

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================
# Bot Management API
# ============================================================

@app.route('/api/bots', methods=['GET'])
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
def upload_knowledge_files(bot_id):
    """Upload knowledge files for a bot"""
    try:
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

        # Get bot's knowledge base directory
        kb_dir = bot.get_knowledge_base_path()

        # Save files
        saved_files = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(kb_dir, filename)
                file.save(file_path)
                saved_files.append(filename)

        # Update bot's knowledge files list
        bot.knowledge_files = list(set(bot.knowledge_files + saved_files))
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

@app.route('/api/bots/<bot_id>/knowledge/<filename>', methods=['DELETE'])
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

@app.route('/api/bots/<bot_id>/chat', methods=['POST'])
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
