import sqlite3
import json
import os
import time
from datetime import datetime

# Ensure the database directory exists
os.makedirs('data', exist_ok=True)
DB_PATH = os.path.join('data', 'conversations.db')

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    
    # Create messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        thinking TEXT,
        images TEXT, -- Store as JSON list of base64 strings or file paths
        attachment_filename TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
    )
    ''')

    # Add attachment_filename column to messages if it doesn't exist (simple migration)
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN attachment_filename TEXT")
        print("Added attachment_filename column to messages table.")
    except sqlite3.OperationalError as e:
        # Ignore error if column already exists
        if 'duplicate column name' not in str(e):
            raise

    # Create conversation documents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
    )
    ''')

    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create user_quotas table for storing message quotas and attachment limits
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_quotas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL UNIQUE,
        daily_message_limit INTEGER DEFAULT 100,
        monthly_message_limit INTEGER DEFAULT 3000,
        max_attachment_size_kb INTEGER DEFAULT 5120,
        messages_used_today INTEGER DEFAULT 0,
        messages_used_month INTEGER DEFAULT 0,
        last_reset_date DATE DEFAULT CURRENT_DATE,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()
    conn.close()

def create_conversation(user_id, title, model):
    """Create a new conversation for a user and return its ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (user_id, title, model) VALUES (?, ?, ?)",
        (user_id, title, model)
    )
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def get_conversation(conversation_id):
    """Get a conversation by ID with all its messages"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get conversation details
    cursor.execute(
        "SELECT * FROM conversations WHERE id = ?",
        (conversation_id,)
    )
    conversation = cursor.fetchone()
    
    if not conversation:
        conn.close()
        return None
    
    # Get all messages for this conversation
    cursor.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conversation_id,)
    )
    messages = cursor.fetchall()
    
    # Convert to dictionary
    result = dict(conversation)
    result['messages'] = [dict(message) for message in messages]
    
    conn.close()
    return result

def get_all_conversations():
    """Get all conversations with their latest message"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.*, 
               (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count,
               (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message
        FROM conversations c
        ORDER BY updated_at DESC
    """)
    
    conversations = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return conversations

def get_user_conversations(user_id):
    """Get all conversations for a specific user with their latest message"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.*, 
               (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count,
               (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message
        FROM conversations c
        WHERE c.user_id = ?
        ORDER BY updated_at DESC
    """, (user_id,))
    conversations = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return conversations

def get_user_conversation(user_id, conversation_id):
    """Get a conversation by ID only if it belongs to the user"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id)
    )
    conversation = cursor.fetchone()
    if not conversation:
        conn.close()
        return None
    cursor.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conversation_id,)
    )
    messages = cursor.fetchall()
    result = dict(conversation)
    result['messages'] = [dict(message) for message in messages]
    conn.close()
    return result

def add_message(conversation_id, role, content, thinking=None, images=None, attachment_filename=None):
    """Add a message to a conversation, optionally including images."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Check if conversation exists
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            return None
            
        # Update conversation's updated_at timestamp
        cursor.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))
        
        # Insert the message
        images_json = json.dumps(images) if images else None
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, thinking, images, attachment_filename) VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, thinking, images_json, attachment_filename)
        )
        conn.commit()
        
        return cursor.lastrowid
    except Exception as e:
        print(f"Database error in add_message: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_conversation_messages(conversation_id):
    """Get all messages for a conversation, ordered by timestamp"""
    conn = sqlite3.connect(DB_PATH)
    # Set row_factory before creating cursor
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, role, content, thinking, images, created_at, attachment_filename FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,)
        )
        messages = []
        for row in cursor.fetchall():
            # Convert Row object to dict
            row_dict = {key: row[key] for key in row.keys()}
            
            # Deserialize images if present
            if row_dict.get('images'):
                try:
                    row_dict['images'] = json.loads(row_dict['images'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode images JSON for message {row_dict['id']}")
                    row_dict['images'] = None # Or handle as error
            else:
                 row_dict['images'] = None
            messages.append(row_dict)
        return messages
    except Exception as e:
        print(f"Database error in get_conversation_messages: {e}")
        return []
    finally:
        conn.close()

def update_conversation_title(conversation_id, title):
    """Update a conversation's title"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (title, conversation_id)
    )
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0

def update_conversation_model(conversation_id, model):
    """Update a conversation's model"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE conversations SET model = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (model, conversation_id)
    )
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0

def delete_conversation(conversation_id):
    """Delete a conversation and all its messages"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0

def delete_message(message_id):
    """Delete a specific message by its ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        conn.commit()
        deleted_count = cursor.rowcount
        print(f"Attempted to delete message {message_id}, rows affected: {deleted_count}") # Debug log
        return deleted_count > 0
    except Exception as e:
        print(f"Database error in delete_message: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# --- Conversation Document Functions ---

def add_conversation_document(conversation_id, filename, file_path):
    """Add a document to a conversation"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO conversation_documents (conversation_id, filename, file_path)
        VALUES (?, ?, ?)
        ''', (conversation_id, filename, file_path))
        
        conn.commit()
        document_id = cursor.lastrowid
        return document_id
    except Exception as e:
        print(f"Error adding conversation document: {e}")
        return None
    finally:
        conn.close()

def get_conversation_documents(conversation_id):
    """Get all documents for a conversation"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        SELECT id, conversation_id, filename, file_path, created_at
        FROM conversation_documents
        WHERE conversation_id = ?
        ORDER BY created_at DESC
        ''', (conversation_id,))
        
        documents = [dict(row) for row in cursor.fetchall()]
        return documents
    except Exception as e:
        print(f"Error getting conversation documents: {e}")
        return []
    finally:
        conn.close()

def delete_conversation_document(document_id):
    """Delete a conversation document"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        DELETE FROM conversation_documents
        WHERE id = ?
        ''', (document_id,))
        
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting conversation document: {e}")
        return False
    finally:
        conn.close()

# --- User Management Functions ---

def create_user(username, password_hash, role='user'):
    """Create a new user with a hashed password and role."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
        ''', (username, password_hash, role))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        print(f"Integrity error creating user: {e}")
        return None
    finally:
        conn.close()

def get_user_by_username(username):
    """Retrieve a user by username."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id):
    """Get a user by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, username, password_hash, role, created_at FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if user:
            # Convert to dict
            user_dict = dict(user)
            # Add display_name field for compatibility with frontend
            user_dict['display_name'] = user_dict['username']
            return user_dict
        return None
    except Exception as e:
        print(f"Error getting user by ID: {e}")
        return None
    finally:
        conn.close()

def update_user_display_name(user_id, display_name):
    """Update a user's display name."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First, check if display_name column exists
    try:
        cursor.execute("SELECT display_name FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Add display_name column if it doesn't exist
        cursor.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
    
    try:
        cursor.execute('''
            UPDATE users 
            SET display_name = ?
            WHERE id = ?
        ''', (display_name, user_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating user display name: {e}")
        return False
    finally:
        conn.close()

def update_user_password(user_id, password_hash):
    """Update a user's password hash."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE users 
            SET password_hash = ?
            WHERE id = ?
        ''', (password_hash, user_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating user password: {e}")
        return False
    finally:
        conn.close()

def get_all_users():
    """Get all users."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, username, role, created_at FROM users ORDER BY id')
        users = [dict(row) for row in cursor.fetchall()]
        # Add display_name field for compatibility with frontend
        for user in users:
            user['display_name'] = user['username']
        return users
    except Exception as e:
        print(f"Error getting all users: {e}")
        return []
    finally:
        conn.close()

def update_user_role(user_id, role):
    """Update a user's role."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE users 
            SET role = ?
            WHERE id = ?
        ''', (role, user_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating user role: {e}")
        return False
    finally:
        conn.close()

def delete_user(user_id):
    """Delete a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # First delete all user's conversations
        cursor.execute('DELETE FROM conversations WHERE user_id = ?', (user_id,))
        # Then delete the user
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting user: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# --- User Quota Management Functions ---

def get_user_quota(user_id):
    """Get quota information for a user"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # First check if the user has a quota record
        cursor.execute('SELECT * FROM user_quotas WHERE user_id = ?', (user_id,))
        quota = cursor.fetchone()
        
        # If no quota record exists, create one with default values
        if not quota:
            cursor.execute('''
                INSERT INTO user_quotas 
                (user_id, daily_message_limit, monthly_message_limit, max_attachment_size_kb, 
                messages_used_today, messages_used_month, last_reset_date)
                VALUES (?, 100, 3000, 5120, 0, 0, CURRENT_DATE)
            ''', (user_id,))
            conn.commit()
            
            # Fetch the newly created record
            cursor.execute('SELECT * FROM user_quotas WHERE user_id = ?', (user_id,))
            quota = cursor.fetchone()
        
        # Convert to dict
        quota_dict = dict(quota) if quota else {}
        
        # Reset counters if needed
        reset_quotas_if_needed(user_id, quota_dict)
        
        return quota_dict
    except Exception as e:
        print(f"Error getting user quota: {e}")
        return None
    finally:
        conn.close()

def update_user_quota(user_id, daily_limit=None, monthly_limit=None, max_attachment_size=None):
    """Update quota settings for a user"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Set row_factory for the connection
    cursor = conn.cursor()
    
    try:
        # Check if user has a quota record
        cursor.execute('SELECT id FROM user_quotas WHERE user_id = ?', (user_id,))
        quota = cursor.fetchone()
        
        if quota:
            # Update existing record
            update_fields = []
            params = []
            
            if daily_limit is not None:
                update_fields.append('daily_message_limit = ?')
                params.append(daily_limit)
            
            if monthly_limit is not None:
                update_fields.append('monthly_message_limit = ?')
                params.append(monthly_limit)
            
            if max_attachment_size is not None:
                update_fields.append('max_attachment_size_kb = ?')
                params.append(max_attachment_size)
            
            if update_fields:
                query = f"UPDATE user_quotas SET {', '.join(update_fields)} WHERE user_id = ?"
                params.append(user_id)
                cursor.execute(query, params)
                conn.commit()
                
                # Fetch the updated record
                cursor.execute('SELECT * FROM user_quotas WHERE user_id = ?', (user_id,))
                updated_quota = cursor.fetchone()
                if updated_quota:
                    return dict(updated_quota)
                return None
        else:
            # Create new record with specified values
            daily_limit = daily_limit if daily_limit is not None else 100
            monthly_limit = monthly_limit if monthly_limit is not None else 3000
            max_attachment_size = max_attachment_size if max_attachment_size is not None else 5120
            
            cursor.execute('''
                INSERT INTO user_quotas 
                (user_id, daily_message_limit, monthly_message_limit, max_attachment_size_kb, 
                messages_used_today, messages_used_month, last_reset_date)
                VALUES (?, ?, ?, ?, 0, 0, CURRENT_DATE)
            ''', (user_id, daily_limit, monthly_limit, max_attachment_size))
            conn.commit()
            
            # Fetch the newly created record
            cursor.execute('SELECT * FROM user_quotas WHERE user_id = ?', (user_id,))
            new_quota = cursor.fetchone()
            if new_quota:
                return dict(new_quota)
            return None
            
        return False
    except Exception as e:
        print(f"Error updating user quota: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def increment_user_message_count(user_id):
    """Increment the message count for a user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get current quota
        quota = get_user_quota(user_id)
        if not quota:
            return False
        
        # Reset counters if needed
        reset_quotas_if_needed(user_id, quota)
        
        # Increment message counts
        cursor.execute('''
            UPDATE user_quotas 
            SET messages_used_today = messages_used_today + 1,
                messages_used_month = messages_used_month + 1
            WHERE user_id = ?
        ''', (user_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error incrementing message count: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def check_user_quota(user_id):
    """Check if a user has exceeded their quota"""
    quota = get_user_quota(user_id)
    if not quota:
        # If no quota info, default to allowing messages
        return {"allowed": True, "reason": None}
    
    # Check daily limit
    if quota['messages_used_today'] >= quota['daily_message_limit']:
        return {
            "allowed": False, 
            "reason": "daily_limit",
            "limit": quota['daily_message_limit'],
            "used": quota['messages_used_today']
        }
    
    # Check monthly limit
    if quota['messages_used_month'] >= quota['monthly_message_limit']:
        return {
            "allowed": False, 
            "reason": "monthly_limit",
            "limit": quota['monthly_message_limit'],
            "used": quota['messages_used_month']
        }
    
    # User is within quota
    return {
        "allowed": True,
        "daily_limit": quota['daily_message_limit'],
        "daily_used": quota['messages_used_today'],
        "monthly_limit": quota['monthly_message_limit'],
        "monthly_used": quota['messages_used_month'],
        "max_attachment_size_kb": quota['max_attachment_size_kb']
    }

def reset_quotas_if_needed(user_id, quota=None):
    """Reset daily and monthly quotas if needed"""
    if quota is None:
        quota = get_user_quota(user_id)
        if not quota:
            return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        last_reset = datetime.strptime(quota['last_reset_date'], '%Y-%m-%d').date()
        today = datetime.now().date()
        
        updates = []
        params = []
        
        # Reset daily count if it's a new day
        if today > last_reset:
            updates.append("messages_used_today = 0")
            updates.append("last_reset_date = CURRENT_DATE")
        
        # Reset monthly count if it's a new month
        if today.month != last_reset.month or today.year != last_reset.year:
            updates.append("messages_used_month = 0")
        
        if updates:
            query = f"UPDATE user_quotas SET {', '.join(updates)} WHERE user_id = ?"
            params.append(user_id)
            cursor.execute(query, params)
            conn.commit()
            return True
            
        return False
    except Exception as e:
        print(f"Error resetting quotas: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# Initialize the database when the module is imported
init_db()
