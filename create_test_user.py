import sqlite3
import os
from passlib.hash import bcrypt

# Path to the database
DB_PATH = os.path.join('data', 'conversations.db')

def create_test_user(username, password, role='user'):
    """Create a test user in the database"""
    try:
        # Hash the password
        password_hash = bcrypt.hash(password)
        
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            print(f"User {username} already exists")
            
            # Update the user's password
            cursor.execute('UPDATE users SET password_hash = ? WHERE username = ?', 
                          (password_hash, username))
            conn.commit()
            print(f"Updated password for user {username}")
            return True
        
        # Create the user
        cursor.execute('''
            INSERT INTO users (username, password_hash, role, message_quota, attachment_size_limit)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, password_hash, role, 50, 2097152))  # 50 messages, 2MB limit
        
        conn.commit()
        print(f"Created user {username} with role {role}")
        return True
    except Exception as e:
        print(f"Error creating test user: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Create a test user
    create_test_user('testuser1', 'password123')
