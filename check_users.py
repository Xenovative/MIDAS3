import sqlite3
import os

# Path to the database
DB_PATH = os.path.join('data', 'conversations.db')

def list_all_users():
    """List all users in the database"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all users
        cursor.execute('SELECT id, username, role, display_name, message_quota, attachment_size_limit FROM users')
        users = [dict(row) for row in cursor.fetchall()]
        
        print(f"Found {len(users)} users in the database:")
        for user in users:
            print(f"ID: {user['id']}, Username: {user['username']}, Role: {user['role']}, Display Name: {user.get('display_name')}")
            print(f"  Message Quota: {user.get('message_quota')}, Attachment Size Limit: {user.get('attachment_size_limit')}")
            print("-" * 50)
        
        return users
    except Exception as e:
        print(f"Error listing users: {e}")
        return []
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    list_all_users()
