import sqlite3
import os
from passlib.hash import bcrypt

# Path to the database
DB_PATH = os.path.join('data', 'conversations.db')

def reset_admin_password(username, new_password):
    """Reset the password for a user with admin role"""
    try:
        # Hash the new password
        password_hash = bcrypt.hash(new_password)
        
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Find the user with admin role
        cursor.execute('SELECT id FROM users WHERE username = ? AND role = ?', (username, 'admin'))
        user = cursor.fetchone()
        
        if not user:
            print(f"No admin user found with username: {username}")
            return False
        
        # Update the password
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, user[0]))
        conn.commit()
        
        print(f"Password reset successfully for admin user: {username}")
        return True
    except Exception as e:
        print(f"Error resetting password: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Reset the admin password to xenotest1234#
    reset_admin_password('admin', 'xenotest1234#')
