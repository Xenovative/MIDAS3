import sqlite3
import os
from passlib.hash import bcrypt

# Path to the database
DB_PATH = os.path.join('data', 'conversations.db')

def reset_admin_password(username, new_password, email='office@xenovative-ltd.com'):
    """Reset the password and email for a user with admin role"""
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
        
        # Update the password and email
        cursor.execute('UPDATE users SET password_hash = ?, email = ? WHERE id = ?', 
                      (password_hash, email, user[0]))
        conn.commit()
        
        print(f"Admin user {username} updated successfully")
        print(f"- Password reset")
        print(f"- Email set to: {email}")
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
