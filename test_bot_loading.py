import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bot import Bot, BOTS_DIR

def test_bot_loading():
    print("=== Testing Bot Loading ===")
    print(f"BOTS_DIR: {BOTS_DIR}")
    print(f"BOTS_DIR exists: {os.path.exists(BOTS_DIR)}")
    
    if not os.path.exists(BOTS_DIR):
        print("Error: BOTS_DIR does not exist!")
        return
    
    # List all bot files
    bot_files = [f for f in os.listdir(BOTS_DIR) if f.endswith('.json')]
    print(f"\nFound {len(bot_files)} bot files:")
    for i, bot_file in enumerate(bot_files, 1):
        print(f"{i}. {bot_file}")
    
    if not bot_files:
        print("No bot files found. Cannot test loading.")
        return
    
    # Test loading the first bot
    test_bot_id = os.path.splitext(bot_files[0])[0]
    print(f"\nTesting load for bot ID: {test_bot_id}")
    
    bot = Bot.get(test_bot_id)
    if bot:
        print("\nSuccessfully loaded bot:")
        print(f"ID: {bot.id}")
        print(f"Name: {bot.name}")
        print(f"Description: {bot.description}")
        print(f"Knowledge Files: {bot.knowledge_files}")
    else:
        print("\nFailed to load bot!")
        
        # Additional debug info
        print("\nDebug Info:")
        print(f"Full path: {os.path.join(BOTS_DIR, f'{test_bot_id}.json')}")
        print(f"File exists: {os.path.exists(os.path.join(BOTS_DIR, f'{test_bot_id}.json'))}")
        
        try:
            with open(os.path.join(BOTS_DIR, bot_files[0]), 'r') as f:
                content = f.read()
                print(f"\nFile content (first 500 chars):\n{content[:500]}...")
        except Exception as e:
            print(f"Error reading file: {e}")

if __name__ == "__main__":
    test_bot_loading()
