import os
import json
import uuid
from datetime import datetime

# Path to store bot data
BOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'bots')
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'knowledge')

# Ensure directories exist
os.makedirs(BOTS_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

class Bot:
    def __init__(self, id=None, name="New Bot", description="", greeting="Hello! How can I help you today?", 
                 base_model=None, embedding_model="nomic-embed-text", system_prompt="", 
                 knowledge_files=None, parameters=None, created_at=None, updated_at=None):
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.greeting = greeting
        self.base_model = base_model
        self.embedding_model = embedding_model
        self.system_prompt = system_prompt
        self.knowledge_files = knowledge_files or []
        self.parameters = parameters or {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 8192  # Increased from 2048 to 8192 for more comprehensive responses
        }
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
    
    def to_dict(self):
        """Convert Bot object to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "greeting": self.greeting,
            "base_model": self.base_model,
            "embedding_model": self.embedding_model,
            "system_prompt": self.system_prompt,
            "knowledge_files": self.knowledge_files,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create Bot object from dictionary"""
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            greeting=data.get("greeting"),
            base_model=data.get("base_model"),
            embedding_model=data.get("embedding_model"),
            system_prompt=data.get("system_prompt"),
            knowledge_files=data.get("knowledge_files"),
            parameters=data.get("parameters"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )
    
    def save(self):
        """Save bot to file"""
        self.updated_at = datetime.now().isoformat()
        bot_file = os.path.join(BOTS_DIR, f"{self.id}.json")
        with open(bot_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return self
    
    @classmethod
    def get(cls, bot_id):
        """Get bot by ID"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Remove any file extension if present
        if bot_id and isinstance(bot_id, str) and bot_id.endswith('.json'):
            bot_id = bot_id[:-5]
        
        if not bot_id or not isinstance(bot_id, str):
            logger.error(f"Invalid bot_id provided: {bot_id} (type: {type(bot_id)})")
            return None
            
        bot_file = os.path.join(BOTS_DIR, f"{bot_id}.json")
        
        # Debug logging
        logger.info(f"Looking for bot file: {bot_file}")
        logger.info(f"BOTS_DIR exists: {os.path.exists(BOTS_DIR)}")
        logger.info(f"BOTS_DIR path: {os.path.abspath(BOTS_DIR)}")
        
        if not os.path.exists(BOTS_DIR):
            logger.error(f"BOTS_DIR does not exist: {BOTS_DIR}")
            return None
            
        if not os.path.exists(bot_file):
            logger.warning(f"Bot file not found: {bot_file}")
            # List all files in BOTS_DIR for debugging
            try:
                files = os.listdir(BOTS_DIR)
                logger.info(f"Files in BOTS_DIR: {files}")
                json_files = [f for f in files if f.endswith('.json')]
                logger.info(f"JSON files in BOTS_DIR: {json_files}")
                
                # Try case-insensitive search
                for f in files:
                    if f.lower() == f"{bot_id}.json".lower():
                        logger.warning(f"Found case-insensitive match: {f} (requested: {bot_id}.json)")
                        bot_file = os.path.join(BOTS_DIR, f)
                        break
                else:
                    return None  # No match found
                    
            except Exception as e:
                logger.error(f"Error listing BOTS_DIR: {str(e)}", exc_info=True)
                return None
        
        try:
            logger.info(f"Attempting to load bot file: {bot_file}")
            with open(bot_file, 'r', encoding='utf-8') as f:
                bot_data = json.load(f)
                
            if not isinstance(bot_data, dict):
                logger.error(f"Bot file {bot_file} does not contain a JSON object")
                return None
                
            logger.info(f"Successfully loaded bot: {bot_data.get('name', 'Unnamed')} (ID: {bot_data.get('id', 'Unknown')})")
            return cls.from_dict(bot_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {bot_file}: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading bot file {bot_file}: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def get_all(cls):
        """Get all bots"""
        bots = []
        for filename in os.listdir(BOTS_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(BOTS_DIR, filename), 'r') as f:
                    bots.append(cls.from_dict(json.load(f)))
        return sorted(bots, key=lambda x: x.updated_at, reverse=True)
    
    @classmethod
    def delete(cls, bot_id):
        """Delete bot by ID"""
        bot_file = os.path.join(BOTS_DIR, f"{bot_id}.json")
        if os.path.exists(bot_file):
            os.remove(bot_file)
            return True
        return False
    
    def get_knowledge_base_path(self):
        """Get the path to the bot's knowledge base directory"""
        kb_path = os.path.join(KNOWLEDGE_DIR, self.id)
        os.makedirs(kb_path, exist_ok=True)
        return kb_path
        
    def save(self):
        """Save bot to file"""
        self.updated_at = datetime.now().isoformat()
        bot_file = os.path.join(BOTS_DIR, f"{self.id}.json")
        with open(bot_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return self
