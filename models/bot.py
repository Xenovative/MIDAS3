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
            "max_tokens": 2048
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
        bot_file = os.path.join(BOTS_DIR, f"{bot_id}.json")
        if not os.path.exists(bot_file):
            return None
        with open(bot_file, 'r') as f:
            return cls.from_dict(json.load(f))
    
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
