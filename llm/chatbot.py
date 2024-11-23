# llm/chatbot.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class Entity:
    """Represents an entity in the chatbot's knowledge base."""
    name: str
    attributes: Dict
    created_at: datetime = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        entity = cls(
            name=data["name"],
            attributes=data["attributes"]
        )
        entity.created_at = datetime.fromisoformat(data["created_at"])
        return entity

@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        message = cls(
            role=data["role"],
            content=data["content"]
        )
        message.timestamp = datetime.fromisoformat(data["timestamp"])
        return message

class Chatbot:
    def __init__(
        self,
        llm,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        max_history: int = 10,
        entities_file: str = "entities.json",
        history_file: str = "chat_history.json"
    ):
        """
        Initialize the chatbot.
        
        Parameters:
        - llm: LLM instance for generating responses
        - system_prompt: System prompt to guide the bot's behavior
        - context: Additional context for the conversation
        - max_history: Maximum number of messages to keep in history
        - entities_file: File to store entities
        - history_file: File to store chat history
        """
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.context = context
        self.max_history = max_history
        self.entities_file = entities_file
        self.history_file = history_file
        
        self.history: List[Message] = []
        self.entities: Dict[str, Entity] = {}
        
        # Load existing data
        self._load_entities()
        self._load_history()
    
    def _load_entities(self):
        """Load entities from file."""
        try:
            with open(self.entities_file, 'r') as f:
                data = json.load(f)
                self.entities = {
                    name: Entity.from_dict(entity_data)
                    for name, entity_data in data.items()
                }
        except FileNotFoundError:
            self.entities = {}
    
    def _save_entities(self):
        """Save entities to file."""
        with open(self.entities_file, 'w') as f:
            json.dump(
                {name: entity.to_dict() for name, entity in self.entities.items()},
                f,
                indent=2
            )
    
    def _load_history(self):
        """Load chat history from file."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.history = [Message.from_dict(msg_data) for msg_data in data]
        except FileNotFoundError:
            self.history = []
    
    def _save_history(self):
        """Save chat history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(
                [msg.to_dict() for msg in self.history],
                f,
                indent=2
            )
    
    def add_entity(self, name: str, attributes: Dict) -> Entity:
        """Add a new entity."""
        entity = Entity(name, attributes)
        self.entities[name] = entity
        self._save_entities()
        return entity
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        return self.entities.get(name)
    
    def _build_prompt(self, user_input: str) -> str:
        """Build the complete prompt including context and history."""
        prompt_parts = []
        
        # Add system prompt
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}\n")
        
        # Add context
        if self.context:
            prompt_parts.append(f"Context: {self.context}\n")
        
        # Add relevant history
        for msg in self.history[-self.max_history:]:
            prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}")
        
        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        
        return "\n".join(prompt_parts)
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and generate response.
        
        Parameters:
        - user_input: User's message
        
        Returns:
        - str: Assistant's response
        """
        # Add user message to history
        self.history.append(Message("user", user_input))
        
        # Generate complete prompt
        prompt = self._build_prompt(user_input)
        
        # Get response from LLM
        response = self.llm.generate(prompt)
        
        # Add assistant response to history
        self.history.append(Message("assistant", response))
        
        # Save updated history
        self._save_history()
        
        return response
    
    def get_history(self) -> List[Dict]:
        """
        Get conversation history.
        
        Returns:
        - List of message dictionaries
        """
        return [msg.to_dict() for msg in self.history]
    
    def clear_history(self):
        """Clear chat history."""
        self.history = []
        self._save_history()
