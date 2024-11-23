# llm/chatbot.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from .llm import LLM

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
        if "created_at" in data:
            entity.created_at = datetime.fromisoformat(data["created_at"])
        return entity

@dataclass
class HistoryEntry:
    """Represents a single conversation entry in the history."""
    timestamp: datetime
    query: str
    response: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "response": self.response,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HistoryEntry':
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            query=data["query"],
            response=data["response"],
            metadata=data.get("metadata", {})
        )

class Chatbot:
    def __init__(
        self,
        llm: LLM,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        max_history: int = 10,
        entities_file: str = "chatbot_entities.json",
        history_file: str = "chatbot_history.json",
        verbose: bool = False
    ):
        """
        Initialize Chatbot.
        
        Parameters:
        - llm: LLM instance for generating responses
        - system_prompt: System prompt to guide bot behavior
        - context: Additional context for conversations
        - max_history: Maximum number of messages to keep in history
        - entities_file: File to store entities
        - history_file: File to store chat history
        - verbose: Whether to print detailed information
        """
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.context = context
        self.max_history = max_history
        self.entities_file = entities_file
        self.history_file = history_file
        self.verbose = verbose
        
        # Load existing data
        self.entities = self._load_entities()
        self.history = self._load_history()
        
    def _load_entities(self) -> Dict[str, Entity]:
        """Load entities from file."""
        try:
            with open(self.entities_file, 'r') as f:
                data = json.load(f)
                return {
                    name: Entity.from_dict(entity_data)
                    for name, entity_data in data.items()
                }
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
            
    def _save_entities(self):
        """Save entities to file."""
        with open(self.entities_file, 'w') as f:
            json.dump(
                {
                    name: entity.to_dict()
                    for name, entity in self.entities.items()
                },
                f,
                indent=2
            )
            
    def _load_history(self) -> List[HistoryEntry]:
        """Load chat history from file."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return [HistoryEntry.from_dict(entry) for entry in data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []
            
    def _save_history(self):
        """Save chat history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(
                [entry.to_dict() for entry in self.history],
                f,
                indent=2
            )
            
    def add_entity(self, name: str, attributes: Dict) -> Entity:
        """Add a new entity."""
        entity = Entity(name=name, attributes=attributes)
        self.entities[name] = entity
        self._save_entities()
        return entity
        
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        return self.entities.get(name)
        
    def _create_prompt(self, query: str) -> str:
        """Create complete prompt with context and history."""
        prompt_parts = []
        
        # Add system prompt
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}\n")
        
        # Add context
        if self.context:
            prompt_parts.append(f"Context: {self.context}\n")
        
        # Add relevant history
        if self.history:
            prompt_parts.append("Previous conversation:")
            for entry in self.history[-self.max_history:]:
                prompt_parts.extend([
                    f"User: {entry.query}",
                    f"Assistant: {entry.response}"
                ])
            prompt_parts.append("")
        
        # Add current query
        prompt_parts.append(f"User: {query}")
        
        return "\n".join(prompt_parts)
        
    def chat(self, query: str) -> str:
        """
        Process user input and generate response.
        
        Parameters:
        - query: User's message
        
        Returns:
        - str: Assistant's response
        """
        if self.verbose:
            print(f"\nProcessing query: {query}")
        
        # Create complete prompt
        prompt = self._create_prompt(query)
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Save to history
        entry = HistoryEntry(
            timestamp=datetime.now(),
            query=query,
            response=response
        )
        self.history.append(entry)
        self._save_history()
        
        if self.verbose:
            print(f"Generated response: {response}")
        
        return response
        
    def get_history(self) -> List[HistoryEntry]:
        """Get chat history."""
        return self.history
        
    def clear_history(self):
        """Clear chat history."""
        self.history = []
        self._save_history()

