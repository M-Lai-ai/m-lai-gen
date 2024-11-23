# llm/chatbot.py

from typing import List, Dict, Optional
import json
from datetime import datetime
from .llm import LLM

class Entity:
    def __init__(self, name: str, attributes: Dict):
        """
        Initialize an entity with a name and attributes.
        
        Parameters:
        - name (str): Name of the entity
        - attributes (dict): Dictionary of entity attributes
        """
        self.name = name
        self.attributes = attributes
        self.created_at = datetime.now()
        
    def to_dict(self) -> Dict:
        """Convert entity to dictionary format."""
        return {
            "name": self.name,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        """Create entity from dictionary format."""
        entity = cls(data["name"], data["attributes"])
        entity.created_at = datetime.fromisoformat(data["created_at"])
        return entity

class Message:
    def __init__(self, role: str, content: str):
        """
        Initialize a message.
        
        Parameters:
        - role (str): Role of the message sender ("user" or "assistant")
        - content (str): Content of the message
        """
        self.role = role
        self.content = content
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create message from dictionary format."""
        message = cls(data["role"], data["content"])
        message.timestamp = datetime.fromisoformat(data["timestamp"])
        return message

class Chatbot:
    def __init__(
        self,
        llm: LLM,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        max_history: int = 10,
        entities_file: str = "entities.json",
        history_file: str = "chat_history.json"
    ):
        """
        Initialize the chatbot.
        
        Parameters:
        - llm (LLM): Instance of LLM class for generating responses
        - system_prompt (str, optional): System prompt to guide the bot's behavior
        - context (str, optional): Additional context for the conversation
        - max_history (int): Maximum number of messages to keep in history
        - entities_file (str): File to store entities
        - history_file (str): File to store chat history
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
        self.load_entities()
        self.load_history()
        
    def add_entity(self, name: str, attributes: Dict) -> Entity:
        """Add a new entity."""
        entity = Entity(name, attributes)
        self.entities[name] = entity
        self.save_entities()
        return entity
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        return self.entities.get(name)
    
    def save_entities(self):
        """Save entities to file."""
        with open(self.entities_file, 'w') as f:
            json.dump({
                name: entity.to_dict() 
                for name, entity in self.entities.items()
            }, f)
    
    def load_entities(self):
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
    
    def save_history(self):
        """Save chat history to file."""
        with open(self.history_file, 'w') as f:
            json.dump([msg.to_dict() for msg in self.history], f)
    
    def load_history(self):
        """Load chat history from file."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.history = [Message.from_dict(msg_data) for msg_data in data]
        except FileNotFoundError:
            self.history = []
    
    def _build_prompt(self, user_input: str) -> str:
        """Build the complete prompt including context and history."""
        messages = []
        
        # Add system prompt
        if self.system_prompt:
            messages.append(f"System: {self.system_prompt}\n")
        
        # Add context
        if self.context:
            messages.append(f"Context: {self.context}\n")
        
        # Add relevant history
        for msg in self.history[-self.max_history:]:
            messages.append(f"{msg.role.capitalize()}: {msg.content}")
        
        # Add current user input
        messages.append(f"User: {user_input}")
        
        return "\n".join(messages)
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and generate response.
        
        Parameters:
        - user_input (str): User's message
        
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
        self.save_history()
        
        return response
    
    def clear_history(self):
        """Clear chat history."""
        self.history = []
        self.save_history()

# Example usage:
if __name__ == "__main__":
    # Initialize LLM
    llm = LLM(
        provider="mistral",
        model="mistral-medium",
        temperature=0.7
    )
    
    # Initialize Chatbot
    chatbot = Chatbot(
        llm=llm,
        system_prompt="You are a helpful assistant specialized in Python programming.",
        context="This is a technical discussion about Python programming.",
        max_history=5
    )
    
    # Add some entities
    chatbot.add_entity("python", {
        "type": "programming_language",
        "version": "3.9",
        "paradigm": ["object-oriented", "functional"]
    })
    
    # Interactive chat loop
    print("Chat started (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.chat(user_input)
        print(f"Assistant: {response}")
