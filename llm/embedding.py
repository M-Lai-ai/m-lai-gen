# llm/embedding.py

from typing import List, Dict, Union, Optional
import requests
import json
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Embedding:
    """A class to handle embeddings from different providers."""
    
    SUPPORTED_PROVIDERS = {
        "cohere": {
            "url": "https://api.cohere.com/v2/embed",
            "models": [
                "embed-english-v3.0",
                "embed-multilingual-v3.0",
                "embed-english-light-v3.0",
                "embed-multilingual-light-v3.0",
                "embed-english-v2.0",
                "embed-english-light-v2.0",
                "embed-multilingual-v2.0"
            ]
        },
        "openai": {
            "url": "https://api.openai.com/v1/embeddings",
            "models": [
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]
        },
        "voyage": {
            "url": "https://api.voyageai.com/v1/embeddings",
            "models": [
                "voyage-3",
                "voyage-3-lite",
                "voyage-2",
                "voyage-finance-2",
                "voyage-law-2"
            ]
        },
        "mistral": {
            "url": "https://api.mistral.ai/v1/embeddings",
            "models": [
                "mistral-embed"
            ]
        }
    }

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Embedding class.
        
        Parameters:
        - provider (str): The embedding provider ("cohere", "openai", "voyage", "mistral")
        - model (str): The model to use for embeddings
        - api_key (str, optional): API key (if not set in environment)
        - **kwargs: Additional provider-specific parameters
        """
        self.provider = provider.lower()
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
            
        self.model = model
        if model not in self.SUPPORTED_PROVIDERS[self.provider]["models"]:
            raise ValueError(f"Unsupported model for {provider}: {model}")
            
        # Get API key from args or environment
        self.api_key = api_key or os.getenv(f"{self.provider.upper()}_API_KEY")
        if not self.api_key:
            raise ValueError(f"No API key provided for {provider}")
            
        self.url = self.SUPPORTED_PROVIDERS[self.provider]["url"]
        self.options = kwargs

    def _create_headers(self) -> Dict:
        """Create headers for API request."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.provider == "cohere":
            headers["Authorization"] = f"Bearer {self.api_key}"
            if self.options.get("client_name"):
                headers["X-Client-Name"] = self.options["client_name"]
                
        elif self.provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        elif self.provider == "voyage":
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        elif self.provider == "mistral":
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers

    def _create_payload(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict:
        """Create payload for API request."""
        if isinstance(texts, str):
            texts = [texts]
            
        payload = {"model": self.model}
        
        if self.provider == "cohere":
            payload.update({
                "texts": texts,
                "input_type": kwargs.get("input_type", "search_document"),
                "truncate": kwargs.get("truncate", "END"),
                "embedding_types": kwargs.get("embedding_types", ["float"])
            })
            
        elif self.provider == "openai":
            payload.update({
                "input": texts,
                "encoding_format": kwargs.get("encoding_format", "float"),
                "dimensions": kwargs.get("dimensions"),
                "user": kwargs.get("user")
            })
            
        elif self.provider == "voyage":
            payload.update({
                "input": texts,
                "input_type": kwargs.get("input_type"),
                "truncation": kwargs.get("truncation", True),
                "encoding_format": kwargs.get("encoding_format")
            })
            
        elif self.provider == "mistral":
            payload.update({
                "input": texts,
                "encoding_format": kwargs.get("encoding_format", "float")
            })
            
        return {k: v for k, v in payload.items() if v is not None}

    def _parse_response(self, response: Dict) -> List[List[float]]:
        """Parse API response to extract embeddings."""
        if self.provider == "cohere":
            return response["embeddings"]["float"]
            
        elif self.provider == "openai":
            return [item["embedding"] for item in response["data"]]
            
        elif self.provider == "voyage":
            return [item["embedding"] for item in response["data"]]
            
        elif self.provider == "mistral":
            return [item["embedding"] for item in response["data"]]
            
        raise ValueError(f"Unsupported provider: {self.provider}")

    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for the input texts.
        
        Parameters:
        - texts: Single text or list of texts to embed
        - **kwargs: Additional provider-specific parameters
        
        Returns:
        - List of embedding vectors
        """
        headers = self._create_headers()
        payload = self._create_payload(texts, **kwargs)
        
        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return self._parse_response(response.json())
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling {self.provider} API: {str(e)}")
            
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Error parsing {self.provider} response: {str(e)}")
