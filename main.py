# main.py

from llm.llm import LLM
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

def main():
    input_text = "Bonjour, comment vas-tu ?"

    # Liste des fournisseurs avec leurs paramètres par défaut
    providers = [
        {
            "provider": "openai",
            "model": "gpt-4",
            "params": {
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        },
        {
            "provider": "anthropic",
            "model": "claude-v1",
            "params": {
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 0.9
            }
        },
        {
            "provider": "mistral",
            "model": "mistral-model-1",
            "params": {
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 0.9,
                "stream": False,
                "tool_choice": "auto",
                "safe_prompt": False
            }
        },
        {
            "provider": "cohere",
            "model": "command-r-plus-08-2024",
            "params": {
                "temperature": 0.5,
                "max_tokens": 100,
                "top_p": 0.9,
                "safe_prompt": True
            }
        }
    ]

    for provider_info in providers:
        provider = provider_info["provider"]
        model = provider_info["model"]
        params = provider_info["params"]

        try:
            # Instanciation du fournisseur avec les paramètres par défaut
            llm = LLM(provider=provider, model=model, **params)
            # Génération de la réponse en fournissant uniquement le texte d'entrée
            response = llm.generate(input_text)
            print(f"**{provider.capitalize()} Response:**\n{response}\n")
        except Exception as e:
            print(f"Erreur avec {provider.capitalize()} : {e}\n")

if __name__ == "__main__":
    main()
