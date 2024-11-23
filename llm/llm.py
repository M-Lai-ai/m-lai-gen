# llm/llm.py

import os
import requests
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class LLM:
    def __init__(
        self,
        provider='openai',  # Fournisseur par défaut
        model=None,
        temperature=0.7,
        max_tokens=1500,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        **kwargs
    ):
        """
        Initialise le fournisseur LLM spécifié avec des paramètres par défaut.

        Paramètres :
        - provider (str) : Fournisseur à utiliser ("openai", "anthropic", "mistral", "cohere").
        - model (str) : Modèle à utiliser pour la génération de texte.
        - temperature (float) : Contrôle la randomisation dans la sortie.
        - max_tokens (int) : Nombre maximum de tokens à générer.
        - top_p (float) : Paramètre de nucleus sampling pour contrôler la diversité.
        - frequency_penalty (float) : Pénalise les nouveaux tokens en fonction de la fréquence.
        - presence_penalty (float) : Pénalise les nouveaux tokens en fonction de la présence.
        - **kwargs : Autres paramètres spécifiques aux fournisseurs.
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.options = kwargs

        # Configuration spécifique au fournisseur
        if self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.url = "https://api.openai.com/v1/chat/completions"
            if not self.api_key:
                raise ValueError("Clé API OpenAI manquante dans les variables d'environnement.")

        elif self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            self.url = "https://api.anthropic.com/v1/messages"
            if not self.api_key:
                raise ValueError("Clé API Anthropic manquante dans les variables d'environnement.")

        elif self.provider == "mistral":
            self.api_key = os.getenv("MISTRAL_API_KEY")
            self.url = "https://api.mistral.ai/v1/chat/completions"
            if not self.api_key:
                raise ValueError("Clé API Mistral manquante dans les variables d'environnement.")

        elif self.provider == "cohere":
            self.api_key = os.getenv("CO_API_KEY")
            self.url = "https://api.cohere.com/v2/chat"
            if not self.api_key:
                raise ValueError("Clé API Cohere manquante dans les variables d'environnement.")

        else:
            raise ValueError(f"Fournisseur invalide : {provider}. Choisissez parmi 'openai', 'anthropic', 'mistral' ou 'cohere'.")

    def generate(self, input_text):
        """
        Génère une réponse en utilisant le fournisseur LLM configuré.

        Paramètres :
        - input_text (str) : Texte d'entrée pour générer la réponse.

        Retour :
        - response_text (str) : Réponse générée par le fournisseur.
        """
        headers = {"Content-Type": "application/json"}

        if self.provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": input_text}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty
            }
            payload.update(self.options)

            try:
                response = requests.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                return f"Erreur lors de la requête OpenAI : {e}"
            except KeyError:
                return "Erreur dans la réponse de l'API OpenAI."

        elif self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "messages": [{"role": "user", "content": input_text}]
            }
            payload.update(self.options)

            try:
                response = requests.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("content", "Aucune réponse trouvée.").strip()
            except requests.exceptions.RequestException as e:
                return f"Erreur lors de la requête Anthropic : {e}"
            except KeyError:
                return "Erreur dans la réponse de l'API Anthropic."

        elif self.provider == "mistral":
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": input_text}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": self.options.get("stream", False),
                "tool_choice": self.options.get("tool_choice", "auto"),
                "safe_prompt": self.options.get("safe_prompt", False)
            }
            # Ajouter les paramètres optionnels pour Mistral
            optional_params = {
                "max_tokens": self.max_tokens,
                "min_tokens": self.options.get("min_tokens"),
                "stop": self.options.get("stop"),
                "random_seed": self.options.get("random_seed"),
                "response_format": self.options.get("response_format"),
                "tools": self.options.get("tools")
            }
            payload.update({k: v for k, v in optional_params.items() if v is not None})

            try:
                response = requests.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "Aucune réponse trouvée.").strip()
            except requests.exceptions.RequestException as e:
                return f"Erreur lors de la requête Mistral : {e}"
            except KeyError:
                return "Erreur dans la réponse de l'API Mistral."

        elif self.provider == "cohere":
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["accept"] = "application/json"
            headers["content-type"] = "application/json"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": {"type": "text", "text": input_text}}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "p": self.top_p  # Cohere utilise 'p' au lieu de 'top_p'
            }

            # Ajouter les paramètres optionnels spécifiques à Cohere
            optional_params = {
                "stream": self.options.get("stream"),
                "tools": self.options.get("tools"),
                "response_format": self.options.get("response_format"),
                "safety_mode": "CONTEXTUAL" if self.options.get("safe_prompt") else None,
                "frequency_penalty": self.options.get("frequency_penalty"),
                "presence_penalty": self.options.get("presence_penalty"),
                "stop_sequences": self.options.get("stop")
            }
            payload.update({k: v for k, v in optional_params.items() if v is not None})

            try:
                response = requests.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                assistant_contents = data.get("message", {}).get("content", [])
                assistant_message = " ".join([block.get("text", "") for block in assistant_contents]).strip()
                return assistant_message
            except requests.exceptions.RequestException as e:
                return f"Erreur lors de la requête Cohere : {e}"
            except KeyError:
                return "Erreur dans la réponse de l'API Cohere."
        else:
            return "Fournisseur invalide."
