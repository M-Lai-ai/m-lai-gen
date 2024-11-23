Certainly! Below is a comprehensive `README.md` for your GitHub repository [M-Lai-ai/m-lai-gen](https://github.com/M-Lai-ai/m-lai-gen.git) in English. This README assumes that your project provides a unified interface to interact with multiple Language Model Providers (OpenAI, Anthropic, Mistral, and Cohere) for generating text responses based on user input.

---

# m-lai-gen

![License](https://img.shields.io/github/license/M-Lai-ai/m-lai-gen)
![Python](https://img.shields.io/github/languages/top/M-Lai-ai/m-lai-gen)
![GitHub issues](https://img.shields.io/github/issues/M-Lai-ai/m-lai-gen)

## Table of Contents

- [m-lai-gen](#m-lai-gen)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Example Code](#example-code)
  - [Example Output](#example-output)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Overview

**m-lai-gen** is a Python library that provides a unified interface to interact with multiple Language Model Providers, including OpenAI, Anthropic, Mistral, and Cohere. This library simplifies the process of generating text responses by allowing users to input text and receive responses without needing to manage different API specifications and configurations for each provider.

## Features

- **Unified Interface**: Interact with multiple LLM providers using a single, consistent API.
- **Default Configurations**: Pre-configured parameters for each provider to streamline usage.
- **Easy Integration**: Minimal setup required to start generating text responses.
- **Extensible**: Easily add support for additional providers in the future.
- **Error Handling**: Comprehensive error management to handle API request issues gracefully.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/M-Lai-ai/m-lai-gen.git
    cd m-lai-gen
    ```

2. **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Set Up Environment Variables**

    Create a `.env` file in the root directory of your project by copying the provided `.env.example` file:

    ```bash
    cp .env.example .env
    ```

2. **Add Your API Keys**

    Open the `.env` file and add your API keys for each provider:

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    MISTRAL_API_KEY=your_mistral_api_key_here
    CO_API_KEY=your_cohere_api_key_here
    ```

    Replace `your_openai_api_key_here`, `your_anthropic_api_key_here`, etc., with your actual API keys.

## Usage

The `LLM` class allows you to generate responses from different providers by simply specifying the provider and input text. Below is an example of how to use the `LLM` class to generate responses.

### Example Code

Create a `main.py` file and add the following code:

```python
# main.py

from llm.llm import LLM
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    input_text = "Hello, how are you?"

    # List of providers with their default parameters
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
            # Instantiate the provider with default parameters
            llm = LLM(provider=provider, model=model, **params)
            # Generate response by providing only the input text
            response = llm.generate(input_text)
            print(f"**{provider.capitalize()} Response:**\n{response}\n")
        except Exception as e:
            print(f"Error with {provider.capitalize()}: {e}\n")

if __name__ == "__main__":
    main()
```

### Running the Example

Ensure your `.env` file is correctly configured with all necessary API keys, then run the script:

```bash
python main.py
```

## Example Output

If all API keys are correctly configured and the models are accessible, running `main.py` should produce output similar to the following:

```
**Openai Response:**
Hello! I'm doing well, thank you. How can I assist you today?

**Anthropic Response:**
Hello! I'm fine, thank you. How can I help you today?

**Mistral Response:**
Hello! I'm good, thank you. How can I assist you today?

**Cohere Response:**
Hello! I'm well, thank you. How may I assist you today?
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

2. **Create a New Branch**

    ```bash
    git checkout -b feature/your-feature-name
    ```

3. **Make Your Changes**

4. **Commit Your Changes**

    ```bash
    git commit -m "Add your message here"
    ```

5. **Push to Your Branch**

    ```bash
    git push origin feature/your-feature-name
    ```

6. **Open a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to reach out:

- **Email**: your.email@example.com
- **GitHub**: [M-Lai-ai](https://github.com/M-Lai-ai)

---

### Additional Files

#### `.env.example`

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
CO_API_KEY=your_cohere_api_key_here
```

#### `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class

# Environment Variables
.env

# IDEs and Editors
.vscode/
.idea/
*.sublime-project
*.sublime-workspace

# Others
*.log
```

#### `requirements.txt`

```txt
requests
python-dotenv
```

#### `LICENSE` (MIT License Example)

```text
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
...
```


