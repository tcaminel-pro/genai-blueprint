"""Provider configuration and API key management.

This module contains shared provider configurations and utilities for
managing API keys across different AI service providers.

"""

import os

from pydantic import SecretStr

# List of implemented LLM providers, with the Python class to be loaded, and the name of the API key environment variable
PROVIDER_INFO = {
    "fake": ("langchain_core", ""),
    "openai": ("langchain_openai", "OPENAI_API_KEY"),
    "deepinfra": ("langchain_community.chat_models.deepinfra", "DEEPINFRA_API_TOKEN"),
    "groq": ("langchain_groq", "GROQ_API_KEY"),
    "ollama": ("langchain_ollama", ""),
    "edenai": ("langchain_community.chat_models.edenai", "EDENAI_API_KEY"),
    "azure": ("langchain_openai", "AZURE_OPENAI_API_KEY"),
    "together": ("langchain_together", "TOGETHER_API_KEY"),
    "deepseek": ("langchain_deepseek", "DEEPSEEK_API_KEY"),
    "openrouter": ("langchain_openai", "OPENROUTER_API_KEY"),
    "huggingface": ("langchain_huggingface", "HUGGINGFACEHUB_API_TOKEN"),
    "mistral": ("langchain_mistralai", "MISTRAL_API_KEY"),
    # NOT TESTED:
    "bedrock": ("langchain_aws", "AWS_ACCESS_KEY_ID"),
    "anthropic": ("langchain_anthropic", "ANTHROPIC_API_KEY"),
    "google": ("langchain_google_vertexai", "GOOGLE_API_KEY"),
    "vllm": ("langchain_community.llms", ""),
}


def get_provider_api_key(provider: str) -> SecretStr | None:
    """Get the API key for a given AI provider.

    Args:
        provider: Name of the AI provider (e.g. "openai", "google")

    Returns:
        The API key as SecretStr if found, None otherwise
    """
    if provider not in PROVIDER_INFO:
        raise ValueError(f"Unknown provider: {provider}. Valid providers are: {list(PROVIDER_INFO.keys())}")
    env_var = PROVIDER_INFO[provider][1]
    return clean_api_key(env_var)


def clean_api_key(env_var: str | None) -> SecretStr | None:
    """Get and clean API key from environment variable.

    Args:
        env_var: Environment variable name to get the API key from

    Returns:
        Cleaned API key value wrapped in SecretStr or None if not found
    """
    if env_var is None or env_var not in os.environ:
        return None
    # Strip any surrounding quotes and whitespace
    r = os.environ[env_var].strip("\"' \t\n\r")
    return SecretStr(r)
