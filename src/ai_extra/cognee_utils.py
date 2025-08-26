"""Utility module for configuring Cognee with LLM information from the factory."""

import os

import cognee

from src.ai_core.llm_factory import get_llm_info
from src.ai_core.providers import get_provider_api_env_var, get_provider_api_key


def set_llm_config(llm_id: str | None = None) -> None:
    """Generate Cognee LLM configuration from LLM factory information."""
    if llm_id is None:
        # Use default from global config
        from src.utils.config_mngr import global_config

        llm_id = global_config().get_str("llm.default_model")

    # Get LLM info from factory
    llm_info = get_llm_info(llm_id)

    api_key_env_var = get_provider_api_env_var(llm_info.provider)
    if api_key_env_var and api_key_env_var not in os.environ:
        raise ValueError(f"API key '{api_key_env_var}' not found in environment variables")
    config = {
        "llm_provider": llm_info.provider,
        "llm_model": llm_info.model,
    }
    api_key = get_provider_api_key(llm_info.provider)

    if api_key:
        config["llm_api_key"] = api_key.get_secret_value()

    cognee.config.set_llm_config(config)


def set_embeddings_config(model_id: str | None = None) -> None:
    """Generate Cognee embedding configuration from embeddings factory information."""

    # taken from https://docs.cognee.ai/how-to-guides/cognee-sdk/embedding-providers-overview
    # the programatic approach does not work

    from src.ai_core.embeddings_factory import EmbeddingsFactory
    from src.utils.config_mngr import global_config

    if model_id is None:
        # Use default from global config
        model_id = global_config().get_str("embeddings.default_model")
    # Get embeddings info from factory
    embeddings_factory = EmbeddingsFactory(embeddings_id=model_id)
    embeddings_info = embeddings_factory.info
    # Set environment variables for cognee
    print(embeddings_info)
    os.environ["EMBEDDING_PROVIDER"] = embeddings_info.provider
    os.environ["EMBEDDING_MODEL"] = f"{embeddings_info.provider}/{embeddings_info.model}"
    os.environ["EMBEDDING_DIMENSIONS"] = str(embeddings_factory.get_dimension())
    if api_key := get_provider_api_key(embeddings_info.provider):
        os.environ["EMBEDDING_API_KEY"] = api_key.get_secret_value()
