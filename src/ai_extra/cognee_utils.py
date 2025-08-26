"""Utility module for configuring Cognee with LLM information from the factory."""

import os

import cognee

from src.ai_core.llm_factory import get_llm_info


def set_cognee_llm_config(llm_id: str | None = None) -> None:
    """Generate Cognee LLM configuration from LLM factory information."""
    if llm_id is None:
        # Use default from global config
        from src.utils.config_mngr import global_config

        llm_id = global_config().get_str("llm.default_model")

    # Get LLM info from factory
    llm_info = get_llm_info(llm_id)
    api_key = os.environ.get(llm_info.api_key)
    if not api_key and llm_info.api_key:
        raise ValueError(f"API key {llm_info.api_key} not found in environment variables")

    # Build the configuration
    config = {
        "llm_provider": llm_info.provider,
        "llm_model": llm_info.model,
    }

    if api_key:
        config["llm_api_key"] = api_key

    cognee.config.set_llm_config(config)

def set_embeddings_config(moel_id: str | None): 
    