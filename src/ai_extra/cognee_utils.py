"""Utility module for configuring Cognee with LLM information from the factory."""

import asyncio
import os
from pathlib import Path

import cognee
from dotenv import load_dotenv

from src.ai_core.providers import get_provider_api_key

load_dotenv()


def set_llm_config(llm_id: str | None = None) -> None:
    """Generate Cognee LLM configuration from LLM factory information."""

    from src.ai_core.llm_factory import LlmFactory

    llm_factory = LlmFactory(llm_id=llm_id)
    llm_info = llm_factory.info
    lc_llm = llm_factory.get()

    config = {}

    api_key = get_provider_api_key(llm_info.provider)
    if api_key:
        config["llm_api_key"] = api_key.get_secret_value()

    from devtools import debug

    debug(config["llm_api_key"])

    if llm_info.provider == "openai":
        config |= {
            "llm_model": llm_info.model,
        }
    elif llm_info.provider == "openrouter":
        config |= {
            "llm_provider": "custom",
            "llm_model": llm_factory.get_litellm_model_name(),
            "llm_endpoint": "https://openrouter.ai/api/v1",
        }

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

    os.environ["EMBEDDING_PROVIDER"] = embeddings_info.provider
    os.environ["EMBEDDING_MODEL"] = f"{embeddings_info.provider}/{embeddings_info.model}"
    os.environ["EMBEDDING_DIMENSIONS"] = str(embeddings_factory.get_dimension())
    if api_key := get_provider_api_key(embeddings_info.provider):
        os.environ["EMBEDDING_API_KEY"] = api_key.get_secret_value()


async def test_config():
    # Test with simple data
    await cognee.add("AI powers Cognee's intelligence.")
    await cognee.cognify()

    result = await cognee.search("What powers Cognee?")
    print(result[0])


def print_config():
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.relational import get_relational_config
    from cognee.infrastructure.databases.vector import get_vectordb_config
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.llm.config import get_llm_config
    from rich import print

    print(get_vectordb_config().to_dict())
    print(get_graph_config().to_dict())
    print(get_relational_config().to_dict())
    print(get_llm_config().to_dict())
    print(get_embedding_config().to_dict())


if __name__ == "__main__":
    cognee_directory_path = str(Path(".cognee_system").resolve())
    cognee.config.system_root_directory(cognee_directory_path)

    # Configure LLM

    LLM_ID = "gpt_4omini_openai"
    EMBEDDINGS_ID = "ada_002_openai"

    LLM_ID = None
    #    EMBEDDINGS_ID = None

    set_llm_config(llm_id=LLM_ID)
    set_embeddings_config(model_id=EMBEDDINGS_ID)

    print_config()

    asyncio.run(test_config())
