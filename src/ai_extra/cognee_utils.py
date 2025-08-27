"""Utility module for configuring Cognee with LLM information from the factory."""

# Disable LiteLLM async-logging early to prevent startup
import asyncio
import os

import cognee
from devtools import debug  # noqa: F401
from dotenv import load_dotenv
from loguru import logger

from src.ai_core.embeddings_factory import EmbeddingsFactory
from src.ai_core.providers import get_provider_api_key
from src.utils.config_mngr import global_config

load_dotenv()


def set_cognee_config(llm_id: str | None = None, embeddings_id: str | None = "ada_002_openai") -> None:
    conf = global_config().merge_with("config/components/cognee.yaml")
    cognee_directory_path = conf.get_dir_path("cognee.default.cognee_system_dir")
    cognee.config.system_root_directory(str(cognee_directory_path.resolve()))

    """Generate Cognee LLM configuration from LLM factory information."""

    from src.ai_core.llm_factory import LlmFactory

    llm_factory = LlmFactory(llm_id=llm_id)
    llm_info = llm_factory.info
    # lc_llm = llm_factory.get()

    config = {}

    api_key = get_provider_api_key(llm_info.provider)
    if api_key:
        config["llm_api_key"] = api_key.get_secret_value()

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

    # set embeddings/  Prpramatic approach does not work (yet?)
    embeddings_factory = EmbeddingsFactory(embeddings_id=embeddings_id)
    embeddings_info = embeddings_factory.info
    if embeddings_info.provider != "openai":
        logger.warning("Emebeddings other than OPenAi my not work")
    os.environ["EMBEDDING_PROVIDER"] = embeddings_info.provider
    os.environ["EMBEDDING_MODEL"] = f"{embeddings_info.provider}/{embeddings_info.model}"
    os.environ["EMBEDDING_DIMENSIONS"] = str(embeddings_factory.get_dimension())
    if api_key := get_provider_api_key(embeddings_info.provider):
        os.environ["EMBEDDING_API_KEY"] = api_key.get_secret_value()

    # define output parser
    os.environ["STRUCTURED_OUTPUT_FRAMEWORK"] = "BAML"

    cognee.config.set_llm_config(config)
    os.environ["LITELLM_DISABLE_LOGGING"] = "True"  # seels useless to avoid error


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


async def test_config():
    await cognee.add("AI powers Cognee's intelligence.")
    await cognee.cognify()

    result = await cognee.search("What powers Cognee?")
    print(f"answser: {result[0]}")


if __name__ == "__main__":
    from cognee.shared.logging_utils import ERROR, get_logger

    # Configure LLM

    LLM_ID = "gpt_4omini_openai"
    EMBEDDINGS_ID = "ada_002_openai"

    LLM_ID = None
    #    EMBEDDINGS_ID = None

    set_cognee_config(llm_id=LLM_ID, embeddings_id=EMBEDDINGS_ID)
    logger = get_logger(level=ERROR)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_config())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
