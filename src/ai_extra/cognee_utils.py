"""Utility module for configuring Cognee with LLM information from the factory."""

# Disable LiteLLM async-logging early to prevent startup
import asyncio
import os

import cognee
from cognee.api.v1.search import SearchType
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


def get_search_type_description(type: SearchType) -> str:
    # https://docs.cognee.ai/reference/search-types
    search_descriptions = {
        SearchType.SUMMARIES: "Vector search on TextSummary content for concise, high-signal hits. Returns summary objects with provenance.",
        SearchType.INSIGHTS: "Finds relevant insights across your knowledge graph.",
        SearchType.CHUNKS: "Returns the most similar text chunks to your query via vector search. Output: Chunk objects with metadata.",
        SearchType.RAG_COMPLETION: "Pulls top-k chunks via vector search, stitches a context window, then asks an LLM to answer. Output: An LLM answer grounded in retrieved chunks.",
        SearchType.GRAPH_COMPLETION: "Finds relevant graph triplets using vector hints, resolves them into readable context, and asks an LLM to answer your question grounded in that context. Output: A natural-language answer with references.",
        SearchType.GRAPH_SUMMARY_COMPLETION: "Builds graph context like GRAPH_COMPLETION, then condenses it before answering. Output: A concise answer grounded in graph context.",
        SearchType.CODE: "Interprets your intent, searches code embeddings and related graph nodes, and assembles relevant source. Output: Structured code contexts and related graph information.",
        SearchType.CYPHER: "Executes your Cypher query against the graph database. Output: Raw query results.",
        SearchType.NATURAL_LANGUAGE: "Infers a Cypher query from your question using the graph schema, runs it, returns the results. Output: Executed graph results.",
        SearchType.GRAPH_COMPLETION_COT: "Combines graph traversal with chain of thought to provide answers to complex multi hop questions.",
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION: "Starts with initial graph context, lets the LLM suggest follow-ups, fetches more graph context, repeats. Output: An answer assembled after expanding the relevant subgraph.",
        SearchType.FEELING_LUCKY: "Uses an LLM to pick the most suitable search mode for your query, then runs it. Output: Results from the selected mode.",
        SearchType.FEEDBACK: "Records user feedback on recent answers and links it to the associated graph elements for future tuning. Output: A feedback record tied to recent interactions.",
    }
    result = search_descriptions.get(type)
    return result or f"unknown Search Type: {type}"


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
