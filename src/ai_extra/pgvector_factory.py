"""PgVector store creation and configuration utilities.

This module provides factory functions for creating and configuring
PgVector stores with support for hybrid search and advanced features.
"""

from typing import Any

from devtools import debug
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
from loguru import logger
from pydantic import PostgresDsn
from sqlalchemy.exc import ProgrammingError

from src.utils.config_mngr import global_config


def create_pg_vector_store(
    embeddings_factory: Any,
    table_name: str,
    config: dict[str, Any],
    conf: dict[str, Any],
) -> Any:
    """Create and configure a PgVector store.

    Args:
        embeddings_factory: Factory for creating embedding models
        table_name: Name of the table to create
        config: Dictionary of vector store specific configuration
        conf: Internal configuration dictionary to store engine details

    Returns:
        Configured PGVectorStore instance
    """
    # Use config dict to override YAML values
    postgres_url = config.get("postgres_url") or global_config().get_str("vector_store.postgres_url")
    schema_name = config.get("postgres_schema") or "public"
    metadata_columns = config.get("metadata_columns") or []

    # Validate metadata_columns format
    if metadata_columns:
        if not isinstance(metadata_columns, list):
            raise ValueError("metadata_columns must be a list")

        for i, column in enumerate(metadata_columns):
            if not isinstance(column, dict):
                raise ValueError(f"metadata_columns[{i}] must be a dictionary")
            if "name" not in column:
                raise ValueError(f"metadata_columns[{i}] missing required key: 'name'")
            if "data_type" not in column:
                raise ValueError(f"metadata_columns[{i}] missing required key: 'data_type'")

    l, _, r = postgres_url.partition("//")
    if not l.startswith("postgres"):
        raise ValueError("postgres_url should start with postgresql://  or postgresql+asyncpg://")
    connection_string = f"postgresql+asyncpg://{r}"
    try:
        PostgresDsn(connection_string)
    except Exception as e:
        raise ValueError(f"Incorrect Postgres URL : {connection_string}") from e

    table_name = table_name
    pg_engine = PGEngine.from_connection_string(url=connection_string)

    # Prepare hybrid search configuration if enabled
    hybrid_search_config = None
    hybrid_search = config.get("hybrid_search", False)
    if hybrid_search:
        hybrid_config = config.get("hybrid_search_config", {})
        hybrid_search_config = HybridSearchConfig(
            tsv_column=hybrid_config.get("tsv_column", "content_tsv"),
            tsv_lang=hybrid_config.get("tsv_lang", "pg_catalog.english"),
            fts_query=hybrid_config.get("fts_query", ""),
            fusion_function=hybrid_config.get("fusion_function"),
            fusion_function_parameters=hybrid_config.get("fusion_function_parameters", {}),
            primary_top_k=hybrid_config.get("primary_top_k", 4),
            secondary_top_k=hybrid_config.get("secondary_top_k", 4),
            index_name=hybrid_config.get("index_name", f"{table_name}_tsv_index"),
            index_type=hybrid_config.get("index_type", "GIN"),
        )
        logger.info(f"Hybrid search enabled with config: {hybrid_search_config}")

    try:
        pg_engine.init_vectorstore_table(
            table_name=table_name,
            schema_name=schema_name,
            vector_size=embeddings_factory.get_dimension(),
            overwrite_existing=False,
            hybrid_search_config=hybrid_search_config,
            metadata_columns=[Column(e["name"], e["data_type"]) for e in metadata_columns],
        )
        logger.info(f"pgvector vector table created: {table_name=} {schema_name=}")
        if hybrid_search and hybrid_search_config:
            logger.info(f"Hybrid search configured with TSV column: {hybrid_search_config.tsv_column}")
    except ProgrammingError as e:
        if "already exists" in str(e).lower():
            logger.debug(f"Use existing pgvector table : {table_name}")
        else:
            raise

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name=table_name,
        schema_name=schema_name,
        embedding_service=embeddings_factory.get(),
        metadata_columns=[e["name"] for e in metadata_columns],
        hybrid_search_config=hybrid_search_config,
    )

    # Apply hybrid search index if enabled
    debug(vector_store)
    if hybrid_search and hybrid_search_config:
        try:
            tsv_index_query = f"""CREATE INDEX langchain_tsv_index ON "{schema_name}"."{table_name}"                   
            USING GIN("content_tsv);"""
            #  Always fail : apply_hybrid_search_index not implemented (only async version exists)
            vector_store._engine._run_as_async(vector_store.__vs.apply_hybrid_search_index())
            logger.info(f"Applied hybrid search index on {table_name}")
        except Exception as e:
            logger.warning(f"Failed to apply hybrid search index: {e}")

    conf["pg_engine"] = pg_engine
    conf["table_name"] = table_name
    conf["schema_name"] = schema_name

    return vector_store
