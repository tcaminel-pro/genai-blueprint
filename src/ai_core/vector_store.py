"""Vector Store Management and Factory System.

This module provides a comprehensive interface for creating, managing, and
interacting with vector stores across multiple storage backends. It supports
advanced document indexing, retrieval, and vector database operations.

Key Features:
- Multi-backend vector store support (Chroma, In-Memory, Sklearn, PgVector)
- Flexible document indexing and deduplication
- Configurable retrieval strategies
- Seamless integration with embedding models
- Advanced search and filtering capabilities
- Generic configuration system with YAML override support
- Hybrid search support for PostgreSQL (vector + full-text search)

Supported Backends:
- Chroma (persistent and in-memory)
- InMemoryVectorStore
- SKLearnVectorStore
- PgVector (with hybrid search support)

Design Patterns:
- Factory Method for vector store creation
- Configurable retrieval strategies
- Generic configuration via dict parameter
- Singleton-like access to vector stores

Example:
    >>> # Create a vector store factory with generic configuration
    >>> factory = VectorStoreFactory(
    ...     id="Chroma",
            table_suffix = "documents",
    ...     embeddings_factory=EmbeddingsFactory(),
    ...     config={"chroma_path": "/custom/path"}
    ... )
    >>>
    >>> # PgVector example with configuration overrides
    >>> pg_factory = VectorStoreFactory(
    ...     id="PgVector",
    ...     embeddings_factory=EmbeddingsFactory(),
    ...     config={"postgres_url": "//user:pass@localhost:5432/db", "postgres_schema": "vector_store"}
    ... )
    >>>
    >>> # PgVector with hybrid search (vector + full-text)
    >>> hybrid_factory = VectorStoreFactory(
    ...     id="PgVector",
    ...     embeddings_factory=EmbeddingsFactory(),
    ...     hybrid_search=True,
    ...     hybrid_search_config={
    ...         "tsv_lang": "pg_catalog.english",
    ...         "fusion_function_parameters": {"primary_results_weight": 0.5, "secondary_results_weight": 0.5},
    ...     }
    ... )
    >>>
    >>> # Add documents to the store
    >>> factory.add_documents([
    ...     Document(page_content="First document"),
    ...     Document(page_content="Second document")
    ... ])
    >>>
    >>> # Perform similarity search
    >>> results = factory.vector_store.similarity_search("query")
    >>>
    >>> # Perform hybrid search (PostgreSQL only)
    >>> from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
    >>> results = hybrid_factory.vector_store.similarity_search(
    ...     "search query",
    ...     hybrid_search_config=HybridSearchConfig(
    ...         tsv_column="content_tsv",
    ...         fusion_function_parameters={"primary_results_weight": 0.5, "secondary_results_weight": 0.5}
    ...     )
    ... )
"""

import os
from collections.abc import Iterable
from typing import Annotated, Any, Literal, get_args

from devtools import debug
from langchain.embeddings.base import Embeddings
from langchain.indexes import IndexingResult, SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PostgresDsn, computed_field, field_validator

from src.ai_core.embeddings import EmbeddingsFactory
from src.utils.config_mngr import global_config, global_config_reload

# List of known Vector Stores (created as Literal so can be checked by MyPy)
VECTOR_STORE_ENGINE = Literal["Chroma", "Chroma_in_memory", "InMemory", "Sklearn", "PgVector"]


class VectorStoreFactory(BaseModel):
    """Factory for creating and managing vector stores with advanced configuration.

    Provides a flexible and powerful interface for creating vector stores with
    support for multiple backends, document indexing, and advanced retrieval
    strategies.

    Attributes:
        id: Identifier for the vector store backend
        embeddings_factory: Factory for creating embedding models
        table_name_prefix: Prefix for generated table/collection names
        config: Dictionary of vector store specific configuration that overrides YAML values
        index_document: Flag to enable document deduplication and indexing
        collection_metadata: Optional metadata for the collection
        cache_embeddings: Flag to enable embedding caching
        hybrid_search: Flag to enable hybrid search (PostgreSQL only)
        hybrid_search_config: Configuration for hybrid search parameters

    Example:
        >>> factory = VectorStoreFactory(
        ...     id="Chroma",
        ...     embeddings_factory=EmbeddingsFactory(),
        ...     config={"chroma_path": "/custom/path"}
        ... )
        >>> factory.add_documents([Document(page_content="example")])
        >>>
        >>> # Hybrid search example
        >>> hybrid_factory = VectorStoreFactory(
        ...     id="PgVector",
        ...     embeddings_factory=EmbeddingsFactory(),
        ...     hybrid_search=True,
        ...     hybrid_search_config={
        ...         "tsv_column": "content_tsv",
        ...         "tsv_lang": "pg_catalog.english"
        ...     }
        ... )
    """

    id: Annotated[VECTOR_STORE_ENGINE | None, Field(validate_default=True)] = None
    embeddings_factory: EmbeddingsFactory
    table_name_prefix: str = "embeddings"
    config: dict[str, Any] = {}
    index_document: bool = False
    collection_metadata: dict[str, str] | None = None
    cache_embeddings: bool = False
    _record_manager: SQLRecordManager | None = None
    _conf: dict = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def table_name(self) -> str:
        """Generate a name by combining collection and embeddings ID.

        Returns:
            Unique collection name to prevent conflicts
        """
        assert self.embeddings_factory
        embeddings_id = self.embeddings_factory.short_name()
        return f"{self.table_name_prefix}_{embeddings_id}"

    @computed_field
    def description(self) -> str:
        """Generate a detailed description of the vector store configuration.

        Returns:
            Comprehensive configuration description string
        """
        r = f"{str(self.id)}/{self.table_name}"
        if self.id == "Chroma":
            r += " => 'on disk'"
        if self.index_document and self._record_manager:
            r += f" indexer: {self._record_manager}"
        r += f" cache_embeddings: {self.cache_embeddings}"
        return r

    @staticmethod
    def known_items() -> list[str]:
        """List all supported vector store backends.

        Returns:
            List of supported vector store engine identifiers
        """
        return list(get_args(VECTOR_STORE_ENGINE))

    @field_validator("id", mode="before")
    def check_known(cls, id: str | None) -> str:
        """Validate and normalize the vector store backend identifier.

        Args:
            id: Vector store backend identifier

        Returns:
            Validated vector store backend identifier

        Raises:
            ValueError: If an unknown vector store backend is specified
        """
        if id is None:
            id = global_config().get_str("vector_store.default")
        if id not in VectorStoreFactory.known_items():
            raise ValueError(f"Unknown Vector Store: {id}")
        return id

    def get(self) -> VectorStore:
        """Create and configure a vector store based on the specified backend.

        Returns:
            Configured vector store instance

        Raises:
            ValueError: If an unsupported vector store backend is specified
        """
        embeddings = self.embeddings_factory.get(cached=self.cache_embeddings)
        vector_store = None
        if self.id in ["Chroma", "Chroma_in_memory"]:
            vector_store = self._create_chroma_vector_store(embeddings)
        elif self.id == "InMemory":
            from langchain_core.vectorstores import InMemoryVectorStore

            vector_store = InMemoryVectorStore(
                embedding=embeddings,
            )
        elif self.id == "Sklearn":
            from langchain_community.vectorstores import SKLearnVectorStore

            vector_store = SKLearnVectorStore(
                embedding=embeddings,
            )
        elif self.id == "PgVector":
            vector_store = self._create_pg_vector_store()
        else:
            raise ValueError(f"Unknown vector store: {self.id}")

        logger.info(f"get vector store  : {self.description}")
        if self.index_document:
            # NOT TESTED
            db_url = global_config().get_str("vector_store.record_manager")
            logger.info(f"vector store record manager : {db_url}")
            namespace = f"{self.id}/{self.table_name}"
            self._record_manager = SQLRecordManager(
                namespace,
                db_url=db_url,
            )
            self._record_manager.create_schema()
        assert vector_store
        return vector_store

    def add_documents(self, docs: Iterable[Document]) -> IndexingResult | list[str]:
        """Add documents to the vector store with optional deduplication.

        Args:
            docs: Iterable of documents to add to the store

        Returns:
            Indexing result or list of document IDs

        Notes:
            Supports two modes of document addition:
            1. Direct addition without indexing
            2. Indexed addition with deduplication
        """
        if not self.index_document:
            return self.get().add_documents(list(docs))
        else:
            vector_store = self.get()
            assert self._record_manager

            info = index(
                docs,
                self._record_manager,
                vector_store,
                cleanup="incremental",
                source_id_key="source",
            )
            return info

    def document_count(self):
        """Count the number of documents in the vector store.

        Returns:
            Number of documents in the store

        Raises:
            NotImplementedError: For unsupported vector store backends
        """
        if self.id in ["Chroma", "Chroma_in_memory"]:
            return self.get()._collection.count()  # type: ignore
        else:
            raise NotImplementedError(f"Don't know how to get collection count for {self.get()}")

    def _create_chroma_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Create and configure a Chroma vector store."""
        from langchain_chroma import Chroma

        if self.id == "Chroma":  # Persistent storage
            store_path = self.config.get("chroma_path") or global_config().get_dir_path(
                "vector_store.chroma_path", create_if_not_exists=True
            )
            persist_directory = str(store_path)
        else:  # Chroma_in_memory
            persist_directory = None
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name=self.table_name,
            collection_metadata=self.collection_metadata,
        )

    def _create_pg_vector_store(self) -> VectorStore:
        """Create and configure a PgVector store."""
        from langchain_postgres import Column, PGEngine, PGVectorStore
        from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
        from sqlalchemy.exc import ProgrammingError

        # Use config dict to override YAML values
        postgres_url = self.config.get("postgres_url") or global_config().get_str("vector_store.postgres_url")
        schema_name = self.config.get("postgres_schema") or "public"
        metadata_columns = self.config.get("metadata_columns") or []

        l, _, r = postgres_url.partition("//")
        if not l.startswith("postgres"):
            raise ValueError("postgres_url should start with postgresql://  or postgresql+asyncpg://")
        connection_string = f"postgresql+asyncpg://{r}"
        try:
            PostgresDsn(connection_string)
        except Exception as e:
            raise ValueError(f"Incorrect Postgres URL : {connection_string}") from e

        table_name = self.table_name
        pg_engine = PGEngine.from_connection_string(url=connection_string)

        # Prepare hybrid search configuration if enabled
        hybrid_search_config = None
        hybrid_search = self.config.get("hybrid_search", False)
        if hybrid_search:
            hybrid_config = self.config.get("hybrid_search_config", {})
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
                vector_size=self.embeddings_factory.get_dimension(),
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
            embedding_service=self.embeddings_factory.get(),
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

        self._conf["pg_engine"] = pg_engine
        self._conf["table_name"] = table_name
        self._conf["schema_name"] = schema_name

        return vector_store

    def clean(self):
        if self.id == "PgVector":
            from langchain_postgres import PGEngine

            if pg_engine := self._conf.get("pg_engine"):
                assert isinstance(pg_engine, PGEngine)
                pg_engine.drop_table(table_name=self._conf["table_name"], schema_name=self._conf["schema_name"])
            else:
                raise NotImplementedError(f"Don't'clean' method for {self.get()}")


def search_one(vc: VectorStore, query: str) -> list[Document]:
    """Perform a similarity search to find the single most relevant document.

    Args:
        vc: Vector store to search in
        query: Search query string

    Returns:
        List containing the most similar document
    """
    return vc.similarity_search(query, k=1)


if __name__ == "__main__":
    """Quick test script for hybrid search functionality."""
    import sys

    try:
        from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
    except ImportError:
        print("langchain-postgres not available. Install with: uv add langchain-postgres")
        sys.exit(1)

    from langchain_core.documents import Document

    from src.ai_core.embeddings import EmbeddingsFactory

    # Test configuration
    os.environ["POSTGRES_USER"] = "tcl"
    os.environ["POSTGRES_PASSWORD"] = "tcl"
    global_config_reload()

    postgres_url = global_config().get_str("vector_store.postgres_url")

    print("🧪 Testing hybrid search with PostgreSQL...")

    # Create embeddings factory
    embeddings_factory = EmbeddingsFactory(embeddings_id="embeddings_768_fake")

    # Create vector store with hybrid search enabled
    factory = VectorStoreFactory(
        id="PgVector",
        embeddings_factory=embeddings_factory,
        config={
            "postgres_url": postgres_url,
            "metadata_columns": [{"name": "test_matadata", "data_type": "TEXT"}],
            "hybrid_search": True,
            "hybrid_search_config": {
                "tsv_column": "content_tsv",
                "tsv_lang": "pg_catalog.english",
                "fusion_function_parameters": {
                    "primary_results_weight": 0.5,
                    "secondary_results_weight": 0.5,
                },
            },
        },
    )

    try:
        # Add test documents
        test_docs = [
            Document(page_content="PostgreSQL is a powerful open-source database system"),
            Document(page_content="Hybrid search combines vector similarity and full-text search"),
            Document(page_content="GIN indexes are used for full-text search in PostgreSQL"),
            Document(page_content="LangChain provides excellent vector store integration"),
        ]

        print("📄 Adding test documents...")
        factory.add_documents(test_docs)

        # Perform hybrid search
        print("🔍 Performing hybrid search...")
        results = factory.get().similarity_search(
            "database search",
            k=2,
            hybrid_search_config=HybridSearchConfig(
                tsv_column="content_tsv",
                fusion_function_parameters={"primary_results_weight": 0.5, "secondary_results_weight": 0.5},
            ),
        )

        print(f"✅ Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")

    except Exception as e:
        print(f"❌ Error: {e}")
        debug(e)
        raise e
        print("💡 Make sure PostgreSQL is running and POSTGRES_URL is set correctly")

    finally:
        # Clean up
        try:
            pass
            # factory.clean()
            # print("🧹 Cleaned up test table")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up - {e}")
