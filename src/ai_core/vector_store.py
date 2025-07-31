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

Supported Backends:
- Chroma (persistent and in-memory)
- InMemoryVectorStore
- SKLearnVectorStore
- PgVector

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
    >>> # Add documents to the store
    >>> factory.add_documents([
    ...     Document(page_content="First document"),
    ...     Document(page_content="Second document")
    ... ])
    >>>
    >>> # Perform similarity search
    >>> results = factory.vector_store.similarity_search("query")
"""

from collections.abc import Iterable
from functools import cached_property
from typing import Annotated, Literal, get_args

from langchain.embeddings.base import Embeddings
from langchain.indexes import IndexingResult, SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_core.runnables import ConfigurableField
from langchain_core.vectorstores.base import VectorStoreRetriever
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from src.ai_core.embeddings import EmbeddingsFactory
from src.utils.config_mngr import global_config

try:
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig

    HAS_HYBRID_SEARCH = True
except ImportError:
    HAS_HYBRID_SEARCH = False

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

    Example:
        >>> factory = VectorStoreFactory(
        ...     id="Chroma",
        ...     embeddings_factory=EmbeddingsFactory(),
        ...     config={"chroma_path": "/custom/path"}
        ... )
        >>> factory.add_documents([Document(page_content="example")])
    """

    id: Annotated[VECTOR_STORE_ENGINE | None, Field(validate_default=True)] = None
    embeddings_factory: EmbeddingsFactory
    table_name_prefix: str = "embeddings"
    config: dict[str, str] = {}
    index_document: bool = False
    collection_metadata: dict[str, str] | None = None
    cache_embeddings: bool = False
    hybrid_search: bool = False
    hybrid_search_config: dict | None = None
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

    @computed_field
    @cached_property
    def vector_store(self) -> VectorStore:
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

    def as_retriever_configurable(self, top_k: int = 4, filter: dict | None = None) -> VectorStoreRetriever:
        """Configure a retriever with a specific number of most relevant documents.

        .. deprecated:: 0.1.0
           This method will be removed in a future version.

        Args:
            top_k: Number of documents to retrieve (default 4)
            filter: Optional filter criteria for documents

        Returns:
            Configurable vector store retriever
        """
        import warnings

        warnings.warn(
            "as_retriever_configurable() is deprecated and will be removed in a future version. "
            "Use as_retriever() with search_kwargs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        search_kwargs = {"k": top_k}
        if filter:
            search_kwargs |= {"filter": filter}

        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs,
        ).configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
            )
        )
        return retriever  # type: ignore

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
            return self.vector_store.add_documents(list(docs))
        else:
            vector_store = self.vector_store
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
            return self.vector_store._collection.count()  # type: ignore
        else:
            raise NotImplementedError(f"Don't know how to get collection count for {self.vector_store}")

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
        from langchain_postgres import PGEngine, PGVectorStore
        from sqlalchemy.exc import ProgrammingError

        # Use config dict to override YAML values
        postgres_url = self.config.get("postgres_url") or global_config().get_str("vector_store.postgres_url")
        schema_name = self.config.get("postgres_schema") or "public"

        connection_string = f"postgresql+asyncpg:{postgres_url}"
        table_name = self.table_name

        pg_engine = PGEngine.from_connection_string(url=connection_string)

        # Prepare hybrid search configuration if enabled
        hybrid_search_config = None
        if self.hybrid_search and HAS_HYBRID_SEARCH:
            hybrid_config = self.hybrid_search_config or {}
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
            )
            logger.info(f"pgvector vector table created: {table_name=} {schema_name=}")
            if self.hybrid_search and hybrid_search_config:
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
            hybrid_search_config=hybrid_search_config,
        )

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
                raise NotImplementedError(f"Don't'clean' method for {self.vector_store}")

    def add_hybrid_search_to_existing_table(
        self,
        tsv_column: str = "content_tsv",
        tsv_lang: str = "pg_catalog.english",
        index_name: str | None = None,
        index_type: str = "GIN",
    ) -> None:
        """Add TSV column and GIN index to an existing PostgreSQL vector store table.

        This method alters an existing table to support hybrid search by:
        1. Adding a TSV (text search vector) column
        2. Creating a GIN index on the TSV column
        3. Setting up a trigger to auto-update the TSV column

        Args:
            tsv_column: Name of the TSV column to add
            tsv_lang: Language configuration for text search
            index_name: Name of the GIN index (defaults to f"{table_name}_tsv_index")
            index_type: Type of index (typically "GIN")

        Example:
            >>> factory = VectorStoreFactory(
            ...     id="PgVector",
            ...     embeddings_factory=embeddings_factory
            ... )
            >>> factory.add_hybrid_search_to_existing_table(
            ...     tsv_column="content_tsv",
            ...     tsv_lang="pg_catalog.english"
            ... )
        """
        if self.id != "PgVector":
            raise ValueError("Hybrid search can only be added to PgVector stores")

        pg_engine = self._conf.get("pg_engine")
        if not pg_engine:
            raise ValueError("PostgreSQL engine not found")

        table_name = self._conf["table_name"]
        schema_name = self._conf["schema_name"]
        content_column = "content"  # Default content column name

        if index_name is None:
            index_name = f"{table_name}_tsv_index"

        # SQL commands to add TSV column and index
        alter_sql = f"""
        ALTER TABLE {schema_name}.{table_name}
        ADD COLUMN IF NOT EXISTS {tsv_column} tsvector;
        """

        # Create GIN index on TSV column
        index_sql = f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {schema_name}.{table_name}
        USING {index_type} ({tsv_column});
        """

        # Create trigger function to auto-update TSV column
        trigger_function_sql = f"""
        CREATE OR REPLACE FUNCTION update_{table_name}_tsv()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.{tsv_column} := to_tsvector('{tsv_lang}', NEW.{content_column});
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """

        # Create trigger to execute function on INSERT/UPDATE
        trigger_sql = f"""
        DROP TRIGGER IF EXISTS trigger_update_{table_name}_tsv ON {schema_name}.{table_name};
        CREATE TRIGGER trigger_update_{table_name}_tsv
            BEFORE INSERT OR UPDATE ON {schema_name}.{table_name}
            FOR EACH ROW
            EXECUTE FUNCTION update_{table_name}_tsv();
        """

        # Execute all SQL commands
        import sqlalchemy

        with pg_engine._engine.connect() as conn:
            conn.execute(sqlalchemy.text(alter_sql))
            conn.execute(sqlalchemy.text(index_sql))
            conn.execute(sqlalchemy.text(trigger_function_sql))
            conn.execute(sqlalchemy.text(trigger_sql))
            conn.commit()

        logger.info(
            f"Added hybrid search support to table {schema_name}.{table_name}: "
            f"TSV column={tsv_column}, GIN index={index_name}"
        )

    def update_existing_records_tsv(
        self, tsv_column: str = "content_tsv", tsv_lang: str = "pg_catalog.english"
    ) -> None:
        """Update TSV values for existing records in the table.

        Args:
            tsv_column: Name of the TSV column to update
            tsv_lang: Language configuration for text search
        """
        if self.id != "PgVector":
            raise ValueError("TSV update can only be applied to PgVector stores")

        pg_engine = self._conf.get("pg_engine")
        if not pg_engine:
            raise ValueError("PostgreSQL engine not found")

        table_name = self._conf["table_name"]
        schema_name = self._conf["schema_name"]
        content_column = "content"  # Default content column name

        update_sql = f"""
        UPDATE {schema_name}.{table_name}
        SET {tsv_column} = to_tsvector('{tsv_lang}', {content_column})
        WHERE {content_column} IS NOT NULL;
        """

        import sqlalchemy

        with pg_engine._engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(update_sql))
            conn.commit()

        logger.info(f"Updated TSV values for {result.rowcount} existing records in {schema_name}.{table_name}")


def search_one(vc: VectorStore, query: str) -> list[Document]:
    """Perform a similarity search to find the single most relevant document.

    Args:
        vc: Vector store to search in
        query: Search query string

    Returns:
        List containing the most similar document
    """
    return vc.similarity_search(query, k=1)
