"""Vector Store Management and Factory System.

This module provides a comprehensive interface for creating, managing, and
interacting with vector stores across multiple storage backends. It supports
advanced document indexing, retrieval, and vector database operations.

Key Features:
- Multi-backend vector store support (Chroma, In-Memory, Sklearn)
- Flexible document indexing and deduplication
- Configurable retrieval strategies
- Seamless integration with embedding models
- Advanced search and filtering capabilities

Supported Backends:
- Chroma (persistent and in-memory)
- InMemoryVectorStore
- SKLearnVectorStore

Design Patterns:
- Factory Method for vector store creation
- Configurable retrieval strategies
- Singleton-like access to vector stores

Example:
    >>> # Create a vector store factory
    >>> factory = VectorStoreFactory(
    ...     id="Chroma",
    ...     embeddings_factory=EmbeddingsFactory(),
    ...     collection_name="my_documents"
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

import os
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, get_args

from langchain.indexes import IndexingResult, SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_chroma import Chroma
from langchain_core.runnables import ConfigurableField
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from python.ai_core.embeddings import EmbeddingsFactory
from python.config import global_config

# List of known Vector Stores (created as Literal so can be checked by MyPy)
VECTOR_STORE_ENGINE = Literal["Chroma", "Chroma_in_memory", "InMemory", "Sklearn"]

default_collection = global_config().get_str("vector_store.default_collection")


def get_vector_vector_store_path() -> str:
    """Determine the appropriate path for storing vector databases.

    Supports both Modal cloud environments and local development setups.

    Returns:
        Path to the vector store storage directory
    """
    if modal_volume := os.environ.get("MODAL_VOLUME_PATH"):
        # Running in Modal environment
        return modal_volume
    else:
        # Local development
        dir = global_config().get_path("vector_store.path", create_if_not_exists = True)
        return str(dir)


class VectorStoreFactory(BaseModel):
    """Factory for creating and managing vector stores with advanced configuration.

    Provides a flexible and powerful interface for creating vector stores with
    support for multiple backends, document indexing, and advanced retrieval
    strategies.

    Attributes:
        id: Identifier for the vector store backend
        embeddings_factory: Factory for creating embedding models
        collection_name: Name of the vector store collection
        index_document: Flag to enable document deduplication and indexing
        collection_metadata: Optional metadata for the collection
        cache_embeddings: Flag to enable embedding caching

    Example:
        >>> factory = VectorStoreFactory(
        ...     id="Chroma",
        ...     embeddings_factory=EmbeddingsFactory(),
        ...     collection_name="documents"
        ... )
        >>> factory.add_documents([Document(page_content="example")])
    """

    id: Annotated[VECTOR_STORE_ENGINE | None, Field(validate_default=True)] = None
    embeddings_factory: EmbeddingsFactory
    collection_name: str = default_collection
    index_document: bool = False
    collection_metadata: dict[str, str] | None = None
    cache_embeddings: bool = False
    _record_manager: SQLRecordManager | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def collection_full_name(self) -> str:
        """Generate a unique collection name by combining collection and embeddings ID.

        Returns:
            Unique collection name to prevent conflicts
        """
        assert self.embeddings_factory
        embeddings_id = self.embeddings_factory.info.id
        return f"{self.collection_name}_{embeddings_id}"

    @computed_field
    def description(self) -> str:
        """Generate a detailed description of the vector store configuration.

        Returns:
            Comprehensive configuration description string
        """
        r = f"{str(self.id)}/{self.collection_full_name}"
        if self.id == "Chroma":
            r += f" => '{get_vector_vector_store_path()}'"
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
        store_path = get_vector_vector_store_path()

        if self.id == "Chroma":
            Path(store_path).mkdir(parents=True, exist_ok=True)

            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=store_path,
                collection_name=self.collection_full_name,
                collection_metadata=self.collection_metadata,
            )
        elif self.id == "Chroma_in_memory":
            vector_store = Chroma(
                embedding_function=embeddings,
                collection_name=self.collection_full_name,
                collection_metadata=self.collection_metadata,
            )
        elif self.id == "InMemory":
            vector_store = InMemoryVectorStore(
                embedding=embeddings,
            )
        elif self.id == "Sklearn":
            from langchain_community.vectorstores import SKLearnVectorStore

            vector_store = SKLearnVectorStore(
                embedding=embeddings,
            )
        else:
            raise ValueError(f"Unknown vector store: {self.id}")

        logger.info(f"get vector store  : {self.description}")
        if self.index_document:
            db_url = f"sqlite:///{get_vector_vector_store_path()}/record_manager_cache.sql"
            logger.info(f"vector store record manager : {db_url}")
            namespace = f"{id}/{self.collection_full_name}"
            self._record_manager = SQLRecordManager(
                namespace,
                db_url=db_url,
            )
            self._record_manager.create_schema()

        return vector_store

    def set_number_of_doc_to_fetch(self, k: int = 4) -> VectorStoreRetriever:
        """Configure a retriever with a specific number of most relevant documents.

        Args:
            k: Number of documents to retrieve (default 4)

        Returns:
            Configurable vector store retriever
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k}).configurable_fields(
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


def search_one(vc: VectorStore, query: str) -> list[Document]:
    """Perform a similarity search to find the single most relevant document.

    Args:
        vc: Vector store to search in
        query: Search query string

    Returns:
        List containing the most similar document
    """
    return vc.similarity_search(query, k=1)
