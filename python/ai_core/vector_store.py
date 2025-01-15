"""
Vector store factory and management system.

This module provides a comprehensive interface for creating and managing vector
stores, supporting multiple storage backends and advanced features for document
indexing and retrieval.

Key Features:
- Support for multiple vector store implementations (Chroma, InMemory, Sklearn)
- Document deduplication and indexing
- Configurable retrieval parameters
- Collection metadata management
- Integration with embedding models

Supported Backends:
- Chroma (persistent and in-memory)
- InMemoryVectorStore
- SKLearnVectorStore

Example:
    >>> # Create vector store factory
    >>> factory = VectorStoreFactory(
    ...     id="Chroma",
    ...     embeddings_factory=EmbeddingsFactory()
    ... )

    >>> # Add documents to store
    >>> factory.add_documents([
    ...     Document(page_content="text1"),
    ...     Document(page_content="text2")
    ... ])

    >>> # Perform similarity search
    >>> results = factory.vector_store.similarity_search("query")
"""

from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal, get_args

from langchain.indexes import IndexingResult, SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_chroma import Chroma
from langchain_core.runnables import ConfigurableField
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from typing_extensions import Annotated

from python.ai_core.embeddings import EmbeddingsFactory
from python.config import global_config

# from langchain_chroma import Chroma  does not work (yet?) with self_query


# List of known Vector Stores (created as Literal so can be checked by MyPy)
VECTOR_STORE_ENGINE = Literal["Chroma", "Chroma_in_memory", "InMemory", "Sklearn"]

default_collection = global_config().get_str("vector_store", "default_collection")


def get_vector_vector_store_path() -> str:
    """Get path to store vector database, using Modal volume if available."""
    if modal_volume := os.environ.get("MODAL_VOLUME_PATH"):
        # Running in Modal environment
        return modal_volume
    else:
        # Local development
        dir = Path(global_config().get_str("vector_store", "path"))
        try:
            dir.mkdir()
        except Exception:
            pass  # TODO : log something
        return str(dir)


class VectorStoreFactory(BaseModel):
    """
    Factory class for creating and managing vector stores.

    This class handles the creation and configuration of vector stores with appropriate
    embeddings, collection management, and document indexing capabilities.

    Attributes:
        id: Vector store type identifier (Chroma or Chroma_in_memory)
        embeddings_factory: Factory to create embeddings for the vector store
        collection_name: Name of the vector store collection
        index_document: Whether to use LangChain indexer for document deduplication
        collection_metadata: Optional metadata for the collection
        _record_manager: Internal SQL manager for document indexing

    Example:
    .. code-block:: python
        store_factory = VectorStoreFactory(
                id="Chroma",
                collection_name="some_name",
                embeddings_factory=EmbeddingsFactory(embeddings_id=None),  # default model
            )
        vector_store = store_factory.vector_store
        vector_store.add_documents([Document(page_content="hello world, metadata={"source": "a source})]
    """

    id: Annotated[VECTOR_STORE_ENGINE | None, Field(validate_default=True)] = None
    embeddings_factory: EmbeddingsFactory
    collection_name: str = default_collection
    index_document: bool = False
    collection_metadata: dict[str, str] | None = None
    _record_manager: SQLRecordManager | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def collection_full_name(self) -> str:
        # Concatenate the collection and and the embeddings ID to avoid errors when changing embeddings
        assert self.embeddings_factory
        embeddings_id = self.embeddings_factory.info.id
        return f"{self.collection_name}_{embeddings_id}"

    @computed_field
    def description(self) -> str:
        r = f"{str(self.id)}/{self.collection_full_name}"
        if self.id == "Chroma":
            r += f" => store: {get_vector_vector_store_path()}"
        if self.index_document == Chroma and self._record_manager:
            r += f" indexer: {self._record_manager}"
        return r

    @staticmethod
    def known_items() -> list[str]:
        return list(get_args(VECTOR_STORE_ENGINE))

    @field_validator("id", mode="before")
    def check_known(cls, id: str | None) -> str:
        if id is None:
            id = global_config().get_str("vector_store", "default")
        if id not in VectorStoreFactory.known_items():
            raise ValueError(f"Unknown Vector Store: {id}")
        return id

    @computed_field
    @cached_property
    def vector_store(self) -> VectorStore:
        """
        Factory for the vector database
        """
        embeddings = self.embeddings_factory.get()  # get the embedding function
        store_path = get_vector_vector_store_path()

        if self.id == "Chroma":
            # Ensure the directory exists
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
                # collection_name=self.collection_full_name,  # TODO (?) : implement  an hash of InMemoryVectorStore
                # collection_metadata=self.collection_metadata,
            )
        elif self.id == "Sklearn":
            from langchain_community.vectorstores import SKLearnVectorStore

            vector_store = SKLearnVectorStore(
                embedding=embeddings,
                # collection_name=self.collection_full_name,
                # collection_metadata=self.collection_metadata,
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
                db_url=db_url,  # @TODO: To improve !!
            )
            self._record_manager.create_schema()

        return vector_store

    def change_top_k(self, k: int = 4) -> VectorStoreRetriever:
        """
        Return a retriever with changed number of most relevant document returned.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k}).configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
            )
        )
        return retriever  # type: ignore

    def add_documents(self, docs: Iterable[Document]) -> IndexingResult | list[str]:
        """
        Add documents to the vector store with optional deduplication.

        Args:
            docs: Iterable of Document objects to add to the store

        Returns:
            IndexingResult if using document indexing, or list of document IDs otherwise

        The method handles two scenarios:
        1. Direct addition to vector store if index_document=False
        2. Indexed addition with deduplication if index_document=True using LangChain's indexer
        """
        # TODO : accept BaseLoader
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
        """
        Return the number of documents in the store.
        """
        # It seems there's no generic way to get the number of docs stored in a Vector Store.
        if self.id in ["Chroma", "Chroma_in_memory"]:
            return self.vector_store._collection.count()  # type: ignore
        else:
            raise NotImplementedError(f"Don'k know how to get collection count for {self.vector_store}")


def search_one(vc: VectorStore, query: str) -> list[Document]:
    """
    Perform a similarity search for a single most relevant document.

    Args:
        vc: Vector store instance to search in
        query: Search query string

    Returns:
        List containing the single most similar document found

    This is a convenience function for quick lookups where only the best match is needed.
    """
    return vc.similarity_search(query, k=1)
