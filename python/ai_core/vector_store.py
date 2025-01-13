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
    >>> from python.ai_core.embeddings import EmbeddingsFactory
    >>> from langchain.schema import Document
    >>> 
    >>> # Create vector store factory
    >>> factory = VectorStoreFactory(
    ...     id="Chroma",
    ...     embeddings_factory=EmbeddingsFactory()
    ... )
    >>> 
    >>> # Add documents to store
    >>> factory.add_documents([
    ...     Document(page_content="Machine learning is powerful"),
    ...     Document(page_content="AI is transforming industries")
    ... ])
    >>> 
    >>> # Perform similarity search
    >>> results = factory.vector_store.similarity_search("AI technology")
"""

# Rest of the existing code remains the same...

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
        >>> from python.ai_core.embeddings import EmbeddingsFactory
        >>> from langchain.schema import Document
        >>> 
        >>> # Create a vector store with default embeddings
        >>> store_factory = VectorStoreFactory(
        ...     id="Chroma",
        ...     collection_name="my_collection",
        ...     embeddings_factory=EmbeddingsFactory()
        ... )
        >>> 
        >>> # Add documents to the vector store
        >>> vector_store = store_factory.vector_store
        >>> vector_store.add_documents([
        ...     Document(page_content="Hello world", metadata={"source": "example"})
        ... ])
    """
    # Existing implementation...

def search_one(vc: VectorStore, query: str) -> list[Document]:
    """
    Perform a similarity search for a single most relevant document.

    Args:
        vc: Vector store instance to search in
        query: Search query string

    Returns:
        List containing the single most similar document found

    Example:
        >>> from python.ai_core.vector_store import VectorStoreFactory
        >>> from langchain.schema import Document
        >>> 
        >>> # Create a vector store and add documents
        >>> factory = VectorStoreFactory()
        >>> factory.vector_store.add_documents([
        ...     Document(page_content="Python is great"),
        ...     Document(page_content="Machine learning is powerful")
        ... ])
        >>> 
        >>> # Search for the most relevant document
        >>> results = search_one(factory.vector_store, "programming language")
        >>> len(results) == 1
        True
    """
    return vc.similarity_search(query, k=1)
