import pytest
from langchain.schema import Document

from src.ai_core.embeddings_factory import EmbeddingsFactory
from src.ai_core.vector_store_factory import VectorStoreFactory
from src.utils.config_mngr import global_config

try:
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig

    HAS_HYBRID_SEARCH = True
except ImportError:
    HAS_HYBRID_SEARCH = False

global_config().select_config("pytest")

EMBEDDINGS_MODEL_ID = "embeddings_768_fake"


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="The quick brown fox jumps over the lazy dog"),
        Document(page_content="Python is a powerful programming language"),
        Document(page_content="Machine learning is transforming many industries"),
    ]


@pytest.mark.parametrize("vector_store_type", ["InMemory", "Chroma_in_memory"])
def test_vector_store_creation_and_search(sample_documents, vector_store_type) -> None:
    """Test vector store creation, document addition, and similarity search.

    Args:
        sample_documents: Fixture providing test documents
        vector_store_type: Parametrized vector store type to test
    """
    # Create vector store factory
    vs_factory = VectorStoreFactory(
        id=vector_store_type,
        embeddings_factory=EmbeddingsFactory(embeddings_id=EMBEDDINGS_MODEL_ID),
    )

    # Add documents
    db = vs_factory.get()
    db.add_documents(sample_documents)

    # Perform similarity search
    query = "programming language"
    results = db.similarity_search(query, k=2)

    assert len(results) == 2
    # assert any("Python" in doc.page_content for doc in results)


def test_vector_store_factory_methods() -> None:
    """Test VectorStoreFactory class methods."""
    # Test known items method
    known_stores = VectorStoreFactory.known_items()
    assert isinstance(known_stores, list)
    assert len(known_stores) > 0


def test_vector_store_retriever() -> None:
    """Test vector store retriever functionality."""
    vs_factory = VectorStoreFactory(
        embeddings_factory=EmbeddingsFactory(embeddings_id=EMBEDDINGS_MODEL_ID),
    )
    db = vs_factory.get()
    db.add_documents(
        [
            Document(page_content="AI is revolutionizing technology"),
            Document(page_content="Machine learning algorithms are complex"),
        ]
    )

    # Test default retriever
    retriever = vs_factory.get().as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke("AI technology")

    assert len(results) == 1


postgres_url = global_config().get_str("vector_store.postgres_url", None)


@pytest.mark.skipif(not HAS_HYBRID_SEARCH, reason="langchain-postgres not available")
@pytest.mark.skipif(not postgres_url, reason="POSTGRES_URL not configured")
def test_pgvector_hybrid_search_creation() -> None:
    """Test PgVector creation with hybrid search enabled."""
    vs_factory = VectorStoreFactory(
        id="PgVector",
        embeddings_factory=EmbeddingsFactory(embeddings_id=EMBEDDINGS_MODEL_ID),
        hybrid_search=True,
        hybrid_search_config={
            "tsv_column": "content_tsv",
            "tsv_lang": "pg_catalog.english",
            "fusion_function_parameters": {
                "primary_results_weight": 0.5,
                "secondary_results_weight": 0.5,
            },
        },
    )

    # Should create successfully
    db = vs_factory.get()
    assert db is not None


@pytest.mark.skipif(not HAS_HYBRID_SEARCH, reason="langchain-postgres not available")
@pytest.mark.skipif(not postgres_url, reason="POSTGRES_URL not configured")
def test_pgvector_hybrid_search_functionality() -> None:
    """Test PgVector hybrid search functionality."""
    vs_factory = VectorStoreFactory(
        id="PgVector",
        embeddings_factory=EmbeddingsFactory(embeddings_id=EMBEDDINGS_MODEL_ID),
        hybrid_search=True,
        hybrid_search_config={
            "tsv_column": "content_tsv",
            "tsv_lang": "pg_catalog.english",
            "fusion_function_parameters": {
                "primary_results_weight": 0.5,
                "secondary_results_weight": 0.5,
            },
        },
    )

    db = vs_factory.get()

    # Add test documents
    test_docs = [
        Document(page_content="PostgreSQL is a powerful database with full-text search"),
        Document(page_content="Vector search uses embeddings for similarity"),
        Document(page_content="Hybrid search combines vector and text search"),
    ]

    db.add_documents(test_docs)

    # Perform hybrid search
    results = db.similarity_search(
        "database search",
        k=2,
        hybrid_search_config=HybridSearchConfig(
            tsv_column="content_tsv",
            fusion_function_parameters={"primary_results_weight": 0.5, "secondary_results_weight": 0.5},
        ),
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.skipif(not HAS_HYBRID_SEARCH, reason="langchain-postgres not available")
@pytest.mark.skipif(not postgres_url, reason="POSTGRES_URL not configured")
def test_pgvector_hybrid_search_with_weights() -> None:
    """Test PgVector hybrid search with different weight configurations."""
    vs_factory = VectorStoreFactory(
        id="PgVector",
        embeddings_factory=EmbeddingsFactory(embeddings_id=EMBEDDINGS_MODEL_ID),
        hybrid_search=True,
        hybrid_search_config={
            "tsv_column": "content_tsv",
            "tsv_lang": "pg_catalog.english",
        },
    )

    db = vs_factory.get()

    # Add test documents
    test_docs = [
        Document(page_content="Vector databases are great for AI applications"),
        Document(page_content="Full-text search indexes text content efficiently"),
        Document(page_content="Hybrid search combines the best of both worlds"),
    ]

    db.add_documents(test_docs)

    # Test with vector bias
    results_vector = db.similarity_search(
        "vector database",
        k=2,
        hybrid_search_config=HybridSearchConfig(
            tsv_column="content_tsv",
            fusion_function_parameters={
                "primary_results_weight": 0.8,
                "secondary_results_weight": 0.2,
            },
        ),
    )

    # Test with text bias
    results_text = db.similarity_search(
        "text search",
        k=2,
        hybrid_search_config=HybridSearchConfig(
            tsv_column="content_tsv",
            fusion_function_parameters={
                "primary_results_weight": 0.2,
                "secondary_results_weight": 0.8,
            },
        ),
    )

    assert len(results_vector) == 2
    assert len(results_text) == 2


@pytest.mark.skipif(not HAS_HYBRID_SEARCH, reason="langchain-postgres not available")
@pytest.mark.skipif(not postgres_url, reason="POSTGRES_URL not configured")
def test_pgvector_basic_connection() -> None:
    """Test basic PgVector connection without hybrid search."""
    vs_factory = VectorStoreFactory(
        id="PgVector",
        embeddings_factory=EmbeddingsFactory(embeddings_id=EMBEDDINGS_MODEL_ID),
    )

    # Should create successfully
    db = vs_factory.get()
    assert db is not None

    # Test basic functionality
    test_docs = [Document(page_content="Test document for connection")]
    db.add_documents(test_docs)

    results = db.similarity_search("test", k=1)
    assert len(results) >= 1
