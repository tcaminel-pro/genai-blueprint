"""Tests for BM25FastRetriever."""

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from ai_extra.bm25s_retriever import BM25FastRetriever


class TestBM25FastRetriever:
    """Test class for BM25FastRetriever."""

    @pytest.fixture
    def sample_documents(self):
        """Provide sample documents for testing."""
        return [
            Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"id": 1}),
            Document(page_content="A fast brown fox leaps over lazy dogs in summer.", metadata={"id": 2}),
            Document(page_content="The lazy dog sleeps in the sun.", metadata={"id": 3}),
            Document(page_content="Python is a programming language used for AI development.", metadata={"id": 4}),
            Document(page_content="Machine learning models require training data.", metadata={"id": 5}),
        ]

    def test_from_texts(self):
        """Test creating retriever from texts."""
        texts = ["hello world", "foo bar", "python programming"]
        metadatas = [{"source": i} for i in range(len(texts))]

        retriever = BM25FastRetriever.from_texts(
            texts=texts,
            metadatas=metadatas,
            k=2
        )

        assert retriever.k == 2
        assert len(retriever.docs) == 3
        assert retriever.docs[0].page_content == "hello world"
        assert retriever.docs[0].metadata["source"] == 0

    def test_from_documents(self, sample_documents):
        """Test creating retriever from documents."""
        retriever = BM25FastRetriever.from_documents(
            documents=sample_documents,
            k=3
        )

        assert retriever.k == 3
        assert len(retriever.docs) == 5
        assert retriever.docs[0].metadata["id"] == 1

    def test_retrieval_basic(self, sample_documents):
        """Test basic retrieval functionality."""
        retriever = BM25FastRetriever.from_documents(
            documents=sample_documents,
            k=2
        )

        results = retriever.invoke("fox")

        assert len(results) == 2
        # Should find documents about foxes
        fox_docs = [doc for doc in results if "fox" in doc.page_content.lower()]
        assert len(fox_docs) > 0

    def test_retrieval_empty_query(self, sample_documents):
        """Test retrieval with empty query."""
        retriever = BM25FastRetriever.from_documents(
            documents=sample_documents,
            k=2
        )

        results = retriever.invoke("")
        assert len(results) == 2  # Should return top documents even for empty query

    def test_retrieval_no_matches(self, sample_documents):
        """Test retrieval with no matching terms."""
        retriever = BM25FastRetriever.from_documents(
            documents=sample_documents,
            k=2
        )

        results = retriever.invoke("zebra elephant")
        assert len(results) == 2  # Should return some documents even without matches

    def test_cache_functionality(self, sample_documents):
        """Test saving and loading from cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "bm25_cache"

            # Create retriever and save to cache
            retriever = BM25FastRetriever.from_documents(
                documents=sample_documents,
                cache_dir=cache_path,
                k=3
            )

            # Load from cache
            cached_retriever = BM25FastRetriever.from_index_file(
                index_file=cache_path,
                k=3
            )

            # Both should return similar results
            original_results = retriever.invoke("fox")
            cached_results = cached_retriever.invoke("fox")

            # Note: cached retriever doesn't store docs, so we only test it doesn't crash
            assert len(cached_results) <= 3

    def test_preprocessing_function(self):
        """Test custom preprocessing function."""
        def custom_preprocess(text: str) -> list[str]:
            return text.upper().split()

        texts = ["hello world", "foo bar"]
        retriever = BM25FastRetriever.from_texts(
            texts=texts,
            preprocess_func=custom_preprocess,
            k=2
        )

        # Should work with custom preprocessing
        results = retriever.invoke("HELLO")
        assert len(results) > 0

    def test_k_parameter(self, sample_documents):
        """Test different k values."""
        retriever = BM25FastRetriever.from_documents(
            documents=sample_documents,
            k=1
        )

        results = retriever.invoke("dog")
        assert len(results) == 1

        retriever.k = 3
        results = retriever.invoke("dog")
        assert len(results) == 3

    def test_bm25_parameters(self, sample_documents):
        """Test passing BM25 parameters."""
        retriever = BM25FastRetriever.from_documents(
            documents=sample_documents,
            bm25_params={"k1": 1.5, "b": 0.75},
            k=2
        )

        results = retriever.invoke("fox")
        assert len(results) == 2  # Should work with custom BM25 parameters
