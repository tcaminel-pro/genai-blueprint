"""
Tests for embeddings functionality.

This module contains basic regression tests for the embeddings factory and utilities.
"""
import pytest
from python.ai_core.embeddings import get_embeddings, EmbeddingsFactory

SENTENCE_1 = "Tokenization is the process of breaking down a text into individual units."
SENTENCE_2 = "Tokens can be words, phrases, or even individual characters."

def test_default_embeddings():
    """Test that default embeddings can be created and used."""
    embedder = get_embeddings()
    vectors = embedder.embed_documents([SENTENCE_1, SENTENCE_2])
    
    # Basic validation of embeddings
    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[1]) > 0

def test_known_embeddings_list():
    """Test that known embeddings list is not empty."""
    embeddings_list = EmbeddingsFactory.known_items()
    assert len(embeddings_list) > 0
    assert isinstance(embeddings_list, list)
