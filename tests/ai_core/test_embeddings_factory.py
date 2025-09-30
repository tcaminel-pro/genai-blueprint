"""Tests for embeddings functionality.

This module contains basic regression tests for the embeddings factory and utilities.
"""

from src.ai_core.embeddings_factory import EmbeddingsFactory, get_embeddings
from src.utils.config_mngr import global_config

SENTENCE_1 = "Tokenization is the process of breaking down a text into individual units."
SENTENCE_2 = "Tokens can be words, phrases, or even individual characters."


global_config().select_config("pytest")


def test_fake_embeddings() -> None:
    """Test that default embeddings can be created and used."""

    # global_config().set("llm.models.default", "embeddings_768_fake")
    embedder = get_embeddings("embeddings_768_fake")
    vectors = embedder.embed_documents([SENTENCE_1, SENTENCE_2])

    # Basic validation of embeddings
    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert len(vectors[1]) == 768


def test_known_embeddings_list() -> None:
    """Test that known embeddings list is not empty."""
    embeddings_list = EmbeddingsFactory.known_items()
    assert len(embeddings_list) > 0
    assert isinstance(embeddings_list, list)
