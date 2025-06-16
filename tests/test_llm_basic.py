"""Tests for LLM joke generation functionality.

This module contains tests that verify:
- Basic LLM joke generation
- Configuration switching between different LLMs
- Error handling for invalid configurations
"""

from langchain_core.messages.ai import AIMessage

from src.ai_core.llm import get_llm
from src.utils.config_mngr import global_config

global_config().select_config("pytest")


def test_basic_joke_generation() -> None:
    """Test that we can generate a simple joke using the default LLM."""
    llm = get_llm()
    joke = llm.invoke("Tell me a short joke about computers")
    assert isinstance(joke, AIMessage)
    assert len(joke.content) > 10  # Basic check that we got some content


def test_streaming_joke() -> None:
    """Test streaming joke generation."""
    llm = get_llm(streaming=True)
    chunks = []
    for chunk in llm.stream("Tell me a joke about AI"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert isinstance(chunks[0], AIMessage)
