"""Tests for LLM joke generation functionality.

This module contains tests that verify:
- Basic LLM joke generation
- Configuration switching between different LLMs
- Error handling for invalid configurations
"""

import pytest
from langchain_core.messages.ai import AIMessage

from python.ai_core.llm import get_configurable_llm, get_llm, llm_config
from python.ai_core.prompts import def_prompt
from python.config import global_config

global_config().select_config("pytest")

another_llm = global_config().get_str("llm", "model2")


def test_basic_joke_generation() -> None:
    """Test that we can generate a simple joke using the default LLM."""
    llm = get_llm()
    joke = llm.invoke("Tell me a short joke about computers")
    assert isinstance(joke, AIMessage)
    assert len(joke.content) > 10  # Basic check that we got some content


def test_configurable_llm_switching() -> None:
    """Test that we can switch LLMs at runtime using config."""
    chain = def_prompt("Tell me a joke about {topic}") | get_configurable_llm()

    # Test with default LLM
    result1 = chain.invoke({"topic": "bears"})

    # Test with specific LLM config
    result2 = chain.with_config(llm_config(llm_id=another_llm)).invoke({"topic": "bears"})

    assert result1 != result2  # Different LLMs should produce different results


def test_invalid_llm_config() -> None:
    """Test error handling for invalid LLM configurations."""
    with pytest.raises(ValueError):
        # This should fail since "invalid_llm" doesn't exist
        get_llm(llm_id="invalid_llm")


def test_streaming_joke() -> None:
    """Test streaming joke generation."""
    llm = get_llm(streaming=True)
    chunks = []
    for chunk in llm.stream("Tell me a joke about AI"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert isinstance(chunks[0], AIMessage)
