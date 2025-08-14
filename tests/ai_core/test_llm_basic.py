"""Tests for LLM joke generation functionality.

This module contains tests that verify:
- Basic LLM joke generation
- Configuration switching between different LLMs
- Error handling for invalid configurations
"""

from langchain_core.messages.human import HumanMessage

from src.ai_core.llm_factory import get_llm

# global_config().select_config("pytest")

LLM_ID = "parrot_local_fake"


def test_basic_call() -> None:
    """Test that we can generate a simple joke using fake LLM."""
    llm = get_llm(llm_id=LLM_ID)
    joke = llm.invoke("Tell me a short joke about computers")
    assert isinstance(joke, HumanMessage), f"{type(joke)}"
    assert len(joke.content) > 10  # Basic check that we got some content


def test_streaming_joke() -> None:
    """Test streaming joke generation."""
    llm = get_llm(llm_id=LLM_ID, streaming=True)
    chunks = []
    for chunk in llm.stream("Tell me a joke about AI"):
        chunks.append(chunk)

    assert len(chunks) > 0
    print(type(chunks[0]))
    assert isinstance(chunks[0], HumanMessage), f"{type(chunks[0])}"
