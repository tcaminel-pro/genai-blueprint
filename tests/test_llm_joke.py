"""
Tests for LLM joke generation functionality.

This module contains tests that verify:
- Basic LLM joke generation
- Configuration switching between different LLMs
- Error handling for invalid configurations
"""

import pytest
from langchain_core.runnables import RunnableConfig

from python.ai_core.llm import get_llm, get_configurable_llm, llm_config
from python.ai_core.prompts import def_prompt

def test_basic_joke_generation():
    """Test that we can generate a simple joke using the default LLM."""
    llm = get_llm()
    joke = llm.invoke("Tell me a short joke about computers")
    assert isinstance(joke, str)
    assert len(joke) > 10  # Basic check that we got some content

def test_configurable_llm_switching():
    """Test that we can switch LLMs at runtime using config."""
    chain = def_prompt("Tell me a joke about {topic}") | get_configurable_llm()
    
    # Test with default LLM
    result1 = chain.invoke({"topic": "bears"})
    
    # Test with specific LLM config
    result2 = chain.with_config(llm_config("gpt_35_openai")).invoke({"topic": "bears"})
    
    assert isinstance(result1, str)
    assert isinstance(result2, str)
    assert result1 != result2  # Different LLMs should produce different results

def test_invalid_llm_config():
    """Test error handling for invalid LLM configurations."""
    with pytest.raises(ValueError):
        # This should fail since "invalid_llm" doesn't exist
        get_llm(llm_id="invalid_llm")

def test_streaming_joke():
    """Test streaming joke generation."""
    llm = get_llm(streaming=True)
    chunks = []
    for chunk in llm.stream("Tell me a joke about AI"):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)
    assert len("".join(chunks)) > 10
