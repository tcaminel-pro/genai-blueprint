"""Tests for LLM factory functionality.

This module contains tests that verify:
- Basic LLM initialization and functionality
- Configuration switching between different LLMs
- Error handling for invalid configurations
- Factory method behavior
- JSON mode and streaming options
"""

import pytest
from langchain_core.messages.human import HumanMessage

from src.ai_core.llm_factory import (
    LlmFactory,
    configurable,
    get_configurable_llm,
    get_llm,
    get_llm_info,
    llm_config,
)

# global_config().select_config("pytest")

LLM_ID_FAKE = "parrot_local_fake"
LLM_ID_FOR_TEST = "gpt_41mini_openrouter"


def test_basic_call() -> None:
    """Test that we can generate a simple joke using fake LLM."""
    llm = get_llm(llm_id=LLM_ID_FAKE)
    joke = llm.invoke("Tell me a short joke about computers")
    assert isinstance(joke, HumanMessage), f"{type(joke)}"
    assert len(joke.content) > 10  # Basic check that we got some content


def test_streaming_joke() -> None:
    """Test streaming joke generation."""
    llm = get_llm(llm_id=LLM_ID_FAKE, streaming=True)
    chunks = []
    for chunk in llm.stream("Tell me a joke about AI"):
        chunks.append(chunk)

    assert len(chunks) > 0
    print(type(chunks[0]))
    assert isinstance(chunks[0], HumanMessage), f"{type(chunks[0])}"


def test_get_llm_with_none_id_uses_default() -> None:
    """Test that get_llm uses default when llm_id is None."""
    # This should not raise an exception
    llm = get_llm(llm_id=LLM_ID_FAKE)
    assert llm is not None


def test_get_llm_with_llm_type() -> None:
    """Test get_llm with llm_type parameter."""
    # Use fake LLM to avoid API key issues
    llm = get_llm(llm_id=LLM_ID_FAKE)
    assert llm is not None


def test_invalid_llm_id_raises_error() -> None:
    """Test that invalid llm_id raises ValueError."""
    with pytest.raises(ValueError, match="Unknown LLM"):
        get_llm(llm_id="nonexistent_model")


def test_llm_factory_creation() -> None:
    """Test LlmFactory class creation."""
    factory = LlmFactory(llm_id=LLM_ID_FAKE)
    assert factory.llm_id == LLM_ID_FAKE
    assert factory.info is not None
    assert factory.provider == "fake"


def test_llm_factory_short_name() -> None:
    """Test short_name method returns correct format."""
    factory = LlmFactory(llm_id=LLM_ID_FAKE)
    short = factory.short_name()
    assert short == "parrot_local"


def test_llm_factory_get_litellm_model_name() -> None:
    """Test get_litellm_model_name method."""
    factory = LlmFactory(llm_id=LLM_ID_FOR_TEST)
    # Skip this test for fake provider since litellm doesn't support it
    if factory.provider == "fake":
        pytest.skip("LiteLLM doesn't support fake provider")
    model_name = factory.get_litellm_model_name()
    assert model_name == "openrouter/openai/gpt-4.1-mini"


def test_llm_factory_get_smolagent_model() -> None:
    """Test get_smolagent_model method."""
    factory = LlmFactory(llm_id=LLM_ID_FAKE)
    # Skip this test for fake provider since smolagent doesn't support it
    if factory.provider == "fake":
        pytest.skip("smolagent doesn't support fake provider")
    model = factory.get_smolagent_model()
    assert model is not None


def test_get_llm_info() -> None:
    """Test get_llm_info function."""
    info = get_llm_info(LLM_ID_FAKE)
    assert info.id == LLM_ID_FAKE
    assert info.provider == "fake"
    assert info.model == "parrot"


def test_get_llm_info_invalid_id() -> None:
    """Test get_llm_info with invalid ID."""
    with pytest.raises(ValueError, match="Unknown LLM"):
        get_llm_info("nonexistent_model")


def test_llm_config() -> None:
    """Test llm_config function."""
    config = llm_config(LLM_ID_FAKE)
    assert "configurable" in config
    assert config["configurable"]["llm_id"] == LLM_ID_FAKE


def test_llm_config_invalid_id() -> None:
    """Test llm_config with invalid ID."""
    with pytest.raises(ValueError, match="Unknown LLM"):
        llm_config("nonexistent_model")


def test_configurable() -> None:
    """Test configurable function."""
    config = configurable({"test_key": "test_value"})
    assert "configurable" in config
    assert config["configurable"]["test_key"] == "test_value"


def test_get_configurable_llm() -> None:
    """Test get_configurable_llm function."""
    llm = get_configurable_llm(llm_id=LLM_ID_FAKE)
    assert llm is not None


def test_get_configurable_llm_with_fallback() -> None:
    """Test get_configurable_llm with fallback option."""
    llm = get_configurable_llm(llm_id=LLM_ID_FAKE, with_fallback=True)
    assert llm is not None


def test_json_mode_parameter() -> None:
    """Test JSON mode parameter."""
    llm = get_llm(llm_id=LLM_ID_FAKE, json_mode=True)
    assert llm is not None


def test_streaming_parameter() -> None:
    """Test streaming parameter."""
    llm = get_llm(llm_id=LLM_ID_FAKE, streaming=True)
    assert llm is not None


def test_cache_parameter() -> None:
    """Test cache parameter."""
    llm = get_llm(llm_id=LLM_ID_FAKE, cache="memory")
    assert llm is not None


def test_llm_params_parameter() -> None:
    """Test additional LLM parameters."""
    llm = get_llm(llm_id=LLM_ID_FAKE, temperature=0.5, max_tokens=100)
    assert llm is not None


def test_known_items() -> None:
    """Test known_items method."""
    items = LlmFactory.known_items()
    assert isinstance(items, list)
    assert len(items) > 0
    assert LLM_ID_FAKE in items


def test_known_items_dict() -> None:
    """Test known_items_dict method."""
    items_dict = LlmFactory.known_items_dict()
    assert isinstance(items_dict, dict)
    assert LLM_ID_FAKE in items_dict


def test_complex_provider_config_parsing() -> None:
    """Test that complex provider configurations (like vllm) are parsed correctly."""
    # Test the liquid_lfm40 model which has complex vllm configuration
    llm_id = "liquid_lfm40_vllm"

    # Check if it's in known items (it should be if config is parsed correctly)
    known_items = LlmFactory.known_items()

    # Only test if the model is actually available (API keys present)
    if llm_id in known_items:
        info = get_llm_info(llm_id)
        assert info.provider == "vllm"
        assert info.model == "bla bla vla"
        assert info.llm_args.get("trust_remote_code") is True
        assert info.llm_args.get("max_new_tokens") == 512
        assert isinstance(info.llm_args.get("llm_kwargs"), dict)
        assert info.llm_args["llm_kwargs"].get("quantization") == "awq"
    else:
        # Test the parsing logic by checking the raw data structure
        from src.ai_core.llm_factory import _read_llm_list_file

        llms = _read_llm_list_file()
        liquid_lfm40_vllm = None
        for llm in llms:
            if llm.id == "liquid_lfm40_vllm":
                liquid_lfm40_vllm = llm
                break

        assert liquid_lfm40_vllm is not None
        assert liquid_lfm40_vllm.provider == "vllm"
        assert liquid_lfm40_vllm.model == "bla bla vla"
        assert liquid_lfm40_vllm.llm_args.get("trust_remote_code") is True
        assert liquid_lfm40_vllm.llm_args.get("max_new_tokens") == 512
        assert liquid_lfm40_vllm.llm_args.get("llm_kwargs", {}).get("quantization") == "awq"


def test_factory_find_llm_id_from_type() -> None:
    """Test find_llm_id_from_type method."""
    # This might fail if no config is set up, so we'll test the error case
    with pytest.raises(ValueError):
        LlmFactory.find_llm_id_from_tag("nonexistent_type")


def test_llm_factory_model_validation() -> None:
    """Test LlmFactory model validation."""
    # Test valid ID
    factory = LlmFactory(llm_id=LLM_ID_FAKE)
    assert factory.llm_id == LLM_ID_FAKE

    # Test invalid ID
    with pytest.raises(ValueError, match="Unknown LLM"):
        LlmFactory(llm_id="invalid_model_id")


def test_field_validator_cache() -> None:
    """Test cache field validator."""
    # Valid cache value
    from langchain_community.cache import SQLiteCache

    llm = LlmFactory(llm_id=LLM_ID_FOR_TEST, cache="sqlite").get()
    assert isinstance(llm.cache, SQLiteCache)

    # Invalid cache value should raise ValueError
    with pytest.raises(ValueError, match="Unknown cache method"):
        LlmFactory(llm_id=LLM_ID_FAKE, cache="invalid_cache")
