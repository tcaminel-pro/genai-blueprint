"""Tests for the Config class and its configuration management features.

These tests verify:
- Basic configuration loading and retrieval
- Environment variable substitution
- Runtime modifications
- Configuration section switching
- Error handling for missing keys
"""

import pytest

from src.utils.config_mngr import OmegaConfig


def test_singleton() -> None:
    """Verify the singleton pattern works correctly."""
    config1 = OmegaConfig.singleton()
    config2 = OmegaConfig.singleton()
    assert config1 is config2


def test_get_str_with_env_var(monkeypatch) -> None:
    """Test environment variable substitution in config values."""
    monkeypatch.setenv("TEST_VAR", "substituted_value")
    config = OmegaConfig.singleton()
    config.set_str("test", "path", "${TEST_VAR}/file.txt")
    assert config.get_str("test.path") == "substituted_value/file.txt"


def test_config_section_switch() -> None:
    """Verify switching between configuration sections works."""
    config = OmegaConfig.singleton()
    original_value = config.get_str("llm.default_model")

    # Switch to a different config section
    config.select_config("training_azure")
    new_value = config.get_str("llm.default_model")

    assert new_value != original_value


def test_runtime_modification() -> None:
    """Verify runtime modifications to config values."""
    config = OmegaConfig.singleton()
    config.set_str("test", "temp_value", "initial")
    assert config.get_str("test.temp_value") == "initial"

    # Modify the value
    config.set_str("test", "temp_value", "modified")
    assert config.get_str("test.temp_value") == "modified"


def test_missing_key() -> None:
    """Verify error handling for missing configuration keys."""
    config = OmegaConfig.singleton()
    with pytest.raises(ValueError):
        config.get_str("nonexistent.key")


def test_get_list() -> None:
    """Verify retrieval of list values from config."""
    config = OmegaConfig.singleton()

    result = config.get_list("chains", "modules")
    assert isinstance(result, list)
