"""Unit tests for the cache.py module."""

from unittest.mock import MagicMock, patch

import pytest

from src.ai_core.cache import CacheMethod, LlmCache


class TestCacheMethod:
    """Test cases for CacheMethod enum."""

    def test_cache_method_values(self) -> None:
        """Test that CacheMethod enum has expected values."""
        assert CacheMethod.MEMORY.value == "memory"
        assert CacheMethod.SQLITE.value == "sqlite"
        assert CacheMethod.NO_CACHE.value == "no_cache"
        assert CacheMethod.DEFAULT.value == "default"


class TestLlmCache:
    """Test cases for LlmCache class."""

    def test_values_method(self) -> None:
        """Test that values() returns correct cache method values."""
        values = LlmCache.values()
        assert "memory" in values
        assert "sqlite" in values
        assert "no_cache" in values
        assert "default" in values
        assert len(values) == 4

    @patch("src.ai_core.cache.global_config")
    def test_from_value_memory(self, mock_config: MagicMock) -> None:
        """Test from_value with memory cache."""
        result = LlmCache.from_value("memory")
        assert result is not None
        assert str(type(result)) == "<class 'langchain_community.cache.InMemoryCache'>"

    @patch("src.ai_core.cache.global_config")
    @patch("pathlib.Path.mkdir")
    @patch("langchain_community.cache.SQLiteCache")
    def test_from_value_sqlite(
        self, mock_sqlite_cache: MagicMock, mock_mkdir: MagicMock, mock_config: MagicMock
    ) -> None:
        """Test from_value with sqlite cache."""
        mock_path = MagicMock()
        mock_config.return_value.get_file_path.return_value = mock_path
        mock_sqlite_instance = MagicMock()
        mock_sqlite_cache.return_value = mock_sqlite_instance

        result = LlmCache.from_value("sqlite")
        assert result == mock_sqlite_instance

    @patch("src.ai_core.cache.global_config")
    def test_from_value_no_cache(self, mock_config: MagicMock) -> None:
        """Test from_value with no_cache."""
        result = LlmCache.from_value("no_cache")
        assert result is None

    @patch("src.ai_core.cache.global_config")
    def test_from_value_default(self, mock_config: MagicMock) -> None:
        """Test from_value with default cache."""
        mock_config.return_value.get_str.return_value = "memory"
        result = LlmCache.from_value("default")
        assert result is not None

    @patch("src.ai_core.cache.global_config")
    def test_from_value_invalid(self, mock_config: MagicMock) -> None:
        """Test from_value with invalid cache method."""
        with pytest.raises(ValueError, match="Unknown cache method 'invalid'"):
            LlmCache.from_value("invalid")

    @patch("src.ai_core.cache.global_config")
    def test_from_value_invalid_default_config(self, mock_config: MagicMock) -> None:
        """Test from_value with invalid default config."""
        mock_config.return_value.get_str.return_value = "invalid"
        mock_config.return_value.get_str.side_effect = None

        with pytest.raises(ValueError, match="Unknown cache method 'invalid'"):
            LlmCache.from_value("default")

    @patch("langchain.globals.set_llm_cache")
    @patch("langchain.globals.get_llm_cache")
    def test_set_method_memory(self, mock_get_cache: MagicMock, mock_set_cache: MagicMock) -> None:
        """Test set_method with memory cache."""
        mock_get_cache.return_value.__class__ = type("DifferentCache", (), {})
        LlmCache.set_method("memory")
        mock_set_cache.assert_called_once()
        # Verify the call was made with an InMemoryCache instance
        call_args = mock_set_cache.call_args[0]
        assert "InMemoryCache" in str(type(call_args[0]))

    @patch("langchain.globals.set_llm_cache")
    @patch("langchain.globals.get_llm_cache")
    def test_set_method_sqlite(self, mock_get_cache: MagicMock, mock_set_cache: MagicMock) -> None:
        """Test set_method with sqlite cache."""
        mock_get_cache.return_value.__class__ = type("DifferentCache", (), {})
        LlmCache.set_method("sqlite")
        mock_set_cache.assert_called_once()

    @patch("langchain.globals.set_llm_cache")
    @patch("langchain.globals.get_llm_cache")
    def test_set_method_no_change(self, mock_get_cache: MagicMock, mock_set_cache: MagicMock) -> None:
        """Test set_method when cache type doesn't change."""
        # Mock that the cache class remains the same
        mock_get_cache.return_value.__class__ = type("InMemoryCache", (), {})
        LlmCache.set_method("memory")
        # Should still be called, but logger won't log change
        mock_set_cache.assert_called_once()

    def test_set_method_none_uses_config(self) -> None:
        """Test set_method with None uses config."""
        # This is more of an integration test, so we'll just verify it doesn't crash
        LlmCache.set_method(None)  # Should use config default
