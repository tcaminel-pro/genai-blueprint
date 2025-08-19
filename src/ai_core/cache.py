"""LLM caching configuration and management.

This module provides a unified interface for configuring and managing caching
mechanisms for Language Learning Models (LLMs). It supports multiple caching
methods including in-memory and SQLite-based persistent caching.

Key Features:
- Configurable through application settings
- Automatic cache directory management
- Runtime cache method switching
- Integration with LangChain's caching system

Example:
    >>> LlmCache.set_method("sqlite")  # Enable persistent SQLite caching
    >>> LlmCache.set_method("memory")  # Switch to in-memory caching
"""

from enum import Enum
from typing import Optional

from langchain_core.caches import BaseCache

from src.utils.config_mngr import global_config


class CacheMethod(Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    NO_CACHE = "no_cache"
    DEFAULT = "default"


class LlmCache:
    """A wrapper above LangChain 'set_llm_cache' to configure and select LLM cache method."""

    @classmethod
    def from_value(cls, value: str | None) -> Optional[BaseCache]:
        """ """
        from langchain_community.cache import InMemoryCache, SQLiteCache
        from loguru import logger

        if value is None:
            value = "default"

        if value == "default":
            from_config = global_config().get_str("llm.cache")
            if from_config and from_config not in LlmCache.values():
                logger.warning(f"Incorrect llm/cache configuration : {from_config}. Should be in {LlmCache.values()} ")
            value = from_config or "no_cache"

        try:
            method = CacheMethod[value.upper()].value
        except KeyError as e:
            raise ValueError(f"Unknown cache method '{value}'. Should be in {LlmCache.values()}") from e

        if method == "sqlite":
            path = global_config().get_file_path("llm.cache_path", check_if_exists=False)
            if path and path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            return SQLiteCache(database_path=str(path))
        elif method == "memory":
            return InMemoryCache()
        elif method == "no_cache":
            return None
        else:
            print(value, method)
            raise ValueError()

    @classmethod
    def values(cls) -> list[str]:
        """Returns possible cache method values."""
        return [method.name.lower() for method in CacheMethod]

    @staticmethod
    def set_method(cache: str | None) -> None:
        """Define caching method. If 'None', take the one defined in configuration. \
        Currently implemented : "memory', 'sqlite'.

        Args:
            cache (str | None): The cache method to set. If None, the default from configuration is used.

        Raises:
            logger.warning: If the default cache configuration is incorrect.
        """
        try:
            from langchain.globals import get_llm_cache, set_llm_cache
        except ImportError:
            # Fallback for older versions of langchain
            from langchain.cache import get_llm_cache, set_llm_cache
        from loguru import logger

        old_cache = get_llm_cache()

        new_cache = LlmCache.from_value(cache)
        set_llm_cache(new_cache)

        if new_cache is None and old_cache is None:
            pass  # No change
        elif new_cache is None or old_cache is None:
            logger.debug(f"LLM cache : {type(new_cache).__name__}")
        elif new_cache.__class__ != old_cache.__class__:
            logger.debug(f"""LLM cache : {str(new_cache.__class__).split(".")[-1].rstrip("'>")}""")


# LlmCache.set_method("default")  # Set default
