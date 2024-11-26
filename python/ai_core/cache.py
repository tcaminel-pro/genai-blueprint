"""
Provides configuration and management of Language Model (LLM) caching strategies.

Key Features:
- Supports multiple cache strategies (memory, SQLite, none)
- Configurable through configuration files
- Logging of cache configuration changes
- Automatic cache directory creation for SQLite caching

"""

from enum import Enum
from pathlib import Path

from langchain.globals import get_llm_cache, set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache
from loguru import logger

from python.config import get_config_str


class LlmCache(str, Enum):
    """
    Enum representing the different types of LLM cache strategies.
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    NONE = "none"
    # Todo : Add Postgres, Similarity, ....

    @classmethod
    def from_value(cls, value):
        """
        Returns the LlmCache member corresponding to the given value.
        If no match is found, returns None.
        """
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return None

    @classmethod
    def values(cls) -> list[str]:
        """
        Returns a list of all possible values for the LlmCache enum.
        """
        return [member.value for member in cls]

    def __repr__(self):
        """
        Returns a string representation of the LlmCache member.
        """
        return str(self.__class__).split(".")[-1].rstrip("'>")


def set_cache(cache: LlmCache | None):
    """
    Define caching strategy. If 'None', take the one defined in configuration.

    Args:
        cache (LlmCache | None): The cache strategy to set. If None, the default from configuration is used.

    Raises:
        logger.warning: If the default cache configuration is incorrect.
    """
    old_cache_cls = get_llm_cache().__class__
    if not cache:
        default = get_config_str("llm", "cache")
        if LlmCache.from_value(default) is None:
            logger.warning(f"Incorrect llm/cache configuration : {default} ")
        cache = LlmCache.from_value(default)
    if cache == LlmCache.MEMORY:
        set_llm_cache(InMemoryCache())
    elif cache == LlmCache.SQLITE:
        path = Path(get_config_str("llm", "cache_path"))
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(path)))
    elif cache == LlmCache.NONE or cache is None:
        set_llm_cache(None)
    new_cache = get_llm_cache()
    if new_cache.__class__ != old_cache_cls:
        logger.info(f"""LLM cache : {str(new_cache.__class__).split('.')[-1].rstrip("'>")}""")
