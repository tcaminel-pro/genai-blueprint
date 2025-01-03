"""
Provides configuration and management of Language Model (LLM) caching method.
A wrapper above LangChain 'set_llm_cache' and 'cache' parameter

Key Features:
- Configurable through configuration files
- Logging of cache configuration changes
- Automatic cache directory creation for SQLite caching

"""

from pathlib import Path
from enum import Enum, auto
from devtools import debug

from langchain.globals import get_llm_cache, set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache, BaseCache
from loguru import logger

from python.config import get_config_str


class CacheMethod(Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    NO_CACHE = "no_cache" 
    DEFAULT = "default"

 
class LlmCache:
    """A wrapper above LangChain 'set_llm_cache' to configure and select LLM cache method."""

    @classmethod
    def from_value(cls, value: str) -> BaseCache | None:
        """
        """
        method = None

        if value == "default":
            from_config = get_config_str("llm", "cache")
            if from_config not in LlmCache.values():
                logger.warning(f"Incorrect llm/cache configuration : {from_config}. Should be in {LlmCache.values()} ")
            value = from_config    

        try:
            method = CacheMethod[value.upper()].value
        except KeyError as e: 
            raise ValueError(f"Unknown cache method '{value}'. Should be in {LlmCache.values()}") from e
          
        if method == "sqlite":
            path = Path(get_config_str("llm", "cache_path"))
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            return SQLiteCache(database_path=str(path))
        elif method == "memory":
            return  InMemoryCache()
        elif method == "no_cache": 
            return None
        else : 
            debug (value, method)
            raise ValueError()


    @classmethod
    def values(cls) -> list[str]:
        """
        Returns possible cache method values.
        """
        return [method.name.lower() for method in CacheMethod]

    @staticmethod
    def set_method(cache: str):
        """
        Define caching method. If 'None', take the one defined in configuration. \
        Currently implemented : "memory', 'sqlite"

        Args:
            cache (str | None): The cache method to set. If None, the default from configuration is used.

        Raises:
            logger.warning: If the default cache configuration is incorrect.
        """
        old_cache = get_llm_cache()

        new_cache = LlmCache.from_value(cache)
        set_llm_cache(new_cache)

        if new_cache.__class__ != old_cache.__class__:
            logger.info(f"""LLM cache : {str(new_cache.__class__).split('.')[-1].rstrip("'>")}""")


LlmCache.set_method("default")  # Set default
