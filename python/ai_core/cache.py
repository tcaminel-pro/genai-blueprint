"""
Helper  for LLM Cache configuration


 """

from enum import Enum

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache
from loguru import logger

from python.config import get_config_str


class LlmCache(str, Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    NONE = "none"
    # Todo : Add Postgres, Similarity, ....

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return None

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]


def set_cache(cache: LlmCache | None):
    """
    Define caching strategy.  If 'None', take the one defined in configuration
    """
    if not cache:
        default = get_config_str("llm", "cache")
        if LlmCache.from_value(default) is None:
            logger.warning(f"Incorrect llm/cache configuration : {default} ")
        cache = LlmCache.from_value(default)
    if cache == LlmCache.MEMORY:
        set_llm_cache(InMemoryCache())
    elif cache == LlmCache.SQLITE:
        set_llm_cache(
            SQLiteCache(database_path=".langchain.db")
        )  # TODO : set path in config file
    elif cache == LlmCache.NONE or cache is None:
        set_llm_cache(None)
