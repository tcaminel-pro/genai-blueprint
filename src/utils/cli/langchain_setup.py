from langchain.globals import set_debug, set_verbose

from src.ai_core.cache import LlmCache
from src.ai_core.llm_factory import LlmFactory
from src.utils.config_mngr import global_config


# pretty print error  message with rich AI!
def setup_langchain(llm_id: str | None, lc_debug: bool, lc_verbose: bool, cache: str):
    set_debug(lc_debug)
    set_verbose(lc_verbose)
    LlmCache.set_method(cache)

    if llm_id is not None:
        if llm_id not in LlmFactory.known_items():
            print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
            return False
        global_config().set("llm.default_model", llm_id)
        return True
