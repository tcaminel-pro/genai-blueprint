from langchain.globals import set_debug, set_verbose
from rich.console import Console
from rich.panel import Panel

from src.ai_core.cache import LlmCache
from src.ai_core.llm_factory import LlmFactory
from src.utils.config_mngr import global_config


def setup_langchain(llm_id: str | None, lc_debug: bool, lc_verbose: bool, cache: str):
    console = Console()
    set_debug(lc_debug)
    set_verbose(lc_verbose)
    LlmCache.set_method(cache)

    if llm_id is not None:
        if llm_id not in LlmFactory.known_items():
            console.print(
                Panel(
                    f"[red]Error:[/red] '{llm_id}' is not a valid model ID.\n"
                    f"Valid options are: {', '.join(LlmFactory.known_items())}",
                    title="Configuration Error",
                    style="red",
                )
            )
            return False
        global_config().set("llm.default_model", llm_id)
        return True
