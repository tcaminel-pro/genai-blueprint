"""Configure LangChain with debug, verbose, cache, and LLM settings.

Provides utilities to setup and configure LangChain components including
debug/verbose modes, cache strategies, and LLM model selection.
"""

from langchain.globals import set_debug, set_verbose
from rich.console import Console
from rich.panel import Panel

from src.ai_core.cache import LlmCache
from src.ai_core.llm_factory import LlmFactory
from src.utils.config_mngr import global_config


def setup_langchain(
    llm_id: str | None, lc_debug: bool | None = None, lc_verbose: bool | None = None, cache: str | None = None
) -> bool:
    """Configure LangChain with the specified settings.

    Args:
        llm_id: The ID of the LLM model to use, or None to keep current model.
        lc_debug: Whether to enable LangChain debug mode.
        lc_verbose: Whether to enable LangChain verbose mode.
        cache: The cache strategy to use (e.g., 'sqlite', 'memory', 'no_cache').

    Returns:
        True if setup completed successfully, False if the specified llm_id is invalid.
    """
    console = Console()
    if lc_debug:
        set_debug(lc_debug)
    if lc_verbose:
        set_verbose(lc_verbose)
    if cache:
        LlmCache.set_method(cache)

    if llm_id is not None:
        if llm_id not in LlmFactory.known_items():
            console.print(
                Panel(
                    f"[red]Error:[/red] '{llm_id}' is not a valid model ID.\n"
                    f"Valid options are: {', '.join(LlmFactory.known_items())}",
                    title="Error",
                    style="red",
                )
            )
            return False
        global_config().set("llm.models.default", llm_id)
    return True
