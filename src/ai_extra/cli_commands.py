import asyncio
from typing import Annotated, Optional

import typer
from langchain.globals import set_debug, set_verbose

# Import modules where runnables are registered
from typer import Option

from src.ai_core.cache import LlmCache
from src.ai_core.llm import LlmFactory
from src.ai_extra.mcp_client import call_react_agent
from src.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def mcp_agent(
        input: str,
        cache: str = "memory",
        lc_verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d")] = False,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ) -> None:
        """
        Quick test
        """
        set_debug(lc_debug)
        set_verbose(lc_verbose)
        LlmCache.set_method(cache)

        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        asyncio.run(call_react_agent(input))
