"""GenAI Lab Command Line Interface.

This module provides the main entry point for the GenAI Lab CLI, offering commands to:
- Run and test LangChain Runnables with various configurations
- Get information about available chains and their schemas
- List available models (LLMs, embeddings, vector stores)
- Execute Fabric patterns for text processing
- Manage LLM configurations and caching

The CLI is built using Typer and supports:
- Interactive command completion
- Help documentation for all commands
- Configuration via environment variables and .env files
- Debug and verbose output modes
- Streaming and non-streaming execution
"""

import importlib
import sys
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from loguru import logger

# Import modules where runnables are registered
from typer import Option

from src.ai_core.llm import LlmFactory
from src.ai_extra.fabric_chain import get_fabric_chain
from src.utils.config_mngr import config_loguru, global_config

load_dotenv(verbose=True)


PRETTY_EXCEPTION = (
    False  #  Alternative : export _TYPER_STANDARD_TRACEBACK=1  see https://typer.tiangolo.com/tutorial/exceptions/
)

cli_app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=PRETTY_EXCEPTION,
)


def define_other_commands(cli_app: typer.Typer) -> None:
    """Define additional utility commands for the CLI.

    Args:
        cli_app: The Typer app instance to add commands to

    Adds commands:
    - echo: Simple command to print a message
    - fabric: Execute Fabric patterns on input text
    """

    @cli_app.command()
    def echo(message: str) -> None:
        print(message)

    @cli_app.command()
    def fabric(
        pattern: Annotated[str, Option("--pattern", "-p")],
        verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        debug_mode: Annotated[bool, Option("--debug", "-d")] = False,
        stream: Annotated[bool, Option("--stream", "-s")] = False,
        # temperature: float = 0.0,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ) -> None:
        """Run 'fabric' pattern on standard input.

        Pattern list is here: https://github.com/danielmiessler/fabric/tree/main/patterns
        Also described here : https://github.com/danielmiessler/fabric/blob/main/patterns/suggest_pattern/user.md

        ex: echo "artificial intelligence" | python python/main_cli.py fabric create_aphorisms --llm-id llama-70-groq
        """
        set_debug(debug_mode)
        set_verbose(verbose)

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            print(f"Error: unknown llm_id. \n Should be in {LlmFactory.known_items()}")
            return

        config = {"llm": llm_id if llm_id else global_config().get_str("llm.default_model")}
        chain = get_fabric_chain(config)
        input = repr("\n".join(sys.stdin))
        input = input.replace("{", "{{").replace("}", "}}")

        if stream:
            for s in chain.stream({"pattern": pattern, "input_data": input}, config):
                print(s, end="", flush=True)
                print("\n")
        else:
            result = chain.invoke({"pattern": pattern, "input_data": input}, config)
            print(result)


def main():
    # modify following  code: accept that 'module' has the form module_path:function  (like src.module1:function).
    # In that case, the function is called
    config_loguru()
    modules = global_config().get_list("commands.modules")
    # Import and register commands from each module
    for module in modules:
        try:
            mod = importlib.import_module(module)
            if hasattr(mod, "register_commands"):
                logger.info(f"register CLI commands from: '{module}'")
                mod.register_commands(cli_app)
            else:
                logger.warning(f"no 'register_commands' found for module {module} ")
        except Exception as ex:
            logger.warning(f"Cannot load module {module}: {ex}")

    # define_llm_related_commands(cli_app)
    # define_lca_related_commands(cli_app)
    define_other_commands(cli_app)

    cli_app()


if __name__ == "__main__":
    main()
