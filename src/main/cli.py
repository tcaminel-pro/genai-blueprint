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

import typer
from dotenv import load_dotenv
from loguru import logger

# Import modules where runnables are registered
from src.utils.config_mngr import global_config
from src.utils.logger_factory import setup_logging

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
        """Echo the message (for test purpose)"""
        print(message)


# NOT WORKING HERE
# @cli_app.callback()
# def callback(logging: bool = False):
#     print("in callback")


def main():
    # We could fo better with Typer @cli_app.callback(), but I haven't succeded
    if "--logging" in sys.argv:
        sys.argv.remove("--logging")
    else:
        logger.disable("src")
    # print(f"in main {argc=}  {argv=}")
    setup_logging()
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
    # cli_app()
    main()
