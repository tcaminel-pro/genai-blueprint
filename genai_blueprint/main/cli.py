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

import sys

import typer
from dotenv import load_dotenv

# Import modules where runnables are registered
from genai_tk.utils.config_mngr import global_config, import_from_qualified
from genai_tk.utils.logger_factory import setup_logging
from loguru import logger

load_dotenv(verbose=True)

PRETTY_EXCEPTION = (
    False  #  Alternative : export _TYPER_STANDARD_TRACEBACK=1  see https://typer.tiangolo.com/tutorial/exceptions/
)

cli_app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=PRETTY_EXCEPTION,
)


def register_commands(cli_app: typer.Typer) -> None:
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
        level = "TRACE"
        sys.argv.remove("--logging")
    else:
        level = None

    setup_logging(level)
    modules = global_config().get_list("cli.commands", value_type=str)
    # Import and register commands from each module
    for module in modules:
        try:
            register_commands = import_from_qualified(module)
            register_commands(cli_app)
        except Exception as ex:
            logger.warning(f"Cannot load module {module}: {ex}")
            # Continue loading other modules instead of crashing

    cli_app()


if __name__ == "__main__":
    # cli_app()
    main()
