"""Streamlit page to interact with Typer CLI commands.

Provides a simple interface to:
- Enter CLI commands
- Execute them using Typer's CliRunner
- Display command output
"""

import importlib
import shlex

import streamlit as st
from devtools import debug
from loguru import logger
from typer.testing import CliRunner

from src.main.cli import cli_app, define_other_commands
from src.utils.config_mngr import global_config


def get_cli_runner() -> CliRunner:
    runner = CliRunner()

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
            logger.exception(f"Cannot load module {module}: {ex}")

    define_other_commands(cli_app)
    return runner


def run_typer_command(command: str) -> str:
    args = shlex.split(command)
    result = get_cli_runner().invoke(cli_app, args)

    if result.exit_code != 0:
        return f"Error: {result.exception}\n{result.output}"
    debug(result.output)
    return result.output


def main() -> None:
    """Main Streamlit page layout and interaction."""
    st.title("Typer CLI Runner")

    # Input for CLI command
    command = st.text_input("Enter CLI command", "echo Hello World")

    # Execute button
    if st.button("Run Command"):
        print(command)
        with st.spinner("Executing command..."):
            try:
                output = run_typer_command(command)
                st.code(output, language="text")
                print("out")
                print(output)
            except Exception as e:
                st.error(f"Command failed: {str(e)}")
                print(f"error: {e}")


if __name__ == "__main__":
    main()
