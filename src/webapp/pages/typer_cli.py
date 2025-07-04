"""Streamlit page to interact with Typer CLI commands.

Provides a simple interface to:
- Enter CLI commands
- Execute them using Typer's CliRunner
- Display command output
"""

import subprocess
import sys
from io import StringIO

import streamlit as st
from typer.testing import CliRunner

# Import the CLI app from main module
from src.main.cli import cli_app


def run_typer_command(command: str) -> str:
    """Execute a Typer CLI command and capture its output.

    Args:
        command: The full command string to execute

    Returns:
        The captured stdout from the command execution
    """
    # Split command into args list
    args = command.split()

    # Create a CliRunner instance
    runner = CliRunner()

    # Capture output
    result = runner.invoke(cli_app, args)

    if result.exit_code != 0:
        return f"Error: {result.exception}\n{result.stdout}"
    return result.stdout


def main() -> None:
    """Main Streamlit page layout and interaction."""
    st.title("Typer CLI Runner")

    # Input for CLI command
    command = st.text_input("Enter CLI command", "echo Hello World")

    # Execute button
    if st.button("Run Command"):
        with st.spinner("Executing command..."):
            try:
                output = run_typer_command(command)
                st.code(output, language="text")
            except Exception as e:
                st.error(f"Command failed: {str(e)}")


if __name__ == "__main__":
    main()
