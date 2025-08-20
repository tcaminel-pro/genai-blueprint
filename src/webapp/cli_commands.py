"""CLI commands for basic authentication management.

This module provides command-line interface commands for:
- Hashing passwords for authentication

"""

from typing import Annotated

import typer


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command("hash-password")
    def hash_password_cmd(
        password: Annotated[str, typer.Argument(help="Password to hash")],
    ) -> None:
        """
        Hash a password for use in the authentication config.

        The hashed password can be added to the auth.yaml file.
        """
        from src.utils.basic_auth import hash_password

        hashed = hash_password(password)
        print(f"Hashed password: {hashed}")

    @cli_app.command("smolagent-shell")
    def smolagent_shell(
        llm_id: Annotated[str, typer.Argument(help="LLM model ID to use")],
        mcp_servers: Annotated[list[str], typer.Option(help="MCP servers to connect to")] = [],
    ) -> None:
        """Start an interactive SmolAgents shell session"""
        from src.utils.cli.smolagents_shell import run_smollagent_shell
        import asyncio

        asyncio.run(run_smollagent_shell(llm_id, mcp_server_names=mcp_servers))
        """
        Hash a password for use in the authentication config.

        The hashed password can be added to the auth.yaml file.
        """
        from src.utils.basic_auth import hash_password

        hashed = hash_password(password)
        print(f"Hashed password: {hashed}")
