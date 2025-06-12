"""CLI commands for authentication management.

This module provides command-line interface commands for:
- Hashing passwords for authentication
- Managing authentication configuration
- Adding, removing, and listing users
- Enabling or disabling authentication
"""
from typing import Annotated

import typer
from typer import Option

from src.ai_core.auth import User, hash_password, load_auth_config, save_auth_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command("hash-password")
    def hash_password_cmd(
        password: Annotated[str, typer.Argument(help="Password to hash")],
    ) -> None:
        """
        Hash a password for use in the authentication config.

        The hashed password can be added to the auth.yaml file.
        """
        hashed = hash_password(password)
        print(f"Hashed password: {hashed}")

    @cli_app.command()
    def auth_config(
        enable: Annotated[bool, Option("--enable", "-e", help="Enable or disable authentication")] = None,
        add_user: Annotated[bool, Option("--add-user", "-a", help="Add a new user")] = False,
        remove_user: Annotated[str, Option("--remove-user", "-r", help="Remove a user by username")] = None,
        list_users: Annotated[bool, Option("--list", "-l", help="List all users")] = False,
    ) -> None:
        """
        Manage authentication configuration.

        This command allows you to:
        - Enable or disable authentication
        - Add new users
        - Remove existing users
        - List all configured users
        """
        # Load the current config
        auth_config = load_auth_config()

        # Enable or disable authentication
        if enable is not None:
            auth_config.enabled = enable
            save_auth_config(auth_config)
            print(f"Authentication {'enabled' if enable else 'disabled'}")

        # Add a new user
        if add_user:
            username = typer.prompt("Username")
            password = typer.prompt("Password", hide_input=True)
            confirm = typer.prompt("Confirm password", hide_input=True)

            if password != confirm:
                print("Passwords do not match")
                return

            # Check if user already exists
            if any(u.username == username for u in auth_config.users):
                # Update existing user
                for user in auth_config.users:
                    if user.username == username:
                        user.password_hash = hash_password(password)
                        break
                print(f"Updated user: {username}")
            else:
                # Add new user
                auth_config.users.append(User(username=username, password_hash=hash_password(password)))
                print(f"Added user: {username}")

            save_auth_config(auth_config)

        # Remove a user
        if remove_user:
            auth_config.users = [u for u in auth_config.users if u.username != remove_user]
            save_auth_config(auth_config)
            print(f"Removed user: {remove_user}")

        # List all users
        if list_users:
            if not auth_config.users:
                print("No users configured")
            else:
                print(f"Authentication: {'enabled' if auth_config.enabled else 'disabled'}")
                print("Configured users:")
                for user in auth_config.users:
                    print(f"  - {user.username}")
