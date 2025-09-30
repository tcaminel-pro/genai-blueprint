"""Basic authentication module for the application.

Provides functionality for:
- Password hashing and verification
- User authentication against a YAML config file
- Session management for Streamlit
"""

import hashlib

import yaml
from pydantic import BaseModel

from src.utils.config_mngr import global_config


class User(BaseModel):
    """User model for authentication."""

    username: str
    password_hash: str


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = False
    users: list[User] = []


def hash_password(password: str) -> str:
    """Hash a password using SHA-256.

    Args:
        password: The plain text password to hash

    Returns:
        The hashed password as a string
    """
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to check against

    Returns:
        True if the password matches, False otherwise
    """
    return hash_password(plain_password) == hashed_password


def load_auth_config() -> AuthConfig:
    """Load authentication configuration from the config file.

    Returns:
        The authentication configuration
    """
    enabled = global_config().get_bool("auth.enabled")
    config_path = global_config().get_file_path("auth.config_file", check_if_exists=True)

    if not config_path.exists():
        return AuthConfig(enabled=False, users=[])

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            return AuthConfig(enabled=False, users=[])

        config_data["enabled"] = enabled  # Default to enabled if not specified

        return AuthConfig(**config_data)
    except Exception:
        # If there's an error loading the config, return a default config
        return AuthConfig(enabled=False, users=[])


def authenticate(username: str, password: str) -> bool:
    """Authenticate a user against the config file.

    Args:
        username: The username to authenticate
        password: The plain text password to verify

    Returns:
        True if authentication is successful, False otherwise
    """
    auth_config = load_auth_config()

    # If authentication is disabled, always return True
    if not auth_config.enabled:
        return True

    # Find the user in the config
    user = next((u for u in auth_config.users if u.username == username), None)
    if not user:
        return False

    # Verify the password
    return verify_password(password, user.password_hash)


def is_authenticated() -> bool:
    """Check if the current session is authenticated.

    Returns:
        True if authenticated, False otherwise
    """
    import streamlit as st

    # If authentication is disabled, always return True
    auth_config = load_auth_config()
    if not auth_config.enabled:
        return True

    # Check if the user is authenticated in the session
    return st.session_state.get("authenticated", False)
