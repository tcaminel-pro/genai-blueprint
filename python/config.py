"""Configuration Manager for the application.

This module handles loading and managing application configuration from a YAML file (app_conf.yaml).
The configuration supports environment variable substitution and multiple environments.

Key Features:
- Loads configuration from app_conf.yaml in current directory or parent directory
- Supports environment variable substitution in config values (e.g., ${HOME})
- Provides hierarchical configuration with default and environment-specific overrides
- Allows runtime configuration modifications
- Implements singleton pattern for global access

Configuration File Structure:
- 'default' section contains base configuration
- Additional sections provide environment-specific overrides
- Environment variables can be used in values (e.g., ${HOME}/path)

Environment Variables:
- BLUEPRINT_CONFIG: Selects which configuration section to use as override
- Other variables can be referenced in config values

Example Usage:
    # Get a config value
    model = global_config().get_str("llm", "default_model")

    # Access singleton config
    config = Config.singleton()
    config.select_config("local")  # Switch to local config
"""

import os
import re
import sys
from collections import ChainMap, defaultdict
from pathlib import Path
from typing import Any, Dict, cast

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from python.utils.singleton import once

load_dotenv(verbose=True, override=True)


CONFIG_FILE = "app_conf.yaml"


os.environ["PWD"] = os.getcwd()  # Hack because PWD is sometime set to a Windows path in WSL


class Config(BaseModel):
    """Application configuration manager.

    Example usage:
        # Get the singleton instance
        config = Config.singleton()

        # Access configuration values
        model = config.get_str("llm", "default_model")

        # Modify configuration at runtime
        config.set_str("llm", "default_model", "gpt-4")

        # Switch to a different configuration section
        config.select_config("production")

        # Access list values
        models = config.get_list("llm", "available_models")
    """

    raw_config: dict[str, dict[str, Any]] = {}
    selected_config_name: str = "default"
    _modified_fields: dict[str, dict[str, Any]] = defaultdict(dict)

    @once()
    def singleton() -> "Config":
        """Returns the singleton instance of Config."""
        """Load configuration from YAML file in current dir or its parent."""
        yml_file = Path.cwd() / CONFIG_FILE
        if not yml_file.exists():
            yml_file = Path.cwd().parent / CONFIG_FILE

        assert yml_file.exists(), f"cannot find {yml_file}"
        logger.info(f"load {yml_file}")

        with open(yml_file) as f:
            data = cast(dict, yaml.safe_load(f))

        config_name = os.environ.get("BLUEPRINT_CONFIG", "default")
        if "BLUEPRINT_CONFIG" in os.environ:
            logger.info(f"Configuration section selected by BLUEPRINT_CONFIG: {config_name}")

        # Validate that the config section exists
        if config_name != "default" and config_name not in data:
            logger.warning(f"Configuration section '{config_name}' not found in {yml_file}. Continue with default")
        # raise ValueError(f"Configuration section '{config_name}' not found")

        return Config(raw_config=data, selected_config_name=config_name)

    @property
    def default_config(self) -> dict[str, dict[str, Any]]:
        """Get the default configuration section."""
        return self.raw_config.get("default", {})

    @property
    def overridden_config(self) -> dict[str, dict[str, Any]]:
        """Get the currently active overridden configuration section."""
        if self.selected_config_name == "default":
            return {}
        if result := self.raw_config.get(self.selected_config_name):
            return result
        else:
            return self.default_config

    def select_config(self, config_name: str) -> None:
        """Select a different configuration section to override defaults."""
        if config_name not in self.raw_config:
            logger.error(f"Configuration section '{config_name}' not found")
            raise ValueError(f"Configuration section '{config_name}' not found")
        logger.info(f"Switching to configuration section: {config_name}")
        self.selected_config_name = config_name

    def _get_config(self, group: str, key: str, default_value: Any | None = None) -> Any:
        """Return the value of a key using ChainMap to query runtime, overridden, and default config.
        Raise an exception if key not found and no default value is given.
        """
        # Create a ChainMap with runtime modifications first, then overridden, then default

        config_map = ChainMap(
            self._modified_fields.get(group, {}),
            self.overridden_config.get(group, {}),
            self.default_config.get(group, {}),
        )

        value = config_map.get(key, default_value)
        if value is None:
            logger.warning(f"Error accessing configuration for {group}/{key}")
            raise ValueError(f"no key {group}/{key} in file {CONFIG_FILE}")
        return value

    # def select_config(self, config: str):
    #     overridden_config = data.get(config, {})
    #     else:
    #         overridden_config = {}

    def get_str(self, group: str, key: str, default_value: str | None = None) -> str:
        """Return the value of a key of type string.
        If it contains an environment variable in the form $(XXX), replace it.
        Raise an exception if key not found and no default value is given.

        Example:
            # In config file:
            # default:
            #   paths:
            #     data: "${HOME}/data"

            # In code:
            data_path = config.get_str("paths", "data")
            # If HOME=/user, returns "/user/data"
        """
        value = self._get_config(group, key, default_value)
        if isinstance(value, str):
            # replace environment variable name by its value
            value = re.sub(r"\${(\w+)}", lambda f: os.environ.get(f.group(1), ""), value)
        else:
            raise ValueError(f"configuration key {group}/{key} is not a string")
        return value

    def get_list(self, group: str, key: str, default_value: list[str] | None = None) -> list:
        """Return the value of a key of type list.
        Raise an exception if key not found and no default value is given.

        Example:
            # In config file:
            # default:
            #   llm:
            #     models: ["gpt-3.5", "gpt-4"]

            # In code:
            models = config.get_list("llm", "models")
            # Returns ["gpt-3.5", "gpt-4"]
        """
        value = self._get_config(group, key, default_value)
        if isinstance(value, list):
            return value
        raise ValueError(f"configuration key {group}/{key} is not a list")

    def set_str(self, group: str, key: str, value: str) -> None:
        """Add or override a key value in the runtime configuration."""
        self._modified_fields[group][key] = value


def global_config() -> Config:
    ### Get the global config singleton ###
    return Config.singleton()


def config_loguru() -> None:
    """Configure the logger."""
    # @TODO: Set config in config file

    FORMAT_STR = "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=FORMAT_STR,
        backtrace=False,
        diagnose=True,
    )


## for quick test ##
if __name__ == "__main__":
    global_config().select_config("training_azure")
    llm = global_config().get_str("llm", "default_model")
    print(llm)

    config = Config.singleton()
    config.set_str("llm", "default_model", "another_llm")
    llm = global_config().get_str("llm", "default_model")
    print(llm)

    llm_list = global_config().get_str("llm", "list")
    print(llm_list)
