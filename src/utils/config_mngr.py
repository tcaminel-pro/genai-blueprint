"""Configuration manager using OmegaConf for YAML-based app configuration.

Handles loading from app_conf.yaml with environment variable substitution and multiple
environments. Supports runtime overrides and implements singleton pattern.

Example:
```python
model = global_config().get("llm.default_model")
global_config().select_config("local")
global_config().set("llm.default_model", "gpt-4")
```
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from upath import UPath

from src.utils.singleton import once

load_dotenv()

# Ensure PWD is set correctly
os.environ["PWD"] = os.getcwd()  # Hack because PWD is sometime set to a Windows path in WSL


class OmegaConfig(BaseModel):
    """Application configuration manager using OmegaConf."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: DictConfig
    selected_config: str

    @property
    def selected(self) -> DictConfig:
        return self.root.get(self.selected_config)

    @once
    def singleton() -> OmegaConfig:
        """Returns the singleton instance of Config."""

        # Load main config file
        app_conf_path = Path("config/app_conf.yaml")
        if not app_conf_path.exists():
            app_conf_path = Path("config/app_conf.yaml").absolute()
        assert app_conf_path.exists(), f"cannot find config file: '{app_conf_path}'"

        config = OmegaConf.load(app_conf_path)
        assert isinstance(config, DictConfig)

        # Load and merge additional config files
        merge_files = config.get("merge", [])
        for file_path in merge_files:
            merge_path = Path(file_path)
            if not merge_path.exists():
                merge_path = Path("config") / file_path
            assert merge_path.exists(), f"cannot find config file: '{merge_path}'"

            merge_config = OmegaConf.load(merge_path)
            config = OmegaConf.merge(config, merge_config)

        OmegaConf.resolve(config)

        # Determine which config to use
        config_name_from_env = os.environ.get("BLUEPRINT_CONFIG")
        config_name_from_yaml = config.get("default_config")
        if config_name_from_env and config_name_from_env not in config:
            logger.warning(
                f"Configuration selected by environment variable 'BLUEPRINT_CONFIG' not found: {config_name_from_env}"
            )
            config_name_from_env = None
        if config_name_from_yaml and config_name_from_yaml not in config:
            logger.warning(f"Configuration selected by key 'default_config' not found: {config_name_from_yaml}")
            config_name_from_yaml = None
        selected_config = config_name_from_env or config_name_from_yaml or "baseline"
        return OmegaConfig(root=config, selected_config=selected_config)

    def select_config(self, config_name: str) -> None:
        """Select a different configuration section to override defaults."""
        if config_name not in self.root:
            logger.error(f"Configuration section '{config_name}' not found")
            raise ValueError(f"Configuration section '{config_name}' not found")
        logger.info(f"Switching to configuration section: {config_name}")
        self.selected_config = config_name

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value using dot notation.
        Args:
            key: Configuration key in dot notation (e.g., "llm.default_model")
            default: Default value if key not found
        Returns:
            The configuration value or default if not found
        """
        # Create merged config with runtime overrides first
        merged = OmegaConf.merge(self.root, self.selected)
        try:
            value = OmegaConf.select(merged, key)
            if value is None:
                if default is not None:
                    return default
                else:
                    raise ValueError(f"Configuration key '{key}' not found")
            return value
        except Exception as e:
            if default is not None:
                return default
            debug(self)
            raise ValueError(f"Configuration key '{key}' not found") from e

    def set(self, key: str, value: Any) -> None:
        """Set a runtime configuration value using dot notation.
        Args:
            key: Configuration key in dot notation (e.g., "llm.default_model")
            value: Value to set
        """
        OmegaConf.update(self.selected, key, value, merge=True)

    def get_str(self, key: str, default: Optional[str] = None) -> str:
        """Get a string configuration value."""
        value = self.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"Configuration value for '{key}' is not a string (its a {type(value)})")
        return value

    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """Get a list configuration value."""
        value = self.get(key, default)
        if not isinstance(value, ListConfig):
            raise ValueError(f"Configuration value for '{key}' is not a list (its a {type(value)})")
        return list(value)

    def get_dict(self, key: str, expected_keys: list | None = None) -> dict:
        """Get a dictionary configuration value.

        Args:
            key: Configuration key in dot notation
            expected_keys: Optional list of required keys to validate against
        Returns:
            The dictionary configuration value
        Raises:
            ValueError: If value is not a dict or if expected keys validation fails
        """
        value = self.get(key)
        if not isinstance(value, DictConfig):
            raise ValueError(f"Configuration value for '{key}' is not a dict (its a {type(value)})")
        result = dict(value)
        if expected_keys is not None:
            missing_keys = [k for k in expected_keys if k not in result]
            if missing_keys:
                raise ValueError(f"Missing required keys '{key}': {', '.join(missing_keys)}")
        return result

    def get_dir_path(self, key: str, create_if_not_exists: bool = False) -> UPath:
        """Get a directory path. Can be local or remote  (https, S3, webdav, sftp,...)

        Args:
            key: Configuration key containing the path
            create_if_not_exists: If True, create directory when missing
        Returns:
            The Path object
        Raises:
            ValueError: If path doesn't exist, is not a directory, or create_if_not_exists=False
        """
        path = UPath(self.get_str(key))
        if not path.exists():
            if create_if_not_exists:
                logger.warning(f"Creating missing directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"Directory path for '{key}' does not exist: '{path}'")
        if not path.is_dir():
            raise ValueError(f"Path for '{key}' is not a directory: '{path}'")
        return path

    def get_file_path(self, key: str, check_if_exists: bool = True) -> UPath:
        """Get a file path. Can be local or remote  (https, S3, webdav, sftp,...)"""
        path = UPath(self.get_str(key))
        if not path.exists() and check_if_exists:
            raise ValueError(f"File path for '{key}' does not exist: '{path}'")
        return path


def global_config() -> OmegaConfig:
    """Get the global config singleton."""
    return OmegaConfig.singleton()


def config_loguru() -> None:
    """
    Configure the logger.
    """

    LOGURU_FORMAT = "<cyan>{time:HH:mm:ss}</cyan>-<level>{level: <7}</level> | <magenta>{file.name}</magenta>:<green>{line} <italic>{function}</italic></green>- <level>{message}</level>"
    # Workaround "LOGURU_FORMAT" does not seems to be taken into account
    format_str = os.environ.get("LOGURU_FORMAT") or LOGURU_FORMAT
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=format_str,
        backtrace=False,
        diagnose=True,
    )


## for quick test ##
if __name__ == "__main__":
    # Get a config value
    model = global_config().get("llm.default_model")
    print(model)
    # Set a runtime override
    global_config().set("llm.default_model", "gpt-4")
    model = global_config().get("llm.default_model")
    print(model)
    print(global_config().get_list("chains.modules"))
    # Switch configurations
    global_config().select_config("training_local")
    model = global_config().get("llm.default_model")
    print(model)
    print(global_config().get_list("chains.modules"))

    global_config().set("llm.default_model", "foo")
    print(global_config().get_str("llm.default_model"))

    global_config().select_config("training_openai")
    print(global_config().get("training_openai.dummy"))

    print(global_config().root.default_config)
    print(global_config().selected.llm.default_model)
