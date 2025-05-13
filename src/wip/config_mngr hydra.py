"""
Configuration Manager using Hydra.

This module provides a simplified interface to Hydra's configuration system.

Key Features:
- Loads configuration from config/ directory
- Supports environment variable substitution
- Provides hierarchical configuration with baseline and environment-specific overrides
- Implements singleton pattern for global access

Example Usage:
    # Get a config value
    model = global_config().get("llm.default_model")
    # Switch configurations
    global_config().select_config("training_local")
"""

from __future__ import annotations

import os
from typing import Any, Optional

from hydra import initialize, compose
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from upath import UPath

from src.utils.singleton import once


class HydraConfig(BaseModel):
    """Application configuration manager using Hydra."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cfg: DictConfig
    config_name: str = "default"

    @once
    def singleton() -> HydraConfig:
        """Returns the singleton instance of Config."""
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="default")
            cfg.paths.project = get_original_cwd()
            return HydraConfig(cfg=cfg)

    def select_config(self, config_name: str) -> None:
        """Select a different configuration."""
        with initialize(version_base=None, config_path="../config"):
            self.cfg = compose(config_name=config_name)
            self.cfg.paths.project = get_original_cwd()
            self.config_name = config_name

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value using dot notation."""
        value = OmegaConf.select(self.cfg, key)
        if value is None and default is not None:
            return default
        return value

    def get_str(self, key: str, default: Optional[str] = None) -> str:
        """Get a string configuration value."""
        value = self.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"Configuration value for '{key}' is not a string")
        return value

    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """Get a list configuration value."""
        value = self.get(key, default)
        if not isinstance(value, list):
            raise ValueError(f"Configuration value for '{key}' is not a list")
        return value

    def get_dict(self, key: str, expected_keys: list | None = None) -> dict:
        """Get a dictionary configuration value."""
        value = self.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Configuration value for '{key}' is not a dict")
        if expected_keys is not None:
            missing_keys = [k for k in expected_keys if k not in value]
            if missing_keys:
                raise ValueError(f"Missing required keys '{key}': {', '.join(missing_keys)}")
        return value

    def get_dir_path(self, key: str, create_if_not_exists: bool = False) -> UPath:
        """Get a directory path."""
        path = UPath(self.get_str(key))
        if not path.exists():
            if create_if_not_exists:
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"Directory path for '{key}' does not exist: '{path}'")
        if not path.is_dir():
            raise ValueError(f"Path for '{key}' is not a directory: '{path}'")
        return path

    def get_file_path(self, key: str, check_if_exists: bool = True) -> UPath:
        """Get a file path."""
        path = UPath(self.get_str(key))
        if not path.exists() and check_if_exists:
            raise ValueError(f"File path for '{key}' does not exist: '{path}'")
        return path


def global_config() -> HydraConfig:
    """Get the global config singleton."""
    return HydraConfig.singleton()

if __name__ == "__main__":
    config = global_config()
    print(config.get("llm.default_model"))
    config.select_config("training_local")
    print(config.get("llm.default_model"))
