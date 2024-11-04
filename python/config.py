"""
Manage the application configuration.

First we read a YAML file "app_conf.yaml".
All configuration parameters are read from "default" section.
The  configuration is then overwritten by the one on section given though the "CONFIGURATION" environment variable
(for example for a  deployment on a specific target).

Last, the configuration can be overwritten through an call to 'set_config_str' (typically for test or config through UI)

"""

import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any, Tuple, cast

import yaml
from loguru import logger

CONFIG_FILE = "app_conf.yaml"


os.environ["PWD"] = os.getcwd()  # Hack because PWD is set to a Windows path sometime in WSL

_modified_fields = defaultdict(dict)


@cache
def yaml_file_config(fn: str = CONFIG_FILE) -> Tuple[dict, dict]:
    # Read the configuration file  found either in the current directory, or its parent

    yml_file = Path.cwd() / fn
    if not yml_file.exists():
        yml_file = Path.cwd().parent / fn

    assert yml_file.exists(), f"cannot find {yml_file}"

    logger.info(f"load {yml_file}")

    with open(yml_file, "r") as f:
        data = cast(dict, yaml.safe_load(f))

    default_conf = data.get("default")

    config = os.environ.get("BLUEPRINT_CONFIG")
    if config:
        logger.info(f"Override config from env. variable: {config}")
        overridden_conf = data.get(config)

    return default_conf, overridden_conf  # type: ignore


def _get_config(group: str, key: str, default_value: Any | None = None) -> Any:
    """
    Return the value of a key, either set by 'set_config', or found in the configuration file.
    Raise an exception if key not found and if not default value is given
    """

    default_conf, overridden_conf = yaml_file_config()
    try:
        default_conf_value = default_conf[group][key]
    except Exception:
        default_conf_value = None
    try:
        overridden_conf_value = overridden_conf[group][key]
    except Exception:
        overridden_conf_value = None
    try:
        runtime_value = _modified_fields[group][key]
    except Exception:
        runtime_value = None

    if default_conf_value is None and overridden_conf_value is None:
        logger.warning(f"Error accessing configuration for {group}/{key}")
        pass

    value = runtime_value or overridden_conf_value or default_conf_value or default_value
    if value is None:
        raise ValueError(f"no key {group}/{key} in file {CONFIG_FILE}")
    return value


def get_config_str(group: str, key: str, default_value: str | None = None) -> str:
    """
    Return the value of a key of type string, either set by 'set_config', or found in the configuration file.
    If it contains an environment variable in the form $(XXX), then replace it.
    Raise an exception if key not found and if not default value is given
    """

    value = _get_config(group, key, default_value)
    if isinstance(value, str):
        # replace environment variable name by its value
        value = re.sub(r"\${(\w+)}", lambda f: os.environ.get(f.group(1), ""), value)
    else:
        raise ValueError("configuration key {group}/{key} is not a string")
    return value


def get_config_list(group: str, key: str, default_value: list[str] | None = None) -> list:
    """
    Return the value of a key of type list, either set by 'set_config', or found in the configuration file.
    Raise an exception if key not found and if not default value is given
    """
    value = _get_config(group, key, default_value)

    if isinstance(value, list):
        return value
    else:
        raise ValueError("configuration key {group}/{key} is not a string")


def set_config_str(group: str, key: str, value: str):
    """
    Add or override a key value
    """
    _modified_fields[group][key] = value
    assert get_config_str(group, key) == value
