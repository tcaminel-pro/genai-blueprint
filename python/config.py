"""
Manage the application configuration. 

First we read a YAML file "app_conf.yaml". 
All configuration parameters are read from "default" section.
The  configuration is then overwritten by the one on section given though the "CONFIGURATION" encironment variable
(for example for a  deployment on a specific target).

Last, the configuration can be overwritten through an call to 'set_config_str' (typically for test or config through UI)

"""

import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

CONFIG_FILE = "app_conf.yaml"


_modified_fields = defaultdict(dict)


@cache
def yaml_file_config(fn: str = CONFIG_FILE) -> dict:
    # Read the configuration file  found either in the current directory, or its parent

    yml_file = Path.cwd() / fn
    if not yml_file.exists():
        yml_file = Path.cwd().parent / fn

    assert yml_file.exists(), f"cannot find {yml_file}"

    with open(yml_file, "r") as f:
        data = yaml.safe_load(f)
    return data


def _get_config(group: str, key: str, default_value: Any | None = None) -> Any:
    """
    Return the value of a key, either set by 'set_config', or found in the configuration file.
    Raise an exception if key not found and if not default value is given
    """

    d = yaml_file_config()
    try:
        default_conf_value = d["default"][group][key]
    except Exception:
        default_conf_value = None

    config = os.environ.get("CONFIGURATION")
    try:
        conf_value = None
        if config:
            config_specific = d.get(config)
            if config_specific is None:
                logger.warning(
                    f"Environment variable CONFIGURATION='{config}', but no corresponding entry in {CONFIG_FILE}"
                )
            conf_value = config_specific[group][key]  # type: ignore
    except Exception:
        pass

    try:
        runtime_value = _modified_fields[group][key]  # type: ignore
    except Exception:
        runtime_value = None

    value = runtime_value or conf_value or default_conf_value or default_value
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


def get_config_list(
    group: str, key: str, default_value: list[str] | None = None
) -> list:
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
