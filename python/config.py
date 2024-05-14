"""
Read the TOML configuration file

"""

import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

import yaml
from loguru import logger

CONFIG_FILE = "app_conf.yaml"


_config = defaultdict(dict)


@cache
def _get_conf_file(fn: str = CONFIG_FILE) -> dict:
    # Read the configuration file  found either in the current directory, or its parent

    yml_file = Path.cwd() / fn
    if not yml_file.exists():
        yml_file = Path.cwd().parent / fn

    assert yml_file.exists(), f"cannot find {yml_file}"

    with open(yml_file, "r") as f:
        data = yaml.safe_load(f)
    return data


def get_config(group: str, key: str, default_value: str | None = None) -> str:
    """
    Return the value of a key, either set by 'set_config', or found in the configuration file.
    If it contains an environment variable in the form $(XXX), then replace it.
    Raise an exception if key not found and if not default value is given
    """
    d = merge_dicts(dict(_get_conf_file()), _config, override=True)

    config = os.environ.get("CONFIGURATION")
    if config:
        d_conf = d.get(config)
        if d_conf is None:
            logger.warning(
                f"Environment variable CONFIGURATION='{config}', but no corresponding entry in {CONFIG_FILE}"
            )
    try:
        default_conf_value = d["default"][group][key]
    except Exception:
        default_conf_value = None
    try:
        conf_value = d_conf[group][key]  # type: ignore
    except Exception:
        conf_value = None

    value = conf_value or default_conf_value or default_value

    if value is None:
        raise ValueError(f"no key {group}/{key} in file {CONFIG_FILE}")

    return re.sub(r"\${(\w+)}", lambda f: os.environ.get(f.group(1), ""), value)


def set_config(group: str, key: str, value: str):
    """
    Add of override a key value
    """
    _config[group][key] = value


def merge_dicts(a: dict, b: dict, override=False, path=[]):
    """
    Utility to merge 2 dictionaries.
    Raise exception if same keys if not 'override' set.
    Note : dict 'a' is modified.  Use 'merge_dicts(dict(a),b)' if it's an issue

    Taken from https://stackoverflow.com/a/7205107"""

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], override, path + [str(key)])
            elif a[key] != b[key] and not override:
                raise Exception("Conflict at " + ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
