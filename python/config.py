"""
Read the TOML configuration file

"""

import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

import yaml

CONFIG_FILE = "app_conf.yaml"


_config = defaultdict(dict)


@cache
def _get_conf_file(fn: str = CONFIG_FILE) -> dict:
    # Read the configuration file  found either in the current directory, or its parent

    toml_file = Path.cwd() / fn
    if not toml_file.exists():
        toml_file = Path.cwd().parent / fn

    assert toml_file.exists(), f"cannot find {toml_file}"

    with open(toml_file, "r") as f:
        data = yaml.safe_load(f)
    return data


def get_config(group: str, key: str, default_value: str | None = None) -> str:
    """
    Return the value of a key, either set by 'set_config', or found in the configuration file.
    If it contains an environment variable in the form $(XXX), then replace it.
    Raise an exception if key not found and if not default value is given
    """
    d = merge_dicts(dict(_get_conf_file()), _config, override=True)

    try:
        value = d[group][key]
        return re.sub(r"\${(\w+)}", lambda f: os.environ.get(f.group(1), ""), value)

    except Exception:
        if default_value:
            return default_value
        else:
            raise ValueError(f"no key {group}/{key} in file {CONFIG_FILE}")


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
