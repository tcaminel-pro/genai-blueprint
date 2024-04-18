"""
Read the TOML configuration file

"""

from functools import cache
import os
from pathlib import Path
import re
from typing import Any
import tomli
from collections import defaultdict

TOML_FILE_NAME = "app_conf.toml"


_config = defaultdict(dict)


@cache
def _get_conf_file(fn: str = TOML_FILE_NAME) -> dict:
    # Read the TOML file  found either in the current directory, or its parent

    toml_file = Path.cwd() / fn
    if not toml_file.exists():
        toml_file = Path.cwd().parent / fn

    assert toml_file.exists(), f"cannot find {toml_file}"

    with open(toml_file, "rb") as f:
        data = tomli.load(f)
    return data


def get_config(group: str, key: str, default_value: str | None = None) -> str:
    """
    Return the value of a key, either set by 'set_config', or found in the configuration file.
    If it contains an environment variable in the form $(XXX), then replace it.
    Raise an exeption if key not found and if not default value is given
    """
    d = merge_dicts(_get_conf_file(), _config)

    try:
        value = d[group][key]
        return re.sub(r"\${(\w+)}", lambda f: os.environ.get(f.group(1), ""), value)

    except Exception as ex:
        if default_value:
            return default_value
        else:
            raise ValueError(f"no key {group}/{key} in file {TOML_FILE_NAME}")


def set_config(group: str, key: str, value: str):
    """ """
    _config[group][key] = value


def merge_dicts(a: dict, b: dict, overide=False, path=[]):
    """Utility to merge 2 dictionnaries.
    Raise exception if same keys if not 'overide' set

    Taken from https://stackoverflow.com/a/7205107"""

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], overide, path + [str(key)])
            elif a[key] != b[key] and not overide:
                raise Exception("Conflict at " + ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
