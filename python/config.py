"""
Read the TOML configuration file

"""

from functools import cache
import os
from pathlib import Path
import re
import tomli

TOML_FILE_NAME = "app_conf.toml"


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
    Return the value of a key in the configuration file.
    If it contains an environment variable in the form $(XXX), then replace it.
    Raise an exeption if key not founf and if not default value is given
    """
    d = _get_conf_file()

    try:
        value = d[group][key]
        return re.sub(r"\${(\w+)}", lambda f: os.environ.get(f.group(1), ""), value)

    except Exception as ex:
        if default_value:
            return default_value
        else:
            raise ValueError(f"no key {group}/{key} in file {TOML_FILE_NAME}")
