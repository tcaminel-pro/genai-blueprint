"""
Install new builtins functions

Taken from https://python-devtools.helpmanual.io/usage/#manual-install

Note:  To make Pylance and Ruff happy:
- create a file __builtins__.pyi  with 'def debug(*args) -> None: ...'  inside
- in pyproject.toml, add 'builtins = ["ic", "debug"]'  in  '[tool.ruff]' 
"""

import sys

if not sys.argv[0].endswith("pytest"):
    import builtins

    try:
        from devtools import debug as debug
    except ImportError:
        pass
    else:
        setattr(builtins, "debug", debug)
        # debug("devtools.debug loaded")
