# taken from https://python-devtools.helpmanual.io/usage/#manual-install


import sys

# we don't install here for pytest as it breaks pytest, it is
# installed later by a pytest fixture

# print("in sitecustomize.py")
if not sys.argv[0].endswith("pytest"):
    import builtins

    try:
        from devtools import debug as dev_debug
    except ImportError:
        pass
    else:
        setattr(builtins, "dev_debug", dev_debug)
