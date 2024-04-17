"""
Entry point for the REST API  

Copyright (C) 2024 Eviden. All rights reserved
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from devtools import debug  # type: ignore
from loguru import logger  # type: ignore

# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on


app = FastAPI()

@app.get("/echo/{message}")
def read_root(message:str):
    logger.info(f"received /echo/{message}")
    return {"msg": message}
