"""
Entry point for the REST API  

Copyright (C) 2024 Eviden. All rights reserved
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from langserve import add_routes

from python.ai_core.chain_registry import (
    get_runnable_registry,
    load_modules_with_chains,
)

# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on

load_modules_with_chains()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)



# test at : http://localhost:8000/joke/playground/

for runnable in get_runnable_registry() :
    add_routes(
        app,
        runnable.get(),
        path="/" + runnable.name.lower()
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
