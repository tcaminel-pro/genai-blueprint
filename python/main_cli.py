"""
Entry point for the Command Line Interface

Copyright (C) 2023 Eviden. All rights reserved
"""

import sys
from pathlib import Path
import typer
from langchain.globals import set_debug, set_verbose
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
)

# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on

from python.ai_chains.lg_rag_example import rag_chain


cli_app = typer.Typer(
     add_completion=True,
     no_args_is_help=True
)


@cli_app.command()
def echo(message: str):
    print(f"you said: {message}")


RUNNABLES: dict[str, Runnable] = {
    "multiply_by_2": RunnableLambda(lambda x : float(x)*2) , 
    "first_RAG": rag_chain }

@cli_app.command()
def run(name: str, input: str, verbose:bool = False, debug:bool = False) -> str: 
    set_debug(debug)
    set_verbose(verbose)

    runnable = RUNNABLES.get(name) 
    if runnable: 
        result = runnable.invoke(input)
        print(result)
    else:
        print(f"Runnable not found: {name}")


if __name__ == "__main__":
    cli_app()
