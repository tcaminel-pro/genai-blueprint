"""
Entry point for the Command Line Interface

Copyright (C) 2023 Eviden. All rights reserved
"""

import sys
from pathlib import Path
import typer
from langchain.globals import set_debug, set_verbose

# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on

from python.ai.q_and_a import create_vector_collection, question_answering


app = typer.Typer(
     add_completion=True,
     no_args_is_help=True,
#     chain=True, 
)


@app.command()
def hello(name: str):
    print(f"Hello {name}")


@app.command()
def create_vectors():
    create_vector_collection()

@app.command()
def question(query: str, verbose:bool = False, debug:bool = False) -> str: 
    set_debug(debug)
    set_verbose(verbose)

    result =  question_answering(query)
    print(result)
    return result


if __name__ == "__main__":
    app()
