"""
Entry point for the Command Line Interface

Copyright (C) 2023 Eviden. All rights reserved
"""

import sys
from pathlib import Path
import importlib
import typer
from langchain.globals import set_debug, set_verbose
from langchain.agents import create_tool_calling_agent, AgentExecutor


# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on

from python.ai_core.chain_registry import get_runnable_registry

RUNNABLES = {"lg_rag_example","lg_tools_example" }
for r in RUNNABLES:
    importlib.import_module(f"python.ai_chains.{r}") 


cli_app = typer.Typer(
     add_completion=True,
     no_args_is_help=True
)


@cli_app.command()
def echo(message: str):
    print(f"you said: {message}")

@cli_app.command()
def run(name: str, input: str | None = None, verbose:bool = False, debug:bool = False, llm: str | None = None): 
    set_debug(debug)
    set_verbose(verbose) 

    runnables_list = sorted([f"'{o.description}'" for o in get_runnable_registry()])
    runnables_list_str = ", ".join(runnables_list) 
    runnable_desc = next((x for x in get_runnable_registry() if x.description == name), None)

    if runnable_desc: 
        if not input: 
            input = runnable_desc.examples[0]
        if llm:
            runnable = runnable_desc.runnable.with_config(configurable={"llm": llm})
        else:
            runnable = runnable_desc.runnable
        if isinstance(runnable_desc.runnable,AgentExecutor ):
            result = runnable.invoke({"input":input})
        else:
            result = runnable.invoke(input)
        print(result)
    else:
        print(f"Runnable not found: '{name}'. Should be in: {runnables_list_str}")


if __name__ == "__main__":
    cli_app()
