"""
Entry point for the Command Line Interface


"""

import importlib
import json
import typer
from typing import Callable
from langchain_core.runnables import Runnable
from langchain.globals import set_debug, set_verbose

from python.ai_core.chain_registry import find_runnable, get_runnable_registry
from python.config import get_config


# Import modules where runnables are registered
RUNNABLES = {"lc_rag_example", "lc_tools_example", "lc_self_query"}
for r in RUNNABLES:
    importlib.import_module(f"python.ai_chains.{r}")


cli_app = typer.Typer(add_completion=True, no_args_is_help=True)


@cli_app.command()
def echo(message: str):
    print(f"you said: {message}")


@cli_app.command()
def run(
    name: str,  # name (description) of the Runnable
    input: str | None = None,  # input
    verbose: bool = False,
    debug: bool = False,
    llm: str | None = None,  # id (our name) of the LLM
):
    """
    Run a given Runnable with parameter 'input". LLM can be changed, otherwise the default one is selected.
    """
    set_debug(debug)
    set_verbose(verbose)

    runnables_list = sorted([f"'{o.name}'" for o in get_runnable_registry()])
    runnables_list_str = ", ".join(runnables_list)

    runnable_desc = find_runnable(name)
    if runnable_desc:
        if not llm:
            llm = get_config("llm", "default_model")
        if not input:
            input = runnable_desc.examples[0]

        result = runnable_desc.invoke(input, {"llm": llm})
        print(result)
    else:
        print(f"Runnable not found: '{name}'. Should be in: {runnables_list_str}")


@cli_app.command()
def chain_info(name: str):
    import pprint

    runnable_desc = find_runnable(name)
    if runnable_desc:
        r = runnable_desc.runnable
        if isinstance(r, Runnable):
            runnable = r
        elif isinstance(r, Callable):
            runnable = r({"llm": None})

        print("type: ", type(runnable))
        try:
            runnable.get_graph().print_ascii()
            print("input type:", runnable.InputType)
            print("output type:", runnable.OutputType)
            print("input schema: ", runnable.input_schema().schema())
            print("output schema: ")
            pprint.pprint(runnable.output_schema().schema(), compact=True, width=120)
        except:
            pass


if __name__ == "__main__":
    cli_app()
