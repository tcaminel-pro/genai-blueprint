"""
Entry point for the Command Line Interface


"""

import importlib
from typing import Callable

import typer
from devtools import pprint
from langchain.globals import set_debug, set_verbose
from langchain_core.runnables import Runnable

from python.ai_core.chain_registry import find_runnable, get_runnable_registry
from python.ai_core.llm import set_cache
from python.config import get_config

# Import modules where runnables are registered
RUNNABLES = {
    "lc_rag_example",
    "lc_tools_example",
    "lc_self_query",
    "lc_advanced_rag_langgraph",
}
for r in RUNNABLES:
    importlib.import_module(f"python.ai_chains.{r}")


def define_commands(cli_app: typer.Typer):
    @cli_app.command()
    def echo(message: str):
        print(f"you said: {message}")

    @cli_app.command()
    def run(
        name: str,  # name (description) of the Runnable
        input: str | None = None,  # input
        verbose: bool = False,
        debug: bool = False,
        cache: str = "sqlite",
        llm: str | None = None,  # id (our name) of the LLM
    ):
        """
        Run a given Runnable with parameter 'input". LLM can be changed, otherwise the default one is selected.
        ''cache' is prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'
        """
        set_debug(debug)
        set_verbose(verbose)
        set_cache(cache)

        runnables_list = sorted([f"'{o.name}'" for o in get_runnable_registry()])
        runnables_list_str = ", ".join(runnables_list)

        runnable_desc = find_runnable(name)
        if runnable_desc:
            if not llm:
                llm = get_config("llm", "default_model")
            if not input:
                input = runnable_desc.examples[0]

            result = runnable_desc.invoke(input, {"llm": llm})
            pprint(result)
        else:
            print(f"Runnable not found: '{name}'. Should be in: {runnables_list_str}")

    @cli_app.command()
    def chain_info(name: str):
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
                pprint(runnable.output_schema().schema())
            except Exception:
                pass


if __name__ == "__main__":
    # _TYPER_STANDARD_TRACEBACK=1

    PRETTY_EXCEPTION = True  #  Alternative : export _TYPER_STANDARD_TRACEBACK=1  see https://typer.tiangolo.com/tutorial/exceptions/

    cli_app = typer.Typer(
        add_completion=True,
        no_args_is_help=True,
        pretty_exceptions_enable=PRETTY_EXCEPTION,
    )
    define_commands(cli_app)

    cli_app()
