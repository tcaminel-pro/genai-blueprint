"""
Entry point for the Command Line Interface,  and commands


"""

import os
import sys
from pathlib import Path
from typing import Callable, Optional

import typer
from devtools import pprint
from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langchain_core.runnables import Runnable

# Import modules where runnables are registered
from typer import Option
from typing_extensions import Annotated

from python.ai_chains.fabric import get_fabric_chain
from python.ai_core.chain_registry import (
    find_runnable,
    get_runnable_registry,
    load_modules_with_chains,
)
from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.llm import LlmFactory, set_cache
from python.ai_core.vector_store import VectorStoreFactory
from python.config import get_config_str

load_dotenv(verbose=True)


load_modules_with_chains()


def define_commands(cli_app: typer.Typer):
    @cli_app.command()
    def echo(message: str):
        print(f"you said: {message}")

    @cli_app.command()
    def run(
        name: str,  # name (description) of the Runnable
        input: str | None = None,  # input
        path: Path | None = None,  # input
        cache: str = "memory",
        temperature: float = 0.0,
        stream: Annotated[bool, Option("--stream", "-s")] = False,
        verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        debug: Annotated[bool, Option("--debug", "-d")] = False,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ):
        """
        Run a given Runnable with parameter 'input". LLM can be changed, otherwise the default one is selected.
        ''cache' is prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'
        """
        set_debug(debug)
        set_verbose(verbose)
        set_cache(cache)

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            print(f"Error: unknown llm_id. \n Should be in {LlmFactory.known_items()}")
            return

        runnables_list = sorted([f"'{o.name}'" for o in get_runnable_registry()])
        runnables_list_str = ", ".join(runnables_list)

        config = {}
        runnable_desc = find_runnable(name)
        if runnable_desc:
            first_example = runnable_desc.examples[0]
            config |= {
                "llm": llm_id if llm_id else get_config_str("llm", "default_model")
            }
            if path:
                config |= {"path": path}
            elif first_example.path:
                config |= {"path": first_example.path}
            if not input:
                input = first_example.query[0]

            if stream:
                for s in runnable_desc.stream(input, config):
                    print(s, end="", flush=True)
                print("\n")
            else:
                result = runnable_desc.invoke(input, config)
                pprint(result)
        else:
            print(f"Runnable not found: '{name}'. Should be in: {runnables_list_str}")

    @cli_app.command()
    def chain_info(name: str):
        """Return information on a given chain, such as input and output schema"""
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

    @cli_app.command()
    def list_models():
        """List the LLMs, embeddings models and vector stores that we know"""
        print("factories:")
        tab = 2 * " "
        print(f"{tab}llm:")
        for model in LlmFactory.known_items():
            print(f"{tab}{tab}- {model}")
        print(f"{tab}embeddings:")
        for model in EmbeddingsFactory.known_items():
            print(f"{tab}{tab}- {model}")
        print(f"{tab}vector_store:")
        for vc in VectorStoreFactory.known_items():
            print(f"{tab}{tab}- {vc}")

    @cli_app.command()
    def fabric(
        pattern: Annotated[str, Option("--pattern", "-p")],
        verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        debug: Annotated[bool, Option("--debug", "-d")] = False,
        cache: str = "memory",
        stream: Annotated[bool, Option("--stream", "-s")] = False,
        temperature: float = 0.0,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ):
        """
        Run 'fabric' pattern on standard input

        Pattern list is here: https://github.com/danielmiessler/fabric/tree/main/patterns
        Also described here : https://github.com/danielmiessler/fabric/blob/main/patterns/suggest_pattern/user.md

        ex: echo "artificial intelligence" | python python/main_cli.py fabric create_aphorisms --llm-id llama-70-groq
        """
        set_debug(debug)
        set_verbose(verbose)
        set_cache(cache)

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            print(f"Error: unknown llm_id. \n Should be in {LlmFactory.known_items()}")
            return

        config = {"llm": llm_id if llm_id else get_config_str("llm", "default_model")}
        chain = get_fabric_chain(config)
        input = "\n".join(sys.stdin)
        if stream:
            for s in chain.stream({"pattern": pattern, "input_data": input}, config):
                print(s, end="", flush=True)
                print("\n")
        else:
            result = chain.invoke({"pattern": pattern, "input_data": input}, config)
            print(result)


if __name__ == "__main__":
    import typer

    PRETTY_EXCEPTION = False  #  Alternative : export _TYPER_STANDARD_TRACEBACK=1  see https://typer.tiangolo.com/tutorial/exceptions/

    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    cli_app = typer.Typer(
        add_completion=True,
        no_args_is_help=True,
        pretty_exceptions_enable=PRETTY_EXCEPTION,
    )
    define_commands(cli_app)

    cli_app()
