import sys
from pathlib import Path
from typing import Annotated, Callable, Optional

import typer
from devtools import pprint
from langchain.globals import set_debug, set_verbose
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# Import modules where runnables are registered
from typer import Option

from src.ai_core.cache import LlmCache
from src.ai_core.chain_registry import ChainRegistry
from src.ai_core.embeddings import EmbeddingsFactory
from src.ai_core.llm import LlmFactory
from src.ai_core.vector_store import VectorStoreFactory
from src.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def llm(
        input: str | None = None,  # input
        cache: str = "memory",
        temperature: Annotated[float, Option("--temperature", "--temp", min=0.0, max=1.0)] = 0.0,
        stream: Annotated[bool, Option("--stream", "-s")] = False,
        lc_verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d")] = False,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ) -> None:
        """
        Invoke an LLM.

        input can be either taken from stdin (Unix pipe), or given with the --input param
        If runnable_name is provided, runs the specified Runnable with the given input.

        The LLM can be changed using --llm-id, otherwise the default one is selected.
        'cache' is the prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'.
        """
        set_debug(lc_debug)
        set_verbose(lc_verbose)
        LlmCache.set_method(cache)

        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        # Check if executed as part ot a pipe
        if not input:
            input = sys.stdin.read()
        if not input or len(input) < 5:
            print("Error: Input parameter or something in stdin is required when no runnable is specified.")
            return

        llm = LlmFactory(
            llm_id=llm_id or global_config().get_str("llm.default_model"),
            json_mode=False,
            streaming=stream,
            cache=cache,
            llm_params={"temperature": temperature},
        ).get()
        chain = llm | StrOutputParser()
        if stream:
            for s in chain.stream(input):
                print(s, end="", flush=True)
            print("\n")
        else:
            result = chain.invoke(input)
            pprint(result)

    @cli_app.command()
    def run(
        runnable_name: str,  # name (description) of the Runnable
        input: str | None = None,  # input
        path: Path | None = None,  # input
        cache: str = "memory",
        temperature: Annotated[float, Option("--temperature", "--temp", min=0.0, max=1.0)] = 0.0,
        stream: Annotated[bool, Option("--stream", "-s")] = False,
        lc_verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d")] = False,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ) -> None:
        """
        Run a Runnable or directly invoke an LLM.

        If no runnable_name is provided, uses the default LLM to directly process the input, that
        can be either taken from stdin (Unix pipe), or given with the --input param
        If runnable_name is provided, runs the specified Runnable with the given input.

        The LLM can be changed using --llm-id, otherwise the default one is selected.
        'cache' is the prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'.
        """
        set_debug(lc_debug)
        set_verbose(lc_verbose)
        LlmCache.set_method(cache)

        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        # Handle input from stdin if no input parameter provided
        if not input and not sys.stdin.isatty():  # Check if stdin has data (pipe/redirect)
            input = sys.stdin.read()
            if len(input.strip()) < 3:  # Ignore very short inputs
                input = None

        chain_registry = ChainRegistry.instance()
        ChainRegistry.load_modules()
        # If runnable_name is provided, proceed with existing logic
        runnables_list = sorted([f"'{o.name}'" for o in chain_registry.get_runnable_list()])
        runnables_list_str = ", ".join(runnables_list)
        runnable_item = chain_registry.find(runnable_name)
        if runnable_item:
            first_example = runnable_item.examples[0]
            llm_args = {"temperature": temperature}
            config = {
                "llm": llm_id if llm_id else global_config().get_str("llm.default_model"),
                "llm_args": llm_args,
            }
            if path:
                config |= {"path": path}
            elif first_example.path:
                config |= {"path": first_example.path}
            if not input:
                input = first_example.query[0]
            chain = runnable_item.get().with_config(configurable=config)
        else:
            print(f"Runnable '{runnable_name}' not found in config. Should be in: {runnables_list_str}")
            return

        if stream:
            for s in chain.stream(input):
                print(s, end="", flush=True)
            print("\n")
        else:
            result = chain.invoke(input)
            pprint(result)

    @cli_app.command()
    def chain_info(name: str) -> None:
        """
        Return information on a given chain, including input and output schema.
        """
        runnable_desc = ChainRegistry.instance().find(name)
        if runnable_desc:
            r = runnable_desc.runnable
            if isinstance(r, Runnable):
                runnable = r
            elif isinstance(r, Callable):
                runnable = r({"llm": None})
            else:
                raise Exception()

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
    def list_models() -> None:
        """
        List the known LLMs, embeddings models, and vector stores.
        """
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
    def llm_info_dump(file_name: Path) -> None:
        """
        Write a list of LLMs in YAML format to the specified file.
        """
        import yaml

        data = [llm.model_dump() for llm in LlmFactory.known_list()]
        with open(file_name, "w") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
