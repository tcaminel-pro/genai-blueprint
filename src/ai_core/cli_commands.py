"""CLI commands for interacting with AI Core functionality.

This module provides command-line interface commands for:
- Running LLMs directly
- Executing registered Runnable chains
- Getting information about available models and chains
- Working with embeddings

The commands are registered with a Typer CLI application and provide:
- Input/output handling (including stdin)
- Configuration of LLM parameters
- Streaming support
- Caching options
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from langchain.globals import set_debug, set_verbose
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from typer import Option

from src.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def config_info() -> None:
        """
        Display current configuration and available API keys.
        """
        from src.ai_core.llm import PROVIDER_INFO

        config = global_config()
        print(f"Selected configuration: {config.selected_config}")

        # Show available API keys
        print("\nAvailable API keys:")
        for provider, (_, key_name) in PROVIDER_INFO.items():
            if key_name and key_name in os.environ:
                print(f"  {provider}: {key_name} = {'*' * 8} (set)")

    @cli_app.command()
    def llm(
        input: Annotated[str | None, typer.Option(help="Input text or '-' to read from stdin")] = None,
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        temperature: Annotated[
            float, Option("--temperature", "--temp", min=0.0, max=1.0, help="Model temperature (0-1)")
        ] = 0.0,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
    ) -> None:
        """
        Invoke an LLM.

        input can be either taken from stdin (Unix pipe), or given with the --input param
        If runnable_name is provided, runs the specified Runnable with the given input.

        The LLM can be changed using --llm-id, otherwise the default one is selected.
        'cache' is the prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'.
        """

        from rich import print as pprint

        from src.ai_core.cache import LlmCache
        from src.ai_core.llm import LlmFactory

        set_debug(lc_debug)
        set_verbose(lc_verbose)
        LlmCache.set_method(cache)

        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        # Check if executed as part ot a pipe
        if not input and not sys.stdin.isatty():
            input = sys.stdin.read()
        if not input or len(input) < 5:
            print("Error: Input parameter or something in stdin is required")
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
        runnable_name: Annotated[str, typer.Argument(help="Name of registered Runnable to execute")],
        input: Annotated[str | None, typer.Option(help="Input text or '-' to read from stdin")] = None,
        path: Annotated[Path | None, typer.Option(help="File path input for the chain")] = None,
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        temperature: Annotated[
            float, Option("--temperature", "--temp", min=0.0, max=1.0, help="Model temperature (0-1)")
        ] = 0.0,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
    ) -> None:
        """
        Run a Runnable or directly invoke an LLM.

        If no runnable_name is provided, uses the default LLM to directly process the input, that
        can be either taken from stdin (Unix pipe), or given with the --input param
        If runnable_name is provided, runs the specified Runnable with the given input.

        The LLM can be changed using --llm-id, otherwise the default one is selected.
        'cache' is the prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'.

        \nex : uv run cli run joke --input "bears"
        """

        from devtools import pprint

        from src.ai_core.cache import LlmCache
        from src.ai_core.chain_registry import ChainRegistry
        from src.ai_core.llm import LlmFactory

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
            input = str(sys.stdin.read())
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
    def chain_info(name: Annotated[str, typer.Argument(help="Name of the chain to inspect")]) -> None:
        """
        Return information on a given chain, including input and output schema.
        """
        from typing import Callable

        from devtools import pprint

        from src.ai_core.chain_registry import ChainRegistry

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
        from src.ai_core.embeddings import EmbeddingsFactory
        from src.ai_core.llm import LlmFactory
        from src.ai_core.vector_store import VectorStoreFactory

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
    def llm_info_dump(file_name: Annotated[Path, typer.Argument(help="Output YAML file path")]) -> None:
        """
        Write a list of LLMs in YAML format to the specified file.
        """
        import yaml

        from src.ai_core.llm import LlmFactory

        data = [llm.model_dump() for llm in LlmFactory.known_list()]
        with open(file_name, "w") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

    @cli_app.command()
    def embedd(
        input: Annotated[str, typer.Argument(help="Text to embed")],
        model_id: Annotated[Optional[str], Option("--model-id", "-m", help="Embeddings model ID")] = None,
    ) -> None:
        """
        Invoke an embedder.

        ex: uv run cli embedd "string to be embedded"
        """
        from src.ai_core.embeddings import EmbeddingsFactory, get_embeddings

        if model_id is not None:
            if model_id not in EmbeddingsFactory.known_items():
                print(f"Error: {model_id} is unknown model id.\nShould be in {EmbeddingsFactory.known_items()}")
                return
            global_config().set("llm.default_model", model_id)

        embedder = get_embeddings(embeddings_id=model_id)
        vector = embedder.embed_documents([input])
        print(f"{vector[0][:20]}...")
        print(f"length: {len(vector[0])}")

    @cli_app.command()
    def list_mcp_tools(
        filter: Annotated[list[str] | None, Option("--filter", "-f", help="Filter tools by server names")] = None,
    ) -> None:
        """Display information about available MCP tools.

        Shows the list of tools from MCP servers along with their descriptions.
        Can be filtered by server names.
        """
        import asyncio

        from src.ai_core.mcp_client import get_mcp_tools_info

        async def display_tools():
            tools_info = await get_mcp_tools_info(filter)
            if not tools_info:
                print("No MCP tools found.")
                return

            for server_name, tools in tools_info.items():
                print(f"\nServer: {server_name}")
                print("-" * (len(server_name) + 8))
                for tool_name, description in tools.items():
                    print(f"  {tool_name}:")
                    print(f"    {description}")
                print()

        asyncio.run(display_tools())

    @cli_app.command()
    def similarity(
        sentences: Annotated[list[str], typer.Argument(help="List of sentences to compare (first is reference)")],
        model_id: Annotated[Optional[str], Option("--model-id", "-m", help="Embeddings model ID")] = None,
    ) -> None:
        """
        Calculate semantic similarity between sentences using cosine similarity.

        The first sentence is used as reference and compared to the others.

        Example:
            uv run cli similarity "This is a test" "This is another test" "Completely different"
        """
        from langchain_community.utils.math import cosine_similarity

        from src.ai_core.embeddings import EmbeddingsFactory, get_embeddings

        if len(sentences) < 2:
            print("Error: At least 2 sentences are required")
            return

        if model_id is not None:
            if model_id not in EmbeddingsFactory.known_items():
                print(f"Error: {model_id} is unknown model id.\nShould be in {EmbeddingsFactory.known_items()}")
                return

        embedder = get_embeddings(embeddings_id=model_id)

        # Generate embeddings for all sentences
        vectors = embedder.embed_documents(sentences)

        # Calculate similarity between first sentence and others
        reference_vector = [vectors[0]]
        other_vectors = vectors[1:]

        similarities = cosine_similarity(reference_vector, other_vectors)

        # Display results
        print("\nSimilarity scores:")
        for i, score in enumerate(similarities[0]):
            print(f"  Sentence 1 vs {i + 2}: {score:.3f}")

    @cli_app.command()
    def list_mcp_prompts(
        filter: Annotated[list[str] | None, Option("--filter", "-f", help="Filter prompts by server names")] = None,
    ) -> None:
        """Display information about available MCP prompts.

        Shows the list of prompts from MCP servers along with their descriptions.
        Can be filtered by server names.

        Example:
            uv run cli list-mcp-prompts
            uv run cli list-mcp-prompts --filter server1 server2
        """
        import asyncio

        from src.ai_core.mcp_client import get_mcp_prompts

        async def display_prompts():
            prompt_info = await get_mcp_prompts(filter)
            if not prompt_info:
                print("No MCP tools found.")
                return

            for server_name, prompts in prompt_info.items():
                print(f"\nServer: {server_name}")
                print("-" * (len(server_name) + 8))
                for name, description in prompts.items():
                    print(f"  {name}:")
                    print(f"    {description}")
                print()

        asyncio.run(display_prompts())
