"""CLI commands for AI Extra functionality.

This module provides command-line interface commands for:
- Running MCP React agents
- Executing SmolAgents with custom tools
- Processing PDF files with OCR
- Running Fabric patterns

The commands are registered with a Typer CLI application and provide:
- Input/output handling (including stdin)
- Configuration of LLM parameters
- Tool integration
- Batch processing capabilities
"""

import asyncio
import sys
from typing import Annotated, Optional

import typer
from langchain.globals import set_debug, set_verbose
from loguru import logger
from smolagents import (
    CodeAgent,
)
from smolagents.default_tools import TOOL_MAPPING

# Import modules where runnables are registered
from typer import Option
from upath import UPath

from src.ai_core.cache import LlmCache
from src.ai_core.llm import LlmFactory
from src.ai_core.mcp_client import call_react_agent, get_mcp_tools_info
from src.ai_extra.mistral_ocr import process_pdf_batch
from src.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def mcp_agent(
        input: str | None = None,
        cache: str = "memory",
        lc_verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d")] = False,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ) -> None:
        """
        Quick test
        """
        set_debug(lc_debug)
        set_verbose(lc_verbose)
        LlmCache.set_method(cache)

        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        if not input and not sys.stdin.isatty():
            input = sys.stdin.read()
        if not input or len(input) < 5:
            print("Error: Input parameter or something in stdin is required")
            return

        asyncio.run(call_react_agent(input))

    @cli_app.command()
    def smolagents(
        prompt: str,
        tools: Annotated[list[str], Option("--tools", "-t")] = [],
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
        imports: list[str] | None = None,
    ) -> None:
        """
        ex: "How many seconds would it take for a leopard at full speed to run through Pont des Arts?" -t web_search
        """
        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        model = LlmFactory(llm_id=llm_id).get_smolagent_model()
        available_tools = []
        for tool_name in tools:
            if "/" in tool_name:
                available_tools.append(Tool.from_space(tool_name))
            else:
                if tool_name in TOOL_MAPPING:
                    available_tools.append(TOOL_MAPPING[tool_name]())
                else:
                    raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

        print(f"Running agent with these tools: {tools}")
        agent = CodeAgent(tools=available_tools, model=model, additional_authorized_imports=imports)

        agent.run(prompt)

    @cli_app.command()
    def ocr_pdf(
        file_patterns: list[str] = typer.Argument(..., help="File patterns to match PDF files (glob patterns)"),  # noqa: B008
        output_dir: str = typer.Option("./ocr_output", help="Directory to save OCR results"),
        use_cache: bool = typer.Option(True, help="Use cached OCR results if available"),
        recursive: bool = typer.Option(False, help="Search for files recursively"),
    ) -> None:
        """Process PDF files with Mistral OCR and save the results as markdown files.

        Example:
            python -m src.ai_extra.mistral_ocr ocr_pdf "*.pdf" "data/*.pdf" --output-dir=./ocr_results
        """
        # Collect all PDF files matching the patterns
        all_files = []
        for pattern in file_patterns:
            path = UPath(pattern)

            # Handle glob patterns
            if "*" in pattern:
                base_dir = path.parent
                if recursive:
                    matched_files = list(base_dir.glob(f"**/{path.name}"))
                else:
                    matched_files = list(base_dir.glob(path.name))
                all_files.extend(matched_files)
            else:
                # Direct file path
                if path.exists():
                    all_files.append(path)

        # Filter for PDF files
        pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]

        if not pdf_files:
            logger.warning("No PDF files found matching the provided patterns.")
            return

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Process the files
        output_path = UPath(output_dir)
        asyncio.run(process_pdf_batch(pdf_files, output_path, use_cache))

        logger.info(f"OCR processing complete. Results saved to {output_dir}")


    @cli_app.command()
    def fabric(
        pattern: Annotated[str, Option("--pattern", "-p")],
        verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        debug_mode: Annotated[bool, Option("--debug", "-d")] = False,
        stream: Annotated[bool, Option("--stream", "-s")] = False,
        # temperature: float = 0.0,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ) -> None:
        """Run 'fabric' pattern on standard input.

        Pattern list is here: https://github.com/danielmiessler/fabric/tree/main/patterns
        Also described here : https://github.com/danielmiessler/fabric/blob/main/patterns/suggest_pattern/user.md

        ex: echo "artificial intelligence" | python python/main_cli.py fabric create_aphorisms --llm-id llama-70-groq
        """
        set_debug(debug_mode)
        set_verbose(verbose)

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            print(f"Error: unknown llm_id. \n Should be in {LlmFactory.known_items()}")
            return

        config = {"llm": llm_id if llm_id else global_config().get_str("llm.default_model")}
        chain = get_fabric_chain(config)
        input = repr("\n".join(sys.stdin))
        input = input.replace("{", "{{").replace("}", "}}")

        if stream:
            for s in chain.stream({"pattern": pattern, "input_data": input}, config):
                print(s, end="", flush=True)
                print("\n")
        else:
            result = chain.invoke({"pattern": pattern, "input_data": input}, config)
            print(result)
