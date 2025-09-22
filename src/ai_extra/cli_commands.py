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
from loguru import logger
from typer import Option
from upath import UPath

from src.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
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

        from src.ai_extra.loaders.mistral_ocr import process_pdf_batch

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
    def gpt_researcher(
        query: Annotated[str, typer.Argument(help="Research query to investigate")],
        config_name: Annotated[
            str, typer.Option("--config", "-c", help="Configuration name from gpt_researcher.yaml")
        ] = "default",
        verbose: Annotated[bool, Option("--verbose", "-v", help="Enable verbose output")] = False,
    ) -> None:
        """
        Run GPT Researcher with configuration from gpt_researcher.yaml.

        Example:
            uv run cli gpt-researcher "Latest developments in AI" --config detailed
            uv run cli gpt-researcher "Climate change impacts" --llm-id gpt-4o
        """
        from src.ai_extra.chains.gpt_researcher_chain import run_gpt_researcher

        try:
            print(f"Running GPT Researcher with config: {config_name}")
            print(f"Query: {query}")

            # Run the research
            result = asyncio.run(run_gpt_researcher(query=query, config_name=config_name, verbose=verbose))

            print("\n" + "=" * 80)
            print("RESEARCH REPORT")
            print("=" * 80)
            print(result.content)

        except Exception as e:
            print(f"Error running GPT Researcher: {str(e)}")
            if verbose:
                import traceback

                traceback.print_exc()

    @cli_app.command()
    def browser_agent(
        task: Annotated[str, typer.Argument(help="The task for the browser agent to execute")],
        headless: Annotated[bool, typer.Option(help="Run browser in headless mode")] = False,
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
    ) -> None:
        """Launch a browser agent to complete a given task.

        Example:
            uv run cli browser-agent "recent news on Atos" --headless
        """
        from browser_use import Agent, BrowserSession

        from src.ai_core.llm_factory import LlmFactory, get_llm_unified
        from src.ai_extra.browser_use_langchain import ChatLangchain

        # Validate the LLM identifier if provided
        if llm is not None:
            resolved_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm)
            if error_msg:
                print(error_msg)
                return

        print(f"Running browser agent with task: {task}")
        browser_session = BrowserSession(
            headless=headless,
            window_size={"width": 800, "height": 600},
        )

        llm_model = ChatLangchain(chat=get_llm_unified(llm=llm))
        agent = Agent(task=task, llm=llm_model, browser_session=browser_session)
        history = asyncio.run(agent.run())
        print(history.final_result())

    @cli_app.command()
    def fabric(
        pattern: Annotated[str, Option("--pattern", "-p", help="Fabric pattern name to execute")],
        verbose: Annotated[bool, Option("--verbose", "-v", help="Enable verbose output")] = False,
        debug_mode: Annotated[bool, Option("--debug", "-d", help="Enable debug mode")] = False,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        # temperature: float = 0.0,
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
    ) -> None:
        """Run 'fabric' pattern on standard input.

        Pattern list is here: https://github.com/danielmiessler/fabric/tree/main/patterns
        Also described here : https://github.com/danielmiessler/fabric/blob/main/patterns/suggest_pattern/user.md

        ex: echo "artificial intelligence" | uv run cli fabric -p "create_aphorisms" --llm-id llama-70-groq
        """
        from langchain.globals import set_debug, set_verbose

        from src.ai_core.llm_factory import LlmFactory
        from src.ai_extra.chains.fabric_chain import get_fabric_chain

        set_debug(debug_mode)
        set_verbose(verbose)

        # Validate the LLM identifier if provided and resolve to ID
        llm_id = None
        if llm is not None:
            resolved_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm)
            if error_msg:
                print(error_msg)
                return
            llm_id = resolved_id

        config = {"llm": llm_id if llm_id else global_config().get_str("llm.models.default")}
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
