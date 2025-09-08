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

from src.ai_extra.tools_smolagents.config_loader import (
    CONF_YAML_FILE,
    load_demo_config,
    process_tools_from_config,
)
from src.ai_extra.tools_smolagents.react_config_loader import (
    REACT_CONF_YAML_FILE,
    load_react_demo_config,
)
from src.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def react_agent(
        input: Annotated[str | None, typer.Option(help="Input query or '-' to read from stdin")] = None,
        mcp: Annotated[
            list[str], typer.Option(help="MCP server names to connect to (e.g. playwright, filesystem, ..)")
        ] = [],
        config: Annotated[
            Optional[str],
            Option("--config", "-c", help="Configuration name from react_agent.yaml (e.g. 'Weather', 'Web Research')"),
        ] = None,
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        chat: Annotated[bool, Option("--chat", "-s", help="Start an interactive chat with agent")] = False,
    ) -> None:
        """
        Run a ReaAct agent connected to MCP Servers.

        Examples:

        # Using MCP servers directly:
        echo "get news from atos.net web site" | uv run cli mcp-agent --mcp playwright --mcp filesystem

        # Using a predefined configuration:
        uv run cli mcp-agent --config "Weather" "What is the wind force in Toulouse?"
        uv run cli mcp-agent --config "Web Research" "Research the latest AI developments"

        Use --chat to start an interactive shell where you can send multiple prompts to the agent.
        """

        from src.ai_core.mcp_client import call_react_agent
        from src.utils.cli.langchain_setup import setup_langchain
        from src.utils.cli.langgraph_agent_shell import run_langgraph_agent_shell

        # Handle configuration loading
        demo_config = None
        config_tools = []
        config_mcp_servers = []

        if config:
            demo_config = load_react_demo_config(config)
            if demo_config is None:
                print(f"Error: Configuration '{config}' not found in {REACT_CONF_YAML_FILE}")
                return

            # Extract configuration parameters
            config_tools = demo_config.get("tools", [])
            config_mcp_servers = demo_config.get("mcp_servers", [])

            print(f"Using ReAct configuration '{config}':")
            if config_tools:
                print(f"  Tools: {', '.join(config_tools)}")
            if config_mcp_servers:
                print(f"  MCP servers: {', '.join(config_mcp_servers)}")

        # Merge MCP servers from config and command line
        final_mcp_servers = list(set(mcp + config_mcp_servers))

        setup_langchain(llm_id, lc_debug, lc_verbose, cache)

        if chat:
            asyncio.run(run_langgraph_agent_shell(llm_id, mcp_server_names=final_mcp_servers))
        else:
            if not input and not sys.stdin.isatty():
                input = sys.stdin.read()
            if not input or len(input) < 5:
                print("Error: Input parameter or something in stdin is required")
                return

            asyncio.run(call_react_agent(input, llm_id=llm_id, mcp_server_filter=final_mcp_servers))

    @cli_app.command()
    def smolagents(
        input: Annotated[str | None, typer.Option(help="Input query or '-' to read from stdin")] = None,
        tools: Annotated[list[str], Option("--tools", "-t", help="Tools to use (web_search, calculator, etc.)")] = [],
        config: Annotated[
            Optional[str],
            Option("--config", "-c", help="Configuration name from codeact_agent.yaml (e.g. 'Titanic', 'MCP')"),
        ] = None,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        imports: list[str] | None = None,
        chat: Annotated[bool, Option("--chat", "-s", help="Start an interactive shell to send prompts")] = False,
    ) -> None:
        """
        Run a SmolAgent CodeAct agent with tools.

        Examples:

        # Using tools directly:
        uv run cli smolagents --input "How many seconds would it take for a leopard at full speed to run through Pont des Arts?" -t web_search
        echo "Tell me about machine learning" | uv run cli smolagents -t web_search

        # Using a predefined configuration:
        uv run cli smolagents --config "Titanic" --input "What is the proportion of female passengers that survived?"
        uv run cli smolagents --config "MCP" --input "What is the current weather in Toulouse?"

        Use --chat to start an interactive shell where you can send multiple prompts to the agent.
        """
        from smolagents import CodeAgent, Tool
        from smolagents.default_tools import TOOL_MAPPING

        from src.ai_core.llm_factory import LlmFactory
        from src.utils.cli.langchain_setup import setup_langchain

        if not setup_langchain(llm_id):
            return

        # Handle configuration loading
        config_tools = []
        config_authorized_imports = []
        final_tools = []
        final_imports = imports or []

        if config:
            demo_config = load_demo_config(config)
            if demo_config is None:
                print(f"Error: Configuration '{config}' not found in {CONF_YAML_FILE}")
                return

            # Extract configuration parameters
            config_tools = process_tools_from_config(demo_config.get("tools", []))
            config_authorized_imports = demo_config.get("authorized_imports", [])

            print(f"Using CodeAct configuration '{config}':")
            if config_tools:
                tool_names = [getattr(t, "name", str(type(t).__name__)) for t in config_tools]
                print(f"  Tools: {', '.join(tool_names)}")
            if config_authorized_imports:
                print(f"  Authorized imports: {', '.join(config_authorized_imports)}")

            # Use config tools and imports
            final_tools.extend(config_tools)
            final_imports.extend(config_authorized_imports)

        model = LlmFactory(llm_id=llm_id).get_smolagent_model()
        available_tools = final_tools.copy()

        # Add command-line specified tools
        for tool_name in tools:
            if "/" in tool_name:
                available_tools.append(Tool.from_space(tool_name))
            else:
                if tool_name in TOOL_MAPPING:
                    available_tools.append(TOOL_MAPPING[tool_name]())
                else:
                    raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

        tool_display = tools + [getattr(t, "name", str(type(t).__name__)) for t in config_tools]
        print(f"Running agent with these tools: {tool_display}")

        if chat:
            print("Chat mode for smolagents is not yet implemented")
            return
            # asyncio.run(run_smolagent_shell(llm_id, mcp_servers=[]))
        else:
            # Handle input from --input parameter or stdin
            if not input and not sys.stdin.isatty():
                input = sys.stdin.read()
            if not input or len(input) < 5:
                print("Error: Input parameter or something in stdin is required")
                return

            agent = CodeAgent(tools=available_tools, model=model, additional_authorized_imports=final_imports)
            agent.run(input)

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
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
    ) -> None:
        """Launch a browser agent to complete a given task.

        Example:
            uv run cli browser-agent "recent news on Atos" --headless
        """
        from browser_use import Agent, BrowserSession

        from src.ai_core.llm_factory import LlmFactory, get_llm
        from src.ai_extra.browser_use_langchain import ChatLangchain

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            print(f"Error: unknown llm_id. \n Should be in {LlmFactory.known_items()}")
            return
        print(f"Running browser agent with task: {task}")
        browser_session = BrowserSession(
            headless=headless,
            window_size={"width": 800, "height": 600},
        )

        llm = ChatLangchain(chat=get_llm(llm_id=llm_id))
        agent = Agent(task=task, llm=llm, browser_session=browser_session)
        history = asyncio.run(agent.run())
        print(history.final_result())

    @cli_app.command()
    def fabric(
        pattern: Annotated[str, Option("--pattern", "-p", help="Fabric pattern name to execute")],
        verbose: Annotated[bool, Option("--verbose", "-v", help="Enable verbose output")] = False,
        debug_mode: Annotated[bool, Option("--debug", "-d", help="Enable debug mode")] = False,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        # temperature: float = 0.0,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
    ) -> None:
        """Run 'fabric' pattern on standard input.

        Pattern list is here: https://github.com/danielmiessler/fabric/tree/main/patterns
        Also described here : https://github.com/danielmiessler/fabric/blob/main/patterns/suggest_pattern/user.md

        ex: echo "artificial intelligence" | uv run cli fabric -p "create_aphorisms" --llm-id llama-70-groq
        """
        from langchain.globals import set_debug, set_verbose

        from ai_extra.chains.fabric_chain import get_fabric_chain
        from src.ai_core.llm_factory import LlmFactory

        set_debug(debug_mode)
        set_verbose(verbose)

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            print(f"Error: unknown llm_id. \n Should be in {LlmFactory.known_items()}")
            return

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
