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
from pathlib import Path
from typing import Annotated, Optional

import typer
from typer import Option
from upath import UPath

from src.ai_core.cache import LlmCache
from src.ai_core.llm import LlmFactory, get_llm
from src.utils.config_mngr import global_config


async def run_mcp_agent_shell(llm_id: str | None, server_filter: list[str] | None = None) -> None:
    """Run an interactive shell for sending prompts to an MCP agent.

    The MCP servers are started once before entering the shell loop.
    The user can type /quit to exit the shell.

    Args:
        llm_id: Optional ID of the language model to use
        server_filter: Optional list of server names to include in the agent
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout

    from src.ai_core.mcp_client import get_mcp_servers_dict
    from src.utils.langgraph import print_astream

    print(f"Starting MCP agent shell with servers: {server_filter if server_filter else 'all'}")
    print("Type /quit to exit; Use up/down arrows to navigate prompt history\n")

    # Initialize the model and MCP client once
    model = get_llm(llm_id=llm_id)
    client = MultiServerMCPClient(get_mcp_servers_dict(server_filter))
    tools = await client.get_tools()
    config = {"configurable": {"thread_id": "1"}}
    agent = create_react_agent(model, tools, checkpointer=MemorySaver())

    # Set up prompt history
    history_file = Path(".blueprint.input.history")
    session = PromptSession(history=FileHistory(str(history_file)))
    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async("> ", auto_suggest=AutoSuggestFromHistory())

            user_input = user_input.strip()
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                break
            if not user_input:
                continue
            resp = agent.astream({"messages": user_input}, config)
            await print_astream(resp)

        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Exiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def mcp_agent(
        input: Annotated[str | None, typer.Option(help="Input query or '-' to read from stdin")] = None,
        server: Annotated[
            list[str], typer.Option(help="MCP server names to connect to (e.g. playwright, filesystem, ..)")
        ] = [],
        all_servers: Annotated[bool, typer.Option(help="Connect to all configured servers")] = False,
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        shell: Annotated[bool, Option("--shell", "-s", help="Start an interactive shell to send prompts")] = False,
    ) -> None:
        """
        Run a ReaAct agent connected to MCP Servers.

        Example:

        echo "get news from atos.net web site" | uv run cli mcp-agent --server playwright --server filesystem

        Use --shell to start an interactive shell where you can send multiple prompts to the agent.
        """
        from langchain.globals import set_debug, set_verbose

        from src.ai_core.mcp_client import call_react_agent

        set_debug(lc_debug)
        set_verbose(lc_verbose)
        LlmCache.set_method(cache)

        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        if shell:
            asyncio.run(run_mcp_agent_shell(llm_id, None if all_servers else server))
        else:
            if not input and not sys.stdin.isatty():
                input = sys.stdin.read()
            if not input or len(input) < 5:
                print("Error: Input parameter or something in stdin is required")
                return

            asyncio.run(call_react_agent(input, llm_id=llm_id, mcp_server_filter=server))

    @cli_app.command()
    def smolagents(
        prompt: Annotated[str, typer.Argument(help="The prompt for the agent to execute")],
        tools: Annotated[list[str], Option("--tools", "-t", help="Tools to use (web_search, calculator, etc.)")] = [],
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        imports: list[str] | None = None,
    ) -> None:
        """
        Run a Smolagent agent possibly having tools.

        ex: uv run cli smolagents "How many seconds would it take for a leopard at full speed to run through Pont des Arts?" -t web_search
        """
        from smolagents import CodeAgent, Tool
        from smolagents.default_tools import TOOL_MAPPING

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
        from loguru import logger

        from src.ai_extra.mistral_ocr import process_pdf_batch

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
        from src.ai_extra.gpt_researcher_chain import run_gpt_researcher

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

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            print(f"Error: unknown llm_id. \n Should be in {LlmFactory.known_items()}")
            return
        print(f"Running browser agent with task: {task}")
        browser_session = BrowserSession(
            headless=headless,
            window_size={"width": 800, "height": 600},
        )

        llm = get_llm(llm_id=llm_id)
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

        from src.ai_extra.fabric_chain import get_fabric_chain

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

    @cli_app.command()
    def extract_projects(
        file_patterns: list[str] = typer.Argument(..., help="File patterns to match Markdown files (glob patterns)"),
        output_dir: str = typer.Option("./extracted_projects", help="Directory to save JSON results"),
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        recursive: bool = typer.Option(False, help="Search for files recursively"),
        use_cache: bool = typer.Option(True, help="Use cached LLM responses if available"),
        batch_size: int = typer.Option(5, help="Number of files to process in each batch"),
    ) -> None:
        """Extract structured project data from Markdown files and save as JSON.

        Example:
            uv run cli extract-projects "*.md" "projects/*.md" --output-dir=./json_output --llm-id gpt-4o
            uv run cli extract-projects "**/*.md" --recursive --output-dir=./data
        """
        import asyncio
        from pathlib import Path
        
        from loguru import logger
        from upath import UPath
        
        from src.ai_core.llm import get_llm
        from src.ai_core.prompts import def_prompt
        from src.demos.ekg.rainbow_model import RainbowProjectAnalysis
        
        if llm_id is not None and llm_id not in LlmFactory.known_items():
            print(f"Error: unknown llm_id. \n Should be in {LlmFactory.known_items()}")
            return
        
        # Setup configuration
        LlmCache.set_method("sqlite" if use_cache else "no_cache")
        if llm_id:
            global_config().set("llm.default_model", llm_id)
        
        # Collect all Markdown files matching the patterns
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
        
        # Filter for Markdown files
        md_files = [f for f in all_files if f.suffix.lower() in [".md", ".markdown"]]
        
        if not md_files:
            logger.warning("No Markdown files found matching the provided patterns.")
            return
        
        logger.info(f"Found {len(md_files)} Markdown files to process")
        
        # Process the files
        output_path = UPath(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        asyncio.run(process_markdown_batch(md_files, output_path, batch_size))
        
        logger.info(f"Project extraction complete. Results saved to {output_dir}")

async def process_markdown_batch(
    md_files: list[UPath], 
    output_dir: UPath, 
    batch_size: int = 5
) -> None:
    """Process a batch of markdown files using LangChain batching."""
    from langchain_core.runnables import RunnableLambda
    from tqdm.asyncio import tqdm_asyncio
    
    from src.ai_core.prompts import def_prompt
    
    # Setup LLM with structured output
    llm = get_llm(temperature=0.0).with_structured_output(RainbowProjectAnalysis)
    
    system = """
    Extract structured information from a project review file. Analyse the files to extracts a comprehensive 360° snapshot of a project: who is involved (team, customer, partners),
    what is being delivered (scope, objectives, technologies), when (start/end dates), where (locations, business lines), how        
    (bidding strategy, pricing, risk mitigation), why (success metrics, differentiators), and how much (TCV, revenue, margin). 

    Details to be extracted is in JSON schema. Answer with a JSON document. Always fill all required fields, possibly with empty list, dict or str.
    """
    
    user = """
    project review file: 
    ---
    {file}
    ---
    """
    
    chain = def_prompt(system=system, user=user) | llm
    
    # Prepare inputs for batch processing
    file_contents = []
    valid_files = []
    
    for file_path in md_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            file_contents.append({"file": content})
            valid_files.append(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    if not file_contents:
        logger.warning("No valid files to process")
        return
    
    # Process in batches
    logger.info(f"Processing {len(valid_files)} files in batches of {batch_size}")
    
    for i in range(0, len(file_contents), batch_size):
        batch_files = valid_files[i:i+batch_size]
        batch_inputs = file_contents[i:i+batch_size]
        
        try:
            # Process batch
            results = await chain.abatch(batch_inputs)
            
            # Save results
            for file_path, result in zip(batch_files, results):
                output_file = output_dir / f"{file_path.stem}_extracted.json"
                output_file.write_text(result.model_dump_json(indent=2))
                logger.info(f"Saved: {output_file}")
                
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Process files individually on batch failure
            for file_path, input_data in zip(batch_files, batch_inputs):
                try:
                    result = await chain.ainvoke(input_data)
                    output_file = output_dir / f"{file_path.stem}_extracted.json"
                    output_file.write_text(result.model_dump_json(indent=2))
                    logger.info(f"Saved (individual): {output_file}")
                except Exception as e2:
                    logger.error(f"Error processing {file_path}: {e2}")
