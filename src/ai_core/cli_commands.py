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
from dotenv import load_dotenv
from typer import Option

from src.utils.config_mngr import global_config

load_dotenv()


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def config_info() -> None:
        """
        Display current configuration, available LLM tags, and API keys.
        """

        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        from src.ai_core.embeddings_factory import EmbeddingsFactory
        from src.ai_core.llm_factory import PROVIDER_INFO, LlmFactory
        from src.ai_core.vector_store_factory import VectorStoreFactory

        config = global_config()
        console = Console()

        console.print(Panel(f"[bold blue]Selected configuration:[/bold blue] {config.selected_config}", expand=False))

        # Default models info
        default_llm = LlmFactory(llm_id=None)
        default_embeddings = EmbeddingsFactory(embeddings_id=None)
        default_vector_store = VectorStoreFactory(id=None, embeddings_factory=default_embeddings)

        models_table = Table(title="Default Components", show_header=True, header_style="bold magenta")
        models_table.add_column("Type", style="cyan")
        models_table.add_column("Model ID", style="green")

        models_table.add_row("LLM", str(default_llm.llm_id))
        models_table.add_row("Embeddings", str(default_embeddings.embeddings_id))
        models_table.add_row("Vector-store", str(default_vector_store.id))

        console.print(models_table)

        # LLM Tags info
        tags_table = Table(title="Available LLM Tags", show_header=True, header_style="bold magenta")
        tags_table.add_column("Tag", style="cyan")
        tags_table.add_column("LLM ID", style="green")
        tags_table.add_column("Status", style="yellow")

        # Get all LLM tags from config under llm.models.*
        llm_models_config = config.get("llm.models", {})
        # Handle both regular dict and OmegaConf DictConfig
        if llm_models_config and hasattr(llm_models_config, "items"):
            for tag, llm_id in llm_models_config.items():
                if tag != "default":  # Skip the default entry as it's shown above
                    # Check if the LLM ID is available (has API keys and module)
                    status = (
                        "[green]✓ available[/green]"
                        if llm_id in LlmFactory.known_items()
                        else "[red]✗ unavailable[/red]"
                    )
                    tags_table.add_row(tag, str(llm_id), status)

        console.print(tags_table)

        # API keys info
        keys_table = Table(title="Available API Keys", show_header=True, header_style="bold magenta")
        keys_table.add_column("Provider", style="cyan")
        keys_table.add_column("Environment Variable", style="green")
        keys_table.add_column("Status", style="yellow")

        for provider, (_, key_name) in PROVIDER_INFO.items():
            if key_name:
                status = "[green]✓ set[/green]" if key_name in os.environ else "[red]✗ not set[/red]"
                keys_table.add_row(provider, key_name, status)

        console.print(keys_table)
        
        # Deep Agents info
        from src.ai_core.deep_agents import deep_agent_factory
        
        agents_table = Table(title="Deep Agents", show_header=True, header_style="bold magenta")
        agents_table.add_column("Type", style="cyan")
        agents_table.add_column("Description", style="green")
        agents_table.add_column("Features", style="yellow")
        
        agents_table.add_row(
            "Research",
            "Comprehensive research with web search",
            "Planning, Search, Notes, Reports"
        )
        agents_table.add_row(
            "Coding",
            "Write, debug, and refactor code",
            "Sub-agents, Testing, Documentation"
        )
        agents_table.add_row(
            "Analysis",
            "Data analysis and insights",
            "Exploration, Patterns, Reports"
        )
        agents_table.add_row(
            "Custom",
            "User-defined agent with custom instructions",
            "Fully configurable"
        )
        
        console.print(agents_table)

    @cli_app.command()
    def llm(
        input: Annotated[str | None, typer.Option(help="Input text or '-' to read from stdin")] = None,
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        temperature: Annotated[
            float, Option("--temperature", "--temp", min=0.0, max=1.0, help="Model temperature (0-1)")
        ] = 0.0,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        raw: Annotated[bool, Option("--raw", "-r", help="Output raw LLM response object")] = False,
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        llm_tag: Annotated[
            Optional[str], Option("--llm-tag", "-tag", help="LLM tag from config (e.g., 'fake', 'powerful_model')")
        ] = None,
    ) -> None:
        """
        Invoke an LLM.

        input can be either taken from stdin (Unix pipe), or given with the --input param
        If runnable_name is provided, runs the specified Runnable with the given input.

        The LLM can be changed using --llm-id or --llm-tag. Tags are defined in config (e.g., 'fake', 'powerful_model').
        If neither is specified, the default model is used.
        'cache' is the prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'.

        Examples:
            uv run cli llm "Tell me a joke" --llm-tag fake
            uv run cli llm "Explain AI" --llm-id parrot_local_fake
        """

        from langchain_core.output_parsers import StrOutputParser
        from rich import print as pprint

        from src.ai_core.llm_factory import LlmFactory
        from src.utils.cli.langchain_setup import setup_langchain

        if not setup_langchain(llm_id, lc_debug, lc_verbose, cache):
            return

        # Check if executed as part ot a pipe
        if not input and not sys.stdin.isatty():
            input = sys.stdin.read()
        if not input or len(input) < 5:
            print("Error: Input parameter or something in stdin is required")
            return

        llm = LlmFactory(
            llm_id=llm_id,
            llm_tag=llm_tag,
            json_mode=False,
            streaming=stream,
            cache=cache,
            llm_params={"temperature": temperature},
        ).get()
        if raw:
            if stream:
                for chunk in llm.stream(input):
                    pprint(chunk)
            else:
                result = llm.invoke(input)
                pprint(result)
        else:
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
        llm_tag: Annotated[
            Optional[str], Option("--llm-tag", "-tag", help="LLM tag from config (e.g., 'fake', 'powerful_model')")
        ] = None,
    ) -> None:
        """
        Run a Runnable or directly invoke an LLM.

        If no runnable_name is provided, uses the default LLM to directly process the input, that
        can be either taken from stdin (Unix pipe), or given with the --input param
        If runnable_name is provided, runs the specified Runnable with the given input.

        The LLM can be changed using --llm-id or --llm-tag. Tags are defined in config (e.g., 'fake', 'powerful_model').
        If neither is specified, the default model is used.
        'cache' is the prompt caching strategy, and it can be either 'sqlite' (default) or 'memory'.

        Examples:
            uv run cli run joke --input "bears"
            uv run cli run joke --input "bears" --llm-tag fake
            uv run cli run joke --input "bears" --llm-id parrot_local_fake
        """

        from devtools import pprint

        from src.ai_core.chain_registry import ChainRegistry
        from src.utils.cli.langchain_setup import setup_langchain

        if not setup_langchain(llm_id, lc_debug, lc_verbose, cache):
            return

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
            # Create a temporary factory to resolve the final llm_id
            try:
                from src.ai_core.llm_factory import LlmFactory

                temp_factory = LlmFactory(llm_id=llm_id, llm_tag=llm_tag)
                final_llm_id = temp_factory.llm_id
            except ValueError as e:
                print(f"Error: {e}")
                return

            config = {
                "llm": final_llm_id,
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
    def list_models() -> None:
        """
        List the known LLMs, embeddings models, and vector stores.
        """
        from rich.columns import Columns
        from rich.console import Console
        from rich.panel import Panel

        from src.ai_core.embeddings_factory import EmbeddingsFactory
        from src.ai_core.llm_factory import LlmFactory
        from src.ai_core.vector_store_factory import VectorStoreFactory

        console = Console()

        # Get all items for each category
        llm_items = LlmFactory.known_items()
        embeddings_items = EmbeddingsFactory.known_items()
        vector_items = VectorStoreFactory.known_items()

        # Format LLM items in several columns
        llm_content = Columns([f"• {item}" for item in llm_items], equal=True, expand=True)
        embeddings_content = Columns([f"• {item}" for item in embeddings_items], equal=True, expand=True)
        vector_content = Columns([f"• {item}" for item in vector_items], equal=True, expand=True)
        llm_panel = Panel(llm_content, title="[bold blue]LLMs[/bold blue]", border_style="blue")
        embeddings_panel = Panel(
            embeddings_content,
            title="[bold green]Embeddings[/bold green]",
            border_style="green",
        )
        vector_panel = Panel(
            vector_content,
            title="[bold magenta]Vector Stores[/bold magenta]",
            border_style="magenta",
        )
        console.print(Panel("Available Models & Components", border_style="bright_blue"))
        console.print(llm_panel)
        console.print()
        bottom_row = Columns([embeddings_panel, vector_panel], equal=True, expand=True)
        console.print(bottom_row)

    @cli_app.command()
    def llm_info_dump(file_name: Annotated[Path, typer.Argument(help="Output YAML file path")]) -> None:
        """
        Write a list of LLMs in YAML format to the specified file.
        """
        import yaml

        from src.ai_core.llm_factory import LlmFactory

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

        from rich.console import Console
        from rich.table import Table

        from src.ai_core.embeddings_factory import EmbeddingsFactory

        if model_id is not None:
            if model_id not in EmbeddingsFactory.known_items():
                print(f"Error: {model_id} is unknown model id.\nShould be in {EmbeddingsFactory.known_items()}")
                return
            global_config().set("llm.models.default", model_id)

        factory = EmbeddingsFactory(
            embeddings_id=model_id,
        )
        embedder = factory.get()
        vector = embedder.embed_documents([input])

        console = Console()
        table = Table(title="Embeddings Summary", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", factory.embeddings_id or "default")
        table.add_row("Vector Length", str(len(vector[0])))
        table.add_row("First 40 Elements", ", ".join(f"{x:.4f}" for x in vector[0][:40]) + " [...]")

        console.print(table)

    @cli_app.command()
    def list_mcp_tools(
        filter: Annotated[list[str] | None, Option("--filter", "-f", help="Filter tools by server names")] = None,
    ) -> None:
        """Display information about available MCP tools.

        Shows the list of tools from MCP servers along with their descriptions.
        Can be filtered by server names.
        """
        import asyncio

        from rich.console import Console
        from rich.table import Table

        from src.ai_core.mcp_client import get_mcp_tools_info

        async def display_tools():
            tools_info = await get_mcp_tools_info(filter)
            if not tools_info:
                print("No MCP tools found.")
                return

            console = Console()
            for server_name, tools in tools_info.items():
                table = Table(
                    title=f"Server: {server_name}",
                    show_header=True,
                    header_style="bold magenta",
                    row_styles=["", "dim"],
                )
                table.add_column("Tool", style="cyan", no_wrap=True)
                table.add_column("Description", style="green")

                for tool_name, description in tools.items():
                    table.add_row(tool_name, description)

                console.print(table)
                print()  # Add space between tables

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

        from src.ai_core.embeddings_factory import (EmbeddingsFactory,
                                                    get_embeddings)

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

        # Display results in table format
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Semantic Similarity", show_header=True, header_style="bold blue")
        table.add_column("Reference Sentence", style="cyan")
        table.add_column("Comparison Sentence", style="green")
        table.add_column("Score", style="magenta", justify="right")

        for i, score in enumerate(similarities[0]):
            table.add_row(sentences[0], sentences[i + 1], f"{score:.3f}")

        console.print(table)

    @cli_app.command()
    def deep_agent(
        agent_type: Annotated[
            str,
            typer.Argument(
                help="Agent type: research, coding, analysis, or custom",
                callback=lambda x: x.lower()
            )
        ],
        input: Annotated[str | None, typer.Option(help="Input text or '-' to read from stdin")] = None,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID")
        ] = None,
        instructions: Annotated[
            Optional[str], Option("--instructions", "-i", help="Custom instructions for the agent")
        ] = None,
        tools: Annotated[
            Optional[list[str]], Option("--tools", "-t", help="Additional tools to include")
        ] = None,
        files: Annotated[
            Optional[list[Path]], Option("--files", "-f", help="Files to include in agent context")
        ] = None,
        output_dir: Annotated[
            Optional[Path], Option("--output-dir", "-o", help="Directory to save agent outputs")
        ] = None,
    ) -> None:
        """
        Run a Deep Agent for complex AI tasks.

        Deep Agents combine planning, file system access, and sub-agents for comprehensive task execution.

        Agent Types:
        - research: Conducts thorough research with web search and report generation
        - coding: Writes, debugs, and refactors code with specialized sub-agents
        - analysis: Analyzes data and generates insights
        - custom: Create a custom agent with your own instructions

        Examples:
            uv run cli deep-agent research "Latest developments in quantum computing"
            uv run cli deep-agent coding "Write a Python function to calculate Fibonacci"
            uv run cli deep-agent analysis --files data.csv "Analyze this dataset"
            echo "Debug this code" | uv run cli deep-agent coding
        """
        import asyncio

        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn

        from src.ai_core.deep_agents import (DeepAgentConfig,
                                             deep_agent_factory,
                                             run_deep_agent)
        from src.ai_core.llm_factory import get_llm

        # Get input from stdin if needed
        if not input and not sys.stdin.isatty():
            input = sys.stdin.read()
        if not input or len(input) < 5:
            print("Error: Input parameter or something in stdin is required")
            return

        console = Console()
        
        # Load files if provided
        file_contents = {}
        if files:
            for file_path in files:
                if file_path.exists():
                    file_contents[file_path.name] = file_path.read_text()
                else:
                    console.print(f"[yellow]Warning: File {file_path} not found[/yellow]")

        async def run_agent():
            # Set the model FIRST if specified, before creating any agents
            if llm_id:
                console.print(f"[cyan]Using model: {llm_id}[/cyan]")
                deep_agent_factory.set_default_model(llm_id)
            
            # Configure the agent based on type
            if agent_type == "research":
                # Import our search tools module to get real search capabilities
                from src.ai_core.search_tools import create_search_function

                # Create the best available search function (raw Python function for deep agents)
                # This will automatically choose between Tavily, Serper, DuckDuckGo, or mock
                internet_search = create_search_function(verbose=True)
                
                agent = deep_agent_factory.create_research_agent(
                    search_tool=internet_search,
                    name="CLI Research Agent",
                    async_mode=True
                )
            
            elif agent_type == "coding":
                agent = deep_agent_factory.create_coding_agent(
                    name="CLI Coding Agent",
                    language="python",
                    async_mode=True
                )
            
            elif agent_type == "analysis":
                agent = deep_agent_factory.create_data_analysis_agent(
                    name="CLI Analysis Agent",
                    async_mode=True
                )
            
            elif agent_type == "custom":
                if not instructions:
                    console.print("[red]Error: Custom agent requires --instructions[/red]")
                    return
                
                config = DeepAgentConfig(
                    name="CLI Custom Agent",
                    instructions=instructions,
                    enable_file_system=True,
                    enable_planning=True,
                    model=llm_id
                )
                
                agent = deep_agent_factory.create_agent(
                    config=config,
                    tools=[],
                    async_mode=True
                )
            
            else:
                console.print(f"[red]Unknown agent type: {agent_type}[/red]")
                return
            
            # Run the agent
            messages = [{"role": "user", "content": input}]
            
            # Show the user's query in a nice format
            console.print("\n[bold magenta]👤 User Query:[/bold magenta]")
            console.print(f"[italic white]{input}[/italic white]\n")
            
            with Progress(
                SpinnerColumn(spinner_name="dots", style="cyan"),
                TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                console=console,
            ) as progress:
                agent_emoji = {
                    "research": "🔍",
                    "coding": "💻",
                    "analysis": "📊",
                    "custom": "⚙️"
                }.get(agent_type, "🤖")
                
                task = progress.add_task(
                    f"{agent_emoji} {agent_type.capitalize()} agent is thinking...", 
                    total=None
                )
                
                try:
                    result = await run_deep_agent(
                        agent=agent,
                        messages=messages,
                        files=file_contents if file_contents else None,
                        stream=stream
                    )
                    
                    progress.update(task, description=f"{agent_emoji} {agent_type.capitalize()} agent completed!")
                    progress.update(task, completed=True)
                except Exception as e:
                    progress.update(task, description=f"❌ Error in {agent_type} agent")
                    progress.update(task, completed=True)
                    raise
            
            # Display results with enhanced markdown rendering
            if "messages" in result and result["messages"]:
                # Get the response content
                response_content = result["messages"][-1].content
                
                # Use Rich's Markdown rendering for better display
                from rich.markdown import Markdown
                from rich.panel import Panel
                
                try:
                    # Render as markdown for better formatting
                    md = Markdown(response_content)
                    # Optionally wrap in a panel for better visual separation
                    # console.print(Panel(md, border_style="cyan", padding=(1, 2)))
                    console.print(md)
                except Exception as e:
                    # Fallback to plain text if markdown parsing fails
                    logger.warning(f"Markdown rendering failed: {e}")
                    console.print(response_content)
                
            # Save files if output directory specified
            if output_dir and "files" in result:
                output_dir.mkdir(parents=True, exist_ok=True)
                console.print("[bold yellow]📁 Saving files...[/bold yellow]")
                for filename, content in result["files"].items():
                    file_path = output_dir / filename
                    file_path.write_text(content)
                    console.print(f"  [green]✓[/green] Saved: [cyan]{file_path}[/cyan]")
        
        # Run the async function
        asyncio.run(run_agent())

    @cli_app.command()
    def list_deep_agents() -> None:
        """
        List all created Deep Agents.
        """
        from rich.console import Console
        from rich.table import Table

        from src.ai_core.deep_agents import deep_agent_factory
        
        console = Console()
        agents = deep_agent_factory.list_agents()
        
        if not agents:
            console.print("[yellow]No Deep Agents have been created yet.[/yellow]")
            return
        
        table = Table(
            title="Deep Agents",
            show_header=True,
            header_style="bold magenta"
        )
        table.add_column("Agent Name", style="cyan")
        table.add_column("Status", style="green")
        
        for agent_name in agents:
            table.add_row(agent_name, "Active")
        
        console.print(table)

    @cli_app.command()
    def deep_agent_demo() -> None:
        """
        Launch the Deep Agent interactive demo (Streamlit app).
        """
        import subprocess
        from pathlib import Path
        
        demo_path = Path(__file__).parent.parent / "demos" / "deep_agent_demo.py"
        
        if not demo_path.exists():
            print(f"Error: Demo file not found at {demo_path}")
            return
        
        print("Launching Deep Agent Demo...")
        print("Open your browser at http://localhost:8501")
        
        try:
            subprocess.run(["streamlit", "run", str(demo_path)])
        except FileNotFoundError:
            print("Error: Streamlit is not installed. Install it with: pip install streamlit")
        except KeyboardInterrupt:
            print("\nDemo stopped.")

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
