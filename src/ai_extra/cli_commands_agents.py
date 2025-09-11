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
from loguru import logger
from typer import Option

from src.ai_extra.tools_smolagents.config_loader import (
    CONF_YAML_FILE,
    load_demo_config,
    process_tools_from_config,
)
from src.ai_extra.tools_smolagents.react_config_loader import (
    REACT_CONF_YAML_FILE,
    load_react_demo_config,
)


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
    def deep_agent(
        input: Annotated[str | None, typer.Argument(help="Input query or '-' to read from stdin")] = None,
        config: Annotated[
            Optional[str],
            Option("--config", "-c", help="Configuration name from deep_agent.yaml (e.g. 'Research', 'Coding')"),
        ] = None,
        stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m", help="LLM model ID")] = None,
        instructions: Annotated[
            Optional[str], Option("--instructions", "-i", help="Custom instructions for the agent")
        ] = None,
        tools: Annotated[Optional[list[str]], Option("--tools", "-t", help="Additional tools to include")] = None,
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

        Examples:

        # Using a predefined configuration:
        uv run cli deep-agent --config "Research" "Latest developments in quantum computing"
        uv run cli deep-agent --config "Coding" "Write a Python function to calculate Fibonacci"
        uv run cli deep-agent --config "Data Analysis" --files data.csv "Analyze this dataset"
        
        # Using custom instructions:
        uv run cli deep-agent --instructions "You are a helpful assistant" "Help me with this task"
        
        # Reading from stdin:
        echo "Debug this code" | uv run cli deep-agent --config "Coding"
        """
        import asyncio

        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn

        from src.ai_core.deep_agents import DeepAgentConfig, deep_agent_factory, run_deep_agent
        from src.ai_extra.tools_smolagents.deep_config_loader import (
            DEEP_AGENT_CONF_YAML_FILE,
            load_deep_agent_demo_config,
        )

        # Get input from stdin if needed
        if not input and not sys.stdin.isatty():
            input = sys.stdin.read()
        if not input or len(input) < 5:
            print("Error: Input parameter or something in stdin is required")
            return

        console = Console()
        
        # Handle configuration loading
        demo_config = None
        config_instructions = None
        config_tools = []
        config_mcp_servers = []
        config_enable_file_system = True
        config_enable_planning = True

        if config:
            demo_config = load_deep_agent_demo_config(config)
            if demo_config is None:
                print(f"Error: Configuration '{config}' not found in {DEEP_AGENT_CONF_YAML_FILE}")
                return

            # Extract configuration parameters
            config_instructions = demo_config.get("instructions", "")
            config_tools = demo_config.get("tools", [])
            config_mcp_servers = demo_config.get("mcp_servers", [])
            config_enable_file_system = demo_config.get("enable_file_system", True)
            config_enable_planning = demo_config.get("enable_planning", True)

            print(f"Using Deep Agent configuration '{config}':")
            if config_tools:
                print(f"  Tools: {len(config_tools)} tool(s) configured")
            if config_mcp_servers:
                print(f"  MCP servers: {', '.join(config_mcp_servers)}")
                
        # Use config instructions if provided, otherwise use command line instructions
        final_instructions = instructions or config_instructions
        
        if not final_instructions:
            if not config:
                print("Error: Either --config or --instructions must be provided")
                return
            else:
                print(f"Error: Configuration '{config}' has no instructions defined")
                return

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

            # Create agent configuration
            agent_config = DeepAgentConfig(
                name=f"CLI {config or 'Custom'} Agent",
                instructions=final_instructions,
                enable_file_system=config_enable_file_system,
                enable_planning=config_enable_planning,
                model=llm_id,
            )

            # TODO: Process tools from config (similar to smolagents)
            # For now, just create the agent with empty tools
            # Tools processing would need to be implemented based on the tool definitions
            # in the YAML config (class:, function:, etc.)
            agent_tools = []
            
            # Create the agent
            agent = deep_agent_factory.create_agent(config=agent_config, tools=agent_tools, async_mode=True)

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
                agent_name = config or "Custom"
                agent_emoji = {"research": "🔍", "coding": "💻", "data analysis": "📊", "web research": "🌐", "documentation writer": "📝", "stock analysis": "📈"}.get(agent_name.lower(), "🤖")

                task = progress.add_task(f"{agent_emoji} {agent_name} agent is thinking...", total=None)

                try:
                    result = await run_deep_agent(
                        agent=agent, messages=messages, files=file_contents if file_contents else None, stream=stream
                    )

                    progress.update(task, description=f"{agent_emoji} {agent_name} agent completed!")
                    progress.update(task, completed=True)
                except Exception:
                    progress.update(task, description=f"❌ Error in {agent_name} agent")
                    progress.update(task, completed=True)
                    raise

            # Display results with enhanced markdown rendering
            if "messages" in result and result["messages"]:
                # Get the response content
                response_content = result["messages"][-1].content

                # Use Rich's Markdown rendering for better display
                from rich.markdown import Markdown

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

        table = Table(title="Deep Agents", show_header=True, header_style="bold magenta")
        table.add_column("Agent Name", style="cyan")
        table.add_column("Status", style="green")

        for agent_name in agents:
            table.add_row(agent_name, "Active")

        console.print(table)
