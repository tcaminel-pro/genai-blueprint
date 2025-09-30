import webbrowser
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.ai_core.llm_factory import get_llm
from src.ai_core.mcp_client import get_mcp_servers_dict
from src.utils.config_mngr import global_config
from src.utils.langgraph import print_astream


async def run_langgraph_agent_shell(
    llm_id: str | None, tools: list[BaseTool] = [], mcp_server_names: list[str] = []
) -> None:
    """Run an interactive shell for sending prompts to a LanggGraph ReAct agent.

    The MCP servers are started once before entering the shell loop.
    The user can type /quit to exit the shell.

    Args:
        llm_id: Optional ID of the language model to use
        server_filter: Optional list of server names to include in the agent
    """
    console = Console()
    last_trace_url: str | None = None

    # Display welcome banner
    welcome_text = Text("🤖 LangGraph Agent", style="bold cyan")
    if mcp_server_names:
        welcome_text.append(f"\nConnected to MCP servers: {', '.join(mcp_server_names)}", style="green")

    console.print(Panel(welcome_text, title="Welcome", border_style="bright_blue"))
    console.print("[dim]Commands: /help, /quit, /trace\nUse up/down arrows to navigate prompt history[/dim]\n")

    model = get_llm(llm_id=llm_id)
    if mcp_server_names:
        with console.status("[bold green]Connecting to MCP servers..."):
            client = MultiServerMCPClient(get_mcp_servers_dict(mcp_server_names))
            tools = tools + await client.get_tools()
            console.print("[green]✓ MCP servers connected[/green]\n")

    config = {"configurable": {"thread_id": "1"}}
    agent = create_react_agent(model, tools, checkpointer=MemorySaver())

    # Set up prompt history
    history_file = Path(".blueprint.input.history")
    session = PromptSession(history=FileHistory(str(history_file)))

    while True:
        try:
            with patch_stdout():
                prompt_style = Style.from_dict(
                    {
                        "prompt": "bold cyan",
                    }
                )
                user_input = await session.prompt_async(
                    ">>> ", style=prompt_style, auto_suggest=AutoSuggestFromHistory()
                )

            user_input = user_input.strip()
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
                break
            if user_input == "/help":
                console.print(
                    Panel(
                        "/help   – show this help\n"
                        "/quit   – exit the shell\n"
                        "/trace  – open last LangSmith trace in browser",
                        title="[bold cyan]Commands[/bold cyan]",
                        border_style="cyan",
                    )
                )
                continue
            if user_input == "/trace":
                if last_trace_url:
                    console.print(f"[dim]Opening trace URL: {last_trace_url}[/dim]")
                    webbrowser.open(last_trace_url)
                else:
                    console.print("[dim]No trace URL available yet.[/dim]")
                continue
            if user_input.startswith("/") and user_input not in {"/quit", "/exit", "/q", "/help", "/trace"}:
                console.print(f"[red]Unknown command: {user_input}[/red]")
                continue
            if not user_input:
                continue

            # Display user prompt with styling
            console.print(Panel(user_input, title="[bold blue]User[/bold blue]", border_style="blue"))

            # Process the response
            with console.status("[bold green]Agent is thinking...\n[/bold green]"):
                if global_config().get_bool("monitoring.langsmith", False):
                    from langchain.callbacks import tracing_v2_enabled

                    with tracing_v2_enabled() as cb:
                        resp = agent.astream({"messages": user_input}, config)
                        await print_astream(resp)
                        last_trace_url = cb.get_run_url()
                else:
                    resp = agent.astream({"messages": user_input}, config)
                    await print_astream(resp)

            console.print()  # Add spacing between interactions

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Received keyboard interrupt. Exiting...[/bold yellow]")
            break
        except Exception as e:
            console.print(Panel(f"[red]Error: {str(e)}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
