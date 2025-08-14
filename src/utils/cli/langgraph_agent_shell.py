from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from src.ai_core.llm_factory import get_llm
from src.ai_core.mcp_client import get_mcp_servers_dict
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

    if mcp_server_names:
        print(f"Starting MCP agent shell with servers: {mcp_server_names}")
    print("Type /quit to exit; Use up/down arrows to navigate prompt history\n")

    model = get_llm(llm_id=llm_id)
    if mcp_server_names:
        client = MultiServerMCPClient(get_mcp_servers_dict(mcp_server_names))
        tools = tools + await client.get_tools()
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
