# complete doc and docstrings AI!
# Utilities to ease use of MCP

import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient, StdioServerParameters

from src.utils.config_mngr import global_config
from src.utils.langgraph import print_astream

load_dotenv()


def get_mcp_servers_from_config() -> dict:
    """
    Read the list of MCP servers and their parameters from the global config file.

    Example configuration:
    ```yaml
        mcp_servers:
            pubmed:
                command: uv
                args: ["tool", "run", "--quiet", "pubmedmcp@0.1.3"]
                transport: stdio
                description: Provides access to PubMed medical research database
     ```
    """
    result = {}
    servers = global_config().get_dict("mcpServers", expected_keys=["args"])
    for name, desc in servers.items():
        desc.pop("description", "")  # not used yet
        desc.pop("example", None)

        if "command" in desc and desc["command"] == "uvx":  # uvx is an alias to 'uv tool run', not always in tha path
            desc["command"] = "uv"
            desc["args"] = ["tool", "run"] + desc["args"]
        if "transport" not in desc:
            desc["transport"] = "stdio"
        # Passing the PATH seems needed for some servers, (ex: Tavily)
        # First saw here : https://github.com/hideya/langchain-mcp-tools-py/blob/7d4ad392a8c6166db0018ebcb98be65f5a16f70a/src/langchain_mcp_tools/langchain_mcp_tools.py#L86
        desc["env"] = {"PATH": os.environ.get("PATH", "")} | dict(desc.get("env", {}))
        if not desc.get("disabled"):
            result[name] = dict(**desc)
        _ = StdioServerParameters(**desc)  # just to test argument types
    return result


## quick test ##
async def call_react_agent(query: str) -> None:
    from langgraph.prebuilt import create_react_agent
    from loguru import logger

    from src.ai_core.llm import get_llm

    model = get_llm()
    async with MultiServerMCPClient(get_mcp_servers_from_config()) as client:
        # async with MultiServerMCPClient(test_servers) as client:
        tools = client.get_tools()
        agent = create_react_agent(model, tools)
        logger.info("invoke MCP agent...")
        resp = agent.astream({"messages": query})
        await print_astream(resp)


if __name__ == "__main__":
    examples = ["what's (3 + 5) x 12?", "list files in current directory", "connect to atos.net and get recent news"]
    asyncio.run(call_react_agent(examples[-1]))
