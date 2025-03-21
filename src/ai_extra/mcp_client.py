# graph.py

import asyncio
from collections import ChainMap
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient, StdioServerParameters
from langgraph.prebuilt import create_react_agent
from loguru import logger

from src.ai_core.llm import get_llm
from src.utils.config_mngr import global_config

model = get_llm()


MATH_SERVER_PATH = Path("/home/tcl/prj/mcp_server/math_server.py")
MATH_SERVER_PATH = Path("/home/tcl/prj/genai-blueprint/src/mcp_server/math_server.py")
assert MATH_SERVER_PATH.exists()


#    "filesystem": MCPServerConfig(
#         params=StdioServerParameters(
#             command="npx",
#             args=["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd().parent)],
#         ),
#         description="Provides access to local filesystem operations",
#     ),

test_servers = {
    "math": {
        "command": "uv",
        "args": ["run", str(MATH_SERVER_PATH)],
        "transport": "stdio",
    }
}


def get_mcp_servers_from_config() -> dict:
    servers = global_config().get_list("mcp_servers")
    result = {}
    for server in servers:
        key = next(iter(server))
        value = dict(ChainMap(*server[key]))  # merge a list of dict into a dict
        if env := value.get("env"):
            value["env"] = dict(ChainMap(*env))
        result[key] = value
        _ = StdioServerParameters(**value)
    return result


async def test():
    async with MultiServerMCPClient(get_mcp_servers_from_config()) as client:
    # async with MultiServerMCPClient(test_servers) as client:
        tools = client.get_tools()
        logger.info("Create agent")
        agent = create_react_agent(model, tools)
        logger.info("invoke MCP agent")
        # math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        # debug(math_response)
        dir_response = await agent.ainvoke({"messages": "list files in current directory"})
        debug(dir_response)


if __name__ == "__main__":
    asyncio.run(test())
