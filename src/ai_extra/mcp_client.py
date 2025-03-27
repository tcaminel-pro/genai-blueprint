"""Utilities for interacting with MCP (Multi-Component Platform) servers.

This module provides functionality to:
- Retrieve and configure MCP servers from application configuration
- Create and manage MCP client connections
- Execute queries using MCP tools with LangChain agents

The main components are:
- get_mcp_servers_from_config: Reads MCP server configurations
- call_react_agent: Executes queries using MCP tools with a ReAct agent
"""

import asyncio
import os
from contextlib import AsyncExitStack
from itertools import chain

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient, StdioServerParameters
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from mcpadapt.core import MCPAdapt
from mcpadapt.langchain_adapter import LangChainAdapter

from src.ai_core.llm import get_llm
from src.utils.config_mngr import global_config
from src.utils.langgraph import print_astream

load_dotenv()

# refactor :
# - create  a function "create_server_parameters" that takes a dict and do the same processing than current get_mcp_servers_from_config and returns a dict
# - Create a new get_mcp_servers_from_json  that takes a JSON string as input


def get_mcp_servers_from_config() -> dict:
    """Retrieve configured MCP servers from application configuration.

    Processes the MCP server configurations from the global config file, handling
    command aliases and environment variables. Validates server parameters.

    Returns:
        Dictionary of server names to their configuration parameters

    Example:
    ```python
    servers = get_mcp_servers_from_config()
    # {'pubmed': {'command': 'uv', 'args': ['tool', 'run', 'pubmedmcp@0.1.3'], ...}}
    ```
    """
    result = {}
    servers = global_config().get_dict("mcpServers")
    for name, desc in servers.items():
        if "command" in desc and desc["command"] == "uvx":  # uvx is an alias to 'uv tool run', not always in tha path
            desc["command"] = "uv"
            desc["args"] = ["tool", "run"] + desc["args"]
        if "transport" not in desc:
            desc["transport"] = "stdio"
        # Passing the PATH seems needed for some servers, (ex: Tavily)
        # First saw here : https://github.com/hideya/langchain-mcp-tools-py/blob/7d4ad392a8c6166db0018ebcb98be65f5a16f70a/src/langchain_mcp_tools/langchain_mcp_tools.py#L86
        desc["env"] = {"PATH": os.environ.get("PATH", "")} | dict(desc.get("env", {}))

        desc.pop("description", "")  # not used yet
        desc.pop("example", None)
        if not desc.get("disabled"):
            desc.pop("disabled", None)
            result[name] = dict(**desc)

        _ = StdioServerParameters(**desc)  # just to test argument types
    return result


async def mcp_agent_runner(
    model: BaseChatModel, servers: list[StdioServerParameters], prompt: str, config: RunnableConfig | None = None
) -> str | None:
    """
    Function using MCPAdapt.  Use to work, but NOT MAINTAINED !!
    """
    # TODO: adapt it for SmolAgent
    # see https://hungvtm.medium.com/building-mcp-servers-and-client-with-smolagents-bd9db2d640e6

    if config is None:
        config = {}
    async with AsyncExitStack() as stack:
        tools_list = [stack.enter_context(MCPAdapt(server, LangChainAdapter())) for server in servers]

        # Merge and flatten tools from all MCP servers
        tools = list(chain.from_iterable(tools_list))

        if _ := config.get("thread_id"):
            memory = MemorySaver()
        else:
            memory = None
        agent_executor = create_react_agent(model, tools, checkpointer=memory)

        result = await agent_executor.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config,
        )
        return result["messages"][-1].content


## quick test ##
async def call_react_agent(query: str, llm_id: str | None = None) -> None:
    """Execute a query using MCP tools with a ReAct agent.

    Creates a ReAct agent with MCP tools and streams the response to the query.

    Args:
        query: The input query to process
    """
    from langgraph.prebuilt import create_react_agent
    from loguru import logger

    model = get_llm(llm_id=llm_id)
    async with MultiServerMCPClient(get_mcp_servers_from_config()) as client:
        # async with MultiServerMCPClient(test_servers) as client:
        tools = client.get_tools()
        agent = create_react_agent(model, tools)
        logger.info("invoke MCP agent...")
        resp = agent.astream({"messages": query})
        await print_astream(resp)


if __name__ == "__main__":
    examples = [
        "what's the weather in Toulouse ? ",
        "list files in current directory",
        "connect to atos.net and get recent news",
    ]
    #  asyncio.run(call_react_agent(examples[-1]))
    asyncio.run(call_react_agent(examples[0]))
