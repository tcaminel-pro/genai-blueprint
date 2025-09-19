"""Utilities for interacting with MCP (Multi-Component Platform) servers.

This module provides functionality to:
- Retrieve and configure MCP servers from application configuration
- Create and manage MCP client connections
- Execute queries using MCP tools with LangChain agents
- Validate and process server configurations
- Run MCP agents with custom queries and server filters

The main components are:
- get_mcp_servers_dict: Retrieves and processes MCP server configurations
- update_server_parameters: Processes individual server configurations
- mcp_agent_runner: Executes queries using MCP tools with a ReAct agent
- call_react_agent: Convenience function for running MCP agents with streaming output

Example usage:
```python
# Get all configured MCP servers
servers = get_mcp_servers_dict()

# Run a query using specific MCP servers
await call_react_agent(
    "What's the weather in Toulouse?",
    mcp_server_filter=["weather"]
)
```
"""

import asyncio
import os
from contextlib import AsyncExitStack
from itertools import chain

from devtools import debug  # noqa: F401
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcpadapt.core import MCPAdapt
from mcpadapt.langchain_adapter import LangChainAdapter

from src.ai_core.llm_factory import get_llm
from src.utils.config_mngr import global_config
from src.utils.langgraph import print_astream

load_dotenv()


def update_server_parameters(server_config: dict) -> dict:
    """Process individual MCP server configuration dictionary.

    Handles command aliases, environment variables, and validation.
    Ensures required parameters are present and properly formatted.

    Args:
        server_config: Raw server configuration dictionary containing:
            - command: The executable command
            - args: List of arguments for the command
            - transport: Communication protocol (defaults to 'stdio')
            - env: Environment variables
            - disabled: Optional flag to disable the server

    Returns:
        Processed server parameters dictionary ready for server instantiation

    Example:
    ```python
    config = {
        "command": "uvx",
        "args": ["tool", "run", "pubmedmcp@0.1.3"]
    }
    processed = update_server_parameters(config)
    # {'command': 'uv', 'args': ['tool', 'run', 'pubmedmcp@0.1.3'], 'transport': 'stdio', 'env': {'PATH': ...}}
    ```
    """
    desc = dict(server_config)
    if (
        "command" in desc and desc["command"] == "uvx"
    ):  # uvx is an alias to 'uv tool run', which is not always in the path
        desc["command"] = "uv"
        desc["args"] = ["tool", "run"] + desc["args"]
    if "transport" not in desc:
        desc["transport"] = "stdio"
    # Passing the PATH seems needed for some servers, (ex: Tavily)
    desc["env"] = {"PATH": os.environ.get("PATH", "")} | dict(desc.get("env", {}))

    desc.pop("description", "")  # not used yet
    desc.pop("example", None)
    desc.pop("disabled", None)  # TODO : reintroduce it
    _ = StdioServerParameters(**desc)  # just to test argument type

    return desc


# def get_mcp_servers_from_json(json_str: str) -> dict:
#     """Retrieve MCP servers from JSON string configuration.

#     Args:
#         json_str: JSON string containing server configurations

#     Returns:
#         Dictionary of server names to their configuration parameters
#     """
#     import json

#     servers = json.loads(json_str)
#     return {name: create_server_parameters(desc) for name, desc in servers.items()}


async def get_mcp_tools_info(filter: list[str] | None = None) -> dict:
    """Get all tools from MCP servers with their names and descriptions."""
    servers = get_mcp_servers_dict(filter)
    tools_info = {}
    for server_name, param_desc in servers.items():
        #        debug(server_name)
        if not param_desc.get("disabled", False):
            server_params = StdioServerParameters(**param_desc)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tools_info[server_name] = {tool.name: tool.description for tool in tools.tools}
    return tools_info


async def get_mcp_prompts(filter: list[str] | None = None) -> dict:
    """Get all prompts  from MCP servers with their names and descriptions."""
    servers = get_mcp_servers_dict(filter)
    prompts_info = {}
    for server_name, param_desc in servers.items():
        if not param_desc.get("disabled", False):
            server_params = StdioServerParameters(**param_desc)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    prompts = await session.list_prompts()
                    prompts_info[server_name] = {p.name: p.description for p in prompts.prompts}
    return prompts_info


def get_mcp_servers_dict(filter: list[str] | None = None) -> dict:
    """Retrieve configured MCP servers from application configuration.

    Processes the MCP server configurations from the global config file, handling
    command aliases and environment variables. Validates server parameters.

    Args:
        filter: List of server names to include. If None, all servers are returned.

    Returns:
        Dictionary of server names to their configuration parameters

    Raises:
        ValueError: If any server in the filter is not found in the configuration

    Example:
    ```python
    servers = get_mcp_servers_from_config()
    # {'pubmed': {'command': 'uv', 'args': ['tool', 'run', 'pubmedmcp@0.1.3'], ...}}
    ```
    """
    servers = global_config().merge_with("config/demos/graph_rag.yaml").get_dict("mcpServers")
    # Filter out servers that are explicitly disabled
    servers = {name: config for name, config in servers.items() if not config.get("disabled", False)}
    if filter is not None:
        missing_servers = [name for name in filter if name not in servers]
        if missing_servers:
            raise ValueError(f"MCP server(s) not found or disabled in configuration: '{', '.join(missing_servers)}'")

    from loguru import logger

    result_dict = {}
    for name, desc in servers.items():
        if filter is None or name in filter:
            try:
                result_dict[name] = update_server_parameters(desc)
            except Exception as e:
                print(f"Skipping MCP server {name} due to configuration error: {str(e)}")
                logger.warning(f"Skipping MCP server {name} due to configuration error: {str(e)}")
    return result_dict


def dict_to_stdio_server_list(param_list: dict) -> list[StdioServerParameters]:
    """Convert a dictionary of server parameters to StdioServerParameters objects.

    Args:
        param_list: Dictionary where keys are server names and values are
                   server parameter dictionaries

    Returns:
        List of StdioServerParameters instances ready for MCP client creation

    Example:
    ```python
    servers = {
        "weather": {"command": "uv", "args": ["tool", "run", "weathermcp"]}
    }
    server_params = dict_to_stdio_server_list(servers)
    # [StdioServerParameters(command='uv', args=['tool', 'run', 'weathermcp'], ...)]
    ```
    """
    return [StdioServerParameters(**desc) for name, desc in param_list.items()]


async def mcp_agent_runner(
    model: BaseChatModel, servers: list[StdioServerParameters], prompt: str, config: RunnableConfig | None = None
) -> str | None:
    """Execute a query using MCP tools with a ReAct agent.

    Creates a ReAct agent with MCP tools and processes the query.
    Note: This function is not actively maintained and may require updates.

    Args:
        model: The language model to use for the agent
        servers: List of StdioServerParameters for MCP server connections
        prompt: The input query to process
        config: Optional RunnableConfig for the agent execution

    Returns:
        The final response content from the agent, or None if no response

    Example:
    ```python
    model = get_llm()
    servers = dict_to_stdio_server_list(get_mcp_servers_dict())
    response = await mcp_agent_runner(model, servers, "What's the weather?")
    ```
    """
    # TODO: adapt it for SmolAgent
    # see https://hungvtm.medium.com/building-mcp-servers-and-client-with-smolagents-bd9db2d640e6

    if config is None:
        config = {}
    async with AsyncExitStack() as stack:
        tools_list = []
        for server in servers:
            mcp_adapt = MCPAdapt(server, LangChainAdapter())
            tools = await stack.enter_async_context(mcp_adapt)
            tools_list.append(tools)

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
async def call_react_agent(
    query: str, llm_id: str | None = None, mcp_server_filter: list | None = None, additional_tools: list | None = None
) -> None:
    """Execute a query using MCP tools with a ReAct agent and stream the response.

    Creates a ReAct agent with MCP tools and streams the response to the query.
    This is a convenience wrapper around mcp_agent_runner with streaming output.

    Args:
        query: The input query to process
        llm_id: Optional ID of the language model to use
        mcp_server_filter: Optional list of server names to include in the agent
        additional_tools: Optional list of additional tools to include in the agent

    Example:
    ```python
    await call_react_agent(
        "What's the weather in Paris?",
        mcp_server_filter=["weather"]
    )
    ```
    """
    from langgraph.prebuilt import create_react_agent
    from loguru import logger

    model = get_llm(llm_id=llm_id)
    client = MultiServerMCPClient(get_mcp_servers_dict(mcp_server_filter))
    try:
        # Get MCP tools
        mcp_tools = await client.get_tools()

        # Combine MCP tools with additional tools from configuration
        all_tools = list(mcp_tools)
        if additional_tools:
            all_tools.extend(additional_tools)

        agent = create_react_agent(model, all_tools)

        tool_names = [getattr(t, "name", str(type(t).__name__)) for t in all_tools]
        logger.info(f"ReAct agent created with {len(all_tools)} tools: {', '.join(tool_names)}")

        resp = agent.astream({"messages": [HumanMessage(content=query)]})
        await print_astream(resp)
    finally:
        # Clean up the client if it has a close method
        if hasattr(client, "close"):
            await client.close()


if __name__ == "__main__":
    examples = [
        "what's the weather in Toulouse ? ",
        "list files in current directory",
        "connect to atos.net and get recent news",
    ]
    #  asyncio.run(call_react_agent(examples[-1]))
    # asyncio.run(call_react_agent(examples[0]))
    asyncio.run(call_react_agent(";\n".join(examples), mcp_server_filter=["filesystem", "weather", "playwright"]))
