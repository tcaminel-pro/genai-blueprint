# Utilities to ease use of MCP

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient, StdioServerParameters

from src.utils.config_mngr import global_config


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
    servers = global_config().get_dict("mcp_servers")

    for name, desc in servers.items():
        desc.pop("description", "")
        desc.pop("example", None)
        _ = StdioServerParameters(**desc)  # quick test of validity
        result[name] = dict(**desc)
    return result


## quick test ##
async def test() -> None:
    import rich
    from langgraph.prebuilt import create_react_agent
    from loguru import logger

    from src.ai_core.llm import get_llm

    model = get_llm()
    async with MultiServerMCPClient(get_mcp_servers_from_config()) as client:
        # async with MultiServerMCPClient(test_servers) as client:
        tools = client.get_tools()
        logger.info("Create agent")
        agent = create_react_agent(model, tools)
        logger.info("invoke MCP agent")
        # math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        # rich.print(math_response)
        dir_response = await agent.ainvoke({"messages": "list files in current directory"})
        rich.print(dir_response)


if __name__ == "__main__":
    asyncio.run(test())
