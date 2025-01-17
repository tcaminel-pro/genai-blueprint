"""
MCP (Model Context Protocol) tool calling from LangChain (WIP !!)

Adapted from : https://github.com/rectalogic/langchain-mcp

"""

import asyncio
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp.toolkit import MCPToolkit
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from python.ai_core.llm import get_llm


async def run(llm: BaseChatModel, tools: list[BaseTool], prompt: str):
    agent = create_react_agent(llm, tools=tools)
    inputs = {"messages": [("user", prompt)]}
    return await agent.ainvoke(inputs)


async def mcp_run(mcp_server_param: StdioServerParameters, llm: BaseChatModel, prompt: str):
    async with stdio_client(mcp_server_param) as (read, write):
        async with ClientSession(read, write) as session:
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()
            return await run(llm, toolkit.get_tools(), prompt)


if __name__ == "__main__":
    MODEL_ID = "llama31_8_groq"
    llm = get_llm(llm_id=MODEL_ID)

    filesystem_mcp_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd().parent)],
    )

    timeserver_mcp_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/mcp-server-time"],
    )

    async def main():
        r = await mcp_run(filesystem_mcp_params, llm, "list the current directory")
        message = r["messages"][-1]
        message.pretty_print()

        r = await mcp_run(timeserver_mcp_params, llm, "current time in New york")
        message = r["messages"][-1]
        message.pretty_print()

    asyncio.run(main())
