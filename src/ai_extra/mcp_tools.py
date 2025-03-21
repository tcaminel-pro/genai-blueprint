"""MCP (Model Context Protocol) tool calling from LangChain (WIP !!).

Adapted from : https://github.com/rectalogic/langchain-mcp

"""

import asyncio
import os
from contextlib import AsyncExitStack
from itertools import chain
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from mcpadapt.langchain_adapter import LangChainAdapter
from pydantic import BaseModel

from src.ai_core.llm import get_llm


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server instance."""

    params: StdioServerParameters
    description: str
    doc: str | None = None


async def mcp_agent_runner(
    model: BaseChatModel, servers: list[StdioServerParameters], prompt, config: RunnableConfig = None
) -> str:
    # mode = config["configurable"].get("mode") or "async"
    # assert mode in ["async", "stream"], "'mode' should be either 'async' or 'stream"

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


if __name__ == "__main__":
    MODEL_ID = "llama31_8_groq"
    MODEL_ID = "claude_haiku35_openrouter"
    llm = get_llm(llm_id=MODEL_ID)

    # Dictionary of MCP server configurations using Pydantic models
    mcp_servers = {
        "filesystem": MCPServerConfig(
            params=StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd().parent)],
            ),
            description="Provides access to local filesystem operations",
        ),
        "timeserver": MCPServerConfig(
            params=StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/mcp-server-time"],
            ),
            description="Provides current time and timezone conversions",
        ),
        "memory": MCPServerConfig(
            params=StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
            ),
            description="Provides short-term memory storage and retrieval",
        ),
        "arxiv": MCPServerConfig(
            params=StdioServerParameters(
                command="uv",
                args=["tool", "run", "@smithery/arxiv-mcp-server", "--storage-path", "/tmp"],
            ),
            description="Provides access to arXiv research papers",
        ),
        "pubmed": MCPServerConfig(
            params=StdioServerParameters(
                command="uvx",
                args=["--quiet", "pubmedmcp@0.1.3"],
                env={"UV_PYTHON": "3.12", **os.environ},
            ),
            description="Provides access to PubMed medical research database",
        ),
        "current time": MCPServerConfig(
            params=StdioServerParameters(command="uvx", args=["MCP-timeserver"]),
            description="get current time",
            doc="https://github.com/SecretiveShell/MCP-timeserver",
        ),
    }

    async def main() -> None:
        # r = await mcp_run(timeserver_mcp_params, llm, "current time in New york")
        # message = r["messages"][-1]
        # # message.pretty_print()

        # r = await mcp_run(memory_mcp_params, llm, "add observation that user name is Thierry")
        # message = r["messages"][-1]
        # message.pretty_print()

        # r = await mcp_run(arxiv_mcp_params, llm, "{'read_paper','paper_id': '2401.12345'})")
        # message = r["messages"][-1]
        # Display and process results from mcp_agent_runner
        # async for event in mcp_agent_runner(
        #     llm, [pubmed_mcp_params], "Find relevant studies on alcohol hangover and treatment.", {}
        # ):
        #     debug(event)

        # async for event in mcp_agent_runner(
        #     llm, [memory_mcp_params, filesystem_mcp_params], "List content of the current directory.", {}
        # ):
        #     print(event)
        result = await mcp_agent_runner(
            llm,
            [mcp_servers["memory"].params, mcp_servers["filesystem"].params],
            "List content of the current directory.",
            {},
        )
        debug(result)

    asyncio.run(main())
