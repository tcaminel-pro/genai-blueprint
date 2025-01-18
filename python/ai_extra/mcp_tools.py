"""
MCP (Model Context Protocol) tool calling from LangChain (WIP !!)

Adapted from : https://github.com/rectalogic/langchain-mcp

"""

import asyncio
import os
from contextlib import AsyncExitStack
from itertools import chain
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from mcpadapt.langchain_adapter import LangChainAdapter

from python.ai_core.llm import get_llm

# or AsyncExitStack ?


async def mcp_agent_runner(model, servers: list[StdioServerParameters], prompt, config: RunnableConfig = {}):
    mode = config["configurable"].get("mode") or "async"
    assert mode in ["async", "stream"], "'mode' should be either 'async' or 'stream"

    async with AsyncExitStack() as stack:
        tools_list = [stack.enter_context(MCPAdapt(server, LangChainAdapter())) for server in servers]

        # Merge and flatten tools from all MCP servers
        tools = list(chain.from_iterable(tools_list))

        if thread_id := config.get("thread_id"):
            memory = MemorySaver()
        else:
            memory = None
        agent_executor = create_react_agent(model, tools, checkpointer=memory)

        # call either agent_executor.astream or agent_executor.ainvoke according to 'mode'
        # and complete typing AI!

        async for event in agent_executor.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config,
        ):
            if agent_msg := event.get("agent"):
                for msg in agent_msg["messages"]:
                    assert isinstance(msg, AIMessage)
                    yield (msg.content)


if __name__ == "__main__":
    MODEL_ID = "llama31_8_groq"
    MODEL_ID = "claude_haiku35_openrouter"
    llm = get_llm(llm_id=MODEL_ID)

    filesystem_mcp_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd().parent)],
    )

    timeserver_mcp_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/mcp-server-time"],
    )

    memory_mcp_params = StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-memory"])

    arxiv_mcp_params = StdioServerParameters(
        command="uv",
        args=["tool", "run", "@smithery/arxiv-mcp-server", "--storage-path", "/tmp"],
    )
    pubmed_mcp_params = StdioServerParameters(
        command="uvx",
        args=["--quiet", "pubmedmcp@0.1.3"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )

    async def main():
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

        async for event in mcp_agent_runner(
            llm, [memory_mcp_params, filesystem_mcp_params], "List content of the current directory.", {}
        ):
            print(event)

    asyncio.run(main())
