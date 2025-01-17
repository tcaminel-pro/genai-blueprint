"""
MCP (Model Context Protocol) tool calling from LangChain (WIP !!)

Adapted from : https://github.com/rectalogic/langchain-mcp

"""

from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from mcpadapt.langchain_adapter import LangChainAdapter

from python.ai_core.llm import get_llm


async def mcp_agent_runner(model, server_params: StdioServerParameters, prompt, config: RunnableConfig = {}):
    async with MCPAdapt(
        server_params,
        LangChainAdapter(),
    ) as tools:
        # Create the agent
        memory = MemorySaver()
        agent_executor = create_react_agent(model, tools, checkpointer=memory)

        async for event in agent_executor.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config,
        ):
            yield(event)

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

    memory_mcp_params = StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-memory"])

    arxiv_mcp_params = StdioServerParameters(
        command="uv",
        args=["tool", "run", "@smithery/arxiv-mcp-server", "--storage-path", "/tmp"],
    )
    pubmed_mcp_params = StdioServerParameters(
        command="uvx",
        args=["--quiet", "pubmedmcp@0.1.3"],
        env={"UV_PYTHON": "3.12"},
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
        # debug(message.pretty_print())

        # 
        # display result of mcp_agent_runner  (possibly change it)  AI!
        mcp_agent_runner(llm, filesystem_mcp_params, "list the current directory", {})

        r = await 

    asyncio.run(main())
