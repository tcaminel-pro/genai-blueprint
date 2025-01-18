"""An example of using the LangChain adapter to adapt MCP tools to LangChain tools.

This example uses the PubMed API to search for studies.
"""

import os
from pathlib import Path

# Import relevant functionality
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from mcpadapt.langchain_adapter import LangChainAdapter

from python.ai_core.llm import get_llm

MODEL_ID = "llama31_8_groq"
MODEL_ID = "claude_haiku35_openrouter"

model = get_llm(llm_id=MODEL_ID)


pubmed_mcp_params = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)


filesystem_mcp_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd().parent)],
)


async def async_main():
    """Fully asynchronous version example."""
    async with MCPAdapt(

        pubmed_mcp_params,
        LangChainAdapter(),
    ) as tools:
        # Create the agent
        memory = MemorySaver()
        agent_executor = create_react_agent(model, tools, checkpointer=memory)

        # Use the agent
        config = {"configurable": {"thread_id": "abc123"}}
        async for event in agent_executor.astream(
            {"messages": [HumanMessage(content="Find relevant studies on alcohol hangover and treatment.")]},
            config,
        ):
            print(event)
            print("----")

        async for event in agent_executor.astream(
            {"messages": [HumanMessage(content="whats the weather where I live?")]},
            config,
        ):
            print(event)
            print("----")


if __name__ == "__main__":
    import asyncio

    # run the sync version
    # main()

    # run the async version
    asyncio.run(async_main())
